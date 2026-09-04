//! The two lines under a result that decide whether anyone clicks it.
//!
//! A ranked list of URIs is not a search engine's output, it is its
//! intermediate state. What a person reads is an extract: the part of the
//! document that made it match, with their words visible in it.
//!
//! This works on the document's text, not on the index. The index stores
//! positions in *tokens*, which say nothing about where a word is in the bytes
//! — and even if it did, the folded term `busqueda` is not what anyone wants
//! to read where the document says `BÚSQUEDA`. So a snippet is cut from the
//! original text, and the only thing shared with the index is the tokenizer,
//! which is why highlighting agrees with matching instead of nearly agreeing.
//!
//! Nothing here is stored. That is a choice with a cost: the caller must be
//! able to get the text back, which for a file corpus means reading the file
//! and for a crawl means it cannot be done at all until there is a document
//! store. Keeping the text in the segment would roughly quadruple it — the
//! index is a third of the text it indexes — for a feature used on ten results
//! out of a hundred thousand.

use crate::tokenizer::{Span, fold, tokenize_spans};

/// An extract, and where the query's words are inside it.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Snippet {
    /// The extract, with ellipses where it was cut.
    pub text: String,
    /// Byte ranges *within `text`* to highlight, in order and non-overlapping.
    pub highlights: Vec<(usize, usize)>,
}

impl Snippet {
    /// The extract with each highlight wrapped, for a caller with no richer
    /// way to show one.
    #[must_use]
    pub fn wrap(&self, before: &str, after: &str) -> String {
        let mut out = String::with_capacity(self.text.len());
        let mut at = 0;
        for &(start, end) in &self.highlights {
            out.push_str(&self.text[at..start]);
            out.push_str(before);
            out.push_str(&self.text[start..end]);
            out.push_str(after);
            at = end;
        }
        out.push_str(&self.text[at..]);
        out
    }
}

const ELLIPSIS: &str = "…";

/// At most this many fragments, however many words were searched for.
///
/// Three ellipsis-separated scraps is already at the edge of readable; a
/// fourth stops being an extract and starts being a concordance.
const MAX_FRAGMENTS: usize = 3;

/// The best `width` bytes of `text` for a search for `terms`.
///
/// "Best" is the window covering the most *distinct* query terms, then the
/// most occurrences, then the earliest — distinct first because a window
/// showing both words a person typed tells them more than one showing the
/// first word four times.
///
/// When no single window can hold every word — which on real documents is the
/// usual case, not the exceptional one — the extract is several fragments
/// joined by ellipses, each chosen to bring a word the others did not have.
/// A two-word search whose snippet shows one of the words has not answered the
/// question it was asked.
///
/// A document containing none of the terms still gets a snippet: its opening.
/// Returning nothing would be correct and useless, and it happens more often
/// than it should — a document can match on its title or its anchor text
/// while its body never says the word.
#[must_use]
pub fn best(text: &str, terms: &[String], width: usize) -> Snippet {
    let wanted: Vec<String> = terms
        .iter()
        .map(|t| fold(t))
        .filter(|t| !t.is_empty())
        .collect();
    if wanted.is_empty() || width == 0 {
        return lead(text, width);
    }

    let spans = tokenize_spans(text);
    let matched: Vec<usize> = spans
        .iter()
        .enumerate()
        .filter(|(_, s)| wanted.contains(&s.text))
        .map(|(i, _)| i)
        .collect();
    if matched.is_empty() {
        return lead(text, width);
    }

    // One fragment per word actually present, capped. Splitting the budget
    // more finely than that would shrink every fragment to buy nothing.
    let mut present: Vec<&str> = matched.iter().map(|&i| spans[i].text.as_str()).collect();
    present.sort_unstable();
    present.dedup();
    let slots = present.len().clamp(1, MAX_FRAGMENTS);

    let mut windows: Vec<(usize, usize)> = Vec::new();
    let mut covered: Vec<&str> = Vec::new();
    for _ in 0..slots {
        let remaining: Vec<String> = present
            .iter()
            .filter(|t| !covered.contains(*t))
            .map(|t| (*t).to_owned())
            .collect();
        if remaining.is_empty() {
            break;
        }
        // Only matches outside the fragments already chosen: a second window
        // over the same words would spend the budget saying nothing new.
        let free: Vec<usize> = matched
            .iter()
            .copied()
            .filter(|i| !windows.iter().any(|&(a, b)| *i >= a && *i <= b))
            .collect();
        let Some(window) = best_window(&spans, &free, &remaining, width / slots) else {
            break;
        };
        for span in &spans[window.0..=window.1] {
            let term = span.text.as_str();
            if present.contains(&term) && !covered.contains(&term) {
                covered.push(term);
            }
        }
        windows.push(window);
        windows.sort_unstable();
    }

    if windows.is_empty() {
        return lead(text, width);
    }
    cut(text, &spans, &windows, &matched, width / windows.len())
}

/// The span range covering the most distinct terms within `width` bytes.
fn best_window(
    spans: &[Span],
    matched: &[usize],
    wanted: &[String],
    width: usize,
) -> Option<(usize, usize)> {
    if matched.is_empty() {
        return None;
    }
    let mut best: Option<(usize, usize, usize, usize)> = None; // distinct, hits, first, last
    for (n, &first) in matched.iter().enumerate() {
        let from = spans[first].start;
        let mut last = first;
        let mut hits = 0usize;
        let mut seen: Vec<&str> = Vec::new();
        for &i in &matched[n..] {
            if spans[i].end.saturating_sub(from) > width {
                break;
            }
            last = i;
            hits += 1;
            let term = spans[i].text.as_str();
            if wanted.iter().any(|w| w == term) && !seen.contains(&term) {
                seen.push(term);
            }
        }
        let candidate = (seen.len(), hits, first, last);
        // Earliest wins ties, so the same document and query always produce
        // the same snippet.
        if best.is_none_or(|(d, h, _, _)| (candidate.0, candidate.1) > (d, h)) {
            best = Some(candidate);
        }
    }
    best.map(|(_, _, first, last)| (first, last))
}

/// Cuts the text around each span window, on character boundaries, with
/// highlights recorded relative to the finished extract.
///
/// Windows arrive sorted. Overlapping byte ranges are merged rather than
/// printed twice, which happens when two windows chosen for different words
/// turn out to sit next to each other.
fn cut(
    text: &str,
    spans: &[Span],
    windows: &[(usize, usize)],
    matched: &[usize],
    width: usize,
) -> Snippet {
    let mut ranges: Vec<(usize, usize)> = Vec::new();
    for &(first, last) in windows {
        let hit_start = spans[first].start;
        let hit_end = spans[last].end;
        // Any room left over goes in front of the first match, so the extract
        // starts mid-sentence rather than mid-word on the match itself.
        let slack = width.saturating_sub(hit_end - hit_start);
        let start = boundary(text, hit_start.saturating_sub(slack / 2), true);
        let end = boundary(text, (hit_end + slack.div_ceil(2)).min(text.len()), false);
        match ranges.last_mut() {
            Some(previous) if start <= previous.1 => previous.1 = previous.1.max(end),
            _ => ranges.push((start, end)),
        }
    }

    let mut out = String::new();
    let mut highlights = Vec::new();
    for (n, &(start, end)) in ranges.iter().enumerate() {
        if start > 0 || n > 0 {
            out.push_str(ELLIPSIS);
        }
        let offset = out.len();
        let piece = text[start..end].trim_end();
        out.push_str(piece);
        highlights.extend(
            matched
                .iter()
                .map(|&i| &spans[i])
                .filter(|s| s.start >= start && s.end <= end)
                .map(|s| (offset + s.start - start, offset + s.end - start))
                .filter(|&(_, e)| e <= offset + piece.len()),
        );
    }
    if ranges.last().is_some_and(|&(_, end)| end < text.len()) {
        out.push_str(ELLIPSIS);
    }
    Snippet {
        text: out,
        highlights,
    }
}

/// The opening of a document, for when nothing matched in its body.
fn lead(text: &str, width: usize) -> Snippet {
    if width == 0 || text.is_empty() {
        // A budget of nothing buys nothing. An ellipsis here would be three
        // bytes spent saying that zero bytes were omitted.
        return Snippet::default();
    }
    let end = boundary(text, width.min(text.len()), false);
    let mut out = text[..end].trim().to_owned();
    if end < text.len() {
        out.push_str(ELLIPSIS);
    }
    Snippet {
        text: out,
        highlights: Vec::new(),
    }
}

/// Moves `at` to a sensible cut point: a character boundary, and a word
/// boundary when one is close enough to be worth reaching for.
///
/// `backwards` decides which way to look for the space. Cutting a word in half
/// is not wrong so much as it looks like a bug in whatever displays it.
fn boundary(text: &str, mut at: usize, backwards: bool) -> usize {
    /// How far to look for whitespace before giving up and cutting mid-word.
    const REACH: usize = 30;

    at = at.min(text.len());
    while at > 0 && at < text.len() && !text.is_char_boundary(at) {
        at += 1;
    }
    if at == 0 || at >= text.len() {
        return at.min(text.len());
    }
    // Look a little way for whitespace, and keep the character cut if there
    // is none - a language without spaces should get a slightly ragged
    // snippet, not none.
    //
    // The reach itself has to land on a character boundary before it can be
    // used as a slice index. `at` is one by construction; `at ± REACH` is not,
    // and slicing at it panics on the first accented word.
    let window = if backwards {
        let from = floor(text, at.saturating_sub(REACH));
        text[from..at]
            .rfind(char::is_whitespace)
            .map(|i| from + i + 1)
    } else {
        let to = ceil(text, (at + REACH).min(text.len()));
        text[at..to].find(char::is_whitespace).map(|i| at + i)
    };
    window.filter(|&i| text.is_char_boundary(i)).unwrap_or(at)
}

/// The character boundary at or below `at`.
fn floor(text: &str, mut at: usize) -> usize {
    at = at.min(text.len());
    while at > 0 && !text.is_char_boundary(at) {
        at -= 1;
    }
    at
}

/// The character boundary at or above `at`.
fn ceil(text: &str, mut at: usize) -> usize {
    at = at.min(text.len());
    while at < text.len() && !text.is_char_boundary(at) {
        at += 1;
    }
    at
}

#[cfg(test)]
mod tests {
    use super::*;

    fn terms(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| (*s).to_owned()).collect()
    }

    #[test]
    fn the_snippet_contains_the_word_that_was_searched_for() {
        let text = "Un texto largo de relleno que sigue y sigue. Aqui aparece la palabra \
                    busqueda, y despues continua con mas relleno que no interesa a nadie.";
        let s = best(text, &terms(&["busqueda"]), 60);
        assert!(s.text.contains("busqueda"), "{:?}", s.text);
        assert_eq!(s.highlights.len(), 1);
        let (a, b) = s.highlights[0];
        assert_eq!(&s.text[a..b], "busqueda");
    }

    #[test]
    fn highlights_point_at_the_original_spelling_not_the_folded_one() {
        let text = "El motor de BÚSQUEDA de Conexcol";
        let s = best(text, &terms(&["busqueda"]), 100);
        let (a, b) = s.highlights[0];
        assert_eq!(&s.text[a..b], "BÚSQUEDA");
    }

    #[test]
    fn a_query_typed_with_accents_still_matches() {
        let text = "El motor de busqueda";
        let s = best(text, &terms(&["BÚSQUEDA"]), 100);
        assert_eq!(s.highlights.len(), 1, "{s:?}");
    }

    #[test]
    fn both_words_appear_even_when_no_single_window_holds_them() {
        // The common case on a real document, and the one a single window
        // gets wrong: the two words are far apart, so an extract that keeps
        // only the best window shows one of them and drops the other.
        let text = format!(
            "El termino perl aparece aqui al principio del documento. {} \
             Y mucho despues, sin ninguna relacion, aparece motor.",
            "relleno ".repeat(60)
        );
        let s = best(&text, &terms(&["perl", "motor"]), 120);
        assert!(s.text.contains("perl"), "{:?}", s.text);
        assert!(s.text.contains("motor"), "{:?}", s.text);
        assert_eq!(s.highlights.len(), 2, "{s:?}");
        for &(a, b) in &s.highlights {
            let word = &s.text[a..b];
            assert!(word == "perl" || word == "motor", "highlighted {word:?}");
        }
    }

    #[test]
    fn a_snippet_is_never_more_than_three_fragments() {
        let mut text = String::new();
        for i in 0..8 {
            use std::fmt::Write as _;
            let _ = write!(text, "palabra{i} {}", "relleno ".repeat(40));
        }
        let all: Vec<String> = (0..8).map(|i| format!("palabra{i}")).collect();
        let s = best(&text, &all, 300);
        // Fragments are separated by ellipses; at most three fragments means
        // at most four ellipses counting the two outer ones.
        assert!(
            s.text.matches(ELLIPSIS).count() <= 4,
            "too many fragments: {:?}",
            s.text
        );
    }

    #[test]
    fn the_window_prefers_covering_both_words_over_repeating_one() {
        let text = "perl perl perl perl perl. \
                    Relleno intermedio suficientemente largo para separar las dos zonas del \
                    documento y forzar una eleccion. \
                    Aqui estan perl y motor juntos.";
        let s = best(text, &terms(&["perl", "motor"]), 50);
        assert!(
            s.text.contains("motor"),
            "chose the wrong window: {:?}",
            s.text
        );
        assert!(s.text.contains("perl"), "{:?}", s.text);
    }

    #[test]
    fn a_document_that_never_says_the_word_still_gets_an_opening() {
        // It matched on its title or an anchor; a result with no snippet at
        // all looks like a broken result.
        let text = "Este documento habla de otra cosa completamente distinta y no menciona \
                    el termino en ningun momento de su cuerpo.";
        let s = best(text, &terms(&["kubernetes"]), 40);
        assert!(!s.text.is_empty());
        assert!(s.highlights.is_empty());
        assert!(s.text.starts_with("Este documento"), "{:?}", s.text);
        assert!(s.text.ends_with(ELLIPSIS), "{:?}", s.text);
    }

    #[test]
    fn a_short_document_is_shown_whole_without_ellipses() {
        let s = best("motor de busqueda", &terms(&["motor"]), 200);
        assert_eq!(s.text, "motor de busqueda");
        assert!(!s.text.contains(ELLIPSIS));
    }

    #[test]
    fn an_empty_document_does_not_panic() {
        assert_eq!(best("", &terms(&["motor"]), 50), Snippet::default());
        assert_eq!(best("", &[], 50), Snippet::default());
        assert_eq!(best("algo", &terms(&["motor"]), 0).text, "");
    }

    #[test]
    fn every_highlight_is_inside_the_snippet_and_on_a_boundary() {
        let text = "ñandú motor ñandú busqueda ñandú motor ñandú busqueda ñandú motor \
                    ñandú busqueda ñandú motor ñandú busqueda ñandú motor ñandú";
        for width in [10, 20, 40, 80, 160, 500] {
            let s = best(text, &terms(&["motor", "busqueda"]), width);
            for &(a, b) in &s.highlights {
                assert!(b <= s.text.len(), "width {width}: {a}..{b} of {:?}", s.text);
                assert!(s.text.is_char_boundary(a) && s.text.is_char_boundary(b));
                let word = &s.text[a..b];
                assert!(
                    word.eq_ignore_ascii_case("motor") || word.eq_ignore_ascii_case("busqueda"),
                    "width {width} highlighted {word:?}"
                );
            }
            assert!(
                s.highlights.windows(2).all(|w| w[0].1 <= w[1].0),
                "overlapping highlights at width {width}"
            );
        }
    }

    #[test]
    fn wrapping_puts_the_markers_around_the_words() {
        let s = best("el motor de busqueda", &terms(&["motor"]), 100);
        assert_eq!(s.wrap("[", "]"), "el [motor] de busqueda");
    }

    #[test]
    fn the_same_text_and_query_always_give_the_same_snippet() {
        let text = "motor aqui y motor alla y motor mas alla, todos iguales de buenos";
        let first = best(text, &terms(&["motor"]), 30);
        for _ in 0..20 {
            assert_eq!(best(text, &terms(&["motor"]), 30), first);
        }
    }

    #[test]
    fn a_snippet_does_not_split_a_multibyte_character() {
        // Every cut point in a text made entirely of multi-byte characters.
        let text = "ñ".repeat(200);
        for width in 1..60 {
            let s = best(&text, &terms(&["x"]), width);
            assert!(
                s.text.chars().all(|c| c == 'ñ' || c == '…'),
                "width {width}"
            );
        }
    }
}
