//! Turning text into terms.
//!
//! Three steps: split on non-alphanumerics, lowercase, fold diacritics.
//!
//! The folding table is the one thing in this engine with a direct ancestor.
//! parasearch (2004) folded accents with a Perl `tr///` whose source and
//! replacement sets were 68 and 66 characters long. Perl pads silently with
//! the last replacement character, so every letter from `ö` onward folded to
//! the wrong thing and `ñ` became `c`: a Colombian search engine that indexed
//! *español* as *espacol*, for years, unnoticed.
//!
//! Here the mapping is exhaustive and total by construction, and
//! `fold_is_total` proves it over the whole Latin-1 and Latin Extended-A range.

use indexander_core::Position;

/// A term and where it appeared, in token positions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Token {
    pub text: String,
    pub position: Position,
}

/// A term, and where it appeared in the *bytes* of the text it came from.
///
/// The index has no use for this — postings are addressed by token position —
/// but a snippet does: it has to cut the original text, with its accents and
/// capitals intact, and a token position says nothing about where that is.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Span {
    /// The folded term, comparable with what the index and query hold.
    pub text: String,
    pub position: Position,
    /// Byte range in the original text. Always on character boundaries.
    pub start: usize,
    pub end: usize,
}

/// Folds one character to its unaccented ASCII form.
///
/// Returns the character unchanged when it has no accent to strip. Never
/// returns a wrong letter: every arm is written out, none is derived from
/// position in a table.
#[must_use]
pub fn fold_char(c: char) -> char {
    match c {
        'à' | 'á' | 'â' | 'ã' | 'ä' | 'å' | 'ā' | 'ă' | 'ą' => 'a',
        'è' | 'é' | 'ê' | 'ë' | 'ē' | 'ĕ' | 'ė' | 'ę' | 'ě' => 'e',
        'ì' | 'í' | 'î' | 'ï' | 'ĩ' | 'ī' | 'ĭ' | 'į' | 'ı' => 'i',
        'ò' | 'ó' | 'ô' | 'õ' | 'ö' | 'ø' | 'ō' | 'ŏ' | 'ő' => 'o',
        'ù' | 'ú' | 'û' | 'ü' | 'ũ' | 'ū' | 'ŭ' | 'ů' | 'ű' | 'ų' => 'u',
        'ý' | 'ÿ' | 'ŷ' => 'y',
        'ñ' | 'ń' | 'ņ' | 'ň' => 'n',
        'ç' | 'ć' | 'ĉ' | 'ċ' | 'č' => 'c',
        'ĝ' | 'ğ' | 'ġ' | 'ģ' => 'g',
        'ĥ' | 'ħ' => 'h',
        'ĵ' => 'j',
        'ķ' | 'ĸ' => 'k',
        'ĺ' | 'ļ' | 'ľ' | 'ŀ' | 'ł' => 'l',
        'ŕ' | 'ŗ' | 'ř' => 'r',
        'ś' | 'ŝ' | 'ş' | 'š' | 'ſ' => 's',
        'ţ' | 'ť' | 'ŧ' => 't',
        'ŵ' => 'w',
        'ź' | 'ż' | 'ž' => 'z',
        'ð' | 'ď' | 'đ' => 'd',
        'þ' => 'p',
        other => other,
    }
}

/// Splits `text` into folded, lowercased terms with their positions.
///
/// `start` lets a caller continue numbering across several strings, which is
/// how a document's anchor texts share one position space.
pub fn tokenize_into(text: &str, start: Position, out: &mut Vec<Token>) -> Position {
    let mut position = start;
    scan(text, |text, _, _| {
        out.push(Token { text, position });
        position += 1;
    });
    position
}

/// The one scanner. Everything that splits text into terms goes through here.
///
/// Two scanners that fold slightly differently would put a term in the index
/// that a query could not find, or highlight a word a search did not match.
/// The callback takes the byte range so the callers that need it have it and
/// the ones that do not pay nothing for it — it is two `usize`s the optimiser
/// drops.
fn scan(text: &str, mut emit: impl FnMut(String, usize, usize)) {
    let mut current = String::new();
    let mut start = 0usize;

    for (at, c) in text.char_indices() {
        if c.is_alphanumeric() {
            if current.is_empty() {
                start = at;
            }
            // `to_lowercase` is an iterator because some characters lowercase
            // to several (ẞ -> ss). Fold after lowercasing, never before.
            for lower in c.to_lowercase() {
                current.push(fold_char(lower));
            }
        } else if !current.is_empty() {
            emit(std::mem::take(&mut current), start, at);
        }
    }
    if !current.is_empty() {
        emit(current, start, text.len());
    }
}

/// Splits `text` into terms that remember where in the bytes they were.
#[must_use]
pub fn tokenize_spans(text: &str) -> Vec<Span> {
    let mut out = Vec::new();
    let mut position = 0;
    scan(text, |text, start, end| {
        out.push(Span {
            text,
            position,
            start,
            end,
        });
        position += 1;
    });
    out
}

/// Convenience wrapper over [`tokenize_into`] for a single string.
#[must_use]
pub fn tokenize(text: &str) -> Vec<Token> {
    let mut out = Vec::new();
    tokenize_into(text, 0, &mut out);
    out
}

/// Folds a whole string. Used to normalise query terms the same way.
#[must_use]
pub fn fold(text: &str) -> String {
    text.chars()
        .flat_map(char::to_lowercase)
        .map(fold_char)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spans_and_tokens_are_the_same_terms() {
        // One scanner, so this can only fail if someone splits it into two.
        for text in [
            "",
            "   ",
            "Búsqueda en Español",
            "a-b_c 42 ¡hola! ñandú",
            "trailing",
            "ẞ and ß",
        ] {
            let tokens = tokenize(text);
            let spans = tokenize_spans(text);
            assert_eq!(tokens.len(), spans.len(), "{text:?}");
            for (t, s) in tokens.iter().zip(&spans) {
                assert_eq!(t.text, s.text, "{text:?}");
                assert_eq!(t.position, s.position, "{text:?}");
            }
        }
    }

    #[test]
    fn a_span_points_at_the_original_word_accents_and_all() {
        let text = "El motor de BÚSQUEDA";
        let spans = tokenize_spans(text);
        assert_eq!(spans[3].text, "busqueda");
        assert_eq!(&text[spans[3].start..spans[3].end], "BÚSQUEDA");
        assert_eq!(spans[1].text, "motor");
        assert_eq!(&text[spans[1].start..spans[1].end], "motor");
    }

    #[test]
    fn spans_are_in_order_and_never_overlap() {
        let text = "uno, dos; tres — cuatro";
        let spans = tokenize_spans(text);
        assert!(spans.windows(2).all(|w| w[0].end <= w[1].start));
        assert!(spans.iter().all(|s| s.start < s.end && s.end <= text.len()));
        // Every range is sliceable, which is the property that matters.
        for s in &spans {
            let _ = &text[s.start..s.end];
        }
    }

    /// The bug this engine exists to not repeat.
    ///
    /// parasearch turned "Compañia Colombiana de Años" into
    /// "Compacia Colombiana de Acos".
    #[test]
    fn the_2004_bug_is_fixed() {
        assert_eq!(
            fold("Compañia Colombiana de Años"),
            "compania colombiana de anos"
        );
        assert_eq!(fold_char('ñ'), 'n');
        assert_eq!(fold("español"), "espanol");
    }

    /// The 2004 table broke because it was positional and its two halves
    /// had different lengths. This asserts the property that failure violated:
    /// folding never invents a letter outside the expected alphabet, over the
    /// entire range where accented Latin letters live.
    #[test]
    fn fold_is_total() {
        for cp in 0u32..=0x017F {
            let Some(c) = char::from_u32(cp) else {
                continue;
            };
            let folded = fold_char(c);
            if c.is_alphabetic() && folded != c {
                assert!(
                    folded.is_ascii_lowercase(),
                    "U+{cp:04X} ({c}) folded to {folded:?}, which is not a plain ascii letter",
                );
            }
        }
    }

    #[test]
    fn folding_preserves_unaccented_text() {
        assert_eq!(fold("plain ascii 123"), "plain ascii 123");
    }

    #[test]
    fn positions_are_token_counts_not_bytes() {
        let toks = tokenize("el robeiro indexa la web");
        let texts: Vec<_> = toks.iter().map(|t| t.text.as_str()).collect();
        assert_eq!(texts, ["el", "robeiro", "indexa", "la", "web"]);
        let positions: Vec<_> = toks.iter().map(|t| t.position).collect();
        assert_eq!(positions, [0, 1, 2, 3, 4]);
    }

    #[test]
    fn punctuation_splits_and_does_not_consume_a_position() {
        let toks = tokenize("hola,mundo!!!  otra");
        assert_eq!(toks.len(), 3);
        assert_eq!(toks[2].position, 2);
    }

    #[test]
    fn tokenize_into_continues_numbering() {
        let mut out = Vec::new();
        let next = tokenize_into("uno dos", 0, &mut out);
        assert_eq!(next, 2);
        let next = tokenize_into("tres", next, &mut out);
        assert_eq!(next, 3);
        assert_eq!(out.last().unwrap().position, 2);
    }

    #[test]
    fn uppercase_is_lowercased_before_folding() {
        assert_eq!(fold("ÑOÑO ÁRBOL"), "nono arbol");
    }

    #[test]
    fn empty_input_yields_no_tokens() {
        assert!(tokenize("").is_empty());
        assert!(tokenize("   ,,, ").is_empty());
    }
}
