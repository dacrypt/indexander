//! Parsing what a person typed into something the engine can execute.
//!
//! The syntax is deliberately small, and it is the syntax people already
//! expect from a search box:
//!
//! ```text
//! motor de busqueda      every term must appear
//! "motor de busqueda"    the words must appear adjacent, in order
//! -perl                  documents containing this term are dropped
//! ```

use crate::tokenizer::fold;

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Query {
    /// Terms every result must contain.
    pub required: Vec<String>,
    /// Terms no result may contain.
    pub excluded: Vec<String>,
    /// Sequences that must appear adjacent and in order, within one field.
    pub phrases: Vec<Vec<String>>,
}

impl Query {
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.required.is_empty() && self.phrases.is_empty()
    }

    /// Every term the query cares about positively, for scoring.
    #[must_use]
    pub fn scoring_terms(&self) -> Vec<String> {
        let mut terms = self.required.clone();
        for phrase in &self.phrases {
            terms.extend(phrase.iter().cloned());
        }
        terms.sort_unstable();
        terms.dedup();
        terms
    }
}

/// Parses a query string. Never fails: anything unparseable is treated as text,
/// which is what a search box should do.
///
/// # Panics
///
/// Never. The one `expect` inside is guarded by the `match` arm that proves
/// the vector has exactly one element.
#[must_use]
pub fn parse(input: &str) -> Query {
    let mut query = Query::default();
    let mut chars = input.chars().peekable();
    let mut current = String::new();
    let mut negated = false;

    // Flush the accumulated word into the right bucket.
    macro_rules! flush_word {
        () => {
            if !current.is_empty() {
                let folded = fold(&current);
                current.clear();
                for term in folded.split(|c: char| !c.is_alphanumeric()) {
                    if term.is_empty() {
                        continue;
                    }
                    if negated {
                        query.excluded.push(term.to_owned());
                    } else {
                        query.required.push(term.to_owned());
                    }
                }
                negated = false;
            }
        };
    }

    while let Some(c) = chars.next() {
        match c {
            '"' => {
                flush_word!();
                let mut phrase = String::new();
                for c in chars.by_ref() {
                    if c == '"' {
                        break;
                    }
                    phrase.push(c);
                }
                let terms: Vec<String> = fold(&phrase)
                    .split(|c: char| !c.is_alphanumeric())
                    .filter(|t| !t.is_empty())
                    .map(str::to_owned)
                    .collect();
                match terms.len() {
                    0 => {}
                    // A one-word "phrase" is just a term.
                    1 => query
                        .required
                        .push(terms.into_iter().next().expect("len 1")),
                    _ => query.phrases.push(terms),
                }
            }
            '-' if current.is_empty() => negated = true,
            c if c.is_whitespace() => flush_word!(),
            c => current.push(c),
        }
    }
    flush_word!();
    let _ = negated;

    query.required.dedup();
    query
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bare_terms_are_all_required() {
        let q = parse("motor de busqueda");
        assert_eq!(q.required, ["motor", "de", "busqueda"]);
        assert!(q.excluded.is_empty());
        assert!(q.phrases.is_empty());
    }

    #[test]
    fn accents_and_case_are_folded_like_the_index() {
        let q = parse("BÚSQUEDA en Español");
        assert_eq!(q.required, ["busqueda", "en", "espanol"]);
    }

    #[test]
    fn quotes_make_a_phrase() {
        let q = parse(r#"perl "motor de busqueda" 2004"#);
        assert_eq!(q.phrases, [["motor", "de", "busqueda"]]);
        assert_eq!(q.required, ["perl", "2004"]);
    }

    #[test]
    fn a_one_word_phrase_is_just_a_term() {
        let q = parse(r#""perl""#);
        assert!(q.phrases.is_empty());
        assert_eq!(q.required, ["perl"]);
    }

    #[test]
    fn minus_excludes() {
        let q = parse("buscador -google");
        assert_eq!(q.required, ["buscador"]);
        assert_eq!(q.excluded, ["google"]);
    }

    #[test]
    fn a_hyphen_inside_a_word_does_not_negate() {
        let q = parse("anchor-text");
        assert_eq!(q.required, ["anchor", "text"]);
        assert!(q.excluded.is_empty());
    }

    #[test]
    fn an_unclosed_quote_still_parses() {
        let q = parse(r#"perl "motor de"#);
        assert_eq!(q.phrases, [["motor", "de"]]);
    }

    #[test]
    fn empty_query_is_empty() {
        assert!(parse("").is_empty());
        assert!(parse("   ").is_empty());
        assert!(parse("-only-negatives").is_empty());
    }

    #[test]
    fn scoring_terms_include_phrase_words_without_duplicates() {
        let q = parse(r#"perl "perl moderno""#);
        assert_eq!(q.scoring_terms(), ["moderno", "perl"]);
    }
}
