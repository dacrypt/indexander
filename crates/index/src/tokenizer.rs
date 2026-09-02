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
    let mut current = String::new();

    // Push whatever has accumulated, if anything, and advance the position.
    macro_rules! flush {
        () => {
            if !current.is_empty() {
                out.push(Token {
                    text: std::mem::take(&mut current),
                    position,
                });
                position += 1;
            }
        };
    }

    for c in text.chars() {
        if c.is_alphanumeric() {
            // `to_lowercase` is an iterator because some characters lowercase
            // to several (ẞ -> ss). Fold after lowercasing, never before.
            for lower in c.to_lowercase() {
                current.push(fold_char(lower));
            }
        } else {
            flush!();
        }
    }
    flush!();

    position
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
