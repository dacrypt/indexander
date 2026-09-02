//! Pulling text and links out of HTML.
//!
//! This is a scanner, not a parser: it never builds a tree, it walks the bytes
//! once and keeps what a crawler needs — the title, the visible text, the
//! links, and crucially the *anchor text* of each link, which is what a page
//! says about the page it points to.
//!
//! Real HTML is malformed, so nothing here can fail. Unclosed tags, stray
//! angle brackets and mismatched quotes all have to produce something usable,
//! because the alternative is dropping the document. Every one of those cases
//! has a test.

/// A link found in a document, with the words that pointed at it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Link {
    /// The raw `href`, still relative and undecoded. Resolving it needs the
    /// page's own URL, which this module does not know about.
    pub href: String,
    /// The visible text between `<a>` and `</a>`, collapsed to single spaces.
    pub text: String,
}

/// Everything the crawler wants from one HTML document.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Extracted {
    pub title: String,
    pub text: String,
    pub links: Vec<Link>,
    /// From `<base href="...">`, if present: relative links resolve against
    /// this instead of the document's own URL.
    pub base: Option<String>,
    /// `<meta name="robots" content="noindex">` — do not index this page.
    pub noindex: bool,
    /// `<meta name="robots" content="nofollow">` — do not follow its links.
    pub nofollow: bool,
}

/// Elements whose contents are code or styling, never prose.
const SKIPPED: [&str; 5] = ["script", "style", "noscript", "template", "svg"];

/// Scans `html` once, collecting text, links and metadata.
#[must_use]
#[allow(clippy::too_many_lines)]
pub fn extract(html: &str) -> Extracted {
    let mut out = Extracted::default();
    let bytes = html.as_bytes();
    let mut i = 0usize;

    // Where the text of the currently open <a> started, if one is open.
    let mut anchor: Option<(String, usize)> = None;
    let mut in_title = false;
    let mut title = String::new();
    let mut text = String::new();

    while i < bytes.len() {
        if bytes[i] != b'<' {
            // Ordinary text. Runs of whitespace collapse to one space so that
            // markup-driven line breaks do not become part of the content.
            let start = i;
            while i < bytes.len() && bytes[i] != b'<' {
                i += 1;
            }
            let chunk = &html[start..i];
            push_collapsed(chunk, &mut text);
            if in_title {
                push_collapsed(chunk, &mut title);
            }
            continue;
        }

        // A comment or a doctype: skip to its end, whatever it contains.
        if html[i..].starts_with("<!--") {
            i = html[i + 4..]
                .find("-->")
                .map_or(bytes.len(), |at| i + 4 + at + 3);
            continue;
        }
        if html[i..].starts_with("<!") {
            i = html[i..].find('>').map_or(bytes.len(), |at| i + at + 1);
            continue;
        }

        let Some(end) = html[i..].find('>').map(|at| i + at) else {
            // An unclosed tag at end of input: treat the rest as text, which
            // is what browsers do and what keeps the content.
            push_collapsed(&html[i..], &mut text);
            break;
        };
        let tag = &html[i + 1..end];
        i = end + 1;

        let closing = tag.starts_with('/');
        let name_source = if closing { &tag[1..] } else { tag };
        let name: String = name_source
            .chars()
            .take_while(char::is_ascii_alphanumeric)
            .collect::<String>()
            .to_ascii_lowercase();

        if name.is_empty() {
            continue;
        }

        if !closing && SKIPPED.contains(&name.as_str()) {
            // Skip to the matching close tag; its contents are not prose.
            let close = format!("</{name}");
            i = html[i..]
                .to_ascii_lowercase()
                .find(&close)
                .map_or(bytes.len(), |at| {
                    let from = i + at;
                    html[from..].find('>').map_or(bytes.len(), |g| from + g + 1)
                });
            continue;
        }

        match (name.as_str(), closing) {
            ("title", false) => in_title = true,
            ("title", true) => in_title = false,
            ("base", false) => {
                if let Some(href) = attribute(tag, "href") {
                    out.base.get_or_insert(href);
                }
            }
            ("meta", false) => {
                let is_robots =
                    attribute(tag, "name").is_some_and(|n| n.eq_ignore_ascii_case("robots"));
                if is_robots && let Some(content) = attribute(tag, "content") {
                    let content = content.to_ascii_lowercase();
                    out.noindex |= content.contains("noindex") || content.contains("none");
                    out.nofollow |= content.contains("nofollow") || content.contains("none");
                }
            }
            ("a", false) => {
                // An <a> inside an unclosed <a> closes the previous one.
                if let Some((href, from)) = anchor.take() {
                    out.links.push(Link {
                        href,
                        text: text[from.min(text.len())..].trim().to_owned(),
                    });
                }
                if let Some(href) = attribute(tag, "href") {
                    let rel_nofollow = attribute(tag, "rel")
                        .is_some_and(|r| r.to_ascii_lowercase().contains("nofollow"));
                    if !rel_nofollow {
                        anchor = Some((href, text.len()));
                    }
                }
            }
            ("a", true) => {
                if let Some((href, from)) = anchor.take() {
                    out.links.push(Link {
                        href,
                        text: text[from.min(text.len())..].trim().to_owned(),
                    });
                }
            }
            // Block-level elements separate words that would otherwise run
            // together: "<p>foo</p><p>bar</p>" is two words, not "foobar".
            (
                "p" | "div" | "br" | "li" | "tr" | "td" | "h1" | "h2" | "h3" | "h4" | "h5" | "h6"
                | "section" | "article" | "header" | "footer" | "blockquote",
                _,
            ) if !text.ends_with(' ') && !text.is_empty() => text.push(' '),
            _ => {}
        }
    }

    // A document that ends mid-link still yields that link.
    if let Some((href, from)) = anchor {
        out.links.push(Link {
            href,
            text: text[from.min(text.len())..].trim().to_owned(),
        });
    }

    out.title = decode_entities(title.trim());
    out.text = decode_entities(text.trim());
    for link in &mut out.links {
        link.text = decode_entities(&link.text);
        link.href = decode_entities(link.href.trim());
    }
    out
}

/// Appends `chunk` to `out`, collapsing every run of whitespace to one space.
fn push_collapsed(chunk: &str, out: &mut String) {
    for c in chunk.chars() {
        if c.is_whitespace() {
            if !out.ends_with(' ') && !out.is_empty() {
                out.push(' ');
            }
        } else {
            out.push(c);
        }
    }
}

/// Reads `name="value"` out of a tag's attribute text.
///
/// Handles double quotes, single quotes and no quotes, because all three occur
/// and a crawler that only understands one of them loses links.
fn attribute(tag: &str, name: &str) -> Option<String> {
    let lower = tag.to_ascii_lowercase();
    let mut from = 0usize;
    while let Some(at) = lower[from..].find(name) {
        let at = from + at;
        from = at + name.len();

        // Must be a whole attribute name, not a suffix of another one.
        let before_ok = at == 0
            || lower.as_bytes()[at - 1].is_ascii_whitespace()
            || lower.as_bytes()[at - 1] == b'"'
            || lower.as_bytes()[at - 1] == b'\'';
        if !before_ok {
            continue;
        }
        let rest = lower[from..].trim_start();
        if !rest.starts_with('=') {
            continue;
        }
        let equals = lower[from..].find('=')? + from + 1;
        let value = tag[equals..].trim_start();
        let quote = value.chars().next()?;
        return Some(if quote == '"' || quote == '\'' {
            value[1..]
                .find(quote)
                .map_or_else(|| value[1..].to_owned(), |e| value[1..=e].to_owned())
        } else {
            value
                .split(|c: char| c.is_whitespace())
                .next()
                .unwrap_or("")
                .trim_end_matches('/')
                .to_owned()
        });
    }
    None
}

/// Expands the handful of HTML entities that actually appear in text.
#[must_use]
pub fn decode_entities(text: &str) -> String {
    if !text.contains('&') {
        return text.to_owned();
    }
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(at) = rest.find('&') {
        out.push_str(&rest[..at]);
        rest = &rest[at..];
        // An entity is at most a few characters; beyond that it is a stray `&`.
        let Some(end) = rest[..rest.len().min(12)].find(';') else {
            out.push('&');
            rest = &rest[1..];
            continue;
        };
        let entity = &rest[1..end];
        let replacement = match entity {
            "amp" => Some('&'),
            "lt" => Some('<'),
            "gt" => Some('>'),
            "quot" => Some('"'),
            "apos" | "#39" => Some('\''),
            "nbsp" | "#160" => Some(' '),
            // The Latin-1 letters, by name. These are not exotic: Spanish and
            // Portuguese pages of the era wrote every accent this way, and a
            // crawler that skips them indexes "Ca&ntilde;on" as two words
            // neither of which anyone will ever search for.
            _ if !entity.starts_with('#') => named_latin1(entity),
            other => other
                .strip_prefix('#')
                .and_then(|n| {
                    n.strip_prefix('x')
                        .or_else(|| n.strip_prefix('X'))
                        .map_or_else(
                            || n.parse::<u32>().ok(),
                            |hex| u32::from_str_radix(hex, 16).ok(),
                        )
                })
                .and_then(char::from_u32),
        };
        if let Some(c) = replacement {
            out.push(c);
            rest = &rest[end + 1..];
        } else {
            out.push('&');
            rest = &rest[1..];
        }
    }
    out.push_str(rest);
    out
}

/// The named entities for accented Latin letters, plus the few symbols that
/// turn up in prose. Everything else falls through to the numeric forms.
fn named_latin1(name: &str) -> Option<char> {
    Some(match name {
        "aacute" => 'á',
        "eacute" => 'é',
        "iacute" => 'í',
        "oacute" => 'ó',
        "uacute" => 'ú',
        "Aacute" => 'Á',
        "Eacute" => 'É',
        "Iacute" => 'Í',
        "Oacute" => 'Ó',
        "Uacute" => 'Ú',
        "agrave" => 'à',
        "egrave" => 'è',
        "igrave" => 'ì',
        "ograve" => 'ò',
        "ugrave" => 'ù',
        "Agrave" => 'À',
        "Egrave" => 'È',
        "Igrave" => 'Ì',
        "Ograve" => 'Ò',
        "Ugrave" => 'Ù',
        "acirc" => 'â',
        "ecirc" => 'ê',
        "icirc" => 'î',
        "ocirc" => 'ô',
        "ucirc" => 'û',
        "Acirc" => 'Â',
        "Ecirc" => 'Ê',
        "Icirc" => 'Î',
        "Ocirc" => 'Ô',
        "Ucirc" => 'Û',
        "auml" => 'ä',
        "euml" => 'ë',
        "iuml" => 'ï',
        "ouml" => 'ö',
        "uuml" => 'ü',
        "Auml" => 'Ä',
        "Euml" => 'Ë',
        "Iuml" => 'Ï',
        "Ouml" => 'Ö',
        "Uuml" => 'Ü',
        "atilde" => 'ã',
        "otilde" => 'õ',
        "ntilde" => 'ñ',
        "Atilde" => 'Ã',
        "Otilde" => 'Õ',
        "Ntilde" => 'Ñ',
        "ccedil" => 'ç',
        "Ccedil" => 'Ç',
        "aring" => 'å',
        "Aring" => 'Å',
        "aelig" => 'æ',
        "AElig" => 'Æ',
        "oslash" => 'ø',
        "Oslash" => 'Ø',
        "yacute" => 'ý',
        "yuml" => 'ÿ',
        "szlig" => 'ß',
        "iexcl" => '¡',
        "iquest" => '¿',
        "laquo" => '«',
        "raquo" => '»',
        "deg" => '°',
        "middot" => '·',
        "ndash" => '–',
        "mdash" => '—',
        "hellip" => '…',
        "lsquo" => '\u{2018}',
        "rsquo" => '\u{2019}',
        "ldquo" => '\u{201C}',
        "rdquo" => '\u{201D}',
        "euro" => '€',
        "pound" => '£',
        "yen" => '¥',
        "cent" => '¢',
        "copy" => '©',
        "reg" => '®',
        "times" => '×',
        "divide" => '÷',
        "plusmn" => '±',
        "frac12" => '½',
        "sup2" => '²',
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn title_and_text_come_out_separately() {
        let e =
            extract("<html><head><title>Indexander</title></head><body>Un motor.</body></html>");
        assert_eq!(e.title, "Indexander");
        assert!(e.text.contains("Un motor."));
    }

    #[test]
    fn links_carry_their_anchor_text() {
        let e = extract(r#"<p>ver <a href="/parasearch">el buscador colombiano</a> ahora</p>"#);
        assert_eq!(e.links.len(), 1);
        assert_eq!(e.links[0].href, "/parasearch");
        assert_eq!(e.links[0].text, "el buscador colombiano");
    }

    #[test]
    fn attributes_may_be_quoted_any_of_three_ways() {
        for html in [
            r#"<a href="/x">t</a>"#,
            r"<a href='/x'>t</a>",
            r"<a href=/x>t</a>",
            r#"<a  class="c"  href = "/x" >t</a>"#,
        ] {
            let e = extract(html);
            assert_eq!(e.links.len(), 1, "failed on {html}");
            assert_eq!(e.links[0].href, "/x", "failed on {html}");
        }
    }

    #[test]
    fn script_and_style_contents_are_not_text() {
        let e = extract(
            "<body>real<script>var x = 'fake';</script><style>.a{color:red}</style>words</body>",
        );
        assert!(!e.text.contains("fake"));
        assert!(!e.text.contains("color"));
        assert!(e.text.contains("real"));
        assert!(e.text.contains("words"));
    }

    #[test]
    fn a_link_inside_script_is_not_a_link() {
        let e = extract(r#"<script>document.write('<a href="/no">x</a>')</script>"#);
        assert!(e.links.is_empty());
    }

    #[test]
    fn comments_are_ignored() {
        let e = extract("<!-- <a href=\"/hidden\">no</a> --><p>yes</p>");
        assert!(e.links.is_empty());
        assert_eq!(e.text.trim(), "yes");
    }

    #[test]
    fn block_elements_keep_words_apart() {
        let e = extract("<p>uno</p><p>dos</p>");
        assert_eq!(e.text, "uno dos");
    }

    #[test]
    fn entities_are_decoded_in_text_and_anchors() {
        let e = extract(r#"<a href="/a?x=1&amp;y=2">Ca&ntilde;&oacute;n &amp; m&aacute;s</a>"#);
        assert_eq!(e.links[0].href, "/a?x=1&y=2");
        assert_eq!(e.links[0].text, "Cañón & más");
    }

    #[test]
    fn numeric_entities_decimal_and_hex() {
        assert_eq!(decode_entities("A&#241;o &#x41;&#x42;"), "Año AB");
    }

    /// Found by a failing test, not by reading the spec: Spanish-language
    /// HTML of the era wrote every accent as a named entity.
    #[test]
    fn latin1_named_entities_are_decoded() {
        assert_eq!(
            decode_entities("Compa&ntilde;ia Colombiana de A&ntilde;os"),
            "Compañia Colombiana de Años"
        );
        assert_eq!(decode_entities("&iquest;b&uacute;squeda?"), "¿búsqueda?");
        assert_eq!(decode_entities("&Ntilde;"), "Ñ");
    }

    #[test]
    fn a_stray_ampersand_survives() {
        assert_eq!(decode_entities("Tom & Jerry"), "Tom & Jerry");
        assert_eq!(decode_entities("&notanentity;"), "&notanentity;");
    }

    #[test]
    fn rel_nofollow_links_are_not_returned() {
        let e = extract(r#"<a href="/paid" rel="nofollow sponsored">ad</a><a href="/ok">ok</a>"#);
        assert_eq!(e.links.len(), 1);
        assert_eq!(e.links[0].href, "/ok");
    }

    #[test]
    fn meta_robots_is_read() {
        let e = extract(r#"<meta name="robots" content="noindex, nofollow">"#);
        assert!(e.noindex);
        assert!(e.nofollow);

        let e = extract(r#"<meta name="ROBOTS" content="none">"#);
        assert!(e.noindex && e.nofollow);

        let e = extract(r#"<meta name="viewport" content="noindex">"#);
        assert!(!e.noindex, "only the robots meta counts");
    }

    #[test]
    fn base_href_is_captured() {
        let e = extract(r#"<base href="https://example.com/docs/">"#);
        assert_eq!(e.base.as_deref(), Some("https://example.com/docs/"));
    }

    #[test]
    fn malformed_html_still_yields_content() {
        // Unclosed tags, an unterminated tag, a stray bracket.
        let e = extract("<p>uno<a href=\"/x\">dos<p>tres < cuatro <div");
        assert!(e.text.contains("uno"));
        assert!(e.text.contains("tres"));
        assert_eq!(e.links.len(), 1);
        assert_eq!(e.links[0].href, "/x");
    }

    #[test]
    fn nested_anchors_do_not_lose_the_outer_link() {
        let e = extract(r#"<a href="/a">uno<a href="/b">dos</a>"#);
        assert_eq!(e.links.len(), 2);
        assert_eq!(e.links[0].href, "/a");
        assert_eq!(e.links[1].href, "/b");
    }

    #[test]
    fn an_href_suffix_is_not_mistaken_for_href() {
        let e = extract(r#"<a data-href="/wrong" href="/right">x</a>"#);
        assert_eq!(e.links[0].href, "/right");
    }

    #[test]
    fn empty_and_textless_documents_are_handled() {
        assert_eq!(extract("").text, "");
        assert_eq!(extract("<html></html>").text, "");
        let e = extract(r#"<a href="/x"></a>"#);
        assert_eq!(e.links[0].text, "");
    }

    #[test]
    fn whitespace_is_collapsed() {
        let e = extract("<p>uno \n\n   dos\t\ttres</p>");
        assert_eq!(e.text, "uno dos tres");
    }
}
