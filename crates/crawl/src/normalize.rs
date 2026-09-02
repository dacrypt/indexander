//! Turning an `href` into a canonical URL.
//!
//! Two URLs that fetch the same bytes must produce the same string, or the
//! crawler will fetch a page many times and the index will hold it many times.
//! Getting this wrong is the classic way a crawler falls into a hole and never
//! comes out.

use url::Url;

/// Query parameters that identify a campaign, not a resource. Dropping them
/// collapses a great many duplicates into one.
const TRACKING: [&str; 9] = [
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "utm_id",
    "gclid",
    "fbclid",
    "mc_eid",
];

/// Resolves `href` against `base` and canonicalises the result.
///
/// Returns `None` for anything that is not a fetchable http(s) page:
/// `mailto:`, `javascript:`, `#fragment`, and malformed input.
#[must_use]
pub fn resolve(base: &Url, href: &str) -> Option<Url> {
    let href = href.trim();
    if href.is_empty() || href.starts_with('#') {
        return None;
    }
    let mut url = base.join(href).ok()?;
    if !matches!(url.scheme(), "http" | "https") {
        return None;
    }

    // A fragment names a place inside a document, not another document.
    url.set_fragment(None);

    // The default port is implied; writing it out makes the same page look
    // like two.
    let default_port = match url.scheme() {
        "http" => 80,
        _ => 443,
    };
    if url.port() == Some(default_port) {
        url.set_port(None).ok()?;
    }

    // An empty path is the root path.
    if url.path().is_empty() {
        url.set_path("/");
    }

    // Hosts are case-insensitive; the `url` crate already lowercases them, but
    // a trailing dot is a distinct string for the same host.
    if let Some(host) = url.host_str() {
        if let Some(trimmed) = host.strip_suffix('.') {
            let trimmed = trimmed.to_owned();
            url.set_host(Some(&trimmed)).ok()?;
        }
    }

    strip_tracking(&mut url);
    Some(url)
}

/// Removes tracking parameters, and the `?` itself if nothing is left.
fn strip_tracking(url: &mut Url) {
    if url.query().is_none() {
        return;
    }
    let kept: Vec<(String, String)> = url
        .query_pairs()
        .filter(|(k, _)| !TRACKING.contains(&k.as_ref()))
        .map(|(k, v)| (k.into_owned(), v.into_owned()))
        .collect();

    if kept.is_empty() {
        url.set_query(None);
    } else {
        let mut serializer = url.query_pairs_mut();
        serializer.clear();
        for (k, v) in kept {
            serializer.append_pair(&k, &v);
        }
        drop(serializer);
    }
}

/// The host of a URL, used to group requests for politeness.
#[must_use]
pub fn host_of(url: &Url) -> String {
    url.host_str().unwrap_or_default().to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base() -> Url {
        Url::parse("https://example.com/docs/page.html").expect("valid")
    }

    fn r(href: &str) -> Option<String> {
        resolve(&base(), href).map(|u| u.to_string())
    }

    #[test]
    fn relative_links_resolve_against_the_page() {
        assert_eq!(
            r("other.html").unwrap(),
            "https://example.com/docs/other.html"
        );
        assert_eq!(r("/root.html").unwrap(), "https://example.com/root.html");
        assert_eq!(r("../up.html").unwrap(), "https://example.com/up.html");
        assert_eq!(r("//other.org/x").unwrap(), "https://other.org/x");
    }

    #[test]
    fn fragments_are_dropped_because_they_are_not_documents() {
        assert_eq!(
            r("page.html#section").unwrap(),
            "https://example.com/docs/page.html"
        );
        assert_eq!(r("#section"), None);
    }

    #[test]
    fn non_fetchable_schemes_are_refused() {
        assert_eq!(r("mailto:a@b.com"), None);
        assert_eq!(r("javascript:void(0)"), None);
        assert_eq!(r("ftp://example.com/x"), None);
        assert_eq!(r("tel:+123"), None);
        assert_eq!(r(""), None);
        assert_eq!(r("   "), None);
    }

    #[test]
    fn default_ports_are_removed() {
        assert_eq!(
            r("https://example.com:443/x").unwrap(),
            "https://example.com/x"
        );
        assert_eq!(
            r("http://example.com:80/x").unwrap(),
            "http://example.com/x"
        );
        // A non-default port is part of the identity and stays.
        assert!(r("http://example.com:8080/x").unwrap().contains(":8080"));
    }

    #[test]
    fn hosts_are_case_folded_and_lose_a_trailing_dot() {
        assert_eq!(r("https://EXAMPLE.COM/x").unwrap(), "https://example.com/x");
        assert_eq!(
            r("https://example.com./x").unwrap(),
            "https://example.com/x"
        );
    }

    #[test]
    fn tracking_parameters_are_stripped_and_real_ones_kept() {
        assert_eq!(
            r("/x?utm_source=twitter&utm_campaign=launch").unwrap(),
            "https://example.com/x"
        );
        assert_eq!(
            r("/x?page=2&utm_source=twitter").unwrap(),
            "https://example.com/x?page=2"
        );
        assert_eq!(r("/x?q=perl").unwrap(), "https://example.com/x?q=perl");
    }

    #[test]
    fn the_same_page_written_differently_normalises_to_one_string() {
        let variants = [
            "https://example.com:443/docs/other.html#top",
            "https://EXAMPLE.com/docs/other.html",
            "other.html",
            "./other.html",
            "/docs/../docs/other.html",
        ];
        let canonical: Vec<String> = variants.iter().filter_map(|v| r(v)).collect();
        assert_eq!(canonical.len(), variants.len());
        assert!(
            canonical.windows(2).all(|w| w[0] == w[1]),
            "variants did not converge: {canonical:?}"
        );
    }

    #[test]
    fn an_empty_path_becomes_root() {
        assert_eq!(r("https://example.com").unwrap(), "https://example.com/");
    }

    #[test]
    fn host_of_returns_the_host() {
        assert_eq!(host_of(&base()), "example.com");
    }
}
