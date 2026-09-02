//! Parsing and applying `robots.txt`.
//!
//! The format has no specification worth the name, only a 1994 memo and
//! twenty-five years of what crawlers actually do. The behaviour implemented
//! here is the one everybody converged on, and which RFC 9309 later wrote down:
//!
//! * Records are groups of `User-agent` lines followed by rules.
//! * A crawler obeys the group whose agent matches it most specifically,
//!   falling back to the `*` group. It obeys **one** group, not the union.
//! * The longest matching `Allow` or `Disallow` path wins. On a tie, `Allow`
//!   wins, because the conservative reading of an ambiguous rule is the one
//!   that does not silently drop a site out of the index.
//! * `Disallow:` with an empty value allows everything. `Disallow: /` forbids
//!   everything. This asymmetry is the single most common way to break a
//!   crawler, so it has its own tests.
//! * Anything unparseable is ignored rather than treated as forbidding.

use std::time::Duration;

/// One rule from a group.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Rule {
    path: String,
    allow: bool,
}

/// The rules that apply to one crawler, already selected from the file.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Robots {
    rules: Vec<Rule>,
    crawl_delay: Option<Duration>,
    sitemaps: Vec<String>,
}

impl Robots {
    /// A `robots.txt` that permits everything, used when a site has none or
    /// when fetching it failed with a client error.
    #[must_use]
    pub fn allow_all() -> Self {
        Self::default()
    }

    /// A `robots.txt` that forbids everything, used when a site's file could
    /// not be fetched because of a server error: unreachable is not consent.
    #[must_use]
    pub fn deny_all() -> Self {
        Self {
            rules: vec![Rule {
                path: "/".to_owned(),
                allow: false,
            }],
            ..Self::default()
        }
    }

    #[must_use]
    pub fn crawl_delay(&self) -> Option<Duration> {
        self.crawl_delay
    }

    #[must_use]
    pub fn sitemaps(&self) -> &[String] {
        &self.sitemaps
    }

    /// Parses `text`, keeping only the rules that apply to `agent`.
    ///
    /// `agent` is matched case-insensitively as a prefix, which is how
    /// `User-agent: Googlebot` is meant to match `Googlebot-Image`.
    #[must_use]
    pub fn parse(text: &str, agent: &str) -> Self {
        let agent = agent.to_ascii_lowercase();

        // Rules collected per group, plus how specific that group's best
        // matching agent line was. Specificity is the length of the matched
        // name, so `indexander` beats `index` beats `*`.
        let mut best: Option<(usize, Vec<Rule>, Option<Duration>)> = None;
        let mut current_agents: Vec<String> = Vec::new();
        let mut current_rules: Vec<Rule> = Vec::new();
        let mut current_delay: Option<Duration> = None;
        let mut sitemaps = Vec::new();
        // A `User-agent` line after rules starts a new group; one after
        // another `User-agent` line extends the same group.
        let mut in_rules = false;

        // Score the group we just finished and keep it if it is the best so far.
        macro_rules! close_group {
            () => {
                if !current_agents.is_empty() {
                    let specificity = current_agents
                        .iter()
                        .filter_map(|a| {
                            if a == "*" {
                                Some(0)
                            } else if agent.starts_with(a.as_str()) {
                                Some(a.len())
                            } else {
                                None
                            }
                        })
                        .max();
                    if let Some(specificity) = specificity {
                        if best.as_ref().is_none_or(|(b, _, _)| specificity >= *b) {
                            best = Some((
                                specificity,
                                std::mem::take(&mut current_rules),
                                current_delay,
                            ));
                        }
                    }
                }
                current_agents.clear();
                current_rules.clear();
                current_delay = None;
                in_rules = false;
            };
        }

        for line in text.lines() {
            // Everything after `#` is a comment, anywhere on the line.
            let line = line.split('#').next().unwrap_or("").trim();
            if line.is_empty() {
                continue;
            }
            let Some((field, value)) = line.split_once(':') else {
                continue;
            };
            let field = field.trim().to_ascii_lowercase();
            let value = value.trim();

            match field.as_str() {
                "user-agent" => {
                    if in_rules {
                        close_group!();
                    }
                    current_agents.push(value.to_ascii_lowercase());
                }
                "disallow" | "allow" => {
                    in_rules = true;
                    // `Disallow:` with nothing after it is not a rule at all;
                    // it is the documented way of saying "allow everything".
                    if field == "disallow" && value.is_empty() {
                        continue;
                    }
                    current_rules.push(Rule {
                        path: value.to_owned(),
                        allow: field == "allow",
                    });
                }
                "crawl-delay" => {
                    in_rules = true;
                    if let Ok(seconds) = value.parse::<f64>() {
                        if seconds.is_finite() && seconds >= 0.0 {
                            current_delay = Some(Duration::from_secs_f64(seconds.min(300.0)));
                        }
                    }
                }
                // Sitemaps are global, not part of any group.
                "sitemap" => sitemaps.push(value.to_owned()),
                _ => {}
            }
        }
        close_group!();
        // The macro's final resets are dead on the last call, by construction.
        let _ = (&current_delay, &in_rules, &current_agents);

        let (_, rules, crawl_delay) = best.unwrap_or_default();
        Self {
            rules,
            crawl_delay,
            sitemaps,
        }
    }

    /// Whether `path` (with its query, if any) may be fetched.
    #[must_use]
    pub fn allows(&self, path: &str) -> bool {
        let mut decision = true;
        let mut best_len = 0usize;

        for rule in &self.rules {
            if !path_matches(&rule.path, path) {
                continue;
            }
            // Longest rule wins; on a tie, Allow wins over Disallow.
            let len = rule.path.len();
            if len > best_len || (len == best_len && rule.allow) {
                best_len = len;
                decision = rule.allow;
            }
        }
        decision
    }
}

/// Matches a rule path against a request path, honouring `*` and `$`.
///
/// These wildcards are not in the original memo but every major crawler
/// supports them and site owners write rules assuming they work.
fn path_matches(pattern: &str, path: &str) -> bool {
    if pattern.is_empty() {
        return false;
    }
    let anchored = pattern.ends_with('$');
    let pattern = if anchored {
        &pattern[..pattern.len() - 1]
    } else {
        pattern
    };

    let mut segments = pattern.split('*');
    let Some(first) = segments.next() else {
        return true;
    };
    if !path.starts_with(first) {
        return false;
    }
    let mut cursor = first.len();

    let mut last_empty = pattern.ends_with('*');
    for segment in segments {
        last_empty = segment.is_empty();
        if segment.is_empty() {
            continue;
        }
        match path[cursor..].find(segment) {
            Some(at) => cursor += at + segment.len(),
            None => return false,
        }
    }

    if anchored && !last_empty {
        return cursor == path.len();
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    const AGENT: &str = "indexander";

    #[test]
    fn no_rules_allows_everything() {
        let r = Robots::parse("", AGENT);
        assert!(r.allows("/"));
        assert!(r.allows("/anything/at/all"));
    }

    #[test]
    fn empty_disallow_allows_everything() {
        // The classic trap: `Disallow:` and `Disallow: /` are opposites.
        let r = Robots::parse("User-agent: *\nDisallow:", AGENT);
        assert!(r.allows("/anything"));
    }

    #[test]
    fn disallow_slash_forbids_everything() {
        let r = Robots::parse("User-agent: *\nDisallow: /", AGENT);
        assert!(!r.allows("/"));
        assert!(!r.allows("/anything"));
    }

    #[test]
    fn prefix_rules_only_match_their_prefix() {
        let r = Robots::parse("User-agent: *\nDisallow: /private", AGENT);
        assert!(!r.allows("/private"));
        assert!(!r.allows("/private/x"));
        assert!(r.allows("/public"));
    }

    #[test]
    fn the_longest_matching_rule_wins() {
        let r = Robots::parse(
            "User-agent: *\nDisallow: /a/\nAllow: /a/b/\nDisallow: /a/b/c/",
            AGENT,
        );
        assert!(!r.allows("/a/x"));
        assert!(r.allows("/a/b/x"));
        assert!(!r.allows("/a/b/c/x"));
    }

    #[test]
    fn allow_wins_a_tie() {
        let r = Robots::parse("User-agent: *\nDisallow: /x\nAllow: /x", AGENT);
        assert!(r.allows("/x"));
    }

    #[test]
    fn the_most_specific_agent_group_is_the_one_obeyed() {
        let text = "\
User-agent: *
Disallow: /

User-agent: indexander
Disallow: /private
";
        let r = Robots::parse(text, AGENT);
        // Our group replaces the wildcard group; it does not add to it.
        assert!(r.allows("/public"));
        assert!(!r.allows("/private"));
    }

    #[test]
    fn an_unrelated_specific_group_is_ignored() {
        let text = "\
User-agent: googlebot
Disallow: /

User-agent: *
Disallow: /private
";
        let r = Robots::parse(text, AGENT);
        assert!(r.allows("/public"));
        assert!(!r.allows("/private"));
    }

    #[test]
    fn agent_matching_is_a_case_insensitive_prefix() {
        let text = "User-agent: INDEX\nDisallow: /no";
        let r = Robots::parse(text, "indexander/1.0");
        assert!(!r.allows("/no"));
    }

    #[test]
    fn consecutive_agent_lines_share_one_group() {
        let text = "\
User-agent: alpha
User-agent: indexander
Disallow: /shared
";
        let r = Robots::parse(text, AGENT);
        assert!(!r.allows("/shared"));
    }

    #[test]
    fn wildcards_and_anchors_work() {
        let r = Robots::parse("User-agent: *\nDisallow: /*.pdf$", AGENT);
        assert!(!r.allows("/docs/manual.pdf"));
        assert!(r.allows("/docs/manual.pdf.html"));
        assert!(r.allows("/docs/manual.html"));
    }

    #[test]
    fn a_trailing_star_matches_any_suffix() {
        let r = Robots::parse("User-agent: *\nDisallow: /search*", AGENT);
        assert!(!r.allows("/search"));
        assert!(!r.allows("/search?q=x"));
        assert!(r.allows("/other"));
    }

    #[test]
    fn comments_and_blank_lines_are_ignored() {
        let text = "\
# a comment
User-agent: *   # trailing comment

Disallow: /private   # another
";
        let r = Robots::parse(text, AGENT);
        assert!(!r.allows("/private"));
        assert!(r.allows("/public"));
    }

    #[test]
    fn garbage_lines_do_not_forbid_anything() {
        let r = Robots::parse("!!! not a robots file\n\u{0}\nrandom text", AGENT);
        assert!(r.allows("/anything"));
    }

    #[test]
    fn crawl_delay_is_read_and_capped() {
        let r = Robots::parse("User-agent: *\nCrawl-delay: 2.5", AGENT);
        assert_eq!(r.crawl_delay(), Some(Duration::from_millis(2500)));

        // A hostile or mistaken value must not stall the crawler forever.
        let r = Robots::parse("User-agent: *\nCrawl-delay: 99999", AGENT);
        assert_eq!(r.crawl_delay(), Some(Duration::from_secs(300)));

        let r = Robots::parse("User-agent: *\nCrawl-delay: not-a-number", AGENT);
        assert_eq!(r.crawl_delay(), None);
    }

    #[test]
    fn sitemaps_are_collected_regardless_of_group() {
        let text = "\
Sitemap: https://example.com/sitemap.xml
User-agent: googlebot
Disallow: /
Sitemap: https://example.com/news.xml
";
        let r = Robots::parse(text, AGENT);
        assert_eq!(r.sitemaps().len(), 2);
    }

    #[test]
    fn deny_all_denies_and_allow_all_allows() {
        assert!(!Robots::deny_all().allows("/"));
        assert!(!Robots::deny_all().allows("/anything"));
        assert!(Robots::allow_all().allows("/anything"));
    }

    #[test]
    fn rules_apply_to_the_query_string_too() {
        let r = Robots::parse("User-agent: *\nDisallow: /*?sort=", AGENT);
        assert!(!r.allows("/list?sort=price"));
        assert!(r.allows("/list?page=2"));
    }
}
