//! Reading judgements somebody else made.
//!
//! The format is TREC's, because every published test collection already
//! speaks it and inventing another one would only mean writing converters.
//! A qrels line is four whitespace-separated fields:
//!
//! ```text
//! topic-id   iteration   document-id   relevance
//! ```
//!
//! The iteration column is a historical artifact of how NIST pooled runs. It
//! is ignored here, as it is nearly everywhere, but it must be present so that
//! files written for other tools load unchanged.
//!
//! Topics are one query per line, the id and then the query text:
//!
//! ```text
//! 1   inverted index compression
//! 2   "block max" wand
//! ```
//!
//! Both formats skip blank lines and anything starting with `#`, so a
//! judgement file can carry a note about who judged it and when — which, given
//! how much a number from here depends on that, it should.

use std::collections::BTreeMap;

use crate::metrics::Judged;

/// A query, with the id its judgements are filed under.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Topic {
    pub id: String,
    pub text: String,
}

/// Judgements, indexed by topic.
///
/// Ordered rather than hashed: a run should be evaluated in a stable order so
/// two reports of the same thing are diffable.
#[derive(Debug, Clone, Default)]
pub struct Qrels {
    topics: BTreeMap<String, Judged>,
}

/// What went wrong reading a file, and on which line.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseError {
    pub line: usize,
    pub message: String,
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "line {}: {}", self.line, self.message)
    }
}

impl std::error::Error for ParseError {}

impl Qrels {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Parses a TREC qrels file.
    ///
    /// A malformed line is an error rather than a skip. Silently dropping a
    /// judgement changes every number downstream and leaves nothing to notice
    /// it by.
    pub fn parse(text: &str) -> Result<Self, ParseError> {
        let mut qrels = Self::new();
        for (n, line) in text.lines().enumerate() {
            let Some(line) = content(line) else { continue };
            let fields: Vec<&str> = line.split_whitespace().collect();
            let [topic, _iteration, document, relevance] = fields[..] else {
                return Err(ParseError {
                    line: n + 1,
                    message: format!(
                        "expected 4 fields (topic, iteration, document, relevance), found {}",
                        fields.len()
                    ),
                });
            };
            // Negative grades appear in some collections to mark documents
            // judged actively harmful. Nothing here ranks below "not
            // relevant", so they clamp to zero rather than fail the load.
            let grade: i32 = relevance.parse().map_err(|_| ParseError {
                line: n + 1,
                message: format!("relevance {relevance:?} is not a number"),
            })?;
            let grade = u8::try_from(grade.clamp(0, i32::from(u8::MAX))).unwrap_or(u8::MAX);
            qrels
                .topics
                .entry(topic.to_owned())
                .or_default()
                .judge(document, grade);
        }
        Ok(qrels)
    }

    /// Judgements for one topic, if it has any.
    #[must_use]
    pub fn get(&self, topic: &str) -> Option<&Judged> {
        self.topics.get(topic)
    }

    #[must_use]
    pub fn topic_count(&self) -> usize {
        self.topics.len()
    }

    /// Every topic id that has at least one relevant document.
    #[must_use]
    pub fn scorable(&self) -> Vec<&str> {
        self.topics
            .iter()
            .filter(|(_, j)| j.is_scorable())
            .map(|(id, _)| id.as_str())
            .collect()
    }
}

/// Parses a topics file: one query per line, id first.
pub fn parse_topics(text: &str) -> Result<Vec<Topic>, ParseError> {
    let mut topics = Vec::new();
    for (n, line) in text.lines().enumerate() {
        let Some(line) = content(line) else { continue };
        let (id, query) = line.split_once(char::is_whitespace).ok_or(ParseError {
            line: n + 1,
            message: "expected an id and a query, found one field".to_owned(),
        })?;
        let query = query.trim();
        if query.is_empty() {
            return Err(ParseError {
                line: n + 1,
                message: format!("topic {id} has no query text"),
            });
        }
        topics.push(Topic {
            id: id.to_owned(),
            text: query.to_owned(),
        });
    }
    Ok(topics)
}

/// The meaningful part of a line: `None` for blanks and comments.
fn content(line: &str) -> Option<&str> {
    let line = line.trim();
    if line.is_empty() || line.starts_with('#') {
        None
    } else {
        Some(line)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_qrels_file_loads_into_topics() {
        let q = Qrels::parse("1 0 doc-a 1\n1 0 doc-b 0\n2 0 doc-c 2\n").unwrap();
        assert_eq!(q.topic_count(), 2);
        let one = q.get("1").unwrap();
        assert_eq!(one.grade("doc-a"), 1);
        assert_eq!(one.grade("doc-b"), 0);
        assert!(one.was_judged("doc-b"));
        assert_eq!(q.get("2").unwrap().grade("doc-c"), 2);
    }

    #[test]
    fn blank_lines_and_comments_are_skipped() {
        let q = Qrels::parse("# who judged this, and when\n\n1 0 doc-a 1\n").unwrap();
        assert_eq!(q.topic_count(), 1);
    }

    #[test]
    fn a_malformed_line_is_an_error_naming_the_line() {
        let err = Qrels::parse("1 0 doc-a 1\nnonsense\n").unwrap_err();
        assert_eq!(err.line, 2);
        assert!(err.message.contains("4 fields"), "{}", err.message);
    }

    #[test]
    fn a_relevance_that_is_not_a_number_is_an_error() {
        let err = Qrels::parse("1 0 doc-a yes\n").unwrap_err();
        assert_eq!(err.line, 1);
        assert!(err.message.contains("not a number"), "{}", err.message);
    }

    #[test]
    fn a_negative_grade_clamps_to_not_relevant() {
        let q = Qrels::parse("1 0 spam -1\n").unwrap();
        let one = q.get("1").unwrap();
        assert_eq!(one.grade("spam"), 0);
        assert!(one.was_judged("spam"));
        assert!(!one.is_scorable());
    }

    #[test]
    fn tabs_are_whitespace_like_anything_else() {
        let q = Qrels::parse("1\t0\tdoc-a\t1\n").unwrap();
        assert_eq!(q.get("1").unwrap().grade("doc-a"), 1);
    }

    #[test]
    fn only_topics_with_something_relevant_are_scorable() {
        let q = Qrels::parse("1 0 a 1\n2 0 b 0\n").unwrap();
        assert_eq!(q.scorable(), ["1"]);
    }

    #[test]
    fn topics_keep_the_whole_query_including_its_spaces_and_quotes() {
        let topics = parse_topics("1  inverted index\n2\t\"block max\" wand\n").unwrap();
        assert_eq!(
            topics,
            [
                Topic {
                    id: "1".into(),
                    text: "inverted index".into()
                },
                Topic {
                    id: "2".into(),
                    text: "\"block max\" wand".into()
                },
            ]
        );
    }

    #[test]
    fn a_topic_without_a_query_is_an_error() {
        assert_eq!(parse_topics("7\n").unwrap_err().line, 1);
        assert_eq!(parse_topics("7   \n").unwrap_err().line, 1);
    }
}
