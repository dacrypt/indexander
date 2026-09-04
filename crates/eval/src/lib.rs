//! Measuring whether the results are *good*, which is a different question
//! from whether they are *consistent*.
//!
//! Every other test in this repository checks that two paths agree: that a
//! merged index scores like the segments it came from, that a cluster returns
//! what a single index would, that a rebuild is byte-identical. Those are
//! worth having, and they are all satisfied by a ranker that puts the worst
//! document first — as long as it does so reproducibly.
//!
//! This crate is the other half. It has no opinion about what is relevant; it
//! takes relevance as input, from one of two places:
//!
//! - [`qrels`], the format research collections publish human judgements in.
//!   Someone decided, and this only does the arithmetic.
//! - [`sampling`], which builds *known-item* queries: a span lifted out of a
//!   document, whose one correct answer is the document it came from. Nobody
//!   judges anything, and the measure is still real, because every document
//!   containing those terms is a candidate and only ranking separates them.
//!
//! Everything here is pure. No files, no clock, no index — the caller brings
//! a ranked list of URIs and this says how good it was.

pub mod metrics;
pub mod qrels;
pub mod sampling;
pub mod ties;

pub use metrics::{Judged, Scores};
pub use qrels::{Qrels, Topic};
