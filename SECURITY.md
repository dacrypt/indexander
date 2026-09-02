# Security

## Scope

This project parses untrusted input in two places, and both matter:

- **Segment files.** `Segment::from_bytes` will be handed corrupt or hostile
  bytes eventually. It must return `Err`, never panic and never read out of
  bounds. `a_truncated_segment_is_rejected_rather_than_misread` is the start of
  that coverage, not the end of it.
- **Query strings.** `query::parse` is total by design: it has no failure mode
  and no unbounded recursion.

The crate sets `unsafe_code = "warn"` at workspace level. There is currently no
`unsafe` in the tree.

## Reporting

Open a private security advisory through GitHub's "Report a vulnerability"
button on this repository. If that is not possible, open a normal issue saying
only that you have found something and how to reach you — no details.
