# Contributing to Gray Box

Thanks for taking the time to contribute. Gray Box is intentionally small — the
bar for a change isn't "does it work," it's "is this the smallest, cleanest
thing that could plausibly work." This doc explains how to get set up and what
we expect from a PR.

---

## Before you start

- **Open an issue first for anything non-trivial.** Bug fixes with a clear
  repro are fine to PR directly. New commands, new page types, new config
  knobs, or anything touching the extraction prompt should be discussed first
  — it's easy to add scope, hard to remove it later.
- **Read [`README.md`](README.md), specifically "Design principles."** The
  three guarantees that shape almost every design decision in this repo:
  - Capture must never fail or lose data.
  - The LLM only reasons — it never touches the filesystem directly. All
    writes are deterministic Python.
  - Answers are grounded or honest, never invented.
  A PR that weakens any of these needs a very good reason.

---

## Development setup

```bash
git clone https://github.com/Aaryanverma/graybox
cd graybox
pip install -e ".[test]"
```

Setup your `config.yaml`. See `config.example.yaml` for the full set of options.

**Always branch from a fresh `main`.** This repo moves quickly between
sessions/contributors — pull immediately before starting work, not just
before opening the PR.

---

## Making changes

- **Surgical over sweeping.** Prefer the smallest diff that correctly fixes
  the issue or adds the feature. If a fix touches five files, ask yourself
  whether it should touch two.
- **No config knobs the user won't intuitively understand.** If you're
  tempted to add a new `retrieval.*` or `embeddings.*` setting, first check
  whether the value can instead be normalized/derived at the source (see
  `embedding_index.py`'s fixed-anchor calibration for the pattern). A new
  knob is a last resort, not a default move.
  - `min_score`, `dedup_threshold`, and any semantic score all live on the
    **same 0–1 scale**, where closer to 1.0 always means "more similar/more
    confident." Keep new scoring code on that same scale rather than
    inventing a new convention.
- **The LLM doesn't touch the filesystem.** Organizer/Curate/Summarizer parse
  or request structured output from the LLM, then hand off to deterministic
  Python (`storage.py`) for every actual write. Don't add a code path where
  the LLM's output is written to disk without passing through that layer.
- **The inbox is immutable.** Nothing should ever rewrite or delete a file
  under `inbox/` outside of `forget.py`'s explicit `--purge` path.
- **Match existing patterns** in the module you're touching before
  introducing a new one — e.g. dataclasses for structured returns, `Hit`/
  `Query`/`Engine` for anything search-related, `_maybe_record` for anything
  that should show up in page history.

---

## Testing

- **New tests must fail against pre-fix code before they're accepted.** If
  you're fixing a bug, write the test first, confirm it fails on `main`, then
  fix it and confirm it passes. A PR that only adds passing tests alongside a
  fix doesn't demonstrate the bug was real.
- **Run the full suite before opening a PR:**
  ```bash
  pytest
  ```
- **Verify against live output, not just static reasoning.** If you're
  claiming a bug is fixed, actually run the reproduction case end-to-end
  (`capture` → `organize` → `ask`, or whatever the relevant flow is) and show
  the before/after behavior in the PR description, not just a description of
  what the code change does.
- Coverage currently sits at 267+ passing tests. `cli.py` and `dashboard.py`
  are excluded from coverage requirements (see `pyproject.toml`) since
  they're thin presentation layers over tested logic — new logic belongs in
  a testable module, not inline in either of those files.

---

## Submitting a PR

1. **Fork, branch, and keep the diff focused.** One logical change per PR.
2. **Description should include:**
   - What broke / what was missing, with a concrete repro if it's a bug fix.
   - What you changed and why (especially if you deviated from the obvious
     approach).
   - Test output showing failing → passing, or `organize --dry-run` /
     `ask` output for behavior changes.
3. **Drop-in replacement files, not partial patches**, when a file changes
   substantially — makes review faster and avoids ambiguity about what state
   the file ends up in.
4. **Expect the maintainer to modify what you submit.** Reviews are direct
   and reference exact file/function/commit — this isn't personal, it's how
   the project stays small. If your proposed code gets changed before merge,
   that's normal; the goal is the best version of the fix landing, not your
   exact diff landing untouched.
5. **CI must pass** (`run-tests.yml`) before review.

---

## Documentation & community content

- Prefer honesty over polish. If a feature has a real limitation (e.g. the
  embedding index is a linear scan with no ANN, not a "real" vector DB),
  say so plainly in the README rather than implying more than what's there.
- Keep explanations concrete and first-principles — avoid marketing language
  ("blazing fast," "seamless") in favor of what actually happens
  mechanically.

---

## Reporting bugs

Open an issue with:
- Gray Box version / commit, and how you installed it (`pip install graybox`
  vs. `pip install -e .`).
- Your `config.yaml` with any API keys redacted.
- The exact command that triggered the issue.
- What you expected vs. what happened — actual output/error, not a summary.

If it's an extraction/retrieval quality issue (wrong entity type, missed
relation, ungrounded answer), include the raw note text and, if possible,
`organize --dry-run` or `ask` output, since these are the fastest way to
reproduce LLM-adjacent bugs deterministically.

---

## Code of conduct

Be direct, be kind, assume good faith. Technical disagreement is fine and
expected; we're optimizing for the smallest correct solution, and that
sometimes means someone's larger PR gets scoped down. That's about the code,
not the contributor.
