# Evidence template

Only the text from `<!-- pr-review-loop:start -->` to
`<!-- pr-review-loop:end -->` goes into the PR body. Fill in every
`{{...}}` placeholder. Repeat a table row once for each item. When a table
has no items, keep the table and give it one row of `(none)` cells. Keep the
rows of earlier runs; add the new run's rows after them.

Values:

- `{{MATCH_NOTE}}`: `matches the PR head` or `differs from the PR head`.
- `{{OUTCOME}}`: `Clean`, `Cap reached`, or `Incomplete: <reason>`.
- `{{FIX}}`: the fix commit, then the test that failed before and passes
  after, or the direct check used instead. Mark fixes made in the final
  round of a run with `not reviewed by a fresh reviewer`.
- `{{SEVERITY}}`: `Critical`, `Important`, `Minor`, or `Not graded` for a
  "Declined to judge" item the author set aside.
- `{{FOLLOW_UP}}`: an issue reference such as `#831`, or `not filed`.
- When only one model family was available, end the Result line with
  `Only one model family was available.`

<!-- pr-review-loop:start -->
## Review evidence

**Evidence head:** `{{EVIDENCE_HEAD}}` ({{MATCH_NOTE}}). If the PR head differs from this commit, later pushes were not reviewed.

**Result:** Run {{RUN}}: {{OUTCOME}} after {{RUN_ROUNDS}} rounds. {{TOTAL}} findings across all runs: {{FIXED}} fixed, {{DECLINED}} declined, {{DEFERRED}} deferred.

| Run | Round | Trigger | Model | Commit reviewed | Findings | Verdict |
|---|---|---|---|---|---|---|
| {{RUN}} | {{ROUND}} | {{TRIGGER}} | {{MODEL}} | `{{REVIEWED_SHA}}` | {{FINDING_COUNT}} | {{VERDICT}} |

### Fixed
| ID | Severity | Finding | Fix |
|---|---|---|---|
| {{ID}} | {{SEVERITY}} | {{FINDING}} (`{{FILE_LINE}}`) | {{FIX}} |

### Declined
| ID | Severity | Finding | Reason |
|---|---|---|---|
| {{ID}} | {{SEVERITY}} | {{FINDING}} | {{REASON}} |

### Deferred
| ID | Severity | Finding | Follow-up |
|---|---|---|---|
| {{ID}} | {{SEVERITY}} | {{FINDING}} | {{FOLLOW_UP}} |

### Validation on `{{EVIDENCE_HEAD}}` (evidence head)
- `{{COMMAND}}`: {{COMMAND_RESULT}}

### Human reviewer callouts
- {{CALLOUT}}

<details><summary>Run {{RUN}} round {{ROUND}} full report</summary>

{{REPORT}}

</details>
<!-- pr-review-loop:end -->
