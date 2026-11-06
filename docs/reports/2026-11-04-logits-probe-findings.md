# Gate 6 replay and logits probe, 2026-11-04

Host: vast.ai instance 59116864, NVIDIA RTX PRO 6000 Blackwell Max-Q, 97887 MiB.
ds4 revision 5379c1c9ac31, ds4-py be45b7c, model sha256 ca22ae2f...6261c0, backend cuda, ctx 100000.
Raw outputs: `bakeoff-evidence/2026-11-04-gate6-crossfeed/raw/` and `bakeoff-evidence/2026-11-04-logits-probe/raw/` at the repository root (git-excluded).

## Question

Does the ds4-py worker drive libds4 differently from the C agent, so that identical
prompt tokens produce different greedy tokens?

## Answer

No. Given the same sync shape, the Python worker and the C driver produce the same
logits to within float32 noise. Exact token replay fails because the native library
itself is not bitwise stable across prefill chunking and across hosts.

## Evidence

### Gate 6 replay with split sync (commit e90a259), 17 C traces, 200 rounds

| Rounds | Passed | Tool-call rounds | Non-tool rounds |
|---|---:|---|---|
| 200 | 59 | 58/183 | 1/17 |

Compare the previous full run on the original host with single-shot sync: 91/91
tool-call rounds passed, 0/196 non-tool rounds.

### Cross-feed of looping ds4-py prompts through the C driver

| Case | First divergence vs ds4-py | Loop detected | Period |
|---|---:|---|---:|
| affine-cipher run-1 round 2 | 4 | yes, 12 repeats | 183 |
| scale-generator run-1 round 2 | 25 | yes, 9 repeats | 546 |

The C driver loops on ds4-py prompts too. The loop is a model-and-prompt property
under greedy decoding, not a ds4-py decode defect.

### Logits probe, three rounds, 16 greedy steps, top-12 logits per step

Variants: v1 split sync at the C cached boundary, v2 single-shot sync, v3 split plus a
redundant re-sync. C driver run with the same split and with no split.

| Case | ds4-py v1 vs C split | ds4-py v2 vs C no-split | v1 vs v2 | v1 vs retained C trace | v2 vs retained C trace |
|---|---|---|---|---:|---:|
| A affine-cipher run-1 ord 2 | same ranking, max delta 5e-8 | same, 5e-8 | different ranking, delta 0.63 | diverges at 0 | diverges at 0 |
| B scale-generator full ord 0 | same, 4e-8 | same, 5e-8 | different ranking, delta 1.51 | diverges at 11 | diverges at 10 |
| C scale-generator run-2 ord 1 (control) | same, 5e-8 | same, 5e-8 | different ranking, delta 0.59 | matches | matches |

Observations:

- Every ds4-py variant matches its C counterpart on all 16 tokens. Python and C are
  numerically equivalent drivers.
- Split versus single-shot sync changes the logits by up to 1.5 and flips the argmax
  in cases A and B. Prefill chunk boundaries are part of the numeric result.
- The redundant re-sync in v3 prefilled zero tokens and left the logits bit-identical
  to v1. The worker's sync bookkeeping is correct.
- Case B is the same prompt shape as the `d163c2c` diagnostic that reproduced the C
  trace exactly (first_divergence -1) on the original host. On this host the same C
  driver with the same split diverges at token 11. The retained trace cannot be
  reproduced by any driver on different hardware.
- Case A is a `</think>` versus `We` decision where the top three logits sit within
  0.11 of each other. Near-ties like this flip under any accumulation change.
- During the Gate 6 replay, ds4-py emitted `This` for case A; the fresh probe emitted
  `We`. The worker chunks a prefill by total length (219-token chunks for a 3140-token
  prefix, 190-token chunks for a 4272-token prompt), so a session that already holds
  earlier rounds prefills with different boundaries than a fresh one.

## Conclusions

1. The ds4-py runtime is equivalent to the C agent. No sync, chunking, or sampling
   defect was found. Commit e90a259 is not a regression; it also is not a fix, because
   matching the C boundary is insufficient when the host differs.
2. Gate 6 as written, exact token replay of traces recorded on another host, tests a
   property the native library does not have. The gate needs either same-host,
   same-chunking replay, or a logits-equivalence criterion like the probe above.
3. Gate 2 cap exits are greedy repetition loops that both drivers reproduce on the
   same prompt. Which prompts loop depends on near-tie argmax decisions, which are
   chaotic under chunking and hardware changes. With four prompts per arm, the 9/12
   versus 3/12 split is weak evidence that ds4-py's longer rendering loops more. A
   fair Gate 2 needs sampling with a fixed seed or many more trials per arm.
