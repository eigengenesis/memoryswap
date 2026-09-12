# Where Does Qwen3.5 Keep Its World?

Status: prospective implementation plan, written 2026-09-11. No chess-model result or GPU feasibility result is established by this document.

Research question: Which memory channel of frozen Qwen3.5 carries information needed for predictions consistent with an inferred chess position, and does that influence survive additional moves?

Implement this as a NEW standalone script, `qwen35_chess_memory.py`. Preserve the existing MuSiQue, Atlas, and monitorability scripts and their outputs. This study has different data, interventions, and claims. Reuse inspected infrastructure ideas, not their training pipelines or experiment-specific imports.

## 1. Scope and intended conclusion

The experiment combines three measurements on the same histories:

1. Recoverability: can a fixed linear readout recover pieces and selected rule state from a model activation?
2. Behavioral influence: does swapping a memory channel move the model's own next-move preferences toward the donor position?
3. Continuation: does that influence persist as the model processes further legal moves?

Cache swaps and behavioral effects are the primary causal evidence. Readouts characterize accessible information. Readout failure alone cannot establish that information is absent. A cache intervention affecting both readouts and behavior does not establish that the exact features read by the probe mediate the behavior.

Both attention and recurrent layers remain active after intervention. Thus this protocol tests the contribution of memory carried across a boundary; it does not uniquely identify which layers perform the subsequent update computation. The channels already exchange information during prefill. Do not describe them as independent computational systems.

The connection to Atlas is conceptual: investigate a possible native workspace for compositional state transitions. This experiment neither trains nor validates the old Atlas operators, tomography score, subspace projection, or continual-learning method.

### Prior work and attribution

- [Qwen3.5-4B-Base model card](https://huggingface.co/Qwen/Qwen3.5-4B-Base): the official base checkpoint; its documented language stack interleaves Gated DeltaNet and full-attention layers.
- [What Attention Recalls and Recurrence Controls in Hybrid Language Models](https://arxiv.org/abs/2609.04434): already introduces split-prefill and crossed recurrent/KV caches, including Qwen3.5. Credit those interventions. The proposed extension is to inferred chess positions and continuation, not invention of cache swapping. This limited source check does not establish novelty across all literature.
- [Neel Nanda's Othello work](https://www.neelnanda.io/mechanistic-interpretability/othello): relevant precedent for relating board readouts to causal behavior. General Qwen competence on chess cannot be assumed from a model trained specifically on Othello.
- [Pinned Qwen implementation](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/models/qwen3_5/modeling_qwen3_5.py) and [cache implementation](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/cache_utils.py): reference sources for the backend. Actual compatibility must pass the real-model tests below.

## 2. Files to implement

| File | Responsibility |
| --- | --- |
| `qwen35_chess_memory.py` | Data preparation, frozen backend, cache operations, probe fitting, evaluation, analysis, plotting, CLI |
| `tests/test_qwen35_chess_memory.py` | Verifier, splits, token boundaries, cache isolation, scoring, statistics, artifact tests |
| `requirements_qwen35_chess_memory.txt` | Reproducible minimal environment, pinned after the real preflight |
| `data/qwen35_chess_memory/config.json` | Model, protocol choices, sample counts, and seeds |
| `data/qwen35_chess_memory/manifest.json` | Final histories, splits, pairs, candidates, continuations, exclusions, source hashes |
| `docs/qwen35_chess_memory_runbook.md` | Exact tested Kaggle commands and resume instructions, written with implementation |

Use `outputs/qwen35_chess_memory/<run-id>/` for every artifact. The new runner must not import `qwen_atlas_transfer`, `qwen35_operator_atlas`, or `latent_interface_monitorability`: these bring assumptions about learned connectors, synthetic states, and chat templates.

Small hashing, atomic-write, environment-recording, and plotting helpers can be adapted after inspection. The existing Transformers 5.17.0 loading pattern is a starting point, not proof that the new cache operations work.

## 3. Frozen configuration

| Setting | Value |
| --- | --- |
| Primary model | `Qwen/Qwen3.5-4B-Base` |
| Resource fallback | `Qwen/Qwen3.5-2B-Base`, only for documented memory/runtime infeasibility before behavioral results |
| Model/tokenizer revision | Resolve real immutable commit IDs during prepare; record them before any behavioral evaluation |
| Initial Transformers version | `5.17.0`; record exact Torch, CUDA, tokenizers, and kernel versions |
| Model mode | Frozen, `eval()`, inference without gradients; no adapters or weight training |
| Precision | FP16 weights; preserve the implementation's internal FP32 states where applicable |
| Device/batch | One T4, batch size 1 |
| Attention backend | `eager` initially, held fixed throughout the measured run |
| Maximum evaluated input length | 512 tokens including query and candidate continuation; reject overlength examples, never truncate |
| History format | UCI moves from the standard starting position |
| Probe feature | Final normalized language-model activation at the last token of the fixed query suffix |
| Probe family | Linear multi-output ridge classifier on one-hot targets |
| Ridge grid | `[0.1, 1, 10, 100, 1000]` |
| Interventions | Full, three zeroing controls, and both crossed-cache directions |
| Continuation horizons | `0, 1, 2, 4` plies; a ply is one player's move |
| Data/probe seed | `11` |
| Bootstrap seed | `41` |
| Bootstrap replicates | `10000` |

Use the Transformers class compatible with this pinned multimodal checkpoint for text-only forwards; inspect its returned language cache. Do not assume that an `AutoModelForCausalLM` substitution works. Do not install optional fast kernels midway through the experiment. If a backend change is necessary, preserve the failed preflight and create a new version before examining test outcomes.

Two T4s do not automatically provide one combined memory pool. A second GPU may process independent frozen evaluation shards after correctness is established; it is not required by the plan. Do not quantize the primary model to rescue a behavioral failure.

## 4. Real data and exact labels

### Source and collection

Use the January 2013 standard-rated PGN archive linked by the [official Lichess database](https://database.lichess.org/). It is a small archived source of real games, listed at roughly 18 MB compressed. Record its download URL, retrieval date, compressed SHA256, and available publisher checksum. Do not download a current multi-gigabyte monthly archive.

Stream-decompress with `zstandard`, parse using `chess.pgn`, and enumerate moves using `python-chess`. Scan at most the first 20,000 valid games in archive order for the initial candidate pool. If structural quotas cannot be filled, the only allowed expansion is the remainder of this same archive, before running the model. Record whether expansion was needed.

Keep standard chess from the normal initial board. Reject variants, custom starting FENs, malformed/illegal movetext, null moves, and insufficiently long games. For every accepted move, explicitly require membership in `board.legal_moves` before `board.push(move)`: `push` itself does not validate legality. Use only the main line. Strip comments, player metadata, results, annotations, clock text, and engine evaluations from model input.

The verifier supplies labels and chooses evaluation contrasts. Its board state, FEN, legal-move sets, and conclusions never become model input or cache content. No LLM-generated labels or manual language audit is required.

### Labels

Fix square order to `a1, b1, ..., h1, a2, ..., h8` and piece class order to:

`empty, P, N, B, R, Q, K, p, n, b, r, q, k`.

Store 70 categorical targets: 64 square labels, side to move, four castling-right bits in `WK,WQ,BK,BQ` order, and one en-passant target (`none` or a square). This gives 907 output scores for a 64*13 + 2 + 4*2 + 65 linear classifier.

Use raw FEN en-passant semantics consistently: `board.fen(en_passant="fen")` includes the passed-over square after a double pawn move even when capture is unavailable. Store both this FEN and the legal move set. Unit-test this distinction using the [python-chess documentation](https://python-chess.readthedocs.io/en/latest/core.html).

Also store halfmove clock and fullmove number for matching. They are not probe targets. Neither the 70 labels nor a single FEN encodes repetition history. Call the all-head metric "piece-and-rule-state exact match", not complete game-state recovery. This experiment makes no claim about draw rules or optimal play.

### Splits and counts

Assign each unique game a partition using a stable hash of its full UCI move sequence and seed 11. Buckets modulo 100: pilot `0..9`, train `10..59`, development `60..74`, test `75..99`. Identical games receive the same partition.

| Partition | Target size | Purpose |
| --- | --- | --- |
| Pilot | 20 pairs / 40 distinct game histories | Implementation diagnostics and timing; never final evidence |
| Train | 500 states from 500 games | Fit the probe and cheap baseline probes |
| Development | 50 pairs / 100 histories | Probe regularization and competence gates |
| Test | 100 pairs / 200 histories | Fixed primary comparisons |
| Transposition control | Up to 30 additional test triplets | Same-position transfer with a matched different-position donor |

Training cutoffs range from ply 12 through 44. Pair cutoffs range from ply 12 through 40, leaving four later moves when needed. Balance training as evenly as possible across side to move and bands `12..19`, `20..27`, `28..35`, `36..44`, selecting by stable hash within each cell. Do not accidentally train only at even plies.

Exclude repeated selected move prefixes and identical selected piece-and-rule target states across partitions. Apply deterministic priority pilot, train, development, test; later partitions skip collisions and use the next structurally eligible item. Check this exclusion for evaluated continuation targets as well as horizon-zero targets. Keep training target states unique; keep ordinary evaluation targets unique except the intentional same-position duplicates within a transposition triplet. This excludes selected target-state overlap, not common earlier opening positions in the histories. Report shared-opening overlap as a limitation.

Allocate transposition test triplets before ordinary test pairs, with no game reused between them. Every selected game appears in only one evaluation cluster. This keeps resampling straightforward. Test construction may inspect verifier labels and tokenization, but may not inspect model outputs, probe outputs, or scores.

### Ordinary donor pairs

Pair A and B must have equal ply count, exact tokenized prefix length, side to move, total piece count, and check status. Their occupancy must differ on at least two squares. Select without replacement using stable hash order. Assign A/B orientation by hash, not by easier performance.

For each pair construct moves `a` and `b` such that:

- `a` is legal in A and illegal in B.
- `b` is legal in B and illegal in A.
- Their complete scored continuation strings have equal token counts in the fixed query context.
- All scored token IDs are identical across cache conditions and both donor contexts.

Hash-sort legal moves and take the first eligible cross-product pair; use one such candidate pair per donor pair in the primary study. Include all tokens of promotion moves. Reject a donor pair if no eligible candidate pair exists. Do not choose candidates by their language-model likelihoods.

Use up to the target number of structurally eligible pairs. At least 25 development pairs and 50 primary test pairs are required for the full protocol. If fewer exist after the fixed archive expansion, stop and report insufficient assay coverage; do not silently loosen matching.

Save each exclusion reason, actual count, material distribution, ply distribution, and number of changed squares. These matched examples constitute a diagnostic distribution, not an unbiased estimate of all human chess.

Record when the source squares of `a` and `b` were last touched in each history. Flag an older-state subgroup when neither source square appears in either history's last four moves, counting castling rook moves and en-passant captures. Report this subgroup without choosing different candidates or removing other pairs. It helps expose dependence on recent move text; it does not itself rule out history-based heuristics. Also record whether the histories share their final two moves and report coverage of that subgroup.

## 5. Canonical token representation

Use one fixed plain-text representation for this BASE checkpoint:

```text
Chess game from the standard starting position. Moves are in UCI notation.
Moves: e2e4 e7e5 g1f3 ...
Next move:
```

Define `context_text` as the fixed first line plus `Moves: ` and the space-separated move history, without trailing whitespace. Define `query_text` exactly as `"\nNext move:"`. No chat template or thinking-mode switch is part of this base-model protocol.

The saved cache boundary is the end of `context_text`, BEFORE the query. Append the identical query to a clone only when extracting a probe feature or evaluating a candidate. Readouts come from the final query token and therefore measure information accessible after that fixed query computation, not a direct probe of every cache tensor.

Score candidate text `" " + uci_move + "\n"`, including the ending delimiter, as the full continuation after the query. This is conditional next-move preference; it is not unconstrained game-playing accuracy.

Create `render_context`, `tokenize_context_query`, and `candidate_continuation_ids` as the only tokenization entry points. Use the same explicit special-token policy everywhere, initially `add_special_tokens=False`.

For every row, assert:

1. Tokenizing context+query starts with exactly the context IDs.
2. The query suffix IDs match the shared query IDs for every row.
3. Tokenizing context+query+candidate starts with exactly context+query IDs.
4. Candidate suffix IDs match across A, B, all conditions, and the independent full-forward reference.
5. Appending continuation moves preserves the already-cached context prefix IDs.

Do not repair token mismatches by slicing character offsets or assuming standalone word tokens. A systematic boundary failure is an implementation/preparation failure requiring a recorded representation amendment before behavioral results, not a reason to discard most examples silently. Record isolated structural rejections before final freeze.

The pilot is not a prompt search. If the frozen base model cannot perform this assay in the defined representation, report that limitation. Changing UCI to SAN, adding examples, or switching to instruction-tuned weights would be a separate amendment and study.

## 6. Cache backend and invariants

Define a `CacheSnapshot` containing all cache tensors plus non-tensor state needed to resume: sequence length, positions/masks, initialization flags, layer mapping, state indices, and any relevant model-side positional metadata. In the pinned implementation, recurrent layers expose recurrent states and convolution states; preserve both. Attention layers expose keys and values. Verify actual attributes rather than silently accepting unknown layouts.

Notation:

- `R_A`: every recurrent state AND convolution buffer from A.
- `K_A`: every full-attention key AND value tensor from A.
- `C_AA = (R_A, K_A)`, `C_BA = (R_B, K_A)`, `C_AB = (R_A, K_B)`, `C_BB = (R_B, K_B)`.

The first letter always names the recurrent donor. Use this convention in filenames, plots, tests, and formulas.

Required backend operations:

```text
load_backend(config) -> backend, runtime_manifest
prefill(context_ids) -> immutable_snapshot
clone_snapshot(snapshot) -> independent_snapshot
validate_snapshot(snapshot) -> layer_schema_and_byte_counts
assemble_snapshot(recurrent_donor, kv_donor) -> snapshot
zero_channels(snapshot, recurrent=False, attention=False) -> snapshot
advance(snapshot, appended_move_ids) -> new_snapshot
read_feature(snapshot, query_ids) -> final_normalized_hidden_vector
score_candidate(snapshot, query_ids, candidate_ids) -> token_logprobs
snapshot_digest(snapshot) -> exact_tensor_and_metadata_digest
```

No operation may mutate an input snapshot. Every condition, candidate, and query evaluation receives its own clone. Avoid `deepcopy` as an untested assumption: check every tensor's storage independence. Caches can contain nested lists and in-place-updated buffers.

Use one-token cached continuation after the initial prefill for the primary implementation. This reduces ambiguity around carrying Gated DeltaNet state through multi-token calls. Validate against full fresh-prefix processing. Pass/restore positions explicitly as required by the pinned model; do not let the previous donor's model-side RoPE metadata leak into the next call.

For zeroing controls, zero tensor CONTENTS while preserving buffer shapes, past length, initialization flags, rotary positions, and attention-mask shape. Do not set a channel to `None`, shorten its cache, or reset the sequence counter. These controls preserve nonsemantic scaffolding and have distribution-shift limitations. In particular, zeroed KV entries still participate in the attention computation. Label them literally as zeroing, not as a model containing no attention.

The required conditions are:

| ID | Memory at the boundary | Role |
| --- | --- | --- |
| `AA` | `R_A, K_A` | Full A reference |
| `BB` | `R_B, K_B` | Full B reference |
| `BA` | `R_B, K_A` | Recurrent donor changed |
| `AB` | `R_A, K_B` | KV donor changed |
| `R_zero_K` | `R_A, zero(K_A)`; repeat for B | Attention-content erasure diagnostic |
| `zero_R_K` | `zero(R_A), K_A`; repeat for B | Recurrent-content erasure diagnostic |
| `zero_both` | `zero(R_A), zero(K_A)`; repeat for B | Neither original semantic cache retained |

Also perform identity reassembly and restoration controls. Restoring original tensors should recover original outputs numerically; this verifies the intervention machinery, not a semantic discovery.

Do not save every full cache to disk. Stream one pair/triplet at a time, keep only its source snapshots and working clones, and recompute from immutable token IDs when resuming. Save feature vectors and compact numerical results. Record actual cache byte counts; the recurrent matrices can be large enough that hundreds of saved snapshots would consume many gigabytes.

For long prefill calls, avoid retaining vocabulary logits for every token. Use the pinned model's supported final-logit selection or verified language-backbone path; compare it with the reference forward in preflight. Preserve FP32 internal recurrent tensors rather than forcing them to FP16 when cloning.

## 7. Pilot and stop conditions

### Gate A: engineering, before scientific interpretation

Run on the separate pilot partition. All checks must pass:

- Frozen parameters, evaluation mode, no gradients on pretrained parameters, finite activations/logits/caches.
- Correct language hidden dimension and exactly the layer types stated by the loaded text config.
- Applying the frozen output head to the selected normalized hidden feature reproduces the corresponding reference logits; a pre-normalization or wrong-layer activation cannot silently replace it.
- Snapshot copy, identity reassembly, and serialize/restore fixtures preserve every tensor and metadata value.
- Repeating the same token-by-token computation after copy/restore changes candidate token log-probabilities by at most `1e-5` nats. An input snapshot's digest never changes.
- Full fresh-prefix+query/candidate and cached-prefix+query/candidate agree on at least 95% of next-token argmaxes over fixed pilot fixtures; relative L2 hidden error is at most `0.01`, and mean absolute candidate log-probability difference is at most `0.05` nats/token. Record maxima as well as means. These are numerical checks, not allowances for different text.
- Cache poisoning/carry check: equal-length A then B versus B then A gives the same per-example results; condition and candidate evaluation order also leave results unchanged.
- Token boundary, candidate-score alignment, same-length transplantation, and post-continuation sequence-length assertions pass.
- Run the maximum planned 512-token workload with donor snapshots, working clones, and candidate scoring. Record peak allocated/reserved GPU memory and runtime; downloading weights alone is not a fit test.

Failure requires debugging or an explicit versioned amendment. Never reinterpret an implementation failure as absence of chess memory. Use the predeclared 2B-Base fallback only if the 4B resource test fails before behavioral evaluation; record the real revision and rerun all gates.

### Gate B: does the base model respond to the actual position?

On all available development pairs, use the model's own candidate sequence log-probabilities. For each pair, score the SAME `a,b` under full A and full B. Count a success when A prefers `a` and separately when B prefers `b`; ties receive half credit. Average the two decisions within each pair, then across pairs.

Proceed to the full behavioral experiment if this balanced accuracy is at least `0.65` and its paired bootstrap 95% interval has lower bound above `0.50`. This is an assay-competence threshold, not an accuracy target to optimize against. Report the fraction where BOTH full donors choose their legal candidate as well.

A fixed candidate prior alone cannot win both decisions in a balanced pair. Nevertheless, success establishes sensitivity on this contrast, not perfect board tracking or strategic chess skill.

### Gate C: probe feasibility

Fit the readout using all 500 training features and evaluate development features as specified below. Consider it informative for the intended board analysis if its changed-square accuracy exceeds the strongest cheap baseline by at least five percentage points and the paired improvement interval is above zero.

If B passes and C fails, continue the frozen behavioral assay and explicitly report the readout failure. If C passes but B fails, do not claim that recovered boards guide native behavior; stop the expensive causal rollout and write the limited feasibility result. If both fail, stop. Do not fit a nonlinear probe, finetune the model, or change the domain within this frozen study.

### Time/throughput gate

By about two active hours, establish cache correctness or stop for a documented technical limitation. By about three hours, decide whether the scientific assay can proceed. Measure seconds per actual pair, including candidate scoring and cloning, and project the full workload before unsealing test outputs. Feasibility is not established by GPU utilization or a quick model load.

If necessary before test evaluation, use the predeclared compact sample schedule: 50 test pairs, 16 continuation pairs, and 20 transposition triplets, in stable hash order. Otherwise use 100, 32, and 30 respectively. Training/development data, conditions, thresholds, horizons, and analysis stay identical. Choose using measured runtime only and record it in the freeze. If even the compact schedule does not fit the remaining budget, report the pilot rather than dropping controls.

## 8. Linear probe and cheap baselines

Extract only the final normalized hidden vector at the last query token; do not flatten an entire KV or recurrent cache into millions of probe features. This is a deliberate budget/interpretation limit. Use the same feature location for every condition and horizon.

Fit 70 independent linear categorical heads jointly by ridge regression on concatenated one-hot targets. Center targets; center and standardize feature coordinates using TRAIN statistics only. Features with training standard deviation below `1e-6` become zero. Leave intercepts unregularized.

Objective: `sum((XW + b - Y)^2) + lambda * sum(W^2)`, without an implicit division by sample count. A dual solve over the 500 training rows avoids a large feature-space inverse. Convert head scores to labels by argmax. Scores are regression outputs, not calibrated probabilities.

Choose lambda on development accuracy for squares whose true class differs from the initial-board class. Break ties by higher full square accuracy, then larger lambda. Freeze weights, normalization, label order, and lambda before test evaluation. Every intervention and continuation uses this SAME probe, fitted on untouched full-cache activations only.

Required baselines:

1. Initial board: predicts starting occupancy, with no model features.
2. Training-frequency prior: most common class for each square/head in training data.
3. Orderless input baseline: same probe family applied to the mean frozen INPUT embeddings of all move-history tokens, excluding header/query.
4. Last-move input baseline: same probe family applied to the mean frozen INPUT embeddings of only the last UCI move's contextual token span.
5. Label-permutation control: permute entire training target rows with seed 11 and fit the same family. Preserve within-board correlations; report its held-out scores.

Use the same lambda grid and development rule for baseline probes. Record labels used for development selection separately from the 500 training labels. None of these baselines accesses verifier state as features.

Readout metrics: full square accuracy, macro recall across supported piece classes, accuracy on occupied squares, changed-square accuracy versus the initial board, exact 64-square match, each rule-head score/support, and piece-and-rule-state exact match. Report class support and unseen training classes. High unchanged-square accuracy, side-to-move accuracy from ply parity, or always predicting no en-passant is not evidence of rich state tracking.

## 9. Primary held-out causal test

For each test pair, evaluate `AA, BB, BA, AB` and all three zeroing controls for both recipients. Save predictions and complete candidate token log-probabilities for every condition, including failures. Run condition order with a reproducible per-pair permutation and verify source digests afterward.

Let `d(C) = log P(b | C, query) - log P(a | C, query)`, where each log-probability sums all candidate continuation tokens. Positive values favor B's legal move. Candidates have matched token lengths; do not switch between summed and averaged likelihood in different conditions.

Compute per pair:

```text
recurrent_effect = 0.5 * ((d(BA) - d(AA)) + (d(BB) - d(AB)))
kv_effect        = 0.5 * ((d(AB) - d(AA)) + (d(BB) - d(BA)))
interaction      = d(BB) - d(BA) - d(AB) + d(AA)
full_difference  = d(BB) - d(AA)
channel_contrast = kv_effect - recurrent_effect
```

The preregistered primary contrast is mean `channel_contrast` at horizon zero, with a paired 95% interval. Also report both channel effects, their recipient-specific components, interaction, and full_difference. These interventions have unequal numbers of layers/bytes: a larger effect is not greater information per parameter or an intrinsic architectural advantage.

Do not divide by full_difference in the primary analysis: it may be near zero or have the wrong sign. Do not call an effect a percentage of reasoning mediated.

For probes, restrict donor-following to squares where true A and B labels differ. For every prediction report A-label, B-label, or neither. Keep "neither" in the denominator. Also report the mean B-minus-A ridge score on those squares, named a score difference rather than a probability.

Primary results include ALL structurally selected test pairs. A secondary analysis may show pairs where both full donors make the correct behavioral choice, but label that conditioning and show its coverage. It cannot replace the all-pair result.

Define reference labels explicitly: `AA` uses A, `BB` uses B, and each zeroing condition uses its original recipient. A crossed cache has no independently given single gold board. Report its agreement with A and with B separately, along with A/B/neither donor-following. Never select the closer board per example and call that accuracy. Do not use the chess verifier to repair an invalid decoded board before scoring; report the invalid prediction as produced.

A coherent donor-dependent shift in behavior is stronger than a generic score collapse. Zeroing damage alone cannot establish that a channel uniquely contains the state, since it can break the normal cross-channel computation.

## 10. Same-position transposition control

Find test histories A and A' from distinct real games, reaching exactly the same six-field FEN under the chosen en-passant convention, at equal ply and equal token length. Their move histories must actually differ.

Add a different-position history B, matching ply, token length, side, total pieces, and check status. Match token Hamming distance: the number of differing token positions between A and B must be within two of the number between A and A'. Require at least two board squares to differ for A versus B. Choose by hash without model inspection and without reusing a game.

This prevents the comparison being dominated by a tiny textual change for A' and an unrelated transcript for B. Store the actual distances and any remaining imbalance.

Evaluate full A, A', and B; crossed A/A' caches; and crossed A/B caches. Use A-versus-B legal/illegal candidate contrasts to assess behavior in the same-position control. Since next-move preferences can depend on move history even when positions match, do not require identical full-model distributions for A and A'. Show both full baselines.

For state accuracy, compare the mean same-position crossed accuracy to the mean of full A/A'. For legal-candidate preference accuracy, compare analogously. Call a score preserved only if the paired interval's lower bound exceeds a preregistered minus-five-percentage-point margin. Non-significance of damage is not preservation.

The result can support partial invariance to move history in this assay. It cannot prove that a cache is a pure board-state representation: same-position caches can still carry different style, history, and distributed computations.

At least 20 complete triplets are required for the planned transposition comparison. If the fixed archive does not supply them, report the actual availability and mark this control unavailable/underpowered. Do not manufacture reordered games and call them observed human histories. The primary swap result can still be reported with this limitation; do not claim the complete planned milestone was reached.

## 11. Continued state updates

Select the first 32 structurally eligible main test pairs by hash, or 16 under the compact schedule. Eligibility is determined before observing outputs.

For each pair, first try the next four observed moves from A's real game. Require every move to be legal when replayed from BOTH A and B, one at a time. If this fails, try B's observed four-move suffix. If both fail, the pair remains in the primary horizon-zero evaluation but is ineligible for continuation. Never sample moves from the model to decide which pair survives.

The chosen suffix is observed in one real game; applying it to the other position is a verifier-checked counterfactual. Describe this honestly. The intervention study is controlled even though its source games are real.

Require that positions remain distinguishable and that length-matched A-only/B-only legal candidate pairs exist at horizons 1, 2, and 4. Freeze those candidates by the same hash rule. If there are fewer than 16 eligible pairs in the full archive, mark the continuation result underpowered; do not inject illegal moves or fabricate extra observations.

At horizon zero, build each condition from the original snapshots. Advance it with the same suffix tokens. Both channel types run normally and accumulate new state after this one intervention. Do not erase a channel again at every move: that would be a different experiment.

At horizons 1, 2, and 4, clone the trajectory state, append the fixed query, and measure the same probe and native candidate scores. The query and scored candidate tokens must never be fed back into the trajectory that advances to the next horizon.

Evaluate the four crossed/full conditions plus the three zeroing controls for both recipient orientations. For zeroing conditions, "zeroed at the boundary" is the correct label; their erased channels can subsequently refill.

Report two distinct sets of squares:

- Squares whose gold labels change between the original position and the current horizon: tests updating against a baseline that simply keeps the original predicted board.
- Squares where A and B still differ and which the shared suffix has not touched, using from/to squares plus rook squares for castling and the captured square for en-passant: tests persistence of donor-specific information beyond the new move text.

Also report overall and exact board scores and the behavioral effects at every horizon. Record empty metric subsets as unavailable, not zero. A static horizon-zero prediction and each untouched full-cache trajectory are the update baselines.

For crossed trajectories, compare predictions separately with A advanced by the shared suffix and B advanced by that suffix. For zeroing trajectories, use the original recipient advanced by the suffix. Apply the changed-square and untouched-square definitions under each named reference, and keep that reference fixed in figures. There is no privileged gold hybrid trajectory to infer from whichever prediction looks better.

Interpretation: channel influence surviving continuation supports persistent carry-over through the active hybrid model. It does not show that the recurrent layers alone execute chess transitions or that attention contains only a transcript lookup.

## 12. Statistics and interpretation

Keep per-example results and resample paired clusters, not individual squares or candidate tokens. Ordinary A/B pairs are clusters; transposition A/A'/B triplets are clusters. Never reuse a game across these clusters. Keep both directions, all horizons, all conditions, and baseline observations together during resampling.

Use 10,000 bootstrap resamples with seed 41 and 95% percentile intervals. The horizon-zero channel contrast is primary. Treat other channel comparisons, horizons, rule heads, and subgroups as secondary estimates; report all of them and avoid claiming a discovery because one among many unplanned intervals excludes zero. These intervals concern sampling of the chosen game contrasts, not variation across model training seeds.

Do not report individual squares as independent sample size. Caption figures with number of independent pairs/triplets and actual histories. Report failed numerical evaluations explicitly. A non-finite forward is a technical failure, not a missing row to omit from the denominator; stop the affected stage for diagnosis.

| Observation | Supported reading | Unsupported leap |
| --- | --- | --- |
| B's recurrent cache shifts behavior and readouts toward B | Recurrent memory contributes donor-specific information at this intervention boundary | Recurrence is the sole board representation |
| KV swaps dominate the matched behavioral contrast | KV carry-over has greater measured influence in this setup | The model necessarily reconstructs everything from raw text |
| Both erasures hurt, swaps give no coherent donor direction | Dependence on intact hybrid computation; attribution remains unresolved | A clean distributed board circuit has been found |
| Probe follows B, behavior does not | Accessible information and measured behavior disagree | The model uses the decoded board |
| Behavior follows B, linear probe fails | Behavior is sensitive to donor information; this readout misses the relevant structure | No explicit state exists, or information is broadly hidden |
| Both channels can support some behavior after erasure | Potential redundancy or recovery in the remaining active network | Both independently implement the same world model |
| Same-position swaps work, different-position swaps steer behavior | Evidence consistent with state-sensitive transfer beyond this degree of history change | A pure canonical state space has been proven |
| Additional moves erase the difference | Influence did not persist under this continuation | The original memory contained no state |
| Full model fails the competence gate | This base model/format/assay did not support the intended test | Qwen cannot represent worlds or the hypothesis is disproven |

## 13. Figures and readable outputs

Generate real figures only after evaluation using Matplotlib; export SVG, PDF, and 300-dpi PNG. Use a white background, dark text, one consistent blue for KV, orange for recurrent, dark gray for full, and pale gray for zeroing. Keep the same mapping everywhere and use line styles as well as color. Use readable labels and units, not raw condition variable names alone.

1. `01_memory_conditions`: small cache-intervention schematic plus two aligned panels for balanced legal-candidate accuracy and changed-square probe accuracy of untouched and zeroed recipient conditions, with uncertainty intervals and baseline lines. Put exact board recovery in a clearly labeled inset/table even when near zero. Put crossed conditions in the donor-following figure, since they have no unique gold recipient.
2. `02_donor_following`: recurrent/KV effect estimates in log-probability units; A/B/neither probe outcomes on differing squares; transposition-control comparison with its sample count. Do not conflate regression scores with log-probabilities on one axis.
3. `03_continuation`: horizons 0,1,2,4 for native behavioral influence, updating on changed squares, and retention on untouched differing squares. Include full and static-prediction baselines, sample counts, and shaded paired intervals.

Include one compact board illustration for the median primary-effect pair, selected by an explicit deterministic rule, with source boards and decoded hybrid board. Also include a failure example if any exist. Show all chosen example IDs and selection rules; label illustrations as examples, not the population estimate.

Check exported plots for clipped labels, color consistency, correct axis limits, intervals, and whether a reader can identify the intervention without reading the code. Do not hide absent controls behind empty decorative panels: label them unavailable and explain why.

## 14. Required tests

### Local tests, no model download

1. UCI replay and square/class mapping round-trip correctly.
2. Castling, en-passant, promotion, side to move, and rights changes match hand-specified fixtures.
3. Illegal/null moves and nonstandard starts are rejected.
4. Game/prefix/selected-target split exclusions are deterministic; deliberate within-triplet transpositions remain permitted.
5. Opposing behavioral candidates are exclusively legal for the correct donor and have matched contextual token lengths.
6. A move suffix legal only for one donor is rejected for continuation.
7. Same-position matching includes all six FEN fields and the token-distance rule.
8. Mock hybrid snapshots preserve metadata, copy both recurrent and convolution states, and never share mutable tensors.
9. Zeroing preserves lengths/flags and removes the specified tensor contents only; zero-both cannot accidentally reload either donor.
10. Scoring sums all target tokens including the delimiter, does not score the query as a target, and handles unequal token lengths by structural rejection.
11. A scoring/query branch cannot contaminate another candidate or later trajectory.
12. Probe normalization sees training rows only; ridge heads reconstruct known linearly decodable fixtures; permutation controls are independent of test labels.
13. Factorial-effect formulas have the expected signs on a constructed lookup table; donor "neither" remains in the denominator.
14. Bootstrap keeps all observations from a pair/triplet together; empty subsets produce unavailable values.
15. Freeze/resume rejects changed model revision, tokenizer, script, data, conditions, or probe state; resumed rows match uninterrupted execution and are never duplicated.

### Real-model preflight

Run Gate A on the chosen GPU and checkpoint. Test mock passing separately from real model passing in the report. Do not use a fake backend's scores as experiment results. Record every numerical comparison and the candidate/token IDs used.

## 15. Stages, artifacts, and implementation order

Implement these stage names as CLI subcommands. The example sequence below is the intended interface, not a claim that the commands already exist:

```text
prepare -> validate-data -> preflight -> extract-features -> fit-probe
        -> pilot -> freeze -> evaluate -> analyze -> verify
```

`prepare` resolves model/tokenizer revisions, downloads/parses the fixed archive, builds deterministic candidate pools, pairs, continuations, and label manifests. `validate-data` runs verifier and split/token assertions without loading model weights.

`preflight` handles cache correctness, model immutability, and resource timing on pilot histories. `extract-features` processes train/development only. `fit-probe` fits readouts/baselines, selects lambda on development, and writes their exact feature contracts.

`pilot` evaluates the development behavioral gate, probe gate, and runtime projection. `freeze` records the chosen full/compact schedule and hashes config, manifest, script, dependency lock, tokenizer files, model revision, probes, and this plan. It must refuse if Gate A/B do not permit the requested main study.

`evaluate` is the first scientific scoring of test examples. It runs all required conditions on the frozen rows and uses an append-only per-cluster journal with atomic completion markers. Resume completed clusters without selecting a different sample. A process restart must load exactly the same freeze and source hashes. If a real code bug is discovered after test access, preserve the old run, document the correction and prior access, create a new version, and rerun every affected comparison. Such a rerun is a disclosed correction, not a newly untouched confirmation set.

`analyze` produces summary JSON/CSV and plots using saved outputs only, without loading model weights. `verify` replays verifier labels, checks sample coverage, hashes, numerical gates, and protocol adherence, and reports completed versus unavailable sections.

Required run artifacts:

```text
environment.json
config.json
manifest.json
data_audit.json
preflight.json
probe_config.json
probe_weights.npz
baseline_probe_weights.npz
pilot.json
freeze.json
features/train.npz
features/development.npz
results/primary.jsonl
results/transpositions.jsonl
results/continuations.jsonl
summary.json
metrics.csv
figures/
verification.json
time_log.csv
decisions.md
```

Save enough per-candidate token IDs and log-probabilities to recompute every behavioral metric; enough per-head predictions/scores and gold labels to recompute probe metrics; and every pair's actual context text and continuation. Do not keep only aggregate summaries.

Record model weight-file hashes and full loaded-parameter digests before/after the measured model run using streaming tensor chunks. Any sampled parameter hash must be explicitly labeled as sampled and cannot replace the full immutability check. Keep the probe fit on cached CPU features separate from the frozen model.

Build the Kaggle ZIP only after local tests pass. Include the new script, tests, requirements, plan/runbook, and frozen data/config; exclude credentials, model weights, downloaded archive, unrelated experiments, and generated cache dumps. A Kaggle real preflight remains necessary after upload.

## 16. Time budget and milestones

These are planning caps, not measured runtimes. Maintain a real active-time ledger. The new filename does not reset time already spent on this application task; use the quoted program rules and report earlier work honestly. Track any separately allowed executive-summary time separately.

| Additional active time | Work and stopping point |
| --- | --- |
| 0:00-0:30 | Freeze config, implement verifier/data preparation and token contracts |
| 0:30-2:00 | Implement backend/cache operations and tests; establish real cache correctness |
| 2:00-3:00 | Extract train/dev features, fit probes, run competence/timing gates |
| 3:00-3:30 | Finish validation, select schedule using timing, write freeze |
| 3:30-6:30 | Run primary, transposition, and continuation evaluations with resumable logging |
| 6:30-7:30 | Analyze, render/inspect figures, verify artifacts and limitations |
| 7:30-9:30 | User writes main research report from evidence and decision log |
| 9:30-11:30 | Reproduction checks, packaging, executive summary, or limited debugging buffer within the applicable cap |

Do not spend the writing reserve searching for a positive effect. If the time cap is reached, summarize the completed stages and explicitly label the remaining tests unrun. Never lower gates, remove a weak condition, or increase the sample after seeing test results.

The minimum valid feasibility output is a verified cache implementation plus an honest full-model competence/probe result. That can document useful progress, but it is not equivalent to the complete planned causal study and cannot promise a competitive application.

The full milestone is one reproducible account of which cache intervention changes position-sensitive behavior, how that relates to linear board recovery, whether the effect tolerates same-position history changes, and how it behaves after further moves. The measured answer may be a dissociation, a shared dependence, or an unresolved attribution.

## 17. Writeup evidence structure

The user should write the application text in their own words, consistent with the program's current AI-use policy. This file is an implementation/protocol document, not application prose.

Start the report with a 1-3 page executive summary containing the question, exact model/task, one intervention diagram, the strongest measured result, strongest counterevidence, and the main limitation. Then include methods, all main figures, a compact hypotheses/results table, failures, prior-work attribution, reproduction steps, and the active-time/LLM-use account.

Useful reflection evidence includes errors actually caught: a shallow cache clone, a missing convolution buffer, incorrectly reset positions, a candidate that is illegal in both boards, a query contaminating the rollout, a deceptive unchanged-square baseline, or an unsupported claim that probes establish use. Do not claim any of these occurred unless recorded during the real work.

The final claim should name the model, UCI representation, selected positions, intervention boundary, and outcome. Expensive compute, a positive effect, and acceptance are not completion criteria; correct implementation and conclusions supported by the saved evidence are.
