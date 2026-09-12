# MemorySwap

### Which memory carries the chess position?

A causal study of the two memory systems inside **Qwen3.5-4B-Base**.

[Read the notebook](notebooks/qwen35_chess_memory_results_colab.ipynb) · [Open in Colab](https://colab.research.google.com/github/eigengenesis/memoryswap/blob/main/notebooks/qwen35_chess_memory_results_colab.ipynb) · [Download the artifacts](https://github.com/eigengenesis/memoryswap/releases/tag/v1.0) · [Protocol](docs/qwen35_chess_memory_plan.md)

---

Give a language model a chess game as a sequence of moves. It sees no board. Where does it keep the information needed to distinguish a legal next move from an illegal one?

Qwen3.5 has two kinds of memory: recurrent states in 24 Gated DeltaNet layers, and key/value caches in eight full-attention layers. I exchanged these memories between matched game histories, then measured which position influenced the model's move preference.

**Swapping attention memory had the larger donor-specific effect. But erasing recurrent memory reduced performance to chance.** The two interventions reveal different aspects of how the model depends on its memory.

## The result at a glance

| Across 100 held-out position pairs | Estimate | 95% bootstrap interval |
| :--- | ---: | ---: |
| Recurrent donor effect | 0.35 nats | 0.19 to 0.52 |
| Attention KV donor effect | 3.51 nats | 2.96 to 4.08 |
| KV minus recurrent | **3.16 nats** | **2.60 to 3.71** |

An effect measures the shift in log-probability toward the donor position's legal move. Larger values mean stronger donor influence.

![Donor influence from recurrent and attention memory, mixed board readouts, and same-position controls](figures/02_donor_following.png)

*Left: KV swaps had the larger behavioral effect. Middle: decoded boards mixed information from both positions. Right: the same-position behavioral control began at chance, limiting what its unchanged score can tell us.*

## What was swapped?

Each pair contained two real Lichess histories, **A** and **B**, matched on ply count, tokenized length, side to move, piece count, and check status. A chess rules engine identified move **a**, legal only in A, and move **b**, legal only in B. Candidate selection did not use model scores.

| Cache condition | Recurrent memory | Attention memory |
| :--- | :--- | :--- |
| AA | A | A |
| BA | B | A |
| AB | A | B |
| BB | B | B |

For each condition, the experiment scores the full candidate continuations:

```text
preference = log P(move b) - log P(move a)
```

Positive values favor B's move. The factorial comparison separates recurrent effects, KV effects, and their interaction. Additional controls erase memory contents, swap histories reaching the same position, and continue the game for up to four more plies.

Only linear ridge readouts were trained. **The language model stayed frozen throughout.**

## The complication that matters

With intact memory, balanced legal-candidate accuracy was **78.5%**. Erasing KV content reduced it to **55.5%**; erasing recurrent content reduced it to **49.5%**.

![Legal-candidate accuracy and linear board readout under memory erasure](figures/01_memory_conditions.png)

*Both memory channels matter under erasure. Because zeroing creates an abnormal internal state, the performance loss can include general disruption.*

Board information was partially readable from the final query activation: **61.0%** accuracy on squares changed from the starting position, against **53.0%** for the strongest cheap baseline. Exact recovery of all 64 squares was only **0.5%**. Crossed memories did not yield a clean reconstruction of either donor's board.

KV donor influence also persisted after additional moves. Both kinds of layers processed those moves, so the experiment does not isolate the computation that updates the position.

<details>
<summary><strong>View the continuation results</strong></summary>

![Behavioral influence, board updating, and donor following after additional moves](figures/03_continuation.png)

Continuation results use 32 independent pairs at 0, 1, 2, and 4 additional plies. Shaded regions show paired bootstrap intervals.

</details>

## Scope

This result supports a specific claim: **boundary-carried attention KV content had greater donor-specific causal influence on move preference in this frozen model and chess assay.**

It does not identify an exclusive storage location for a complete board. Whole-channel swaps can create unfamiliar internal states; the two layer types exchange information during normal processing; and the linear readout has not been shown to mediate the model's behavior. The study covers one checkpoint and one matched chess distribution.

Earlier Atlas composition experiments motivated the question of whether internal state can transfer between computations. No Atlas interface or continual-learning mechanism was trained in this study.

## Read, inspect, reproduce

| What you want to do | Start here |
| :--- | :--- |
| Read the results with figures | [Results notebook](notebooks/qwen35_chess_memory_results_colab.ipynb) |
| Check hashes and recompute primary effects on CPU | [Open in Colab](https://colab.research.google.com/github/eigengenesis/memoryswap/blob/main/notebooks/qwen35_chess_memory_results_colab.ipynb) and run all cells |
| Inspect numerical results | [Summary](results/summary.json) · [Metrics](results/metrics.csv) · [Verification](results/verification.json) |
| Inspect held-out examples | [Primary journal](results/primary.jsonl) · [Continuations](results/continuations.jsonl) · [Transpositions](results/transpositions.jsonl) |
| Rerun the model experiment | [Runbook](docs/qwen35_chess_memory_runbook.md) · [Script](qwen35_chess_memory.py) |
| Review the design and decisions | [Protocol](docs/qwen35_chess_memory_plan.md) · [Decision log](results/decisions.md) |

The run used 500 probe-training positions, 100 development positions, 100 held-out primary pairs, 32 continuation pairs, and 30 transposition triplets. Intervals use 10,000 paired cluster-bootstrap samples.

The [release bundle](https://github.com/eigengenesis/memoryswap/releases/tag/v1.0) contains the preserved journals, configuration, verification report, and original figures. It excludes model weights, model caches, extracted features, and probe arrays. Reading and checking saved results requires no GPU; rerunning the full experiment requires downloading the model and following the runbook.

<details>
<summary><strong>Run the local tests</strong></summary>

```bash
python -m pip install -r requirements_qwen35_chess_memory.txt
python -m pytest tests/test_qwen35_chess_memory.py -q
```

</details>

## Credits

The crossed-cache design follows Afendulev et al., [*What Attention Recalls and Recurrence Controls in Hybrid Language Models*](https://arxiv.org/abs/2609.04434). Related board-state work includes Neel Nanda's [*Actually, Othello-GPT Has A Linear Emergent World Representation*](https://www.neelnanda.io/mechanistic-interpretability/othello).

Model: [Qwen3.5-4B-Base](https://huggingface.co/Qwen/Qwen3.5-4B-Base). Games: [Lichess Open Database](https://database.lichess.org/). Board and legality labels: [python-chess](https://python-chess.readthedocs.io/).
