# Qwen3.5 Chess Memory Runbook

The frozen scientific protocol is `docs/qwen35_chess_memory_plan.md`. Run every
stage separately and preserve failed artifacts. A passed local test suite does
not replace the real Qwen3.5 preflight on Kaggle.

## Kaggle Setup

Enable one T4 GPU. Upload and extract the supplied ZIP into `/kaggle/working`.
Install the frozen dependencies, restart the notebook session, then set the
Hugging Face cache before importing Transformers:

```bash
cd /kaggle/working
python -m pip install -q -U -r requirements_qwen35_chess_memory.txt
```

```python
import os
os.environ["HF_HOME"] = "/kaggle/working/hf-cache"
```

Use a new run directory. The commands below use:

```text
/kaggle/working/outputs/qwen35_chess_memory/qwen4b-seed11-v1
```

The packaged real-game manifest supplies both preregistered schedules exactly:
the full schedule has 100 primary pairs, 32 continuation pairs, and 30 natural
transposition triplets; the compact schedule has 50, 16, and 20 respectively.
Do not relax the eligibility rules or fabricate trajectories.

During the outcome-blind implementation audit, an early allocator stopped
searching for continuations after accepting 100 ordinary pairs and produced
only 3 continuations. A second candidate omitted transpositions from the
initial-scan quota check and produced 0 triplets. Both candidates were
superseded before any model inference. The final allocator searches the frozen
continuation quota prospectively and requires every structural quota before an
initial archive scan can terminate.

## Ordered Commands

Prepare the immutable real-game manifest and run its verifier:

```bash
python qwen35_chess_memory.py prepare \
  --config data/qwen35_chess_memory/config.json \
  --manifest data/qwen35_chess_memory/manifest.json \
  --run-dir /kaggle/working/outputs/qwen35_chess_memory/qwen4b-seed11-v1 \
  --allow-remote-downloads \
  --resume
```

```bash
python qwen35_chess_memory.py validate-data \
  --config data/qwen35_chess_memory/config.json \
  --manifest data/qwen35_chess_memory/manifest.json \
  --run-dir /kaggle/working/outputs/qwen35_chess_memory/qwen4b-seed11-v1 \
  --resume
```

Run engineering Gate A. Stop and preserve `preflight.json` if
`all_gates_passed` is false:

```bash
python qwen35_chess_memory.py preflight \
  --config data/qwen35_chess_memory/config.json \
  --manifest data/qwen35_chess_memory/manifest.json \
  --run-dir /kaggle/working/outputs/qwen35_chess_memory/qwen4b-seed11-v1 \
  --allow-remote-downloads
```

Only a documented 4B memory/runtime failure before behavioral results permits
the 2B fallback. Preserve the failed 4B run, start a new directory, and pass its
failure JSON to `prepare --use-fallback --fallback-reason PATH`. Use a new
manifest path as well; the runner refuses to relabel or overwrite the packaged
4B manifest as a 2B study.

After Gate A passes, extract only train/development features and fit the frozen
linear readouts:

```bash
python qwen35_chess_memory.py extract-features \
  --config data/qwen35_chess_memory/config.json \
  --manifest data/qwen35_chess_memory/manifest.json \
  --run-dir /kaggle/working/outputs/qwen35_chess_memory/qwen4b-seed11-v1 \
  --allow-remote-downloads
```

```bash
python qwen35_chess_memory.py fit-probe \
  --config data/qwen35_chess_memory/config.json \
  --manifest data/qwen35_chess_memory/manifest.json \
  --run-dir /kaggle/working/outputs/qwen35_chess_memory/qwen4b-seed11-v1
```

Run development Gates B/C and the actual-pair timing projection:

```bash
python qwen35_chess_memory.py pilot \
  --config data/qwen35_chess_memory/config.json \
  --manifest data/qwen35_chess_memory/manifest.json \
  --run-dir /kaggle/working/outputs/qwen35_chess_memory/qwen4b-seed11-v1 \
  --allow-remote-downloads
```

Stop before held-out evaluation if Gate B fails. Gate C failure still permits
the frozen behavioral assay, but not an informative linear-board-readout claim.

Freeze while test journals are still absent. Replace `28800` with the honest
seconds remaining in the research budget. The runner chooses full or compact
using the measured timing only:

```bash
python qwen35_chess_memory.py freeze \
  --config data/qwen35_chess_memory/config.json \
  --manifest data/qwen35_chess_memory/manifest.json \
  --run-dir /kaggle/working/outputs/qwen35_chess_memory/qwen4b-seed11-v1 \
  --remaining-seconds 28800
```

Run the frozen test clusters, analyze saved outputs, and verify everything:

```bash
python qwen35_chess_memory.py evaluate \
  --config data/qwen35_chess_memory/config.json \
  --manifest data/qwen35_chess_memory/manifest.json \
  --run-dir /kaggle/working/outputs/qwen35_chess_memory/qwen4b-seed11-v1 \
  --allow-remote-downloads
```

```bash
python qwen35_chess_memory.py analyze \
  --config data/qwen35_chess_memory/config.json \
  --manifest data/qwen35_chess_memory/manifest.json \
  --run-dir /kaggle/working/outputs/qwen35_chess_memory/qwen4b-seed11-v1
```

```bash
python qwen35_chess_memory.py verify \
  --config data/qwen35_chess_memory/config.json \
  --manifest data/qwen35_chess_memory/manifest.json \
  --run-dir /kaggle/working/outputs/qwen35_chess_memory/qwen4b-seed11-v1
```

For an interrupted `extract-features`, `pilot`, or `evaluate`, rerun the same
command with `--resume`. Never use resume to overwrite a completed artifact or
to change selected IDs. A source-code repair after freeze requires preserving
the old run and starting a disclosed new version.

## Local Verification

```bash
PYENV_VERSION=ml-env PYTHONDONTWRITEBYTECODE=1 \
PYTHONPYCACHEPREFIX=/tmp/qwen35-chess-memory-pycache \
MPLCONFIGDIR=/tmp/qwen35-chess-memory-matplotlib \
XDG_CACHE_HOME=/tmp/qwen35-chess-memory-cache \
pyenv exec python -m pytest -p no:cacheprovider \
  tests/test_qwen35_chess_memory.py -q
```
