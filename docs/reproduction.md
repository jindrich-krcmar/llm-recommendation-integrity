# Reproduction

There are two paths: rescore the included final answers without a GPU, or retrain
and regenerate all conditions. The published result files are historical outputs;
new runs write to ignored `outputs/` directories and separate result files.

## 1. Verify the published results

Clone the repository and run from its root with Python 3.12:

```text
python -m recommendation_integrity.evaluation.published
python -m unittest discover -s tests
```

No package installation, model download, or GPU is required for this path. The
verifier recomputes all final summaries, conditional outcomes, six paired
comparisons, grouped uncertainty, biased-policy agreement, and repair gates from
the compressed saved completions. Expected output includes `answers_rescored: 18000`
and `all_declared_metrics_and_gates_match: true`.

The gzip files contain JSONL. `cases.jsonl.gz` stores 3,600 complete cases once;
each condition file stores its case ID, raw completion, generated token IDs, and
prompt length. Scores and repeated prompt text are regenerated, not trusted from
saved prediction fields. File hashes are in `provenance.json`; `summary.json`
contains the audited aggregates. This verifies saved outputs, not fresh model
execution. Token IDs are preserved for inspection but not decoded by the
standard-library verifier.

## 2. Prepare a training environment

Use a fresh checkout/environment for a full rerun. Training/evaluation output
directories must not already exist; the runners deliberately refuse overwrites.
Do not delete historical runs to follow these instructions in an existing checkout.

The experiment was run on native Windows using Python 3.12.6, NVIDIA CUDA,
BF16, PyTorch 2.11.0+cu128, Transformers 5.21.0, PEFT 0.21.2, TRL 1.16.0,
Accelerate 1.15.0, and Datasets 5.1.0. The complete tested dependency snapshot is
[`requirements-windows.lock.txt`](../requirements-windows.lock.txt).

Model training/generation currently requires a CUDA GPU with BF16 support and
an appropriate NVIDIA driver. The loaders explicitly select `cuda:0`; they do
not implement CPU, MPS, ROCm, multi-GPU, or distributed execution. A cloud VM
must expose its NVIDIA GPU to the guest/container. GPU smoke checks must pass
before spending time on the experiment.

### Tested Windows/CUDA route

With Python 3.12 available as `python`, run in PowerShell:

```powershell
.\scripts\Setup-Environment.ps1
.\.venv\Scripts\Activate.ps1
python -m pip check
```

The setup script creates `.venv`, installs the CUDA PyTorch wheel and locked
dependencies, and runs CUDA/BF16, bitsandbytes NF4, and tiny random-model SFT/DPO
smoke checks. NF4 is an environment check; the actual experiment uses BF16 LoRA,
not quantized training. These checks are not experimental results.

In a new terminal, reactivate `.venv`; optionally dot-source
`. .\scripts\Enter-Project.ps1` to set project-local pip/temp caches. Model runners
set their own project-local Hugging Face cache. No API keys are needed for the
public pinned checkpoint.

### Untested Linux/CUDA starting point

[`scripts/setup-linux.sh`](../scripts/setup-linux.sh) targets a Linux CUDA host
with Python 3.12 already installed. It installs the same core training-package
versions, then runs the GPU and tiny-training checks:

```bash
bash scripts/setup-linux.sh
source .venv/bin/activate
python -m pip check
```

This route has **not been executed or dependency-resolved on Linux**. The direct
dependency pins in [`requirements-linux.in`](../requirements-linux.in) are a
candidate specification, not a Linux lockfile; transitive packages may differ
from Windows. Wheel availability and driver compatibility must be checked on
the target host. Consult [PyTorch's installation selector](https://pytorch.org/get-started/locally/)
when choosing a CUDA build. If versions need adjustment, record the resulting
environment, rerun the integration checks, and treat the rerun as a different
environment rather than silently claiming the original dependency lock.

macOS can be used to inspect/rescore the published data with standard Python.
The current training and generation code does not support macOS CPU/MPS.

## 3. Cache and verify the pinned model

With the chosen environment active, the remaining commands use `python` on
either platform:

```text
python scripts/check_model.py
python scripts/check_model.py --offline
python -m unittest discover -s tests
```

The first model check downloads the exact model/tokenizer revision in
[`configs/model.json`](../configs/model.json). The second verifies fresh-process
offline loading. Subsequent experiment runners use only that local snapshot and
disable remote model code. Their manifests record package versions and source
hashes. These scripts check loading/generation; their single example is not a
baseline accuracy estimate.

## 4. Train the matched SFT branches

```text
python -m recommendation_integrity train-sft --config configs/sft_aligned.json --output-dir outputs/sft-aligned-01
python -m recommendation_integrity train-sft --config configs/sft_biased.json --output-dir outputs/sft-biased-01
python -m recommendation_integrity compare-sft --check-training-only
```

Each branch generates 600 training groups, uses the same seed/prompts/settings,
and completes 675 updates. The audit checks initialization equality, label-only
differences, frozen bases, saved data, and completed budgets. Neither branch
continues from the other. The earlier `sft_pilot.json` capability exercise is
not needed to reproduce the five final conditions.

## 5. Prepare preferences and train both mitigation conditions

```text
python -m recommendation_integrity prepare-dpo
python -m recommendation_integrity train-dpo --data-dir outputs/dpo-data-01 --sft-run outputs/sft-biased-01 --output-dir outputs/dpo-01
python -m recommendation_integrity train-dpo --objective-config configs/dpo_sft.json --data-dir outputs/dpo-data-01 --sft-run outputs/sft-biased-01 --output-dir outputs/dpo-sft-01
```

Preparation reserves the final 200 groups, then creates 300 disjoint preference
groups, their 1,800 pairs, and the 720 development validation pairs. Both training
commands restart from the complete biased SFT adapter and use a separate identical
frozen biased reference. Each completes 339 updates. Do not initialize DPO+SFT
from the failed pure-DPO adapter.

The prepared manifest freezes all settings from `dpo.json`. `dpo_sft.json` adds
only the fixed objective weights. Editing the config after preparation does not
change the already prepared recipe. Both runners save Trainer state and durable
metrics for training/validation curves. A runtime cap that prevents completion
produces an incomplete status and an error; do not treat it as a successful run.

## 6. Optional development comparisons

These reproduce the development reports without using the final set:

```text
python -m recommendation_integrity evaluate --output-dir outputs/pilot-m0-01
python -m recommendation_integrity evaluate --adapter outputs/sft-aligned-01/adapter --output-dir outputs/eval-aligned-01
python -m recommendation_integrity evaluate --adapter outputs/sft-biased-01/adapter --output-dir outputs/eval-biased-01
python -m recommendation_integrity compare-sft
python -m recommendation_integrity evaluate --adapter outputs/dpo-01/adapter --output-dir outputs/eval-dpo-01
python -m recommendation_integrity compare-dpo
python -m recommendation_integrity evaluate --adapter outputs/dpo-sft-01/adapter --output-dir outputs/eval-dpo-sft-01
python -m recommendation_integrity compare-dpo --dpo-run outputs/dpo-sft-01 --after outputs/eval-dpo-sft-01 --output results/dpo_sft_summary.json
```

The original experiment used development observations to choose its single
stabilization variant. The reproduction recipes are now fixed. Further tuning
against the published final data would require a fresh holdout for a new confirmatory test.

## 7. Freeze and run the final comparison

```text
python -m recommendation_integrity final-evaluation prepare
python -m recommendation_integrity final-evaluation run
python -m recommendation_integrity final-evaluation compare --output results/final_comparison_recheck.json
```

Preparation binds the five checkpoints, scientific code, packages, model files,
case file, metrics, and thresholds before final generation. Running evaluates
all five in separate processes and writes 18,000 raw answers under
`outputs/final-evaluation-01/`, plus `results/final_comparison.json`. It skips
only fully completed, audited conditions; partial directories are not silently
overwritten or resumed. Comparing again needs no model loading but requires the
original frozen local artifacts and environment metadata.

On the tested environment, the four training runs used about 43 minutes in total,
and final generation about 32 minutes, excluding installation/model download and
additional development evaluation. These are observations, not runtime guarantees.
Different software, hardware, kernels, or platforms may change outputs or runtime;
whole-run manifests also contain timestamps. The included saved-output verifier
provides an exact numerical audit independent of retraining reproducibility.

## 8. Export a separate result bundle

The tracked `results/published/` files are the original results. To export a new
run without replacing them:

```text
python scripts/export_publication.py --destination outputs/reproduction-publication
python -m recommendation_integrity.evaluation.published --directory outputs/reproduction-publication
```

The exporter audits the frozen local run, stores one shared case file, projects
lossless completion/token fields into deterministic gzip files, and copies the
numerical summary. It computes figure data directly from the audited answers.
No weights, absolute local paths, account information, or development discussions
are exported. If rendered assets are needed in a bundle, pass `--figure-directory`
containing its `final_comparison.png`, `.svg`, and matching `plotted_metrics.csv`;
the exporter rejects figure data from a different run.

The original figure values are included as `results/published/plotted_metrics.csv`
and are independently recomputed by the verifier. Optional figure rendering uses
Matplotlib, outside the training dependency lock:

```text
python -m pip install -r requirements-plots.txt
python scripts/plot_results.py
```

This writes PNG/SVG to `outputs/figures/` using the published data. For a new run:

```text
python scripts/plot_results.py --metrics outputs/reproduction-publication/plotted_metrics.csv --output-directory outputs/reproduction-figures
```

Figure rendering is separate from the standard-library verification and model
training. Font/backend differences may change image bytes while leaving plotted
values unchanged. See the [final report](../results/final_report.md) for interpretation.
