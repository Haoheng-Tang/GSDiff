# AGENTS.md

Guidance for coding agents working in this repository.

## Project Overview

This is the official implementation of the AAAI 2025 paper "GSDiff: Synthesizing Vector Floorplans via Geometry-enhanced Structural Graph Generation". It is a Python/PyTorch research codebase for generating vector floorplans with unconstrained, topology-constrained, boundary-constrained, and LIFULL variants.

The repo is organized as:

- `gsdiff/`: model definitions and geometry/rendering utilities.
  - `house_nn1.py`, `house_nn2.py`, `house_nn3.py`: core node/edge Transformer models for RPLAN-style experiments.
  - `heterhouse_*.py`, `boundary_*.py`, `bubble_diagram_*.py`: architecture variants for different constraints, feature sizes, and LIFULL.
  - `utils.py`, `utils_lifull.py`: geometry, cycle extraction, visualization, and rendering helpers.
- `datasets/`: dataset loaders plus RPLAN and LIFULL preprocessing scripts.
  - `rplan-process*.py`, `rplan-extract.py`, `move.py`: RPLAN data preparation pipeline.
  - `rplang_edge_semantics_simplified*.py`, `rplang_bubble_diagram*.py`, `lifull*.py`: `torch.utils.data.Dataset` implementations.
- `scripts/`: training, testing, autoencoder, image diffusion, and metric scripts.
  - `trainval_main_*.py`: main node-generation training for unconstrained/topology/boundary settings.
  - `trainval_simplified_edge_*.py`: edge model training.
  - `test_main.py`, `test_topo.py`, `test_boun.py`, `test-final-lifull1.py`: inference/evaluation entry points.
  - `metrics/`: local FID/KID implementations.
- Top-level `evalmetric-*.py`: post-hoc metric aggregation and analysis scripts.
- `retrieval.py`: large retrieval/utility script.

## Environment

The intended environment is Python 3.10 with PyTorch 2.0.1 and CUDA 11.8. The dependency files are conda-style exports, not standard pip-only requirement files.

Typical setup:

```bash
conda create -n gsdiff --file requirements.txt
conda activate gsdiff
```

`requirements_full.txt` is a larger linux-64 environment export. Prefer `requirements.txt` unless you are reproducing the exact original environment.

Important dependencies include `torch`, `torchvision`, `numpy`, `opencv-python`, `Pillow`, `networkx`, `scipy`, `scikit-image`, `scikit-learn`, `shapely`, `tensorboardx`, `tqdm`, and `pytorch-fid`.

## Data Layout

Large datasets and generated artifacts are intentionally not included.

RPLAN data expected by loaders:

- `datasets/rplang-v3-withsemantics/{train,val,test}`
- `datasets/rplang-v3-withsemantics-withboundary/{train,val,test}`
- `datasets/rplang-v3-withsemantics-withboundary-v2/{train,val,test}`
- `datasets/rplang-v3-bubble-diagram/{train,val,test}`

The preprocessing pipeline in `README.md` starts from `datasets/rplandata/Data/floorplan_dataset`, then runs `datasets/rplan-extract.py`, `datasets/rplan-process1.py` through `datasets/rplan-process10.py`, then `datasets/move.py`.

LIFULL data expected by loaders:

- `datasets/lifulldata/annot_json/instances_train.json`
- `datasets/lifulldata/annot_json/instances_val.json`
- `datasets/lifulldata/annot_json/instances_test.json`
- `datasets/lifulldata/annot_npy/*.npy`

Model weights are expected under `outputs/...`; test results are written under `test_outputs/...`. The README links downloadable checkpoints that should be placed in `outputs`.

## Running Scripts

Most scripts are direct entry points and have hard-coded settings near the top. There is little/no `argparse`.

Examples:

```bash
python scripts/test_main.py
python scripts/test_topo.py
python scripts/test_boun.py
python scripts/trainval_main_unconstrained.py
python scripts/trainval_simplified_edge_unconstrained.py
python evalmetric-no-constrain-fid-kid.py
```

Before running a training or test script, inspect and adjust:

- `sys.path.append(...)` entries, many of which point to the original author's absolute Linux paths.
- `device = 'cuda:0'` or `device = 'cuda:1'`.
- `output_dir = ...`; many scripts call `os.makedirs(output_dir, exist_ok=False)` and will fail if the directory already exists.
- Checkpoint paths under `outputs/...`.
- Batch sizes and `num_workers`, especially on Windows or CPU-only machines.

Dataset loaders commonly use paths like `../datasets/...`. These resolve relative to the process working directory, so run scripts from the directory their paths expect, or patch the paths carefully.

## Validation And Testing

There is no formal pytest/unittest suite in this repo. Validation is usually script-level:

- For syntax/import sanity, run targeted Python files with the configured environment and expected data/checkpoints.
- For model evaluation, run the corresponding `scripts/test_*.py` script, then aggregate with the appropriate top-level `evalmetric-*.py`.
- FID/KID code renders ground truth and predictions to image folders before computing metrics.

Avoid launching full training/evaluation runs casually: defaults can use the full dataset, GPU, many workers, and long loops.

## Code Conventions And Cautions

- Preserve the research-script style unless the task explicitly asks for a refactor. Many constants encode experiment variants.
- Be conservative with path changes. Hard-coded absolute paths, relative dataset paths, and checkpoint names are part of how the scripts are currently operated.
- Do not delete or overwrite `outputs/`, `test_outputs/`, dataset folders, or checkpoint files unless explicitly requested.
- Many utility files contain non-ASCII comments that appear mojibaked in some editors. Avoid broad reformatting or encoding churn.
- Keep tensor shape contracts intact. Common fixed sizes include 53 nodes and 2809 flattened edges (`53 * 53`), with coordinates normalized around `[-1, 1]`.
- Models often separate node generation and edge prediction stages. Check imports carefully before swapping `house_nn*` or `heterhouse_*` variants.
- If adding a new entry point, prefer explicit top-of-file configuration matching existing scripts over introducing a new framework-wide config system.
- If making scripts more portable, use `Path(__file__).resolve()` or repo-root detection narrowly, and verify that existing relative data expectations still work.
- Generated image metrics should render GT and predictions with identical resolution, colors, and anti-aliasing settings.

## Git Hygiene

- Treat the working tree as shared with the user. Do not revert unrelated changes.
- Keep edits narrowly scoped to the requested behavior.
- Large datasets, generated images, checkpoints, and metric outputs should normally remain untracked.
