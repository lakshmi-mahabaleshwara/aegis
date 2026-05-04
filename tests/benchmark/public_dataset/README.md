# Public Benchmark Dataset

## Purpose

This directory holds the public DICOM collections used to benchmark AEGIS
against competing de-identification tools. The datasets are freely
available, contain a mix of modalities, burnt-in annotations, and DICOM
metadata that exercises the full AEGIS pipeline.

All datasets are described in a single registry — [`datasets.yaml`](datasets.yaml) —
which is consumed by both the preparation script and the benchmarking
notebook. **To add a new dataset, append an entry there**; no script or
notebook edits required (unless the source type is new).

---

## Layout

| Path | Purpose |
|------|---------|
| [`datasets.yaml`](datasets.yaml) | Single source of truth — modalities, sources, target dirs, defaults. |
| [`pydicom_samples/`](pydicom_samples/) | In-repo smoke tier (56 files), prepared from pydicom's built-in test data. |
| [`benchmark_large_datasets.ipynb`](benchmark_large_datasets.ipynb) | Interactive download + benchmark + plots for the larger registered datasets. |
| `ground_truth/` *(optional)* | PHI-detection annotations for precision/recall/F1 scoring. |

---

## Smoke tier — `pydicom_samples`

**Source**: [pydicom](https://github.com/pydicom/pydicom) built-in test data
**Files**: 56 DICOMs · **Modalities**: CT, MR, US, CR, OT · **License**: MIT

These cover a wide range of transfer syntaxes, pixel formats, character
sets, and edge cases (big-endian, JPEG 2000, RLE, multi-frame,
truncated). Small enough to commit; runs offline.

**Prepare:**

```bash
python scripts/prepare_pydicom_samples.py
```

Filters (accepted modalities, suffix, pixel-data requirement) come from
the `pydicom_samples` entry in [`datasets.yaml`](datasets.yaml). Override
the registry path or target dir with `--registry` / `--target`; see
`--help` for the full surface. The script is re-runnable.

**Run the benchmark:**

```bash
python scripts/bench_aegis.py \
    --input  tests/benchmark/public_dataset/pydicom_samples \
    --config monai_aegis/config/config.yaml \
    --output tests/benchmark/results/ \
    --mode   single
```

See [`scripts/bench_aegis.py`](../../../scripts/bench_aegis.py) for the
full flag surface (including `--ground-truth` and `--mode series`).

---

## Larger datasets

Larger collections (TCIA HNSCC, RSNA Pneumonia, BUSI, NIH Chest X-ray 14)
are registered in [`datasets.yaml`](datasets.yaml) and driven by
[`benchmark_large_datasets.ipynb`](benchmark_large_datasets.ipynb).
The notebook handles:

- **Selection** — pick a registered key (`tcia_hnscc`, `rsna_pneumonia`, …).
- **Preparation** — dispatches on `source.type` (`kaggle`, `tcia`, `http`).
  Auth-gated sources print the next step rather than failing silently.
- **Benchmark** — invokes `scripts/bench_aegis.py` with defaults from
  `benchmark_defaults` in the registry, overridable per-entry.
- **Visualization** — runtime, redacted-region, and status histograms
  from the latest results CSV.

To add a new dataset, append an entry to [`datasets.yaml`](datasets.yaml).
If the `source.type` is one the notebook doesn't yet know about, register
a preparer in section 4's `PREPARERS` dict.

---

## Ground-truth annotations (optional)

For PHI-detection precision/recall/F1 benchmarking, create ground-truth
labels for a subset of images:

```
tests/benchmark/public_dataset/ground_truth/
├── annotations.csv          # See schema below
└── README_annotations.md    # Labelling protocol
```

**`annotations.csv` schema:**

```csv
file_path,text,x_min,y_min,x_max,y_max,is_phi,phi_type
pydicom_samples/CT_small.dcm,John Doe,10,5,120,25,true,PATIENT
pydicom_samples/CT_small.dcm,GE Medical,300,5,420,25,false,
```

Pass the file via `--ground-truth` to `bench_aegis.py` to enable
per-file precision / recall / F1 columns in the results CSV.

---

## .gitignore

Large datasets should NOT be committed. The `pydicom_samples` directory
is small enough to commit; add these for the rest:

```
tests/benchmark/public_dataset/tcia_*
tests/benchmark/public_dataset/rsna_*
tests/benchmark/public_dataset/busi_*
tests/benchmark/public_dataset/nih_*
tests/benchmark/public_dataset/ground_truth/
tests/benchmark/results/
```
