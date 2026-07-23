# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

A cytoskeletal organization classifier for microscopy images. It segments cells from two-channel
CZI images (a nuclear stain in the `nuclei` channel, e.g. DAPI, + a cytoskeletal stain in the
`cells` channel, e.g. tubulin — see the `--nuclei-channel`/`--cells-channel` CLI options), extracts
per-cell morphological/intensity/texture features, and trains a Random Forest classifier to
distinguish `doxpos` vs `doxneg` conditions (dox = doxycycline-inducible system), based on the
condition label embedded in each filename.

The codebase is currently a single script, `script_20260722_213507.py` (filename is an
auto-generated timestamp, not meaningful). The `main()` pipeline runs end-to-end: discovery,
segmentation, feature extraction, CSV export, Random Forest training/cross-validation, and
feature-importance reporting (CSV + bar plot).

## Environment & running

Dependencies are managed with [pixi](https://pixi.sh) (see `pixi.toml` / `pixi.lock`). There is a
single environment, `cell-class`.

```sh
# install/sync the environment
pixi install

# run the pipeline script inside the pixi environment
pixi run python script_20260722_213507.py

# see all CLI options (input/output folders, channel indices, z-slice)
pixi run python script_20260722_213507.py --help

# or drop into an activated shell
pixi shell -e cell-class
```

No `[tasks]` are defined in `pixi.toml`, no test suite, and no linter/formatter config exist yet.

Key dependencies: `bioio` (CZI reading), `numpy`, `pandas`, `scikit-image`, `scikit-learn`,
`matplotlib`.

## Pipeline architecture

The script runs a linear pipeline over a folder of `.czi` files:

1. **Discovery & labeling** (`extract_condition_from_filename`) — condition (`doxpos`/`doxneg`) is
   inferred purely from substrings in the filename; files that match neither are skipped with a
   warning.
2. **Channel loading** (`load_image_channels`) — uses `bioio.BioImage` to load a file, then indexes
   into the nuclei and cells channels (`--nuclei-channel`, default 0; `--cells-channel`, default 1)
   at a single z-slice (`--z-slice`, default 0). Handles TCZYX and CZYX dimension orderings from
   `img.data`; any other dimensionality (e.g. a bare ZYX single-channel image) is unsupported and
   logged/skipped rather than guessed at.
3. **Nuclei segmentation** (`segment_nuclei`) — median filter denoise, Li's threshold
   (`filters.threshold_li`) on the nuclei channel, small-object removal, connected-component
   labeling (`skimage.measure.label`). Hole filling (`morphology.remove_small_holes`) is present in
   the code but currently commented out.
4. **Cell segmentation** (`segment_cells`) — seeded watershed on the cells channel, using nuclei
   labels as markers and the Sobel gradient of the (Gaussian-smoothed) cells channel as the
   elevation map, so flooding stops at real intensity edges between touching cells rather than at a
   purely geometric distance from the foreground mask boundary.
5. **Per-cell feature extraction** (`extract_morphological_features`) — for each labeled cell:
   shape features (area, perimeter, eccentricity, solidity, aspect ratio via
   `skimage.measure.regionprops`, unaffected by intensity normalization below), nuclei/cells
   intensity stats, cells texture (Laplacian-based granularity, precomputed once per image),
   nuclei–cells correlation, and a cells coefficient-of-variation feature. Cells smaller than 10 px
   are dropped.

   Before any intensity feature is computed, both channels are divided by `nuclei_reference` —
   this cell's own mean nuclei-channel (DAPI) intensity — to correct for per-cell/per-image
   illumination and staining-intensity differences. Three features are deliberately *not* computed
   because normalization makes them redundant: `nuclei_mean` (would be ≈1.0 for every cell — it's
   the reference itself), `nuclei_cv` (would be numerically ≈ `nuclei_std`, since the mean it
   divides by is ≈1), and `cells_nuclei_ratio` (post-normalization, `cells_mean` is already
   numerically identical to raw `cells_mean / nuclei_mean`). `nuclei_cells_correlation` is
   unaffected by the normalization — Pearson correlation is invariant to positive scalar rescaling.
6. **Aggregation** (`process_single_image`, `main`) — images are processed in parallel via
   `concurrent.futures.ProcessPoolExecutor` (`--workers`, default `os.cpu_count()`); features from
   all cells across all images are concatenated into one `pandas.DataFrame`, written to
   `results/cell_features.csv`, then used to train/cross-validate a `RandomForestClassifier`
   predicting condition from the extracted features (after `StandardScaler` normalization).
   Cross-validation uses `sklearn.model_selection.LeaveOneGroupOut` grouped by `image_file` (one
   fold per image, manually looped with `sklearn.base.clone` so each fold gets a fresh untrained
   classifier) rather than a plain K-fold split — cells from the same image are not independent
   samples (they share illumination/staining/segmentation-threshold batch effects), so a
   cell-level split would leak every image into every fold and inflate the accuracy estimate.
   Per-fold accuracy is printed with the held-out image's filename, which is useful diagnostic
   signal on its own with only a handful of images (e.g. one image scoring far below the rest is
   worth investigating before trusting the aggregate number).

Configuration (input/output/QC folders, channel indices, z-slice, worker count) is exposed via
`argparse` CLI options (`parse_args`), with defaults matching the original hardcoded values.
Thresholds and other algorithm parameters (e.g. `median_filter_size`, `gaussian_sigma`) remain
function-default arguments, not CLI-configurable.

Since each worker starts its own `bioio-bioformats` JVM instance (via `scyjava`), parallelism
trades memory/startup cost for throughput — `--workers 1` restores fully sequential behavior. The
per-cell progress bar inside `process_single_image` (`show_progress`) is only enabled when
`--workers 1`, since multiple processes writing `tqdm` bars to the same terminal at once garbles
the output; the outer per-image progress bar (in `main`) always runs single-process and stays
clean regardless of worker count.

## Conventions worth preserving

- Segmentation/normalization functions consistently guard divide-by-zero with a small epsilon
  (`+ 1e-8`) rather than special-casing.
- Feature extraction returns `None` (not raising) for cells that fail validity checks, and callers
  filter `None` out before concatenation — follow this pattern rather than raising exceptions for
  expected per-cell segmentation issues.