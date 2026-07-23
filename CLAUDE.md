# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

A cytoskeletal organization classifier for microscopy images. It segments cells from two-channel
CZI images (DAPI nuclear stain + tubulin), extracts per-cell morphological/intensity/texture
features, and trains a Random Forest classifier to distinguish `doxpos` vs `doxneg` conditions
(dox = doxycycline-inducible system), based on the condition label embedded in each filename.

The codebase is currently a single script, `script_20260722_213507.py` (filename is an
auto-generated timestamp, not meaningful). It is a work in progress — as of this writing the
`main()` pipeline is incomplete, ending mid-way through the feature-importance reporting section.

## Environment & running

Dependencies are managed with [pixi](https://pixi.sh) (see `pixi.toml` / `pixi.lock`). There is a
single environment, `cell-class`.

```sh
# install/sync the environment
pixi install

# run the pipeline script inside the pixi environment
pixi run python script_20260722_213507.py

# or drop into an activated shell
pixi shell -e cell-class
```

No `[tasks]` are defined in `pixi.toml`, no test suite, and no linter/formatter config exist yet.

Key dependencies: `bioio` (CZI reading), `numpy`, `pandas`, `scipy`, `scikit-image`,
`scikit-learn`, `matplotlib`.

## Pipeline architecture

The script runs a linear pipeline over a folder of `.czi` files:

1. **Discovery & labeling** (`extract_condition_from_filename`) — condition (`doxpos`/`doxneg`) is
   inferred purely from substrings in the filename; files that match neither are skipped with a
   warning.
2. **Channel loading** (`load_image_channels`) — uses `bioio.BioImage` to load a file, then indexes
   into the DAPI and tubulin channels (`DAPI_CHANNEL = 0`, `TUBULIN_CHANNEL = 1`) at a single
   z-slice (`Z_SLICE = 0`). Handles TCZYX, CZYX, and ZYX dimension orderings from `img.data`.
3. **Nuclei segmentation** (`segment_nuclei`) — percentile threshold on normalized DAPI intensity,
   small-object removal, hole filling, connected-component labeling.
4. **Cell segmentation** (`segment_cells`) — seeded watershed on the tubulin channel, using nuclei
   labels as markers and a distance transform as the elevation map.
5. **Per-cell feature extraction** (`extract_morphological_features`) — for each labeled cell:
   shape features (area, perimeter, eccentricity, solidity, aspect ratio via
   `skimage.measure.regionprops`), DAPI/tubulin intensity stats, tubulin texture (local variance,
   Laplacian-based granularity), DAPI–tubulin correlation, and coefficient-of-variation features.
   Cells smaller than 10 px are dropped.
6. **Aggregation** (`process_single_image`, `main`) — features from all cells across all images are
   concatenated into one `pandas.DataFrame`, written to `results/cell_features.csv`, then used to
   train/cross-validate a `RandomForestClassifier` predicting condition from the extracted
   features (after `StandardScaler` normalization).

Configuration (input/output folders, channel indices, z-slice, thresholds) is set via module-level
constants near the top of the script rather than CLI args or a config file.

## Conventions worth preserving

- Segmentation/normalization functions consistently guard divide-by-zero with a small epsilon
  (`+ 1e-8`) rather than special-casing.
- Feature extraction returns `None` (not raising) for cells that fail validity checks, and callers
  filter `None` out before concatenation — follow this pattern rather than raising exceptions for
  expected per-cell segmentation issues.