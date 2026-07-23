import argparse
import concurrent.futures
import os
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bioio import BioImage
from skimage import morphology, measure, filters, io
from skimage.segmentation import watershed
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

warnings.filterwarnings('ignore')


def parse_args():
    """Parse command-line configuration for the pipeline."""
    parser = argparse.ArgumentParser(
        description='Cytoskeletal organization classifier: segments cells from CZI images, '
                    'extracts per-cell features, and trains a doxpos/doxneg classifier.'
    )
    parser.add_argument(
        '--input-folder', type=Path,
        default=Path('C:/Users/barryd/Downloads/wetransfer_example-images_2026-07-21_1330/Images for Dave'),
        help='Folder containing CZI files (default: %(default)s)'
    )
    parser.add_argument(
        '--output-folder', type=Path, default=Path('./results'),
        help='Folder to write results to (default: %(default)s)'
    )
    parser.add_argument(
        '--qc-folder', type=Path, default=None,
        help='Folder to write segmentation QC images to (default: <output-folder>/segmentation_qc)'
    )
    parser.add_argument(
        '--nuclei-channel', type=int, default=0,
        help='Channel index of the nuclei stain (default: %(default)s)'
    )
    parser.add_argument(
        '--cells-channel', type=int, default=1,
        help='Channel index of the cells stain (default: %(default)s)'
    )
    parser.add_argument(
        '--z-slice', type=int, default=0,
        help='Z-slice index to use (default: %(default)s)'
    )
    parser.add_argument(
        '--workers', type=int, default=os.cpu_count(),
        help='Number of images to process in parallel (default: %(default)s, i.e. all CPU cores)'
    )

    args = parser.parse_args()
    if args.qc_folder is None:
        args.qc_folder = args.output_folder / 'segmentation_qc'
    return args


def extract_condition_from_filename(filename):
    """Extract condition (doxneg or doxpos) from filename."""
    filename_lower = filename.lower()
    if 'doxpos' in filename_lower:
        return 'doxpos'
    elif 'doxneg' in filename_lower:
        return 'doxneg'
    else:
        return None


def load_image_channels(image_path, nuclei_channel, cells_channel, z_slice):
    """Load CZI image and extract nuclei and cells channels from first z-slice."""
    try:
        img = BioImage(str(image_path))

        # Get data shape and extract first z-slice
        data = img.data

        print(f"    Image shape: {data.shape} (dims: {img.dims.order})")

        # Handle different possible dimension orders
        if len(data.shape) == 5:  # TCZYX
            nuclei = data[0, nuclei_channel, z_slice, :, :].astype(np.float32)
            cells = data[0, cells_channel, z_slice, :, :].astype(np.float32)
        elif len(data.shape) == 4:  # CZYX
            nuclei = data[nuclei_channel, z_slice, :, :].astype(np.float32)
            cells = data[cells_channel, z_slice, :, :].astype(np.float32)
        else:
            print(f"    Unsupported dimensionality: {len(data.shape)}D")
            return None, None

        print(f"    Nuclei range: [{nuclei.min():.1f}, {nuclei.max():.1f}], "
              f"Cells range: [{cells.min():.1f}, {cells.max():.1f}]")
        return nuclei, cells
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        return None, None


def segment_nuclei(nuclei_image, median_filter_size=2):
    """Segment nuclei from the nuclei channel using Li's thresholding and morphological operations."""
    # Median filter to denoise while preserving edges
    footprint = np.ones((median_filter_size, median_filter_size), dtype=bool)
    nuclei_filtered = filters.median(nuclei_image, footprint=footprint)

    # Apply Li's threshold
    threshold = filters.threshold_li(nuclei_filtered)
    binary = nuclei_filtered > threshold

    # Remove small objects (hole filling disabled below)
    binary = morphology.remove_small_objects(binary, min_size=10)
    # binary = morphology.remove_small_holes(binary, max_size=binary.size)

    # Label connected components
    nuclei_labels, num_nuclei = measure.label(binary, connectivity=1, return_num=True)
    print(f"    Found {num_nuclei} nuclei (Li threshold={threshold:.1f})")

    return nuclei_labels


def segment_cells(nuclei_labels, cells_image, gaussian_sigma=2):
    """Segment cell boundaries using seeded watershed from nuclei."""
    # Gaussian smoothing to denoise while preserving cell-scale structure
    cells_smoothed = filters.gaussian(cells_image, sigma=gaussian_sigma, preserve_range=True)

    # Apply triangle threshold to obtain foreground mask
    threshold = filters.threshold_triangle(cells_smoothed)
    mask = cells_smoothed > threshold

    # Elevation map for watershed: gradient of the smoothed cells signal, so
    # flooding stops at intensity edges between cells rather than at a purely
    # geometric distance from the mask boundary
    elevation = filters.sobel(cells_smoothed)

    # Apply watershed seeded by nuclei
    cell_labels = watershed(elevation, markers=nuclei_labels, mask=mask)
    num_cells = len(np.unique(cell_labels)) - (1 if 0 in cell_labels else 0)
    print(f"    Segmented {num_cells} cells (triangle threshold={threshold:.1f})")

    return cell_labels


def save_segmentation_qc(image_path, nuclei_labels, cell_labels, qc_folder):
    """Save raw nuclei and cell label images as PNGs for visual QC."""
    nuclei_path = qc_folder / f'{image_path.stem}_nuclei_labels.png'
    cell_path = qc_folder / f'{image_path.stem}_cell_labels.png'

    io.imsave(nuclei_path, nuclei_labels.astype(np.uint16), check_contrast=False)
    io.imsave(cell_path, cell_labels.astype(np.uint16), check_contrast=False)

    print(f"    Saved label images: {nuclei_path.name}, {cell_path.name}")


def extract_morphological_features(cell_mask, nuclei_image, cells_image, cells_laplacian,
                                   cell_label, debug=False):
    """Extract morphological and intensity features from a single cell."""
    t_start = time.perf_counter()
    features_dict = {}

    # Get cell region
    cell_region = cell_mask == cell_label

    if np.sum(cell_region) < 10:  # Skip very small cells
        return None

    # Shape and size features
    t0 = time.perf_counter()
    region_props = measure.regionprops(cell_region.astype(int))[0]
    features_dict['area'] = region_props.area
    features_dict['perimeter'] = region_props.perimeter
    features_dict['eccentricity'] = region_props.eccentricity
    features_dict['solidity'] = region_props.solidity
    features_dict['aspect_ratio'] = region_props.major_axis_length / (region_props.minor_axis_length + 1e-8)
    t1 = time.perf_counter()

    # Nuclei intensity features
    nuclei_values = nuclei_image[cell_region]
    features_dict['nuclei_mean'] = np.mean(nuclei_values)
    features_dict['nuclei_std'] = np.std(nuclei_values)
    features_dict['nuclei_max'] = np.max(nuclei_values)
    features_dict['nuclei_min'] = np.min(nuclei_values)

    # Cells intensity features
    cells_values = cells_image[cell_region]
    features_dict['cells_mean'] = np.mean(cells_values)
    features_dict['cells_std'] = np.std(cells_values)
    features_dict['cells_max'] = np.max(cells_values)
    features_dict['cells_min'] = np.min(cells_values)
    t2 = time.perf_counter()

    # Granularity (Laplacian) - precomputed once per image by the caller
    features_dict['cells_granularity'] = np.mean(np.abs(cells_laplacian[cell_region]))
    t3 = time.perf_counter()

    # Intensity ratio and colocalization
    features_dict['cells_nuclei_ratio'] = (features_dict['cells_mean'] / (features_dict['nuclei_mean'] + 1e-8))
    nuclei_cells_correlation = np.corrcoef(nuclei_values, cells_values)[0, 1]
    features_dict['nuclei_cells_correlation'] = nuclei_cells_correlation if not np.isnan(
        nuclei_cells_correlation) else 0

    # Spatial intensity distribution (coefficient of variation)
    features_dict['nuclei_cv'] = features_dict['nuclei_std'] / (features_dict['nuclei_mean'] + 1e-8)
    features_dict['cells_cv'] = features_dict['cells_std'] / (features_dict['cells_mean'] + 1e-8)
    t4 = time.perf_counter()

    if debug:
        tqdm.write(
            f"    [debug cell {cell_label}] shape={t1 - t0:.3f}s intensity={t2 - t1:.3f}s "
            f"texture_lookup={t3 - t2:.3f}s ratios={t4 - t3:.3f}s total={t4 - t_start:.3f}s"
        )

    return features_dict


def process_single_image(image_path, nuclei_channel, cells_channel, z_slice, qc_folder, show_progress=True):
    """Process a single CZI image and extract features for all cells."""
    print(f"Processing {image_path.name}...")
    condition = extract_condition_from_filename(image_path.name)
    if condition is None:
        print(f"Warning: Could not determine condition for {image_path.name}")
        return None
    print(f"  Condition: {condition}")

    # Load channels
    nuclei, cells = load_image_channels(image_path, nuclei_channel, cells_channel, z_slice)
    if nuclei is None or cells is None:
        print(f"Warning: Failed to load image {image_path.name}")
        return None

    # Segment nuclei and cells
    nuclei_labels = segment_nuclei(nuclei)
    cell_labels = segment_cells(nuclei_labels, cells)
    save_segmentation_qc(image_path, nuclei_labels, cell_labels, qc_folder)

    # Precompute whole-image granularity map once (previously recomputed per cell)
    t0 = time.perf_counter()
    cells_laplacian = filters.laplace(cells)
    print(f"    Precomputed granularity map in {time.perf_counter() - t0:.2f}s")

    # Extract features for each cell
    cell_features_list = []
    unique_labels = np.unique(cell_labels)
    skipped = 0

    non_bg_labels = unique_labels[unique_labels != 0]
    first_label = non_bg_labels[0] if len(non_bg_labels) else None

    for cell_label in tqdm(unique_labels, desc=f"  Extracting features ({image_path.name})", unit="cell",
                           disable=not show_progress):
        if cell_label == 0:  # Skip background
            continue

        cell_features = extract_morphological_features(
            cell_labels, nuclei, cells, cells_laplacian,
            cell_label, debug=(cell_label == first_label)
        )
        if cell_features is not None:
            cell_features['condition'] = condition
            cell_features['image_file'] = image_path.name
            cell_features_list.append(cell_features)
        else:
            skipped += 1

    print(f"    Kept {len(cell_features_list)} cells, skipped {skipped} (too small)")

    if cell_features_list:
        return pd.DataFrame(cell_features_list)
    else:
        return None


def main():
    """Main analysis pipeline."""
    args = parse_args()
    args.output_folder.mkdir(exist_ok=True)
    args.qc_folder.mkdir(exist_ok=True)

    print("Starting cytoskeletal organization classifier analysis...")
    print(f"Input folder: {args.input_folder.resolve()}")
    print(f"Output folder: {args.output_folder.resolve()}")

    # Find all CZI files
    czi_files = list(args.input_folder.glob('*.czi'))
    if not czi_files:
        print(f"No CZI files found in {args.input_folder}")
        return

    print(f"Found {len(czi_files)} CZI files")
    print(f"Using {args.workers} worker process(es)")

    # Process all images in parallel and collect features. Each worker gets its own
    # BioImage/Bioformats JVM instance, so per-cell progress bars are disabled when
    # running with more than one worker to avoid multiple processes fighting over the
    # same terminal line; the outer per-image progress bar remains single-process.
    show_progress = args.workers == 1
    all_features = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                process_single_image, image_path, args.nuclei_channel, args.cells_channel,
                args.z_slice, args.qc_folder, show_progress
            ): image_path
            for image_path in czi_files
        }
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures),
                            desc="Processing images", unit="image"):
            image_path = futures[future]
            try:
                image_features = future.result()
            except Exception as exc:
                print(f"Error processing {image_path.name}: {exc}")
                continue
            if image_features is not None:
                all_features.append(image_features)
                print(f"  Extracted features from {len(image_features)} cells ({image_path.name})")

    if not all_features:
        print("No cells extracted from any images")
        return

    # Combine all features into single DataFrame
    features_df = pd.concat(all_features, ignore_index=True)
    print(f"\nTotal cells processed: {len(features_df)}")
    print(f"Condition distribution:\n{features_df['condition'].value_counts()}")

    # Save raw features
    features_df.to_csv(args.output_folder / 'cell_features.csv', index=False)
    print(f"\nRaw features saved to {args.output_folder / 'cell_features.csv'}")

    # Prepare data for classification
    feature_columns = [col for col in features_df.columns if col not in ['condition', 'image_file']]
    X = features_df[feature_columns].values
    y = (features_df['condition'] == 'doxpos').astype(int).values

    # Handle any NaN or inf values
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train Random Forest classifier with cross-validation
    print("\nTraining Random Forest classifier...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10, min_samples_split=5)
    cv_scores = cross_val_score(clf, X_scaled, y, cv=5, scoring='accuracy')

    print(f"Cross-validation accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

    # Train final model on all data
    clf.fit(X_scaled, y)

    # Get feature importance
    importance_df = pd.DataFrame({
        'feature': feature_columns,
        'importance': clf.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nFeature importances:")
    print(importance_df.to_string(index=False))
    importance_df.to_csv(args.output_folder / 'feature_importance.csv', index=False)

    # Plot feature importance
    plt.figure(figsize=(8, 6))
    plt.barh(importance_df['feature'], importance_df['importance'])
    plt.xlabel('Importance')
    plt.title('Feature Importance (Random Forest)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(args.output_folder / 'feature_importance.png')
    print(f"\nFeature importance plot saved to {args.output_folder / 'feature_importance.png'}")


if __name__ == '__main__':
    main()
