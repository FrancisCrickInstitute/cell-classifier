import numpy as np
import pandas as pd
from pathlib import Path
from bioio import BioImage
from scipy import ndimage
from scipy.ndimage import label
from skimage import morphology, feature, measure, filters, io
from skimage.segmentation import watershed
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Configuration parameters
INPUT_FOLDER = Path('C:/Users/davej/Downloads/wetransfer_example-images_2026-07-21_1330/Images for Dave')  # Folder containing CZI files
OUTPUT_FOLDER = Path('./results')
QC_FOLDER = OUTPUT_FOLDER / 'segmentation_qc'

DAPI_CHANNEL = 0
TUBULIN_CHANNEL = 1
Z_SLICE = 0  # Use first z-slice only

# Create output folders
OUTPUT_FOLDER.mkdir(exist_ok=True)
QC_FOLDER.mkdir(exist_ok=True)

def extract_condition_from_filename(filename):
    """Extract condition (doxneg or doxpos) from filename."""
    filename_lower = filename.lower()
    if 'doxpos' in filename_lower:
        return 'doxpos'
    elif 'doxneg' in filename_lower:
        return 'doxneg'
    else:
        return None

def load_image_channels(image_path):
    """Load CZI image and extract DAPI and tubulin channels from first z-slice."""
    try:
        img = BioImage(str(image_path))
        
        # Get data shape and extract first z-slice
        data = img.data
        
        print(f"    Image shape: {data.shape} (dims: {img.dims.order})")

        # Handle different possible dimension orders
        if len(data.shape) == 5:  # TCZYX
            dapi = data[0, DAPI_CHANNEL, Z_SLICE, :, :].astype(np.float32)
            tubulin = data[0, TUBULIN_CHANNEL, Z_SLICE, :, :].astype(np.float32)
        elif len(data.shape) == 4:  # CZYX
            dapi = data[DAPI_CHANNEL, Z_SLICE, :, :].astype(np.float32)
            tubulin = data[TUBULIN_CHANNEL, Z_SLICE, :, :].astype(np.float32)
        elif len(data.shape) == 3:  # ZYX
            dapi = img.get_image_data('ZYX')[Z_SLICE, :, :].astype(np.float32)
            tubulin = img.get_image_data('ZYX')[Z_SLICE, :, :].astype(np.float32)
        else:
            print(f"    Unsupported dimensionality: {len(data.shape)}D")
            return None, None

        print(f"    DAPI range: [{dapi.min():.1f}, {dapi.max():.1f}], "
              f"Tubulin range: [{tubulin.min():.1f}, {tubulin.max():.1f}]")
        return dapi, tubulin
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        return None, None

def segment_nuclei(dapi_image, median_filter_size=3):
    """Segment nuclei from DAPI channel using triangle thresholding and morphological operations."""
    # Median filter to denoise while preserving edges
    dapi_filtered = ndimage.median_filter(dapi_image, size=median_filter_size)

    # Apply triangle threshold
    threshold = filters.threshold_triangle(dapi_filtered)
    binary = dapi_filtered > threshold

    # Remove small objects and fill holes
    binary = morphology.remove_small_objects(binary, min_size=50)
    binary = ndimage.binary_fill_holes(binary)

    # Label connected components
    nuclei_labels, num_nuclei = label(binary)
    print(f"    Found {num_nuclei} nuclei (triangle threshold={threshold:.1f})")

    return nuclei_labels

def segment_cells(nuclei_labels, tubulin_image, gaussian_sigma=2):
    """Segment cell boundaries using seeded watershed from nuclei."""
    # Gaussian smoothing to denoise while preserving cell-scale structure
    tubulin_smoothed = ndimage.gaussian_filter(tubulin_image, sigma=gaussian_sigma)

    # Apply triangle threshold to obtain foreground mask
    threshold = filters.threshold_triangle(tubulin_smoothed)
    mask = tubulin_smoothed > threshold

    # Create distance map for watershed
    distance = ndimage.distance_transform_edt(mask)

    # Apply watershed seeded by nuclei
    cell_labels = watershed(-distance, markers=nuclei_labels, mask=mask)
    num_cells = len(np.unique(cell_labels)) - (1 if 0 in cell_labels else 0)
    print(f"    Segmented {num_cells} cells (triangle threshold={threshold:.1f})")

    return cell_labels

def save_segmentation_qc(image_path, nuclei_labels, cell_labels):
    """Save raw nuclei and cell label images as PNGs for visual QC."""
    nuclei_path = QC_FOLDER / f'{image_path.stem}_nuclei_labels.png'
    cell_path = QC_FOLDER / f'{image_path.stem}_cell_labels.png'

    io.imsave(nuclei_path, nuclei_labels.astype(np.uint16), check_contrast=False)
    io.imsave(cell_path, cell_labels.astype(np.uint16), check_contrast=False)

    print(f"    Saved label images: {nuclei_path.name}, {cell_path.name}")

def extract_morphological_features(cell_mask, dapi_image, tubulin_image, cell_label):
    """Extract morphological and intensity features from a single cell."""
    features_dict = {}
    
    # Get cell region
    cell_region = cell_mask == cell_label
    
    if np.sum(cell_region) < 10:  # Skip very small cells
        return None
    
    # Shape and size features
    region_props = measure.regionprops(cell_region.astype(int))[0]
    features_dict['area'] = region_props.area
    features_dict['perimeter'] = region_props.perimeter
    features_dict['eccentricity'] = region_props.eccentricity
    features_dict['solidity'] = region_props.solidity
    features_dict['aspect_ratio'] = region_props.major_axis_length / (region_props.minor_axis_length + 1e-8)
    
    # DAPI intensity features
    dapi_values = dapi_image[cell_region]
    features_dict['dapi_mean'] = np.mean(dapi_values)
    features_dict['dapi_std'] = np.std(dapi_values)
    features_dict['dapi_max'] = np.max(dapi_values)
    features_dict['dapi_min'] = np.min(dapi_values)
    
    # Tubulin intensity features
    tubulin_values = tubulin_image[cell_region]
    features_dict['tubulin_mean'] = np.mean(tubulin_values)
    features_dict['tubulin_std'] = np.std(tubulin_values)
    features_dict['tubulin_max'] = np.max(tubulin_values)
    features_dict['tubulin_min'] = np.min(tubulin_values)
    
    # Tubulin texture features (local variance as organization metric)
    tubulin_local_var = ndimage.generic_filter(tubulin_image, np.var, size=5)
    features_dict['tubulin_texture_var'] = np.mean(tubulin_local_var[cell_region])
    
    # Granularity (using Laplacian)
    tubulin_laplacian = ndimage.laplace(tubulin_image)
    features_dict['tubulin_granularity'] = np.mean(np.abs(tubulin_laplacian[cell_region]))
    
    # Intensity ratio and colocalization
    features_dict['tubulin_dapi_ratio'] = (features_dict['tubulin_mean'] / (features_dict['dapi_mean'] + 1e-8))
    dapi_tubulin_correlation = np.corrcoef(dapi_values, tubulin_values)[0, 1]
    features_dict['dapi_tubulin_correlation'] = dapi_tubulin_correlation if not np.isnan(dapi_tubulin_correlation) else 0
    
    # Spatial intensity distribution (coefficient of variation)
    features_dict['dapi_cv'] = features_dict['dapi_std'] / (features_dict['dapi_mean'] + 1e-8)
    features_dict['tubulin_cv'] = features_dict['tubulin_std'] / (features_dict['tubulin_mean'] + 1e-8)
    
    return features_dict

def process_single_image(image_path):
    """Process a single CZI image and extract features for all cells."""
    condition = extract_condition_from_filename(image_path.name)
    if condition is None:
        print(f"Warning: Could not determine condition for {image_path.name}")
        return None
    print(f"  Condition: {condition}")

    # Load channels
    dapi, tubulin = load_image_channels(image_path)
    if dapi is None or tubulin is None:
        print(f"Warning: Failed to load image {image_path.name}")
        return None

    # Segment nuclei and cells
    nuclei_labels = segment_nuclei(dapi)
    cell_labels = segment_cells(nuclei_labels, tubulin)
    save_segmentation_qc(image_path, nuclei_labels, cell_labels)

    # Extract features for each cell
    cell_features_list = []
    unique_labels = np.unique(cell_labels)
    skipped = 0

    for cell_label in unique_labels:
        if cell_label == 0:  # Skip background
            continue

        cell_features = extract_morphological_features(cell_labels, dapi, tubulin, cell_label)
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
    print("Starting cytoskeletal organization classifier analysis...")
    print(f"Input folder: {INPUT_FOLDER.resolve()}")
    print(f"Output folder: {OUTPUT_FOLDER.resolve()}")

    # Find all CZI files
    czi_files = list(INPUT_FOLDER.glob('*.czi'))
    if not czi_files:
        print(f"No CZI files found in {INPUT_FOLDER}")
        return
    
    print(f"Found {len(czi_files)} CZI files")
    
    # Process all images and collect features
    all_features = []
    for image_path in czi_files:
        print(f"Processing {image_path.name}...")
        image_features = process_single_image(image_path)
        if image_features is not None:
            all_features.append(image_features)
            print(f"  Extracted features from {len(image_features)} cells")
    
    if not all_features:
        print("No cells extracted from any images")
        return
    
    # Combine all features into single DataFrame
    features_df = pd.concat(all_features, ignore_index=True)
    print(f"\nTotal cells processed: {len(features_df)}")
    print(f"Condition distribution:\n{features_df['condition'].value_counts()}")
    
    # Save raw features
    features_df.to_csv(OUTPUT_FOLDER / 'cell_features.csv', index=False)
    print(f"\nRaw features saved to {OUTPUT_FOLDER / 'cell_features.csv'}")
    
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
    importance_df.to_csv(OUTPUT_FOLDER / 'feature_importance.csv', index=False)

    # Plot feature importance
    plt.figure(figsize=(8, 6))
    plt.barh(importance_df['feature'], importance_df['importance'])
    plt.xlabel('Importance')
    plt.title('Feature Importance (Random Forest)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(OUTPUT_FOLDER / 'feature_importance.png')
    print(f"\nFeature importance plot saved to {OUTPUT_FOLDER / 'feature_importance.png'}")


if __name__ == '__main__':
    main()