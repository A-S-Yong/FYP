import numpy as np
import pandas as pd
from pathlib import Path
import pydicom
import cv2
from PIL import Image
import os
import json
from tqdm import tqdm
import random
import warnings
import sys
from sklearn.model_selection import GroupShuffleSplit

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
SEED = 42
BASE_DIR = Path("/Users/ay/Desktop/FYP/data/CBIS-DDSM")
# Used v3_square to be specific about the method used
OUTPUT_BASE = Path("/Users/ay/Downloads/Github/FYP/processed_data_v3_square")

CONFIG = {
    'crop_size': 224,          
    'padding_pixels': 50,      
    'final_size': 224,         
    'val_ratio': 0.20,         
    'random_seed': SEED
}

# Define Output Directories
PROCESSED_GREYSCALE_DIR = OUTPUT_BASE / "processed_greyscale"
PROCESSED_GREYSCALE_CLAHE_DIR = OUTPUT_BASE / "processed_greyscale_clahe"

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Seeds set to: {seed}")

# Check if critical DICOM libraries are installed
def check_dependencies():
    try:
        import pylibjpeg
    except ImportError:
        print("Note: 'pylibjpeg' not found. Try: pip install pylibjpeg pylibjpeg-libjpeg")

# Reads a DICOM file and returns the pixel array
def load_dicom_image(dicom_path):
    try:
        dcm = pydicom.dcmread(str(dicom_path))
        return dcm.pixel_array
    except Exception as e:
        # Returns None if decoding fails
        return None

# Normalizes raw DICOM data to 0-255 range
def normalize_dicom_to_8bit(image):
    if image is None: return None
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# Applies Contrast Limited Adaptive Histogram Equalization
def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image)

# Resolves the complex folder structure of CBIS-DDSM
def fix_image_path(csv_path, base_dir, want_mask=False):
    if pd.isna(csv_path) or csv_path is None: return None
    
    csv_path = str(csv_path).strip().replace("\\", "/")
    patient_folder = csv_path.split("/")[0]
    patient_dir = base_dir / patient_folder
    
    if not patient_dir.exists(): return None
    
    search_pattern = "*ROI*mask*images*" if want_mask else "*full*mammogram*images*"
    candidates = list(patient_dir.rglob(search_pattern))
    
    if not want_mask and not candidates:
        candidates = list(patient_dir.rglob("*cropped*images*"))
        
    if not candidates: return None
    
    dcm_files = list(candidates[0].rglob("*.dcm"))
    return dcm_files[0] if dcm_files else None

# Crop ROI to a square; preserve aspect ratio and pad with zeros to avoid distortion at edges.
def crop_square_preserve_aspect_ratio(full_image, roi_mask, target_size=224, padding=50):
    coords = np.argwhere(roi_mask > 0)
    if len(coords) == 0: return None

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    center_y, center_x = (y_min + y_max) // 2, (x_min + x_max) // 2
    
    roi_h, roi_w = (y_max - y_min), (x_max - x_min)
    crop_size = max(roi_h, roi_w) + (padding * 2)
    half_size = crop_size // 2

    y1, y2 = center_y - half_size, center_y + half_size
    x1, x2 = center_x - half_size, center_x + half_size

    canvas = np.zeros((crop_size, crop_size), dtype=full_image.dtype)

    src_y1, src_y2 = max(0, y1), min(full_image.shape[0], y2)
    src_x1, src_x2 = max(0, x1), min(full_image.shape[1], x2)

    dst_y1 = src_y1 - y1
    dst_y2 = dst_y1 + (src_y2 - src_y1)
    dst_x1 = src_x1 - x1
    dst_x2 = dst_x1 + (src_x2 - src_x1)

    try:
        canvas[dst_y1:dst_y2, dst_x1:dst_x2] = full_image[src_y1:src_y2, src_x1:src_x2]
    except ValueError:
        return None 

    final_img = cv2.resize(canvas, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)
    return final_img

def main():
    set_seed(SEED)
    check_dependencies()
    
    for p in [PROCESSED_GREYSCALE_DIR, PROCESSED_GREYSCALE_CLAHE_DIR]:
        p.mkdir(parents=True, exist_ok=True)
    
    # --- DATA LOADING ---
    train_meta = pd.read_csv(BASE_DIR / "mass_case_description_train_set.csv")
    test_meta = pd.read_csv(BASE_DIR / "mass_case_description_test_set.csv")
    
    train_meta["split"] = "train"
    test_meta["split"] = "test"
    
    df_all = pd.concat([train_meta, test_meta], ignore_index=True)
    df_mass = df_all[df_all["abnormality type"].astype(str).str.lower() == "mass"].copy()
    
    df_mass["pathology"] = df_mass["pathology"].astype(str).str.strip().replace({
        "BENIGN_WITHOUT_CALLBACK": "BENIGN"
    })
    df_mass["label"] = (df_mass["pathology"] == "MALIGNANT").astype(int)
    
    if 'patient_core' not in df_mass.columns:
        df_mass['patient_core'] = df_mass['patient_id'].astype(str).str.extract(r'^(.*?_P_\d+)')[0].fillna(df_mass['patient_id'].astype(str))

    test_df = df_mass[df_mass['split'] == 'test'].copy()
    full_train_df = df_mass[df_mass['split'] == 'train'].copy()
    
    # --- SPLIT TRAIN -> VAL (0.2) ---
    print(f"Splitting Data (Val Ratio: {CONFIG['val_ratio']}):")
    gss = GroupShuffleSplit(n_splits=1, test_size=CONFIG['val_ratio'], random_state=CONFIG['random_seed'])
    
    train_idx, val_idx = next(gss.split(full_train_df, full_train_df['label'], groups=full_train_df['patient_core']))
    
    train_df = full_train_df.iloc[train_idx].copy()
    val_df = full_train_df.iloc[val_idx].copy()
    
    train_df['split'] = 'train'
    val_df['split'] = 'val'
    test_df['split'] = 'test'
    
    df_final = pd.concat([train_df, val_df, test_df], ignore_index=True)
    
    print(f"Final Counts -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    with open(OUTPUT_BASE / 'config.json', 'w') as f:
        json.dump({k: str(v) if isinstance(v, Path) else v for k, v in CONFIG.items()}, f, indent=2)

    # --- PROCESSING LOOP ---
    stats = {
        'success': 0,
        # File path not found on disk
        'missing_file': 0,
        # DICOM found but failed to decode
        'dicom_read_error': 0,
        'failed_crop': 0,
        'shape_mismatch': 0,
        'empty_mask': 0
    }
    
    print("\nStarting Image Processing:")
    for idx, row in tqdm(df_final.iterrows(), total=len(df_final)):
        
        split = row['split']
        class_name = 'malignant' if row['label'] == 1 else 'benign'
        
        save_dir_grey = PROCESSED_GREYSCALE_DIR / split / class_name
        save_dir_clahe = PROCESSED_GREYSCALE_CLAHE_DIR / split / class_name
        
        save_dir_grey.mkdir(parents=True, exist_ok=True)
        save_dir_clahe.mkdir(parents=True, exist_ok=True)
        
        full_path = fix_image_path(row['image file path'], BASE_DIR, want_mask=False)
        mask_path = fix_image_path(row['ROI mask file path'], BASE_DIR, want_mask=True)
        
        # Check if the files exist
        if not full_path or not mask_path:
            stats['missing_file'] += 1
            continue
            
        # Check if they can be read
        full_img = load_dicom_image(full_path)
        mask_img = load_dicom_image(mask_path)
        
        if full_img is None or mask_img is None:
            stats['dicom_read_error'] += 1
            continue
            
        full_img = normalize_dicom_to_8bit(full_img)
        mask_img = normalize_dicom_to_8bit(mask_img)
        
        if full_img.shape != mask_img.shape:
            mask_img = cv2.resize(mask_img, (full_img.shape[1], full_img.shape[0]), interpolation=cv2.INTER_NEAREST)
            stats['shape_mismatch'] += 1
            
        _, mask_binary = cv2.threshold(mask_img, 127, 255, cv2.THRESH_BINARY)
        
        # Check if the mask is empty?
        if np.sum(mask_binary) == 0:
            stats['empty_mask'] += 1
            continue

        cropped = crop_square_preserve_aspect_ratio(
            full_img, 
            mask_binary, 
            target_size=CONFIG['final_size'], 
            padding=CONFIG['padding_pixels']
        )
        
        if cropped is None:
            stats['failed_crop'] += 1
            continue
            
        filename = f"{row.name}_{class_name}.png"
        
        Image.fromarray(cropped).save(save_dir_grey / filename)
        
        clahe_img = apply_clahe(cropped)
        composite = np.stack([cropped, clahe_img, cropped], axis=-1)
        Image.fromarray(composite).save(save_dir_clahe / filename)
        
        stats['success'] += 1
        
    print("\nProcessing Complete.")
    print("Final Statistics:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    main()