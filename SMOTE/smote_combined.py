import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import Counter
import cv2
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
import random
import gc  # For garbage collection

# 1. Data Loading and Preprocessing with Reduced Resolution for Memory Efficiency
def preprocess_dataset(combined_dataset_path, sample_size=None, resize_dim=256):
    """
    Load images from a folder structure where each subfolder is a class
    
    Parameters:
    combined_dataset_path: Path to the combined dataset folder containing class subfolders
    sample_size: Optional limit on number of images to process per class (for testing)
    resize_dim: Size to resize images to before feature extraction
    """
    # Get all class folders
    class_folders = [folder for folder in os.listdir(combined_dataset_path) 
                    if os.path.isdir(os.path.join(combined_dataset_path, folder))]
    
    print(f"Found {len(class_folders)} class folders: {class_folders}")
    
    # Create metadata on the fly from folder structure
    image_data = []
    images = {}
    image_original_sizes = {}
    class_distribution = Counter()

    print("Loading images from class folders...")
    for class_name in class_folders:
        class_path = os.path.join(combined_dataset_path, class_name)
        image_files = [f for f in os.listdir(class_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"Found {len(image_files)} images in class {class_name}")
        
        # Optionally sample for testing
        if sample_size and sample_size < len(image_files):
            random.seed(42)  # For reproducibility
            image_files = random.sample(image_files, sample_size)
            print(f"Using sample of {sample_size} images for class {class_name}")
        
        class_distribution[class_name] += len(image_files)
        
        # Load and process images
        for img_file in tqdm(image_files, desc=f"Processing {class_name}"):
            img_path = os.path.join(class_path, img_file)
            img_id = f"{class_name}_{img_file}"  # Create unique identifier
            
            try:
                # Load image
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Warning: Could not read image {img_path}")
                    continue
                    
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Store original dimensions
                image_original_sizes[img_id] = img.shape
                
                # Use a smaller resolution for SMOTE processing
                img_resized = cv2.resize(img, (resize_dim, resize_dim))
                
                # Apply simple preprocessing
                enhanced = cv2.convertScaleAbs(img_resized, alpha=1.1, beta=10)

                # Flatten the enhanced image for SMOTE
                img_flat = enhanced.flatten()

                image_data.append({
                    'image_id': img_id,
                    'class': class_name,
                    'features': img_flat,
                    'original_path': img_path
                })
                
                # Only store thumbnail to save memory
                thumb = cv2.resize(img, (resize_dim, resize_dim))
                images[img_id] = thumb
                
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")
                continue

    # Free some memory
    gc.collect()
    
    if not image_data:
        raise ValueError("No images were loaded successfully. Check your folder structure.")
    
    # Create X (features) and y (labels) for SMOTE
    X = np.stack([item['features'] for item in image_data])
    y = np.array([item['class'] for item in image_data])

    # Create mapping from index to image_id
    idx_to_img_id = {i: item['image_id'] for i, item in enumerate(image_data)}
    
    # Get paths for original images
    original_paths = {item['image_id']: item['original_path'] for item in image_data}

    # Get class-specific image sizes for better reconstruction
    class_sizes = {}
    for i, class_name in enumerate(y):
        if class_name not in class_sizes:
            class_sizes[class_name] = []
        img_id = idx_to_img_id[i]
        class_sizes[class_name].append(image_original_sizes[img_id])

    # Analyze class distribution
    print("Class distribution before balancing:")
    for class_name, count in class_distribution.items():
        print(f"{class_name}: {count}")

    # Create a simple metadata dataframe for reference
    metadata_df = pd.DataFrame([{
        'image_id': item['image_id'],
        'dx': item['class'],
        'path': item['original_path']
    } for item in image_data])

    return X, y, idx_to_img_id, images, metadata_df, class_distribution, class_sizes, original_paths

# 2. Apply SMOTE with memory-efficient parameters
def apply_tabular_smote(X, y, pca_variance=0.99, min_k_neighbors=3):
    """
    Apply SMOTE to the image data with memory-efficient parameters
    """
    print("Applying SMOTE with memory-efficient parameters...")

    # More aggressive dimension reduction to save memory
    print(f"Using PCA for dimensionality reduction preserving {pca_variance*100}% variance...")

    # Find number of components needed to preserve desired variance
    pca = PCA(n_components=pca_variance)
    X_reduced = pca.fit_transform(X)
    print(f"Reduced dimensions from {X.shape[1]} to {X_reduced.shape[1]}")

    # Check for minimum samples in minority class
    class_counts = Counter(y)
    min_class_samples = min(class_counts.values())
    
    # Adjust k_neighbors to be smaller than the smallest class
    k_neighbors = min(min_class_samples - 1, min_k_neighbors)
    if k_neighbors < 1:
        k_neighbors = 1
        print("Warning: Some classes have very few samples. SMOTE results may be poor.")
    
    print(f"Using k_neighbors={k_neighbors} for SMOTE")
    
    try:
        # Apply SMOTE with adjusted parameters
        smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
        X_res_reduced, y_res = smote.fit_resample(X_reduced, y)
    except ValueError as e:
        print(f"SMOTE error: {str(e)}")
        print("Trying with k_neighbors=1 as fallback...")
        smote = SMOTE(random_state=42, k_neighbors=1)
        X_res_reduced, y_res = smote.fit_resample(X_reduced, y)

    # Transform back to original space
    X_res = pca.inverse_transform(X_res_reduced)

    # Apply clipping to ensure valid pixel values
    X_res = np.clip(X_res, 0, 255)

    # Check new class distribution
    new_class_distribution = Counter(y_res)
    print("Class distribution after SMOTE:")
    for class_name, count in new_class_distribution.items():
        print(f"{class_name}: {count}")

    return X_res, y_res, new_class_distribution

# 3. Reconstruct images with simplified enhancement
def reconstruct_images(X_res, y_res, class_sizes, original_count, resize_dim=256):
    """
    Reconstruct images from SMOTE-generated tabular data with simpler enhancements
    """
    # Size used during preprocessing
    recon_base_size = (resize_dim, resize_dim, 3)

    reconstructed_images = []
    synthetic_count = 0
    
    # Process in batches to save memory
    batch_size = 100
    for batch_start in range(original_count, len(X_res), batch_size):
        batch_end = min(batch_start + batch_size, len(X_res))
        
        for i in range(batch_start, batch_end):
            features = X_res[i]
            label = y_res[i]
            
            # Normalize values to valid pixel range
            features = np.clip(features, 0, 255).astype(np.uint8)

            # Reshape to intermediate dimensions
            img = features.reshape(recon_base_size)
            
            synthetic_count += 1

            # Apply a simpler set of enhancements
            try:
                # 1. Basic color correction
                img = cv2.convertScaleAbs(img, alpha=1.1, beta=10)
                
                # 2. Mild sharpening
                kernel = np.array([[-0.5,-0.5,-0.5], [-0.5,5,-0.5], [-0.5,-0.5,-0.5]])
                img = cv2.filter2D(img, -1, kernel)
                
                # Choose size from same class if available (default to current size)
                final_img = img
                if label in class_sizes and class_sizes[label]:
                    target_size = random.choice(class_sizes[label])
                    # Resize to match original class dimensions, with safe error handling
                    try:
                        final_img = cv2.resize(img, (target_size[1], target_size[0]))
                    except Exception as e:
                        print(f"Resize error: {str(e)}. Using original size.")
            except Exception as e:
                print(f"Image enhancement error: {str(e)}. Using basic reconstruction.")
                final_img = img
            
            # Store as synthetic image
            reconstructed_images.append((final_img, label, f"smote_{label}_{synthetic_count}"))
        
        # Free memory after each batch
        gc.collect()

    return reconstructed_images

# 4. Save reconstructed SMOTE images with efficient batch processing
def save_smote_images(reconstructed_images, original_images, idx_to_img_id, output_dir, original_df, original_paths):
    """
    Save both original and SMOTE-generated images with batch processing
    """
    # Create main output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create balanced directory structure with class subfolders
    for class_name in set(original_df['dx']):
        os.makedirs(os.path.join(output_dir, class_name), exist_ok=True)

    # Create a new metadata dataframe
    new_metadata = []

    # First, copy original images to their class folders
    print("Copying original images to balanced dataset...")
    for idx, img_id in tqdm(idx_to_img_id.items()):
        if img_id in original_paths:
            # Get class name
            class_name = original_df[original_df['image_id'] == img_id].iloc[0]['dx']
            
            # Get source path and create destination path
            src_path = original_paths[img_id]
            filename = os.path.basename(src_path)
            dst_path = os.path.join(output_dir, class_name, filename)
            
            try:
                # Copy the original file
                import shutil
                shutil.copy2(src_path, dst_path)
                
                # Add metadata
                img_metadata = {
                    'image_id': img_id,
                    'dx': class_name,
                    'is_synthetic': 0,
                    'original_path': src_path,
                    'balanced_path': dst_path
                }
                new_metadata.append(img_metadata)
            except Exception as e:
                print(f"Error copying original image {img_id}: {str(e)}")

    # Then, save SMOTE-generated images in batches
    print("Saving SMOTE-generated images...")
    batch_size = 50
    for batch_start in range(0, len(reconstructed_images), batch_size):
        batch_end = min(batch_start + batch_size, len(reconstructed_images))
        batch = reconstructed_images[batch_start:batch_end]
        
        for img, label, img_id in tqdm(batch, leave=False):
            try:
                # Ensure colors are valid
                img = np.clip(img, 0, 255).astype(np.uint8)

                # Save with good quality to class subfolder
                img_path = os.path.join(output_dir, label, f"{img_id}.jpg")
                cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, 99])

                # Create metadata for synthetic image
                img_metadata = {
                    'image_id': img_id,
                    'dx': label,
                    'is_synthetic': 1,
                    'original_path': None,
                    'balanced_path': img_path
                }
                new_metadata.append(img_metadata)
                    
            except Exception as e:
                print(f"Error saving synthetic image {img_id}: {str(e)}")
        
        # Free memory after each batch
        gc.collect()

    # Save new metadata
    try:
        metadata_df = pd.DataFrame(new_metadata)
        metadata_df.to_csv(os.path.join(output_dir, 'smote_metadata.csv'), index=False)
        
        # Generate summary
        class_distribution = Counter(metadata_df['dx'])
        synthetic_count = Counter(metadata_df[metadata_df['is_synthetic'] == 1]['dx'])
        original_count = Counter(metadata_df[metadata_df['is_synthetic'] == 0]['dx'])
        
        print("Final class distribution after SMOTE:")
        for class_name in sorted(class_distribution.keys()):
            print(f"{class_name}: {class_distribution[class_name]} images "
                  f"({original_count[class_name]} original, {synthetic_count[class_name]} synthetic)")
        
        return class_distribution, original_count, synthetic_count
    except Exception as e:
        print(f"Error saving metadata: {str(e)}")
        return Counter([item[1] for item in reconstructed_images]), None, None

# Main execution
if __name__ == "__main__":
    # Paths and configurations - MODIFY THESE TO MATCH YOUR ENVIRONMENT
    combined_dataset_path = r'C:\Users\rdeva\Downloads\SEM3\SEM4\Deep_Learning\combined-dataset'  # Path to the combined dataset folder
    output_dir = r'C:\Users\rdeva\Downloads\SEM3\SEM4\Deep_Learning\outs'  # Output directory
    
    # Parameters
    SAMPLE_SIZE = 500  # Set to a number for testing or None to use all images
    RESIZE_DIM = 256   # Resolution for processing
    
    try:
        # 1. Preprocess dataset with memory efficiency
        print(f"Preprocessing dataset from {combined_dataset_path}...")
        X, y, idx_to_img_id, original_images, metadata_df, original_distribution, class_sizes, original_paths = preprocess_dataset(
            combined_dataset_path, sample_size=SAMPLE_SIZE, resize_dim=RESIZE_DIM)

        # 2. Apply SMOTE with memory-efficient parameters
        print("Applying SMOTE with memory-efficient parameters...")
        X_res, y_res, new_class_distribution = apply_tabular_smote(X, y, pca_variance=0.99)

        # 3. Reconstruct images with simplified enhancement
        print("Reconstructing images with simplified enhancement...")
        original_count = len(original_images)
        reconstructed_images = reconstruct_images(X_res, y_res, class_sizes, original_count, resize_dim=RESIZE_DIM)

        # 4. Save balanced dataset
        print("Saving balanced dataset...")
        final_distribution, original_count, synthetic_count = save_smote_images(
            reconstructed_images, original_images, idx_to_img_id, output_dir, metadata_df, original_paths)

        # 5. Visualize results
        print("Visualizing results...")
        plt.figure(figsize=(12, 8))
        
        # Plot original distribution
        plt.subplot(2, 1, 1)
        pd.Series(original_distribution).plot(kind='bar')
        plt.title('Original Class Distribution')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Plot balanced distribution
        plt.subplot(2, 1, 2)
        df = pd.DataFrame({
            'Original': pd.Series(original_count),
            'Synthetic': pd.Series(synthetic_count)
        })
        df.plot(kind='bar', stacked=True, figsize=(12, 6))
        plt.title('Balanced Class Distribution (Original + Synthetic)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(output_dir, 'distribution.png'))
        plt.close()

        print(f"SMOTE balancing complete! Balanced dataset saved to {output_dir}")
        print(f"Metadata saved to {os.path.join(output_dir, 'smote_metadata.csv')}")
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()