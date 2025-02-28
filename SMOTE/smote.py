import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from collections import Counter
import cv2
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
import random
import shutil
import gc  # For garbage collection

# 1. Data Loading and Preprocessing with Reduced Resolution for Memory Efficiency
def preprocess_dataset(metadata_path, image_dirs, sample_size=None, resize_dim=128):
    """
    Load metadata and preprocess data for SMOTE with memory efficiency
    
    Parameters:
    metadata_path: Path to the CSV metadata file
    image_dirs: List of directories containing images
    sample_size: Optional limit on number of images to process (for testing)
    resize_dim: Size to resize images to before feature extraction
    """
    # Read metadata
    df = pd.read_csv(metadata_path)
    
    # Optionally take a sample for testing
    if sample_size and sample_size < len(df):
        df = df.sample(sample_size, random_state=42)
        print(f"Using sample of {sample_size} images for testing")
    
    # Check if all images exist and load them
    images = {}
    image_data = []
    image_original_sizes = {}

    print("Loading images...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        img_id = row['image_id']
        loaded = False

        for img_dir in image_dirs:
            img_path = os.path.join(img_dir, f"{img_id}.jpg")
            if os.path.exists(img_path):
                try:
                    # Load image at reduced resolution first to check validity
                    img = cv2.imread(img_path)
                    if img is None:
                        print(f"Warning: Could not read image {img_id}")
                        continue
                        
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    # Store original dimensions but not the full image to save memory
                    image_original_sizes[img_id] = img.shape
                    
                    # Use a smaller resolution for SMOTE processing
                    img_resized = cv2.resize(img, (resize_dim, resize_dim))
                    
                    # Apply simpler preprocessing to save computation
                    # Basic enhancement
                    enhanced = cv2.convertScaleAbs(img_resized, alpha=1.1, beta=10)

                    # Flatten the enhanced image
                    img_flat = enhanced.flatten()

                    image_data.append({
                        'image_id': img_id,
                        'class': row['dx'],
                        'features': img_flat,
                    })
                    loaded = True
                    
                    # Only store thumbnail of original to save memory
                    thumb = cv2.resize(img, (resize_dim, resize_dim))
                    images[img_id] = thumb
                    
                    break
                except Exception as e:
                    print(f"Error processing {img_id}: {str(e)}")
                    continue

        if not loaded:
            print(f"Warning: Image {img_id} not found or could not be loaded")

    # Free some memory
    gc.collect()
    
    # Create X (features) and y (labels) for SMOTE
    X = np.stack([item['features'] for item in image_data])
    y = np.array([item['class'] for item in image_data])

    # Create mapping from index to image_id
    idx_to_img_id = {i: item['image_id'] for i, item in enumerate(image_data)}

    # Get class-specific image sizes for better reconstruction
    class_sizes = {}
    for i, class_name in enumerate(y):
        if class_name not in class_sizes:
            class_sizes[class_name] = []
        img_id = idx_to_img_id[i]
        class_sizes[class_name].append(image_original_sizes[img_id])

    # Analyze class distribution
    class_distribution = Counter(y)
    print("Class distribution before balancing:")
    for class_name, count in class_distribution.items():
        print(f"{class_name}: {count}")

    return X, y, idx_to_img_id, images, df, class_distribution, class_sizes

# 2. Apply SMOTE with memory-efficient parameters
def apply_tabular_smote(X, y, pca_variance=0.95, min_k_neighbors=3):
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
def reconstruct_images(X_res, y_res, class_sizes, original_count, resize_dim=128):
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
def save_smote_images(reconstructed_images, original_images, idx_to_img_id, output_dir, original_df):
    """
    Save both original and SMOTE-generated images with batch processing
    """
    os.makedirs(output_dir, exist_ok=True)

    # Create a new metadata dataframe
    new_metadata = []

    # First, save original images
    print("Saving original images...")
    for idx, img_id in tqdm(idx_to_img_id.items()):
        if img_id in original_images:
            # Save image (already resized to save memory)
            img = original_images[img_id]
            try:
                cv2.imwrite(os.path.join(output_dir, f"{img_id}.jpg"),
                            cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                            [cv2.IMWRITE_JPEG_QUALITY, 95])  # Good quality

                # Add metadata
                img_metadata = original_df[original_df['image_id'] == img_id].iloc[0].to_dict()
                new_metadata.append(img_metadata)
            except Exception as e:
                print(f"Error saving original image {img_id}: {str(e)}")

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

                # Save with good quality
                img_path = os.path.join(output_dir, f"{img_id}.jpg")
                cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, 95])

                # Create metadata for synthetic image (handling errors)
                try:
                    class_samples = original_df[original_df['dx'] == label]
                    if len(class_samples) > 0:
                        # Get numeric columns
                        numeric_cols = class_samples.select_dtypes(include=['number']).columns
                        avg_metadata = class_samples[numeric_cols].mean().to_dict() if len(numeric_cols) > 0 else {}
                        
                        # Get categorical columns
                        cat_cols = class_samples.select_dtypes(exclude=['number']).columns
                        most_common_values = {}
                        for col in cat_cols:
                            if col != 'image_id' and len(class_samples[col]) > 0:
                                most_common_values[col] = class_samples[col].value_counts().idxmax()
                        
                        img_metadata = {
                            'image_id': img_id,
                            'dx': label,
                            'is_synthetic': 1,
                            **most_common_values,
                            **avg_metadata
                        }
                        
                        new_metadata.append(img_metadata)
                except Exception as e:
                    print(f"Error creating metadata for {img_id}: {str(e)}")
                    # Add minimal metadata
                    new_metadata.append({
                        'image_id': img_id,
                        'dx': label,
                        'is_synthetic': 1
                    })
                    
            except Exception as e:
                print(f"Error saving synthetic image {img_id}: {str(e)}")
        
        # Free memory after each batch
        gc.collect()

    # Save new metadata
    try:
        pd.DataFrame(new_metadata).to_csv(os.path.join(output_dir, 'smote_metadata.csv'), index=False)
        
        # Generate summary
        new_distribution = pd.DataFrame(new_metadata)['dx'].value_counts()
        print("Final class distribution after SMOTE:")
        print(new_distribution)
        
        return new_distribution
    except Exception as e:
        print(f"Error saving metadata: {str(e)}")
        return Counter([item[1] for item in reconstructed_images])

# Main execution
if __name__ == "__main__":
    # Paths and configurations - MODIFY THESE
    image_dirs = [
        './HAM10000/HAM10000_images_part_1',
        './HAM10000/HAM10000_images_part_2'
    ]
    metadata_path = './HAM10000/HAM10000_metadata.csv'
    output_dir = './smote_output'  # Use a relative path
    
    # Testing with smaller sample and lower resolution
    SAMPLE_SIZE = 500  # Set to None to use all images
    RESIZE_DIM = 64   # Lower resolution for processing
    
    try:
        # 1. Preprocess dataset with memory efficiency
        print("Preprocessing dataset with memory efficiency...")
        X, y, idx_to_img_id, original_images, original_df, original_distribution, class_sizes = preprocess_dataset(
            metadata_path, image_dirs, sample_size=SAMPLE_SIZE, resize_dim=RESIZE_DIM)

        # 2. Apply SMOTE with memory-efficient parameters
        print("Applying SMOTE with memory-efficient parameters...")
        X_res, y_res, new_class_distribution = apply_tabular_smote(X, y, pca_variance=0.9)

        # 3. Reconstruct images with simplified enhancement
        print("Reconstructing images with simplified enhancement...")
        original_count = len(original_images)
        reconstructed_images = reconstruct_images(X_res, y_res, class_sizes, original_count, resize_dim=RESIZE_DIM)

        # 4. Save balanced dataset
        print("Saving balanced dataset...")
        final_distribution = save_smote_images(reconstructed_images, original_images, idx_to_img_id, output_dir, original_df)

        # 5. Visualize results (simplified)
        print("Visualizing results...")
        plt.figure(figsize=(10, 6))
        pd.Series(final_distribution).plot(kind='bar')
        plt.title('Final Class Distribution After SMOTE')
        plt.savefig(os.path.join(output_dir, 'distribution.png'))
        plt.close()

        print("SMOTE balancing complete!")
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()