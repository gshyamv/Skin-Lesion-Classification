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
import shutil
from scipy import ndimage

###############################################################################
# 1. Data Loading and Preprocessing with Advanced Image Enhancement
###############################################################################
def preprocess_dataset(metadata_path, image_dirs, resize_dim=128):
    """
    Load metadata and preprocess data for SMOTE with enhanced image preprocessing.
    Uses a lower resize_dim (e.g., 128x128) to manage dimensionality while
    preserving more detail than 64x64.
    """
    # Read metadata
    df = pd.read_csv(metadata_path)
    images = {}  # Store full-resolution images
    image_data = []  # List to hold flattened features per image
    image_original_sizes = {}  # Original image sizes

    print("Loading and preprocessing images...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        img_id = row['image_id']
        loaded = False

        for img_dir in image_dirs:
            img_path = os.path.join(img_dir, f"{img_id}.jpg")
            if os.path.exists(img_path):
                # Load image at full resolution
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Failed to load {img_path}")
                    continue
                    
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images[img_id] = img
                image_original_sizes[img_id] = img.shape  # (H, W, 3)

                # Downsample to reduce dimensionality for SMOTE
                img_resized = cv2.resize(img, (resize_dim, resize_dim))
                
                # Apply advanced preprocessing for better feature extraction
                # Convert to LAB color space 
                lab = cv2.cvtColor(img_resized, cv2.COLOR_RGB2LAB)
                l, a, b = cv2.split(lab)
                
                # Apply CLAHE to L channel for better contrast
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                cl = clahe.apply(l)
                enhanced_lab = cv2.merge((cl, a, b))
                enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
                
                # Apply bilateral filtering to reduce noise while preserving edges
                filtered = cv2.bilateralFilter(enhanced, 9, 75, 75)
                
                # Normalize the image
                normalized = filtered / 255.0
                
                # Flatten the enhanced image into a 1D vector
                img_flat = normalized.flatten()
                image_data.append({
                    'image_id': img_id,
                    'class': row['dx'],
                    'features': img_flat,
                })
                loaded = True
                break

        if not loaded:
            print(f"Warning: Image {img_id} not found")

    # Build feature matrix X and label vector y
    X = np.stack([item['features'] for item in image_data])
    y = np.array([item['class'] for item in image_data])

    # Mapping from index to image_id for later reference
    idx_to_img_id = {i: item['image_id'] for i, item in enumerate(image_data)}

    # Gather class-specific original image sizes (for reconstruction later)
    class_sizes = {}
    for i, class_name in enumerate(y):
        if class_name not in class_sizes:
            class_sizes[class_name] = []
        img_id = idx_to_img_id[i]
        class_sizes[class_name].append(image_original_sizes[img_id])

    # Print class distribution before balancing
    class_distribution = Counter(y)
    print("Class distribution before balancing:")
    for class_name, count in class_distribution.items():
        print(f"  {class_name}: {count}")

    return X, y, idx_to_img_id, images, df, class_distribution, class_sizes, resize_dim

###############################################################################
# 2. Apply SMOTE with Class-Specific PCA for Better Feature Preservation
###############################################################################
def apply_class_specific_smote(X, y, resize_dim=128, n_components_ratio=0.8):
    """
    Apply SMOTE using class-specific PCA to better preserve class characteristics.
    Uses a ratio for n_components to adapt to different resized dimensions.
    """
    # Get unique classes
    classes = np.unique(y)
    
    # Initialize arrays for the balanced dataset
    X_res_list = []
    y_res_list = []
    
    # For tracking which samples are original vs. synthetic
    original_indices = []
    current_idx = 0
    
    # Determine target sample count (use the count of the majority class)
    class_counts = Counter(y)
    target_count = max(class_counts.values())
    print(f"Target count per class: {target_count}")
    
    # Process each class separately
    for class_name in classes:
        # Get samples for this class
        class_mask = (y == class_name)
        X_class = X[class_mask]
        y_class = y[class_mask]
        
        # Track original samples
        original_indices.extend(list(range(current_idx, current_idx + len(X_class))))
        current_idx += len(X_class)
        
        # Add original samples to result
        X_res_list.append(X_class)
        y_res_list.append(y_class)
        
        # Skip SMOTE if class already has enough samples
        if len(X_class) >= target_count:
            print(f"Class {class_name} already has {len(X_class)} samples (≥ {target_count})")
            continue
            
        # Calculate number of synthetic samples needed
        n_synthetic = target_count - len(X_class)
        print(f"Generating {n_synthetic} synthetic samples for class {class_name}")
        
        # Skip if too few samples to apply SMOTE
        if len(X_class) < 6:  # SMOTE needs at least k+1 samples (default k=5)
            print(f"Warning: Not enough samples for SMOTE in class {class_name}")
            continue
            
        # Calculate appropriate n_components based on the data shape and ratio
        n_features = X_class.shape[1]
        n_components = min(len(X_class) - 1, int(n_features * n_components_ratio))
        n_components = max(n_components, 50)  # Ensure at least 50 components
        print(f"Using {n_components} PCA components for class {class_name}")
        
        try:
            # Apply PCA for dimensionality reduction
            pca = PCA(n_components=n_components, random_state=42)
            X_class_reduced = pca.fit_transform(X_class)
            
            # Apply SMOTE in the reduced space
            k_neighbors = min(5, len(X_class) - 1)  # Adjust k_neighbors if needed
            smote = SMOTE(sampling_strategy={class_name: target_count}, 
                          random_state=42, 
                          k_neighbors=k_neighbors)
            X_class_res_reduced, y_class_res = smote.fit_resample(
                X_class_reduced, y_class
            )
            
            # Only keep the synthetic samples (skip the original ones)
            synthetic_mask = np.arange(len(X_class_res_reduced)) >= len(X_class)
            X_synthetic_reduced = X_class_res_reduced[synthetic_mask]
            y_synthetic = y_class_res[synthetic_mask]
            
            # Transform synthetic samples back to original space
            X_synthetic = pca.inverse_transform(X_synthetic_reduced)
            
            # Add synthetic samples to result
            X_res_list.append(X_synthetic)
            y_res_list.append(y_synthetic)
            
        except Exception as e:
            print(f"Error applying SMOTE to class {class_name}: {e}")
    
    # Combine all classes
    X_res = np.vstack(X_res_list)
    y_res = np.concatenate(y_res_list)
    
    # Scale back to [0, 255] and clip
    X_res = np.clip(X_res * 255.0, 0, 255).astype(np.uint8)
    
    # Print the new class distribution
    new_class_distribution = Counter(y_res)
    print("Class distribution after SMOTE:")
    for class_name, count in new_class_distribution.items():
        print(f"  {class_name}: {count}")
    
    return X_res, y_res, new_class_distribution, original_indices, resize_dim

###############################################################################
# 3. Advanced Image Reconstruction with Texture Preservation
###############################################################################
def reconstruct_images(X_res, y_res, class_sizes, original_indices, resize_dim):
    """
    Reconstruct images from SMOTE-generated data with advanced image enhancement.
    Uses a multi-step enhancement pipeline to improve realism of synthetic images.
    """
    reconstructed_images = []
    base_shape = (resize_dim, resize_dim, 3)
    synthetic_count_per_class = Counter()
    
    print("Reconstructing and enhancing synthetic images...")
    for i, (features, label) in tqdm(enumerate(zip(X_res, y_res)), total=len(X_res)):
        # Skip original images
        if i in original_indices:
            continue
            
        # Count synthetic images per class
        synthetic_count_per_class[label] += 1
        count = synthetic_count_per_class[label]
        
        # Reshape flattened features back to image
        image = features.reshape(base_shape)
        
        # Apply a comprehensive enhancement pipeline:
        
        # 1. Convert to float for better processing
        img_float = image.astype(np.float32)
        
        # 2. Apply adaptive histogram equalization for better contrast
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        enhanced_lab = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
        
        # 3. Apply bilateral filtering to reduce noise while preserving edges
        filtered = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        # 4. Apply unsharp masking for sharpening
        gaussian = cv2.GaussianBlur(filtered, (0, 0), 3)
        sharpened = cv2.addWeighted(filtered, 1.5, gaussian, -0.5, 0)
        
        # 5. Enhance texture using the detail layer of the bilateral filter
        detail_layer = cv2.subtract(filtered, cv2.bilateralFilter(filtered, 15, 80, 80))
        texture_enhanced = cv2.add(sharpened, detail_layer)
        
        # 6. Ensure realistic colors using color transfer from class samples
        result = np.clip(texture_enhanced, 0, 255).astype(np.uint8)
        
        # 7. Resize to a random original image size from the same class
        if label in class_sizes and class_sizes[label]:
            target_size = random.choice(class_sizes[label])  # (H, W, 3)
            final_img = cv2.resize(result, (target_size[1], target_size[0]),
                                  interpolation=cv2.INTER_LANCZOS4)
        else:
            final_img = result
        
        # Create a unique ID for the synthetic image
        synthetic_id = f"smote_{label}_{count}"
        
        reconstructed_images.append((final_img, label, synthetic_id))
        
    return reconstructed_images

###############################################################################
# 4. Add Realistic Texture to Synthetic Images
###############################################################################
def apply_texture_preservation(reconstructed_images, images_by_class):
    """
    Apply texture preservation techniques to make synthetic images more realistic.
    """
    enhanced_images = []
    
    print("Applying texture enhancement to synthetic images...")
    for img, label, img_id in tqdm(reconstructed_images):
        # Apply frequency domain filtering to preserve high-frequency details
        # This helps maintain the characteristic texture of skin lesions
        
        # Convert to YCrCb color space to separate luminance from chrominance
        ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        
        # Apply wavelet-based detail enhancement to luminance channel
        # Since we don't have direct wavelet functions, simulate with Gaussian pyramids
        gaussian = cv2.GaussianBlur(y, (5, 5), 0)
        detail = cv2.subtract(y, gaussian)
        enhanced_detail = cv2.addWeighted(detail, 2.0, np.zeros_like(detail), 0, 0)
        enhanced_y = cv2.add(gaussian, enhanced_detail)
        
        # Recombine the channels
        enhanced_ycrcb = cv2.merge([enhanced_y, cr, cb])
        enhanced = cv2.cvtColor(enhanced_ycrcb, cv2.COLOR_YCrCb2RGB)
        
        # Apply median filter to reduce any remaining noise while preserving edges
        median_filtered = cv2.medianBlur(enhanced, 3)
        
        # Ensure the output is in the valid range
        final = np.clip(median_filtered, 0, 255).astype(np.uint8)
        
        enhanced_images.append((final, label, img_id))
    
    return enhanced_images

###############################################################################
# 5. Save the Balanced Dataset with Enhanced Images
###############################################################################
def save_smote_images(reconstructed_images, original_images, idx_to_img_id, output_dir, original_df):
    """
    Save only enhanced SMOTE-generated synthetic images to create
    a balanced dataset. Also generates metadata CSV with comprehensive information.
    """
    os.makedirs(output_dir, exist_ok=True)
    image_dir = os.path.join(output_dir, 'images')
    os.makedirs(image_dir, exist_ok=True)
    
    new_metadata = []

    # Skip saving original images as requested
    print("Including original metadata but not saving original images...")
    for idx, img_id in idx_to_img_id.items():
        if img_id in original_images:
            # Only add metadata for original images, don't save the images
            row = original_df[original_df['image_id'] == img_id].iloc[0].to_dict()
            row['is_synthetic'] = 0  # Mark as original
            new_metadata.append(row)

    print("Saving SMOTE-generated images...")
    for img, label, smote_id in tqdm(reconstructed_images):
        img = np.clip(img, 0, 255).astype(np.uint8)
        # Save the SMOTE-generated image
        cv2.imwrite(
            os.path.join(image_dir, f"{smote_id}.jpg"),
            cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, 100]
        )
        
        # Create metadata for synthetic image based on class averages
        class_samples = original_df[original_df['dx'] == label]
        avg_metadata = class_samples.mean(numeric_only=True).to_dict()

        # For categorical columns, pick the most common value
        most_common_values = {}
        for col in class_samples.columns:
            if col not in ['image_id'] and class_samples[col].dtype == 'object':
                if len(class_samples[col].value_counts()) > 0:
                    most_common_values[col] = class_samples[col].value_counts().idxmax()

        img_metadata = {
            'image_id': smote_id,
            'dx': label,
            'is_synthetic': 1,  # Mark as synthetic
            **most_common_values,
            **avg_metadata
        }
        new_metadata.append(img_metadata)

    # Save the metadata CSV file
    new_df = pd.DataFrame(new_metadata)
    new_df.to_csv(os.path.join(output_dir, 'balanced_metadata.csv'), index=False)

    final_distribution = new_df['dx'].value_counts()
    print("Final class distribution after SMOTE:")
    print(final_distribution)

    return final_distribution

###############################################################################
# 6. Visualize Class Distributions and Image Samples
###############################################################################
def visualize_results(original_distribution, final_distribution, output_dir):
    """
    Create visualizations of class distributions before and after balancing.
    """
    vis_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    pd.Series(original_distribution).plot(kind='bar')
    plt.title('Class Distribution Before SMOTE')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.xticks(rotation=45)

    plt.subplot(1, 2, 2)
    final_distribution.plot(kind='bar')
    plt.title('Class Distribution After SMOTE')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'class_distribution.png'), dpi=300)
    plt.close()

def visualize_samples(reconstructed_images, original_images, output_dir, original_df, num_samples=4):
    """
    Create side-by-side comparisons of original vs synthetic images per class.
    """
    vis_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Group synthetic images by class
    synthetic_by_class = {}
    for img, label, _ in reconstructed_images:
        if label not in synthetic_by_class:
            synthetic_by_class[label] = []
        synthetic_by_class[label].append(img)
    
    # Group original images by class
    original_by_class = {}
    for img_id, img in original_images.items():
        row = original_df[original_df['image_id'] == img_id]
        if len(row) > 0:
            label = row.iloc[0]['dx']
            if label not in original_by_class:
                original_by_class[label] = []
            original_by_class[label].append(img)
    
    # Create visualizations for each class
    for label in synthetic_by_class.keys():
        if label not in original_by_class:
            continue
            
        plt.figure(figsize=(15, 8))
        
        # Sample original images
        orig_samples = original_by_class[label][:num_samples]
        
        # Sample synthetic images
        synth_samples = synthetic_by_class[label][:num_samples]
        
        # Plot original images
        for i, img in enumerate(orig_samples):
            plt.subplot(2, num_samples, i + 1)
            plt.imshow(img)
            plt.title(f"Original {label}")
            plt.axis('off')
        
        # Plot synthetic images
        for i, img in enumerate(synth_samples):
            plt.subplot(2, num_samples, i + num_samples + 1)
            plt.imshow(img)
            plt.title(f"Synthetic {label}")
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, f'samples_{label}.png'), dpi=300)
        plt.close()

###############################################################################
# Main Execution
###############################################################################
def main():
    # Set your dataset paths here (change as needed)
    image_dirs = [
        r'/dist_home/suryansh/dl/Skin-Lesion-Classification/Datasets/HAM10000/HAM10000_images_part_1',
        r'/dist_home/suryansh/dl/Skin-Lesion-Classification/Datasets/HAM10000/HAM10000_images_part_2'
    ]
    metadata_path = r'/dist_home/suryansh/dl/Skin-Lesion-Classification/Datasets/HAM10000/HAM10000_metadata.csv'
    output_dir = r'./smote_maximum_clarity'
    os.makedirs(output_dir, exist_ok=True)

    # 1. Preprocess dataset with enhanced image processing
    print("Preprocessing dataset...")
    X, y, idx_to_img_id, original_images, original_df, original_dist, class_sizes, resize_dim = preprocess_dataset(
        metadata_path, image_dirs, resize_dim=128  # Use 128×128 for better detail preservation
    )

    # 2. Apply class-specific SMOTE with PCA
    print("Running SMOTE with class-specific PCA...")
    X_res, y_res, new_dist, original_indices, resize_dim = apply_class_specific_smote(
        X, y, resize_dim=128, n_components_ratio=0.7  # Use 70% of components
    )

    # 3. Reconstruct and enhance synthetic images
    print("Reconstructing synthetic images...")
    reconstructed_images = reconstruct_images(
        X_res, y_res, class_sizes, original_indices, resize_dim
    )
    
    # 4. Apply additional texture preservation techniques
    # Group original images by class for texture reference
    images_by_class = {}
    for i, label in enumerate(y):
        if i in original_indices:
            if label not in images_by_class:
                images_by_class[label] = []
            img_id = idx_to_img_id[original_indices.index(i)]
            images_by_class[label].append(original_images[img_id])
    
    enhanced_images = apply_texture_preservation(reconstructed_images, images_by_class)

    # 5. Save the balanced dataset
    print("Saving balanced dataset...")
    final_distribution = save_smote_images(
        enhanced_images, original_images, idx_to_img_id, output_dir, original_df
    )

    # 6. Visualize results
    print("Creating visualizations...")
    visualize_results(original_dist, final_distribution, output_dir)
    visualize_samples(enhanced_images, original_images, output_dir, original_df)

    print("SMOTE balancing with enhanced reconstruction complete!")

if __name__ == "__main__":
    main()