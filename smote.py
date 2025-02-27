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

# 1. Data Loading and Preprocessing with Maximum Resolution
def preprocess_dataset(metadata_path, image_dirs):
    """
    Load metadata and preprocess data for SMOTE with maximum resolution preservation
    """
    # Read metadata
    df = pd.read_csv(metadata_path)

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
                # Load image at full resolution
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Store original image at full resolution
                images[img_id] = img
                image_original_sizes[img_id] = img.shape
                
                # For SMOTE, we need to work with lower-dimensional data
                # But we'll use a higher resolution than before
                img_resized = cv2.resize(img, (256, 256))  # Using higher resolution
                
                # Apply preprocessing to enhance features before flattening
                # Convert to LAB color space which better separates luminance from color
                lab = cv2.cvtColor(img_resized, cv2.COLOR_RGB2LAB)
                
                # Apply CLAHE to L channel for better contrast without affecting color
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                cl = clahe.apply(l)
                
                # Merge back and convert to RGB
                enhanced = cv2.merge((cl, a, b))
                enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
                
                # Apply slight sharpening to preserve edges
                kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                enhanced = cv2.filter2D(enhanced, -1, kernel)
                
                # Flatten the enhanced image
                img_flat = enhanced.flatten()
                
                image_data.append({
                    'image_id': img_id,
                    'class': row['dx'],
                    'features': img_flat,
                })
                loaded = True
                break

        if not loaded:
            print(f"Warning: Image {img_id} not found")

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

# 2. Apply SMOTE with optimal parameters for image clarity
def apply_tabular_smote(X, y):
    """
    Apply SMOTE to the image data with parameters tuned for clarity
    """
    print("Applying SMOTE with optimal parameters for image clarity...")

    # Dimension reduction with minimal information loss
    print("Using PCA for dimensionality reduction with maximum information preservation...")
    
    # Find number of components needed to preserve 99% variance
    pca = PCA(n_components=0.99)
    X_reduced = pca.fit_transform(X)
    print(f"Reduced dimensions from {X.shape[1]} to {X_reduced.shape[1]} while preserving 99% variance")

    # Apply SMOTE with k_neighbors=5 for better local structure preservation
    smote = SMOTE(random_state=42, k_neighbors=5)
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

# 3. Reconstruct images with maximum clarity
def reconstruct_images(X_res, y_res, class_sizes, original_count):
    """
    Reconstruct images from SMOTE-generated tabular data with enhanced clarity
    """
    # Size used during preprocessing
    recon_base_size = (256, 256, 3)
    
    reconstructed_images = []
    synthetic_count = 0

    for i, (features, label) in enumerate(zip(X_res, y_res)):
        # Normalize values to valid pixel range
        features = np.clip(features, 0, 255).astype(np.uint8)

        # Reshape to intermediate dimensions
        img = features.reshape(recon_base_size)
        
        # Determine if this is a synthetic image
        is_synthetic = i >= original_count
        
        if is_synthetic:
            synthetic_count += 1
            
            # Apply a series of enhancement techniques for maximum clarity
            
            # 1. Convert to LAB to separate luminance from color information
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            
            # 2. Apply CLAHE to L channel for better contrast
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            cl = clahe.apply(l)
            
            # 3. Merge back
            enhanced_lab = cv2.merge((cl, a, b))
            enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
            
            # 4. Apply sharpening filter for edge enhancement
            kernel = np.array([[-1,-1,-1], [-1,9.5,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(enhanced, -1, kernel)
            
            # 5. Apply bilateral filter to remove noise while preserving edges
            denoised = cv2.bilateralFilter(sharpened, 9, 75, 75)
            
            # 6. Choose appropriate size from the same class for resizing
            if label in class_sizes and class_sizes[label]:
                target_size = random.choice(class_sizes[label])
                # Resize to match original class dimensions
                final_img = cv2.resize(denoised, (target_size[1], target_size[0]), 
                                      interpolation=cv2.INTER_LANCZOS4)  # Use Lanczos for high-quality resizing
            else:
                # Fallback if no class sizes available
                final_img = denoised
            
            # Store as synthetic image
            reconstructed_images.append((final_img, label, f"smote_{label}_{synthetic_count}"))
        else:
            # This is an original image - skip reconstruction
            pass

    return reconstructed_images

# 4. Save reconstructed SMOTE images with maximum quality
def save_smote_images(reconstructed_images, original_images, idx_to_img_id, output_dir, original_df):
    """
    Save both original and SMOTE-generated images into a balanced dataset with maximum quality
    """
    os.makedirs(output_dir, exist_ok=True)

    # Create a new metadata dataframe
    new_metadata = []

    # First, save original images and their metadata
    print("Saving original images...")
    for idx, img_id in tqdm(idx_to_img_id.items()):
        if img_id in original_images:
            # Save image at full original quality
            img = original_images[img_id]
            cv2.imwrite(os.path.join(output_dir, f"{img_id}.jpg"),
                        cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 100])  # Maximum quality

            # Add metadata
            img_metadata = original_df[original_df['image_id'] == img_id].iloc[0].to_dict()
            new_metadata.append(img_metadata)

    # Then, save SMOTE-generated images with maximum quality
    print("Saving SMOTE-generated images with maximum clarity...")
    for img, label, img_id in tqdm(reconstructed_images):
        # Apply final cleanup and enhancement
        
        # Ensure colors are valid (extra safety check)
        img = np.clip(img, 0, 255).astype(np.uint8)
        
        # Save at maximum quality
        img_path = os.path.join(output_dir, f"{img_id}.jpg")
        cv2.imwrite(img_path, 
                   cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                   [cv2.IMWRITE_JPEG_QUALITY, 100])  # Maximum JPEG quality
        
        # Create metadata for synthetic image
        class_samples = original_df[original_df['dx'] == label]
        avg_metadata = class_samples.mean(numeric_only=True).to_dict()

        most_common_values = {
            col: class_samples[col].value_counts().idxmax()
            for col in class_samples.columns
            if col not in ['image_id'] and class_samples[col].dtype == 'object'
        }

        img_metadata = {
            'image_id': img_id,
            'dx': label,
            'is_synthetic': 1,  # Flag to indicate synthetic image
            **most_common_values,
            **avg_metadata
        }

        new_metadata.append(img_metadata)

    # Save new metadata
    pd.DataFrame(new_metadata).to_csv(os.path.join(output_dir, 'smote_metadata.csv'), index=False)

    # Generate summary
    new_distribution = pd.DataFrame(new_metadata)['dx'].value_counts()
    print("Final class distribution after SMOTE:")
    print(new_distribution)

    return new_distribution

# 5. Visualize the results
def visualize_results(original_distribution, new_distribution, output_dir):
    """
    Create visualization of class distribution before and after SMOTE
    """
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    original_distribution_df = pd.Series(original_distribution)
    original_distribution_df.plot(kind='bar')
    plt.title('Class Distribution Before SMOTE')
    plt.xlabel('Class')
    plt.ylabel('Count')

    plt.subplot(1, 2, 2)
    new_distribution.plot(kind='bar')
    plt.title('Class Distribution After SMOTE')
    plt.xlabel('Class')
    plt.ylabel('Count')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'smote_distribution.png'), dpi=300)  # High DPI for clear visualization
    plt.close()

# 6. Show sample of original vs synthetic images
def visualize_samples(original_images, reconstructed_images, output_dir, original_df, num_samples=5):
    """
    Visualize samples of original vs SMOTE-generated images
    """
    # Get unique classes
    classes = list(set(label for _, label, _ in reconstructed_images))

    for class_name in classes:
        plt.figure(figsize=(15, 8))

        # Find original images of this class
        original_class_images = []
        for img_id, img in original_images.items():
            if len(original_class_images) >= num_samples:
                break
            if any(row['image_id'] == img_id and row['dx'] == class_name
                   for _, row in original_df.iterrows()):
                original_class_images.append(img)

        # Find SMOTE images of this class
        smote_class_images = [img for img, label, _ in reconstructed_images
                              if label == class_name][:num_samples]

        # Plot original images
        for i, img in enumerate(original_class_images[:num_samples]):
            plt.subplot(2, num_samples, i + 1)
            plt.imshow(img)
            plt.title(f"Original {class_name}")
            plt.axis('off')

        # Plot SMOTE images
        for i, img in enumerate(smote_class_images[:num_samples]):
            plt.subplot(2, num_samples, i + num_samples + 1)
            plt.imshow(img)
            plt.title(f"SMOTE {class_name}")
            plt.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'samples_{class_name}.png'), dpi=300)  # High DPI
        plt.close()

# Post-processing to further enhance synthetic images
def enhance_synthetic_dataset(output_dir):
    """
    Apply additional enhancement to all synthetic images after generation
    """
    print("Applying final enhancement to all synthetic images...")
    
    # Get all synthetic image paths
    synthetic_paths = [os.path.join(output_dir, f) for f in os.listdir(output_dir) 
                      if f.startswith('smote_') and f.endswith('.jpg')]
    
    for img_path in tqdm(synthetic_paths):
        # Read image
        img = cv2.imread(img_path)
        
        # Apply a series of enhancements
        
        # 1. Apply detail-preserving denoising
        denoised = cv2.fastNlMeansDenoisingColored(img, None, 5, 5, 7, 21)
        
        # 2. Apply unsharp mask for clarity
        gaussian = cv2.GaussianBlur(denoised, (0, 0), 3)
        unsharp = cv2.addWeighted(denoised, 1.5, gaussian, -0.5, 0)
        
        # 3. Apply final JPEG quality optimization
        cv2.imwrite(img_path, unsharp, [cv2.IMWRITE_JPEG_QUALITY, 100])
    
    print(f"Enhanced {len(synthetic_paths)} synthetic images")

# Main execution
if __name__ == "__main__":
    # Paths and configurations
    image_dirs = [
        r'/dist_home/suryansh/dl/Skin-Lesion-Classification/Datasets/HAM10000/HAM10000_images_part_1',
        r'/dist_home/suryansh/dl/Skin-Lesion-Classification/Datasets/HAM10000/HAM10000_images_part_2'
    ]
    metadata_path = r'/dist_home/suryansh/dl/Skin-Lesion-Classification/Datasets/HAM10000/HAM10000_metadata.csv'
    output_dir = r'/smote_maximum_clarity'

    # 1. Preprocess dataset with maximum clarity preservation
    print("Preprocessing dataset with maximum clarity preservation...")
    X, y, idx_to_img_id, original_images, original_df, original_distribution, class_sizes = preprocess_dataset(metadata_path, image_dirs)

    # 2. Apply SMOTE with optimal parameters
    print("Applying SMOTE with optimal parameters for image clarity...")
    X_res, y_res, new_class_distribution = apply_tabular_smote(X, y)

    # 3. Reconstruct images with maximum clarity
    print("Reconstructing images with maximum clarity...")
    original_count = len(original_images)
    reconstructed_images = reconstruct_images(X_res, y_res, class_sizes, original_count)

    # 4. Save balanced dataset with maximum quality
    print("Saving balanced dataset with maximum quality...")
    final_distribution = save_smote_images(reconstructed_images, original_images, idx_to_img_id, output_dir, original_df)

    # 5. Apply additional enhancement to synthetic images
    print("Applying final enhancement pass to maximize clarity...")
    enhance_synthetic_dataset(output_dir)

    # 6. Visualize results
    print("Visualizing results...")
    visualize_results(original_distribution, final_distribution, output_dir)

    # 7. Show sample comparisons
    print("Generating high-quality sample comparisons...")
    visualize_samples(original_images, reconstructed_images, output_dir, original_df)

    print("SMOTE balancing with maximum image clarity complete!")