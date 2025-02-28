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

###############################################################################
# 1. Data Loading and Preprocessing at a Manageable Resolution
###############################################################################
def preprocess_dataset(metadata_path, image_dirs, resize_dim=64):
    """
    Load metadata and preprocess data for SMOTE at a specified resolution
    (default 128x128). This reduces the dimensionality compared to using 256x256.
    """
    # Read metadata
    df = pd.read_csv(metadata_path)
    images = {}  # Store full-resolution images
    image_data = []  # List to hold flattened features per image
    image_original_sizes = {}  # Original image sizes

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
                images[img_id] = img
                image_original_sizes[img_id] = img.shape  # (H, W, 3)

                # Downsample to reduce dimensionality for SMOTE
                img_resized = cv2.resize(img, (resize_dim, resize_dim))

                # Convert to LAB color space and apply CLAHE for better contrast
                lab = cv2.cvtColor(img_resized, cv2.COLOR_RGB2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                cl = clahe.apply(l)
                enhanced = cv2.merge((cl, a, b))
                enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)

                # Apply slight sharpening
                kernel = np.array([[-1, -1, -1],
                                   [-1,  9, -1],
                                   [-1, -1, -1]])
                enhanced = cv2.filter2D(enhanced, -1, kernel)

                # Flatten the enhanced image into a 1D vector
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

    return X, y, idx_to_img_id, images, df, class_distribution, class_sizes

###############################################################################
# 2. Apply SMOTE with PCA for Dimensionality Reduction
###############################################################################
def apply_tabular_smote(X, y, n_components=300):
    """
    Apply PCA → SMOTE → Inverse PCA.
    n_components is fixed to 300 to reduce dimensionality and memory usage.
    """
    print("Applying PCA for dimensionality reduction...")
    pca = PCA(n_components=n_components, random_state=42)
    X_reduced = pca.fit_transform(X)
    print(f"Original shape: {X.shape}, Reduced shape: {X_reduced.shape}")

    print("Applying SMOTE...")
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_res_reduced, y_res = smote.fit_resample(X_reduced, y)

    print("Reconstructing from PCA space (inverse_transform)...")
    X_res = pca.inverse_transform(X_res_reduced)
    X_res = np.clip(X_res, 0, 255)  # Ensure valid pixel range

    new_class_distribution = Counter(y_res)
    print("Class distribution after SMOTE:")
    for class_name, count in new_class_distribution.items():
        print(f"  {class_name}: {count}")

    return X_res, y_res, new_class_distribution

###############################################################################
# 3. Reconstruct Synthetic Images from SMOTE-Generated Data
###############################################################################
def reconstruct_images(X_res, y_res, class_sizes, original_count, recon_dim=128):
    """
    Reconstruct images from SMOTE-generated tabular data.
    Only synthetic images (indices ≥ original_count) are reconstructed.
    Applies an enhancement pipeline for improved clarity.
    """
    recon_base_size = (recon_dim, recon_dim, 3)
    reconstructed_images = []
    synthetic_count = 0

    for i, (features, label) in enumerate(zip(X_res, y_res)):
        # Skip original images
        is_synthetic = (i >= original_count)
        if not is_synthetic:
            continue

        synthetic_count += 1
        features = np.clip(features, 0, 255).astype(np.uint8)
        img = features.reshape(recon_base_size)

        # Enhancement pipeline for synthetic images:
        # 1. Convert to LAB and apply CLAHE
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        enhanced_lab = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)

        # 2. Sharpen using a convolution kernel
        kernel = np.array([[-1, -1, -1],
                           [-1, 9.5, -1],
                           [-1, -1, -1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)

        # 3. Bilateral filtering to remove noise while preserving edges
        denoised = cv2.bilateralFilter(sharpened, 9, 75, 75)

        # Resize to match a random original size from the same class (if available)
        if label in class_sizes and class_sizes[label]:
            target_size = random.choice(class_sizes[label])  # (H, W, 3)
            final_img = cv2.resize(denoised, (target_size[1], target_size[0]),
                                   interpolation=cv2.INTER_LANCZOS4)
        else:
            final_img = denoised

        reconstructed_images.append((final_img, label, f"smote_{label}_{synthetic_count}"))

    return reconstructed_images

###############################################################################
# 4. Save the Balanced Dataset (Original + Synthetic Images)
###############################################################################
def save_smote_images(reconstructed_images, original_images, idx_to_img_id, output_dir, original_df):
    """
    Save both the original images and SMOTE-generated synthetic images to form a
    balanced dataset. Also generates a new metadata CSV.
    """
    os.makedirs(output_dir, exist_ok=True)
    new_metadata = []

    print("Saving original images...")
    for idx, img_id in tqdm(idx_to_img_id.items()):
        if img_id in original_images:
            img = original_images[img_id]
            cv2.imwrite(
                os.path.join(output_dir, f"{img_id}.jpg"),
                cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 100]
            )
            row = original_df[original_df['image_id'] == img_id].iloc[0].to_dict()
            new_metadata.append(row)

    print("Saving SMOTE-generated images...")
    for img, label, smote_id in tqdm(reconstructed_images):
        img = np.clip(img, 0, 255).astype(np.uint8)
        cv2.imwrite(
            os.path.join(output_dir, f"{smote_id}.jpg"),
            cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, 100]
        )
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
            'is_synthetic': 1,
            **most_common_values,
            **avg_metadata
        }
        new_metadata.append(img_metadata)

    new_df = pd.DataFrame(new_metadata)
    new_df.to_csv(os.path.join(output_dir, 'smote_metadata.csv'), index=False)

    final_distribution = new_df['dx'].value_counts()
    print("Final class distribution after SMOTE:")
    print(final_distribution)

    return final_distribution

###############################################################################
# 5. Visualize Class Distributions Before and After SMOTE
###############################################################################
def visualize_results(original_distribution, new_distribution, output_dir):
    plt.figure(figsize=(12, 6))
    if isinstance(original_distribution, dict):
        original_distribution = pd.Series(original_distribution)
    if isinstance(new_distribution, dict):
        new_distribution = pd.Series(new_distribution)

    plt.subplot(1, 2, 1)
    original_distribution.plot(kind='bar')
    plt.title('Class Distribution Before SMOTE')
    plt.xlabel('Class')
    plt.ylabel('Count')

    plt.subplot(1, 2, 2)
    new_distribution.plot(kind='bar')
    plt.title('Class Distribution After SMOTE')
    plt.xlabel('Class')
    plt.ylabel('Count')

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'smote_distribution.png'), dpi=300)
    plt.close()

###############################################################################
# 6. Visualize Sample Comparisons of Original vs. Synthetic Images
###############################################################################
def visualize_samples(original_images, reconstructed_images, output_dir, original_df, num_samples=5):
    """
    Generate side-by-side comparisons of a few original and synthetic images.
    """
    classes = list(set(label for _, label, _ in reconstructed_images))
    os.makedirs(output_dir, exist_ok=True)

    for class_name in classes:
        plt.figure(figsize=(15, 8))

        # Collect original images for this class
        original_class_images = []
        for img_id, img in original_images.items():
            if len(original_class_images) >= num_samples:
                break
            if any((original_df['image_id'] == img_id) & (original_df['dx'] == class_name)):
                original_class_images.append(img)

        # Collect synthetic images for this class
        smote_class_images = [img for (img, lbl, _) in reconstructed_images if lbl == class_name][:num_samples]

        # Plot original images
        for i, img in enumerate(original_class_images):
            plt.subplot(2, num_samples, i + 1)
            plt.imshow(img)
            plt.title(f"Original {class_name}")
            plt.axis('off')

        # Plot synthetic images
        for i, img in enumerate(smote_class_images):
            plt.subplot(2, num_samples, i + num_samples + 1)
            plt.imshow(img)
            plt.title(f"SMOTE {class_name}")
            plt.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'samples_{class_name}.png'), dpi=300)
        plt.close()

###############################################################################
# 7. Optional Final Enhancement Pass on Synthetic Images
###############################################################################
def enhance_synthetic_dataset(output_dir):
    """
    Apply additional enhancement to all synthetic images in output_dir.
    """
    print("Applying final enhancement to all synthetic images...")
    synthetic_paths = [os.path.join(output_dir, f) for f in os.listdir(output_dir)
                       if f.startswith('smote_') and f.endswith('.jpg')]
    for img_path in tqdm(synthetic_paths):
        img = cv2.imread(img_path)
        denoised = cv2.fastNlMeansDenoisingColored(img, None, 5, 5, 7, 21)
        gaussian = cv2.GaussianBlur(denoised, (0, 0), 3)
        unsharp = cv2.addWeighted(denoised, 1.5, gaussian, -0.5, 0)
        cv2.imwrite(img_path, unsharp, [cv2.IMWRITE_JPEG_QUALITY, 100])
    print(f"Enhanced {len(synthetic_paths)} synthetic images")

###############################################################################
# Main Execution
###############################################################################
if __name__ == "__main__":
    # Set your dataset paths here (change as needed)
    image_dirs = [
        r'/dist_home/suryansh/dl/Skin-Lesion-Classification/Datasets/HAM10000/HAM10000_images_part_1',
        r'/dist_home/suryansh/dl/Skin-Lesion-Classification/Datasets/HAM10000/HAM10000_images_part_2'
    ]
    metadata_path = r'/dist_home/suryansh/dl/Skin-Lesion-Classification/Datasets/HAM10000/HAM10000_metadata.csv'
    output_dir = r'./smote_maximum_clarity'

    # 1. Preprocess dataset (using 128×128 resolution to manage memory)
    print("Preprocessing dataset...")
    X, y, idx_to_img_id, original_images, original_df, original_dist, class_sizes = preprocess_dataset(
        metadata_path, image_dirs, resize_dim=64
    )

    # 2. Apply PCA + SMOTE + Inverse PCA
    print("Running SMOTE pipeline...")
    X_res, y_res, new_dist = apply_tabular_smote(X, y, n_components=300)

    # 3. Reconstruct synthetic images (only for samples beyond the original count)
    original_count = len(X)  # Number of original images
    reconstructed_images = reconstruct_images(X_res, y_res, class_sizes, original_count, recon_dim=128)

    # 4. Save the balanced dataset (original + synthetic) and metadata
    final_distribution = save_smote_images(reconstructed_images, original_images, idx_to_img_id, output_dir, original_df)

    # 5. (Optional) Apply an extra enhancement pass on synthetic images
    enhance_synthetic_dataset(output_dir)

    # 6. Visualize class distributions before and after SMOTE
    visualize_results(original_dist, final_distribution, output_dir)

    # 7. Visualize sample comparisons of original vs synthetic images
    visualize_samples(original_images, reconstructed_images, output_dir, original_df)

    print("SMOTE balancing with reduced resolution complete!")
