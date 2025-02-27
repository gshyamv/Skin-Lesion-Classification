import cv2
import numpy as np
import os
import glob

def generate_bounding_box_on_original(image_path, mask_path, output_image_path, output_label_path):
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if image is None or mask is None:
        print(f"Error: Could not load {image_path} or {mask_path}")
        return

    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Find all white pixel coordinates
    white_pixels = np.column_stack(np.where(binary_mask == 255))
    if white_pixels.size == 0:
        print(f"No lesion detected in mask: {mask_path}")
        return

    min_y, min_x = white_pixels.min(axis=0)
    max_y, max_x = white_pixels.max(axis=0)
    
    # Centered bounding box
    center_x = (min_x + max_x) // 2
    center_y = (min_y + max_y) // 2
    half_box = max(max_x - center_x, max_y - center_y)

    left = max(center_x - half_box, 0)
    right = min(center_x + half_box, image.shape[1] - 1)
    top = max(center_y - half_box, 0)
    bottom = min(center_y + half_box, image.shape[0] - 1)

    # Draw bounding box on the original image
    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)

    # Save the image with bounding box
    cv2.imwrite(output_image_path, image)

    # YOLO format: class_id x_center y_center width height
    bbox_width = right - left
    bbox_height = bottom - top
    yolo_x_center = (left + bbox_width / 2) / image.shape[1]
    yolo_y_center = (top + bbox_height / 2) / image.shape[0]
    yolo_width = bbox_width / image.shape[1]
    yolo_height = bbox_height / image.shape[0]

    with open(output_label_path, "w") as f:
        f.write(f"0 {yolo_x_center:.6f} {yolo_y_center:.6f} {yolo_width:.6f} {yolo_height:.6f}\n")


def process_all_images(image_dirs, mask_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    boxed_images_dir = os.path.join(output_dir, 'boxed_images')
    labels_dir = os.path.join(output_dir, 'labels')
    os.makedirs(boxed_images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    image_paths = []
    for img_dir in image_dirs:
        image_paths.extend(glob.glob(os.path.join(img_dir, "*.jpg")))

    total = len(image_paths)
    print(f"Found {total} images to process...")

    for image_path in image_paths:
        image_name = os.path.basename(image_path).replace('.jpg', '')
        mask_path = os.path.join(mask_dir, f"{image_name}_segmentation.png")

        if not os.path.exists(mask_path):
            print(f"Mask not found for image: {image_name}")
            continue

        output_image_path = os.path.join(boxed_images_dir, f"{image_name}_boxed.jpg")
        output_label_path = os.path.join(labels_dir, f"{image_name}.txt")

        generate_bounding_box_on_original(
            image_path, mask_path,
            output_image_path, output_label_path
        )
        print(f"Processed {image_name}")

    print("Bounding boxes on original images completed.")

if __name__ == '__main__':
    image_dirs = [r'C:\profolders\Collage stuff\Sem 4 project\DL\Datasets\HAM10000\HAM10000_images_part_1', r'C:\profolders\Collage stuff\Sem 4 project\DL\Datasets\HAM10000\HAM10000_images_part_2']
    mask_dir = r'C:\profolders\Collage stuff\Sem 4 project\DL\Datasets\HAM10000_segmentations_lesion_tschandl'
    output_dir = 'yolo_dataset'

    process_all_images(image_dirs, mask_dir, output_dir)
