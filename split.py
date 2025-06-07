import os
import shutil
import random
from pathlib import Path

def split_data(image_dir, annotation_dir, output_dir, train_split=0.7, test_split=0.15, valid_split=0.15):
    # Ensure the output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories for train, test, and valid
    for split in ['train', 'test', 'valid']:
        (output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)

    # List all images and annotations
    images = sorted([f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))])
    annotations = sorted([f for f in os.listdir(annotation_dir) if os.path.isfile(os.path.join(annotation_dir, f))])

    # Filter out images that do not have corresponding annotations
    valid_images = []
    valid_annotations = []
    for img in images:
        annotation_name = os.path.splitext(img)[0] + '.txt'  # assuming annotations are in .txt format
        if annotation_name in annotations:
            valid_images.append(img)
            valid_annotations.append(annotation_name)

    # Ensure the number of valid images matches the number of valid annotations
    assert len(valid_images) == len(valid_annotations), "The number of valid images must match the number of valid annotations"

    # Shuffle the data
    combined = list(zip(valid_images, valid_annotations))
    random.shuffle(combined)
    valid_images[:], valid_annotations[:] = zip(*combined)

    # Calculate split indices
    total_images = len(valid_images)
    train_end = int(total_images * train_split)
    test_end = train_end + int(total_images * test_split)

    # Split the data
    train_images, train_annotations = valid_images[:train_end], valid_annotations[:train_end]
    test_images, test_annotations = valid_images[train_end:test_end], valid_annotations[train_end:test_end]
    valid_images, valid_annotations = valid_images[test_end:], valid_annotations[test_end:]

    # Helper function to copy files
    def copy_files(images, annotations, split):
        for img, ann in zip(images, annotations):
            shutil.copy(os.path.join(image_dir, img), os.path.join(output_dir, split, 'images', img))
            shutil.copy(os.path.join(annotation_dir, ann), os.path.join(output_dir, split, 'labels', ann))

    # Copy the files to their respective directories
    copy_files(train_images, train_annotations, 'train')
    copy_files(test_images, test_annotations, 'test')
    copy_files(valid_images, valid_annotations, 'valid')

    print(f"Data successfully split into train, test, and valid sets")

# Example usage
image_dir = 'D:/objectDetection/Deepfish/'
annotation_dir = 'D:/objectDetection/DeepfishA/'
output_dir = 'D:/objectDetection/DeepfishR/'
split_data(image_dir, annotation_dir, output_dir)
