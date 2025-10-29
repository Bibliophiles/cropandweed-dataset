#!/usr/bin/env python3
"""
Dataset Structure Creator for YOLO Pipeline
Creates a unified dataset structure for stratified cross-validation
"""

import os
import json
import shutil
import argparse
from pathlib import Path
import yaml
from collections import defaultdict


from datasets import DATASETS


def setup_dataset_directories(output_dir: Path) -> Path:
    """Create the basic dataset directory structure."""
    dataset_dir = Path(output_dir)
    dataset_dir.mkdir(exist_ok=True)
    
    (dataset_dir / 'images').mkdir(parents=True, exist_ok=True)
    (dataset_dir / 'labels').mkdir(parents=True, exist_ok=True)
    
    return dataset_dir


def find_valid_image_label_pairs(images_dir: Path, labels_dir: Path) -> list:
    """Find all valid image-label pairs."""

    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_files.extend(images_dir.glob(ext))
    
    # Filter images that have corresponding label files
    valid_images = []
    
    print(f"🔍 Scanning for valid image-label pairs...")
    
    for img_file in image_files:
        label_file = labels_dir / f"{img_file.stem}.txt"
        if label_file.exists():
            # Check if label file has content (not empty)
            if label_file.stat().st_size > 0:
                valid_images.append(img_file)
            else:
                print(f"⚠️  Skipping {img_file.name} - empty label file")
        else:
            print(f"⚠️  Skipping {img_file.name} - no corresponding label file")
    
    print(f"\n📊 Dataset Statistics:")
    print(f"   Total images found: {len(image_files)}")
    print(f"   Valid image-label pairs: {len(valid_images)}")
    print(f"   Excluded: {len(image_files) - len(valid_images)}")
    
    if len(valid_images) == 0:
        raise ValueError("No valid image-label pairs found!")
    
    return valid_images


def copy_files_to_dataset(valid_images: list, labels_dir: Path, dataset_dir: Path) -> tuple:
    """Copy valid image-label pairs to the dataset directory."""
    print(f"\n🔄 Copying files to pipeline format...")
    
    copied_images = 0
    copied_labels = 0
    skipped_files = []
    
    for img_file in valid_images:
        try:
            # Copy image
            dst_img = dataset_dir / 'images' / img_file.name
            shutil.copy2(img_file, dst_img)
            copied_images += 1
            
            # Copy corresponding label
            label_file = labels_dir / f"{img_file.stem}.txt"
            dst_label = dataset_dir / 'labels' / f"{img_file.stem}.txt"
            shutil.copy2(label_file, dst_label)
            copied_labels += 1
            
        except Exception as e:
            print(f"❌ Error copying {img_file.name}: {e}")
            skipped_files.append(img_file.name)
    
    print(f"✅ Successfully copied:")
    print(f"   Images: {copied_images}")
    print(f"   Labels: {copied_labels}")
    
    if skipped_files:
        print(f"⚠️  Skipped {len(skipped_files)} files due to errors")
    
    return copied_images, copied_labels


def get_class_names_from_dataset(dataset_name: str, labels_dir: Path) -> list:
    """
    ✅ FIXED: Get proper class names from the dataset configuration.
    """
    if dataset_name not in DATASETS:
        print(f"⚠️  Unknown dataset '{dataset_name}', available: {list(DATASETS.keys())}")
        print(f"⚠️  Falling back to auto-detection from labels...")
        return get_class_names_from_labels_fallback(labels_dir)
    
    dataset = DATASETS[dataset_name]
    
    # Get class IDs that actually exist in the labels
    existing_class_ids = set()
    for label_file in labels_dir.glob('*.txt'):
        with open(label_file, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        class_id = int(line.split()[0])
                        existing_class_ids.add(class_id)
                    except (ValueError, IndexError):
                        continue
    
    if not existing_class_ids:
        raise ValueError("No valid class IDs found in label files!")
    
    # Create class names list based on dataset configuration
    max_class_id = max(existing_class_ids)
    class_names = []
    
    for i in range(max_class_id + 1):
        if i in dataset.get_label_ids():
            # Use the proper name from dataset
            class_names.append(dataset.get_label_name(i))
        else:
            # Fallback for missing classes
            class_names.append(f"class_{i}")
    
    print(f"✅ Using {dataset_name} class names:")
    for i, name in enumerate(class_names):
        if i in existing_class_ids:
            print(f"   {i}: {name} ✓")
        else:
            print(f"   {i}: {name} (not used)")
    
    return class_names


def get_class_names_from_labels_fallback(labels_dir: Path) -> list:
    """Fallback method: Extract class names from label files (generic names)."""
    class_ids = set()
    
    for label_file in labels_dir.glob('*.txt'):
        with open(label_file, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        class_id = int(line.split()[0])
                        class_ids.add(class_id)
                    except (ValueError, IndexError):
                        continue
    
    # Create generic class names as fallback
    max_class_id = max(class_ids) if class_ids else 0
    class_names = [f"class_{i}" for i in range(max_class_id + 1)]
    
    print(f"⚠️  Using generic class names (no dataset mapping found):")
    for i, name in enumerate(class_names):
        print(f"   {i}: {name}")
    
    return class_names


def create_yaml_config(dataset_dir: Path, class_names: list) -> Path:
    """Create data.yaml file for the pipeline."""
    config = {
        'path': str(dataset_dir.absolute()),
        'train': './images',  # Will be overridden by pipeline for each fold
        'val': './images',    # Will be overridden by pipeline for each fold
        'nc': len(class_names),
        'names': class_names
    }
    
    yaml_path = dataset_dir / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"\n✅ Created pipeline config: {yaml_path}")
    print(f"📁 Pipeline dataset structure:")
    print(f"   {dataset_dir}/")
    print(f"   ├── images/           # All images (no splits)")
    print(f"   ├── labels/           # All labels (no splits)")
    print(f"   └── data.yaml         # Configuration file")
    
    print(f"\n📋 Classes ({len(class_names)}):")
    for i, name in enumerate(class_names):
        print(f"   {i}: {name}")
    
    return yaml_path


def analyze_dataset_distribution(dataset_dir: Path, class_names: list) -> dict:
    """Analyze class distribution and data quality."""
    class_counts = defaultdict(int)
    total_annotations = 0
    total_images = 0
    images_per_class = defaultdict(set)
    
    labels_dir = dataset_dir / 'labels'
    
    print(f"\n📊 Analyzing class distribution...")
    
    for label_file in labels_dir.glob('*.txt'):
        total_images += 1
        image_classes = set()
        
        with open(label_file, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        class_id = int(line.split()[0])
                        class_counts[class_id] += 1
                        total_annotations += 1
                        image_classes.add(class_id)
                    except (ValueError, IndexError):
                        print(f"⚠️  Invalid annotation in {label_file.name}: {line.strip()}")
        
        # Track which images contain each class
        for class_id in image_classes:
            images_per_class[class_id].add(label_file.stem)
    
    print(f"\n📈 Dataset Analysis:")
    print(f"   Total images: {total_images}")
    print(f"   Total annotations: {total_annotations}")
    print(f"   Average annotations per image: {total_annotations/total_images:.2f}")
    
    print(f"\n📋 Class Distribution:")
    print(f"{'Class Name':<25} {'ID':<4} {'Annotations':<12} {'Images':<8} {'%Images':<8}")
    print("-" * 75)
    
    for class_id in sorted(class_counts.keys()):
        # ✅ Use actual class name instead of generic
        class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
        ann_count = class_counts[class_id]
        img_count = len(images_per_class[class_id])
        img_percentage = (img_count / total_images) * 100
        
        print(f"{class_name:<25} {class_id:<4} {ann_count:<12} {img_count:<8} {img_percentage:<7.1f}%")
    
    # Data quality checks
    print(f"\n🔍 Data Quality Checks:")
    
    images_dir = dataset_dir / 'images'
    all_images = set(f.stem for f in images_dir.glob('*') if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp'])
    labeled_images = set(f.stem for f in labels_dir.glob('*.txt'))
    
    unlabeled_images = all_images - labeled_images
    if unlabeled_images:
        print(f"   ⚠️  {len(unlabeled_images)} images without labels")
    
    orphaned_labels = labeled_images - all_images
    if orphaned_labels:
        print(f"   ⚠️  {len(orphaned_labels)} labels without images")
    
    if class_counts:
        min_class = min(class_counts.values())
        max_class = max(class_counts.values())
        imbalance_ratio = max_class / min_class
        print(f"   📊 Class imbalance ratio: {imbalance_ratio:.1f}:1")
        
        if imbalance_ratio > 10:
            print(f"   ⚠️  High class imbalance detected!")
    
    return dict(class_counts)


def create_pipeline_dataset(images_dir: str, labels_dir: str, output_dir: str, dataset_name: str) -> Path:
    """
    Main function to create dataset structure for stratified cross-validation pipeline.
    
    Args:
        images_dir: Path to directory containing images
        labels_dir: Path to directory containing YOLO format labels
        output_dir: Path where the unified dataset will be created
        dataset_name: Name of the dataset for identification (must match DATASETS keys)
    
    Returns:
        Path to created dataset directory
    """
    images_path = Path(images_dir)
    labels_path = Path(labels_dir)
    
    # Validate input directories
    if not images_path.exists():
        raise ValueError(f"Images directory not found: {images_dir}")
    if not labels_path.exists():
        raise ValueError(f"Labels directory not found: {labels_dir}")
    
    print(f"🔄 Creating '{dataset_name}' dataset for stratified cross-validation pipeline...")
    
    # Create output directory structure
    dataset_dir = setup_dataset_directories(output_dir)
    
    # Find valid image-label pairs
    valid_images = find_valid_image_label_pairs(images_path, labels_path)
    
    # Copy files to dataset
    copied_images, copied_labels = copy_files_to_dataset(valid_images, labels_path, dataset_dir)
    
    # Get proper class names from dataset configuration
    class_names = get_class_names_from_dataset(dataset_name, dataset_dir / 'labels')
    
    # Create YAML configuration
    yaml_path = create_yaml_config(dataset_dir, class_names)
    
    # Analyze dataset
    class_counts = analyze_dataset_distribution(dataset_dir, class_names)
    
    print(f"\n🎉 '{dataset_name}' dataset preparation complete!")
    print(f"📁 Dataset location: {dataset_dir}")
    print(f"📄 Config file: {yaml_path}")
    print(f"\n📈 Ready for stratified cross-validation pipeline!")
    print(f"   Total images: {copied_images}")
    print(f"   Classes: {len(class_counts)}")
        
    return dataset_dir


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Create YOLO dataset structure for stratified cross-validation pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--images_dir', 
        type=str, 
        required=True,
        help='Path to directory containing images'
    )
    
    parser.add_argument(
        '--labels_dir', 
        type=str, 
        required=True,
        help='Path to directory containing YOLO format labels (.txt files)'
    )
    
    parser.add_argument(
        '--output_dir', 
        type=str, 
        required=True,
        help='Path where the unified dataset will be created'
    )
    
    parser.add_argument(
        '--dataset_name', 
        type=str, 
        required=True,
        choices=list(DATASETS.keys()),  # ✅ Restrict to valid dataset names
        help=f'Dataset variant name (choices: {list(DATASETS.keys())})'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()
    
    try:
        dataset_dir = create_pipeline_dataset(
            images_dir=args.images_dir,
            labels_dir=args.labels_dir,
            output_dir=args.output_dir,
            dataset_name=args.dataset_name
        )
        
        print(f"\n✅ Success! Dataset created at: {dataset_dir}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())