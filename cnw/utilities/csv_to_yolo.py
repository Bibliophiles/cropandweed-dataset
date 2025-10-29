#!/usr/bin/env python3
"""
CSV to YOLO Converter
Converts CSV annotations to YOLO format with configurable dataset variants
"""

import os
import csv
import json
import argparse
from pathlib import Path
from PIL import Image
from datasets import DATASETS


DEFAULT_OUTPUT_DIR = "./output"

def load_image_params(params_dir: Path, base_name: str) -> dict:
    """Load additional parameters for an image."""
    params_file = params_dir / f"{base_name}.txt"
    params = {}
    
    if params_file.exists():
        try:
            with open(params_file, 'r') as f:
                for line in f:
                    if ':' in line:
                        key, value = line.strip().split(':', 1)
                        params[key.strip()] = int(value.strip())
        except Exception as e:
            print(f"⚠️ Error reading params for {base_name}: {e}")
    
    return params


def get_image_dimensions(img_path: Path) -> tuple:
    """Get image dimensions safely."""
    try:
        with Image.open(img_path) as im:
            return im.size  # (width, height)
    except Exception as e:
        print(f"⚠️ Error opening image {img_path.name}: {e}")
        return None, None


def process_csv_annotations(csv_path: Path, dataset) -> list:
    """Process CSV file and return list of valid annotations."""
    annotations = []
    
    try:
        with open(csv_path, newline="") as csvfile:
            reader = csv.reader(csvfile)
            for row_num, row in enumerate(reader, 1):
                if len(row) < 7:
                    continue
                
                try:
                    # CSV format: Left, Top, Right, Bottom, Label ID, Stem X, Stem Y
                    left, top, right, bottom, label_id, stem_x, stem_y = map(int, row)
                    
                    # Map to target dataset class
                    mapped_id = dataset.get_mapped_id(label_id)
                    if mapped_id is None:
                        continue
                    
                    # Validate bounding box
                    if right <= left or bottom <= top:
                        continue
                    
                    annotations.append({
                        'bbox': [left, top, right, bottom],
                        'class_id': mapped_id,
                        'stem_point': [stem_x, stem_y],
                        'original_id': label_id
                    })
                    
                except ValueError:
                    continue
                    
    except Exception as e:
        print(f"⚠️ Error reading {csv_path}: {e}")
    
    return annotations


def csv_to_yolo(images_dir: Path, ann_dir: Path, output_dir: Path, target_dataset: str) -> int:
    """
    Convert CSV annotations to YOLO format.
    
    Returns:
        Number of successfully processed files
    """
    # Get dataset configuration
    if target_dataset not in DATASETS:
        raise ValueError(f"Unknown dataset: {target_dataset}. Available: {list(DATASETS.keys())}")
    
    dataset = DATASETS[target_dataset]
    print(f"📋 Using dataset: {target_dataset}")
    print(f"🏷️  Classes: {[(id, dataset.get_label_name(id)) for id in dataset.get_label_ids()]}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    processed_count = 0
    total_annotations = 0
    skipped_files = []
    
    print(f"\n🔄 Converting CSV annotations to YOLO format...")
    
    for ann_file in ann_dir.glob("*.csv"):
        base_name = ann_file.stem
        img_file = base_name + ".jpg"
        img_path = images_dir / img_file
        
        # Check if corresponding image exists
        if not img_path.exists():
            skipped_files.append(f"{ann_file.name} (no image)")
            continue
        
        # Get image dimensions
        width, height = get_image_dimensions(img_path)
        if width is None or height is None:
            skipped_files.append(f"{ann_file.name} (invalid image)")
            continue
        
        # Process CSV annotations
        annotations = process_csv_annotations(ann_file, dataset)
        if not annotations:
            skipped_files.append(f"{ann_file.name} (no valid annotations)")
            continue
        
        # Convert to YOLO format and write
        yolo_file = output_dir / f"{base_name}.txt"
        valid_annotations = 0
        
        with open(yolo_file, "w") as yolo_f:
            for ann in annotations:
                left, top, right, bottom = ann['bbox']
                class_id = ann['class_id']
                
                # Validate bounding box against image dimensions
                if left < 0 or top < 0 or right > width or bottom > height:
                    continue
                
                # Convert to YOLO format (normalized coordinates)
                x_center = ((left + right) / 2) / width
                y_center = ((top + bottom) / 2) / height
                bbox_w = (right - left) / width
                bbox_h = (bottom - top) / height
                
                # Validate normalized coordinates
                if 0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 < bbox_w <= 1 and 0 < bbox_h <= 1:
                    yolo_f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_w:.6f} {bbox_h:.6f}\n")
                    valid_annotations += 1
        
        if valid_annotations > 0:
            processed_count += 1
            total_annotations += valid_annotations
            print(f"✅ {ann_file.name}: {valid_annotations} annotations")
        else:
            yolo_file.unlink()  # Remove empty file
            skipped_files.append(f"{ann_file.name} (no valid YOLO annotations)")
    
    # Print summary
    print(f"\n📊 YOLO Conversion Summary:")
    print(f"   Successfully processed: {processed_count} files")
    print(f"   Total annotations: {total_annotations}")
    print(f"   Skipped files: {len(skipped_files)}")
    
    if skipped_files and len(skipped_files) <= 10:
        print(f"\n⚠️  Skipped files:")
        for file in skipped_files:
            print(f"   - {file}")
    elif len(skipped_files) > 10:
        print(f"\n⚠️  Skipped {len(skipped_files)} files (showing first 5):")
        for file in skipped_files[:5]:
            print(f"   - {file}")
    
    print(f"\n✅ YOLO labels saved to: {output_dir}")
    
    return processed_count


def csv_to_coco(images_dir: Path, ann_dir: Path, params_dir: Path, output_json: Path, target_dataset: str) -> int:
    """
    Convert CSV annotations to COCO format.
    
    Returns:
        Number of images processed
    """
    # Get dataset configuration
    if target_dataset not in DATASETS:
        raise ValueError(f"Unknown dataset: {target_dataset}. Available: {list(DATASETS.keys())}")
    
    dataset = DATASETS[target_dataset]
    label_ids = dataset.get_label_ids()
    
    # Create categories for COCO format
    categories = [
        {
            "id": int(label_id), 
            "name": dataset.get_label_name(label_id), 
            "supercategory": "plant"
        }
        for label_id in label_ids
    ]
    
    # Initialize COCO structure
    coco = {
        "images": [], 
        "annotations": [], 
        "categories": categories,
        "info": {
            "description": f"CropAndWeed Dataset - {target_dataset}",
            "version": "1.0",
            "year": 2024,
            "contributor": "CropAndWeed Dataset",
            "date_created": "2024"
        }
    }
    
    print(f"\n🔄 Converting to COCO format for dataset: {target_dataset}")
    print(f"📂 Categories: {len(categories)}")
    
    ann_id = 1
    img_id = 1
    processed_images = 0
    
    for ann_file in ann_dir.glob("*.csv"):
        base_name = ann_file.stem
        img_file = base_name + ".jpg"
        img_path = images_dir / img_file
        
        if not img_path.exists():
            continue
        
        # Get image dimensions
        width, height = get_image_dimensions(img_path)
        if width is None or height is None:
            continue
        
        # Load additional parameters if available
        params = load_image_params(params_dir, base_name) if params_dir and params_dir.exists() else {}
        
        # Create image entry
        image_entry = {
            "id": img_id,
            "file_name": img_file,
            "width": width,
            "height": height
        }
        
        # Add parameters if available
        if params:
            image_entry.update({
                "moisture": params.get("moisture", -1),
                "soil": params.get("soil", -1),
                "lighting": params.get("lighting", -1),
                "separability": params.get("separability", -1)
            })
        
        coco["images"].append(image_entry)
        
        # Process annotations
        annotations = process_csv_annotations(ann_file, dataset)
        annotation_count = 0
        
        for ann in annotations:
            left, top, right, bottom = ann['bbox']
            mapped_id = ann['class_id']
            stem_x, stem_y = ann['stem_point']
            
            # Validate bounding box
            if left < 0 or top < 0 or right > width or bottom > height:
                continue
            
            bbox_w = right - left
            bbox_h = bottom - top
            area = bbox_w * bbox_h
            
            if area <= 0:
                continue
            
            # Create COCO annotation
            annotation = {
                "id": ann_id,
                "image_id": img_id,
                "category_id": mapped_id,
                "bbox": [left, top, bbox_w, bbox_h],
                "area": area,
                "iscrowd": 0,
                "stem_point": [stem_x, stem_y]
            }
            
            coco["annotations"].append(annotation)
            ann_id += 1
            annotation_count += 1
        
        if annotation_count > 0:
            processed_images += 1
        
        img_id += 1
    
    # Save COCO format JSON
    with open(output_json, "w") as f:
        json.dump(coco, f, indent=2)
    
    print(f"\n📊 COCO Conversion Summary:")
    print(f"   Images processed: {len(coco['images'])}")
    print(f"   Total annotations: {len(coco['annotations'])}")
    print(f"   Categories: {len(coco['categories'])}")
    print(f"\n✅ COCO annotations saved to: {output_json}")
    
    return processed_images


def convert_annotations(target_dataset: str, output_dir: str = None) -> dict:
    """
    Main conversion function that creates both YOLO and COCO formats.
    
    Args:
        target_dataset: Target dataset variant to use for class mapping
        output_dir: Output directory (optional, uses default if not provided)
    
    Returns:
        Dict with conversion results
    """
    # Setup paths
    images_dir = Path(IMAGES_DIR)
    ann_dir = Path(ANNOTATIONS_DIR)
    params_dir = Path(PARAMS_DIR)
    output_path = Path(output_dir) if output_dir else Path(DEFAULT_OUTPUT_DIR)
    
    # Validate input directories
    if not images_dir.exists():
        raise ValueError(f"Images directory not found: {images_dir}")
    if not ann_dir.exists():
        raise ValueError(f"Annotations directory not found: {ann_dir}")
    if not params_dir.exists():
        print(f"⚠️ Parameters directory not found: {params_dir}")
        params_dir = None
    
    # Validate target dataset
    if target_dataset not in DATASETS:
        raise ValueError(f"Unknown dataset: {target_dataset}. Available: {list(DATASETS.keys())}")
    
    print(f"🚀 Converting annotations for dataset: {target_dataset}")
    print(f"📁 Images: {images_dir}")
    print(f"📄 Annotations: {ann_dir}")
    print(f"📦 Parameters: {params_dir}")
    print(f"📤 Output: {output_path}")
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # Convert to YOLO format
    print(f"\n" + "="*60)
    print(f"🎯 YOLO FORMAT CONVERSION")
    print("="*60)
    
    yolo_output_dir = output_path / f"labels_{target_dataset}"
    processed_yolo = csv_to_yolo(images_dir, ann_dir, yolo_output_dir, target_dataset)
    results['yolo'] = {
        'processed_files': processed_yolo,
        'output_dir': str(yolo_output_dir)
    }
    
    # Convert to COCO format
    print(f"\n" + "="*60)
    print(f"🎯 COCO FORMAT CONVERSION")
    print("="*60)
    
    coco_output_file = output_path / f"annotations_{target_dataset}.json"
    processed_coco = csv_to_coco(images_dir, ann_dir, params_dir, coco_output_file, target_dataset)
    results['coco'] = {
        'processed_images': processed_coco,
        'output_file': str(coco_output_file)
    }
    
    return results


def parse_arguments():
    """Parse command-line arguments for directory paths."""
    parser = argparse.ArgumentParser(
        description='Convert CSV annotations to YOLO format',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--images-dir',
        type=str,
        default='/home/anad0001/cropandweed-dataset-main/data/images',
        help='Path to the images directory'
    )
    
    parser.add_argument(
        '--annotations-dir',
        type=str,
        default='/home/anad0001/cropandweed-dataset-main/data/bboxes/CropAndWeed',
        help='Path to the annotations (bboxes) directory'
    )
    
    parser.add_argument(
        '--params-dir',
        type=str,
        default='/home/anad0001/cropandweed-dataset-main/data/params',
        help='Path to the parameters directory'
    )

    parser.add_argument(
        '--target-dataset',
        type=str,
        required=True,
        help='Target dataset name (e.g., CropAndWeed, Fine24, etc.)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='./output',
        help='Output directory for YOLO and COCO files'
    )
    
    args = parser.parse_args()
    
    # Validate that directories exist
    for dir_name, dir_path in [
        ('Images', args.images_dir),
        ('Annotations', args.annotations_dir),
        ('Params', args.params_dir)
    ]:
        if not os.path.exists(dir_path):
            parser.error(f"{dir_name} directory does not exist: {dir_path}")
    
    return args


def main():
    """Main entry point."""
    args = parse_arguments()
    
    try:
        print(f"🔧 Configuration:")
        print(f"   Images: {IMAGES_DIR}")
        print(f"   Annotations: {ANNOTATIONS_DIR}")
        print(f"   Parameters: {PARAMS_DIR}")
        print(f"   Target dataset: {args.target_dataset}")
        print(f"   Output: {args.output_dir}")
        
        # Convert annotations to both formats
        results = convert_annotations(args.target_dataset, args.output_dir)
        
        # Print final summary
        print(f"\n" + "="*60)
        print(f"🎉 CONVERSION COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"📊 Results:")
        print(f"   YOLO files processed: {results['yolo']['processed_files']}")
        print(f"   YOLO output: {results['yolo']['output_dir']}")
        print(f"   COCO images processed: {results['coco']['processed_images']}")
        print(f"   COCO output: {results['coco']['output_file']}")
        
        print(f"\n🚀 Next steps:")
        print(f"   Use with dataset_structure.py:")
        print(f"   python dataset_structure.py \\")
        print(f"       --images_dir {IMAGES_DIR} \\")
        print(f"       --labels_dir {results['yolo']['output_dir']} \\")
        print(f"       --output_dir ./pipeline_dataset_{args.target_dataset} \\")
        print(f"       --dataset_name {args.target_dataset}")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1


if __name__ == "__main__":
    args = parse_arguments()
    
    IMAGES_DIR = args.images_dir
    ANNOTATIONS_DIR = args.annotations_dir
    PARAMS_DIR = args.params_dir
    
    exit(main())