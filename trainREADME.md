## Table of Contents

1. Prerequisites
2. Initial Setup
3. Repository Structure
4. Dataset Preparation
5. Training Pipeline
6. Managing Training Sessions
7. Monitoring and Results

---

## Prerequisites

### Required Software

- **Visual Studio Code** - IDE for development
- **Cisco Secure Client** - VPN connection to access remote server
- **Python 3.8+** - Programming language
- **Miniconda** - Lightweight conda environment manager
- **tmux** - Terminal multiplexer for persistent sessions

### Hardware Requirements

- Access to GPU server (e.g., `srgpu01`)
- NVIDIA GPU with CUDA support
- Sufficient storage for dataset and model outputs

---

## Initial Setup

### 1. Connect to Remote Server

```bash
# Install VS Code and Cisco Secure Client on your local machine

# Connect to VPN using Cisco Secure Client with your credentials

# SSH into the GPU server from VS Code terminal
ssh srgpu01
```

### 2. Install Miniconda

```bash
# Download Miniconda installer
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# Run installer
bash Miniconda3-latest-Linux-x86_64.sh

# Follow prompts and restart terminal
source ~/.bashrc
```

### 3. Create Conda Environment

```bash
# Create a new conda environment named 'cnw_env'
conda create -n cnw_env python=3.9 -y

# Activate the environment
conda activate cnw_env
```

### 4. Clone Repository

```bash
# Clone the CropAndWeed dataset repository
git clone https://github.com/Bibliophiles/cropandweed-dataset.git

# Navigate to project directory
cd cropandweed-dataset/cnw
```

### 5. Install Dependencies

```bash
# Install required Python packages from requirements.txt
# This includes ultralytics, optuna, scikit-learn, etc.
pip install -r requirements.txt
```

### 6. Download Dataset

```bash
# Run setup script to download and extract dataset
# This creates images, annotations, and mapped dataset variants
python setup.py
```

**What this does:**
- Downloads all dataset images
- Extracts annotation files (bounding boxes, masks, etc.)
- Creates pre-defined dataset variants (CropOrWeed2, CropAndWeed, etc.)

### 7. Install tmux

```bash
# Install tmux for session management
sudo apt-get install tmux  # Ubuntu/Debian
# or
conda install -c conda-forge tmux
```

---

## Repository Structure

```
cropandweed-dataset/
├── cnw/                                    # Main package directory
│   ├── __init__.py                        # Package initializer
│   ├── map_dataset.py                     # Dataset mapping utilities
│   ├── requirements.txt                   # Python dependencies
│   ├── setup.py                           # Dataset download script
│   ├── visualize_annotations.py           # Annotation visualization tool
│   │
│   └── utilities/                         # Utility scripts
│       ├── csv_to_yolo.py                # Convert CSV annotations to YOLO format
│       ├── dataset_structure.py          # Create pipeline-ready dataset structure
│       ├── datasets.py                   # Dataset variant definitions
│       │
│       └── yolo_training_pipeline/       # Main training pipeline
│           ├── main.py                   # Pipeline entry point
│           │
│           ├── config/                   # Configuration files
│           │   ├── base_config.py       # Base training configuration
│           │   └── model_configs.py     # YOLO model variants
│           │
│           ├── data/                     # Data handling
│           │   └── stratified_split.py  # Cross-validation splitting
│           │
│           ├── models/                   # Model wrappers
│           │   └── sklearn_yolo_wrapper.py  # Scikit-learn compatible YOLO
│           │
│           ├── optimization/             # Hyperparameter optimization
│           │   └── optuna_nested_cv.py  # Nested CV with Optuna
│           │
│           └── utils/                    # Utility modules
│               ├── device_utils.py      # GPU/CPU management
│               ├── gpu_monitoring.py    # Hardware monitoring
│               ├── logging_utils.py     # Logging management
│               ├── metrics.py           # Metrics calculation
│               ├── reproducibility.py   # Seed setting & environment checks
│               ├── utils.py             # General utilities
│               └── visualization.py     # Results visualization
│
├── data/                                  # Dataset storage (created by setup.py)
│   ├── images/                           # All dataset images
│   ├── bboxes/                           # Bounding box annotations
│   ├── params/                           # Image parameters
│   └── masks/                            # Semantic segmentation masks
│
├── figures/                               # Documentation figures
├── images/                                # README images
├── README.md                              # Main documentation
└── trainREADME.md                        # This training guide
```

### Key Files Explained

| File | Purpose |
|------|---------|
| `setup.py` | Downloads dataset images and annotations |
| `csv_to_yolo.py` | Converts CSV bounding boxes to YOLO format |
| `dataset_structure.py` | Prepares dataset for stratified cross-validation |
| `datasets.py` | Defines dataset variants (CropOrWeed2, CropAndWeed, etc.) |
| `main.py` | Entry point for training pipeline |
| `optuna_nested_cv.py` | Implements nested cross-validation with hyperparameter optimization |
| `sklearn_yolo_wrapper.py` | Wraps YOLO for scikit-learn compatibility |

---

## Dataset Preparation

### Step 1: Convert Annotations to YOLO Format

Navigate to the utilities directory:

```bash
cd cropandweed-dataset/cnw/utilities
```

Run the conversion script for each target-dataset:

```bash
python csv_to_yolo.py \
  --images-dir /home/anad0001/cropandweed-dataset/data/images \
  --annotations-dir /home/anad0001/cropandweed-dataset/data/bboxes/CropAndWeed \
  --params-dir /home/anad0001/cropandweed-dataset/data/params \
  --target-dataset CropOrWeed2 \
  --output-dir ./converted_annotations
```

#### Parameter Explanation

| Parameter | Description | Example Value |
|-----------|-------------|---------------|
| `--images-dir` | Path to directory containing all images | `/path/to/data/images` |
| `--annotations-dir` | Path to CSV annotation files (use `CropAndWeed` base directory, not variant-specific) | `/path/to/data/bboxes/CropAndWeed` |
| `--params-dir` | Path to image parameter files (metadata) | `/path/to/data/params` |
| `--target-dataset` | Dataset variant to create (choices: `CropOrWeed2`, `Fine24`, `CropsOrWeed9`, `Coarse1`etc.) | `CropOrWeed2` |
| `--output-dir` | Directory where YOLO format labels will be saved | `./converted_annotations` |

**IMPORTANT:** Always use the base `CropAndWeed` directory for `--annotations-dir`, as it contains all annotations that are then filtered based on the `--target-dataset` parameter.

**Output:**
- Creates `converted_annotations/labels_CropOrWeed2/` with YOLO format `.txt` files
- Each `.txt` file contains: `<class_id> <x_center> <y_center> <width> <height>` (normalized coordinates)

### Step 2: Create Pipeline Dataset Structure

Run the dataset structure creation script:

```bash
python dataset_structure.py \
  --images_dir /home/anad0001/cropandweed-dataset/data/images \
  --labels_dir converted_annotations/labels_CropOrWeed2 \
  --output_dir ./pipeline_dataset_CropOrWeed2 \
  --dataset_name CropOrWeed2
```

#### Parameter Explanation

| Parameter | Description | Example Value |
|-----------|-------------|---------------|
| `--images_dir` | Path to original images directory | `/path/to/data/images` |
| `--labels_dir` | Path to YOLO format labels (output from Step 1) | `converted_annotations/labels_CropOrWeed2` |
| `--output_dir` | Directory where unified dataset will be created | `./pipeline_dataset_CropOrWeed2` |
| `--dataset_name` | Must match one of the predefined dataset variants | `CropOrWeed2` |

**What this creates:**
```
pipeline_dataset_CropOrWeed2/
├── images/          # All images (no train/val splits)
├── labels/          # All YOLO labels (no train/val splits)
└── data.yaml        # Dataset configuration file
```

**Why no splits?** The pipeline uses stratified cross-validation, so splits are created dynamically during training to ensure balanced class distribution.

---

## Training Pipeline

### Available Dataset Variants

See `datasets.py` for all variants. Common ones include:

- **`CropOrWeed2`** - Binary classification (crop vs. weed)
- **`CropAndWeed`** - All plant species
- **`CropsOrWeed9`** - 8 crops + 1 weed class
- **`Coarse1`** - 1 vegetation class

### Available YOLO Models

From `model_configs.py`:

| Model | Size | 
|-------|------|
| `yolov5nu` | Nano-U |
| `yolov5su` | Small-U | 
| `yolov5mu` | Medium-U | 
| `yolov5lu` | Large-U | 
| `yolov8n` | Nano | 
| `yolov8s` | Small | 
| `yolov8m` | Medium | 
| `yolov8l` | Large | 
| `yolo11s` | Small | 
| `yolo11m` | Medium | 
| `yolo11l` | Large | 
| `yolo12s` | Small | 
| `yolo12m` | Medium | 
| `yolo12l` | Large | 

### Navigate to Training Pipeline

```bash
cd cropandweed-dataset/cnw/utilities/yolo_training_pipeline
```

### Check Available GPUs

```bash
# View GPU status and availability
nvidia-smi
```

**Example output:**
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.60.13    Driver Version: 525.60.13    CUDA Version: 12.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|   0  Tesla V100-SXM2...  Off  | 00000000:00:1E.0 Off |                    0 |
| N/A   32C    P0    39W / 300W |      0MiB / 16384MiB |      0%      Default |
```

**Note:** Each training session uses 2 GPUs by default (specified by `--gpu_ids`).

---

## Managing Training Sessions

### Create tmux Session

Since each training can take hours/days, use `tmux` to keep sessions running after disconnecting.

```bash
# Create a new named tmux session (e.g., 'yolo_v5')
tmux new -s yolo_v5

# Inside the session:
# 1. Navigate to pipeline directory
cd cropandweed-dataset/cnw/utilities/yolo_training_pipeline

# 2. Activate conda environment
conda activate cnw_env

# 3. Start training (see next section)
```

### Start Training

**Example command:**

```bash
python main.py \
  --model yolov5su \
  --data_path /home/anad0001/cropandweed-dataset/cnw/utilities/pipeline_dataset_CropOrWeed2 \
  --gpu_ids "0,1" \
  --nested_cv \
  --n_trials 20 \
  --epochs 50 \
  --n_folds 5 \
  --n_outer_folds 5
```

#### Complete Parameter Reference

From `base_config.py`:

| Parameter | Description | Default | Example |
|-----------|-------------|---------|---------|
| `--model` | YOLO model variant | **Required** | `yolov5su`, `yolov8s`, `yolo11m` |
| `--data_path` | Path to prepared dataset directory | **Required** | `./pipeline_dataset_CropOrWeed2` |
| `--gpu_ids` | GPU IDs to use (comma-separated) | `"auto"` | `"0,1"`, `"2,3"`, `"cpu"` |
| `--nested_cv` | Enable nested cross-validation | `False` | Add flag to enable |
| `--n_trials` | Number of Optuna hyperparameter trials | `100` | `20` (faster), `100` (thorough) |
| `--epochs` | Maximum training epochs per trial | `100` | `50`, `100`, `200` |
| `--n_folds` | Inner CV folds (for hyperparameter tuning) | `5` | `3`, `5` |
| `--n_outer_folds` | Outer CV folds (for unbiased evaluation) | `5` | `3`, `5` |
| `--patience` | Early stopping patience (epochs) | `20` | `10`, `20`, `50` |
| `--random_state` | Random seed for reproducibility | `42` | Any integer |
| `--output_dir` | Directory for outputs (logs, models, plots) | `./outputs` | `./results` |
| `--workers` | Number of data loading workers | `4` | `2`, `4`, `8` |
| `--log_level` | Logging verbosity | `INFO` | `DEBUG`, `INFO`, `WARNING` |

#### What Happens During Training

1. **Environment Check** (`reproducibility.py`)
   - Validates CUDA availability
   - Sets random seeds for reproducibility
   
2. **Dataset Validation** (`main.py`)
   - Checks `data.yaml` exists
   - Verifies image-label pairs
   
3. **Nested Cross-Validation** (`optuna_nested_cv.py`)
   - **Outer loop**: Splits data into training/test folds
   - **Inner loop**: Optuna hyperparameter optimization using CV
   - **Training**: Best hyperparameters used to train final model
   - **Evaluation**: Model evaluated on held-out test fold
   
4. **Hardware Monitoring** (`gpu_monitoring.py`)
   - Tracks GPU utilization, memory usage
   - Logs CPU and system memory
   
5. **Results Generation** (`visualization.py`)
   - Creates performance plots
   - Generates summary reports
   - Saves metrics to JSON/CSV

### tmux Session Management

```bash
# Detach from session (training continues in background)
# Press: Ctrl + b, then d

# List all sessions
tmux ls

# Re-attach to a session
tmux attach -t yolo_v5

# Kill a session
tmux kill-session -t yolo_v5

# Create multiple sessions for different GPUs
tmux new -s train_gpu01  # Uses GPU 0,1
tmux new -s train_gpu23  # Uses GPU 2,3
tmux new -s train_gpu45  # Uses GPU 4,5
tmux new -s train_gpu67  # Uses GPU 6,7
```

### Example: Running Multiple Models in Parallel

```bash
# Session 1: YOLOv5 on GPUs 0,1
tmux new -s yolov5_train
cd cropandweed-dataset/cnw/utilities/yolo_training_pipeline
conda activate cnw_env
python main.py --model yolov5su --data_path ./pipeline_dataset_CropOrWeed2 --gpu_ids "0,1" --nested_cv --n_trials 20 --epochs 50 --n_folds 5 --n_outer_folds 5
# Detach: Ctrl+b, d

# Session 2: YOLOv8 on GPUs 2,3
tmux new -s yolov8_train
cd cropandweed-dataset/cnw/utilities/yolo_training_pipeline
conda activate cnw_env
python main.py --model yolov8s --data_path ./pipeline_dataset_CropOrWeed2 --gpu_ids "2,3" --nested_cv --n_trials 20 --epochs 50 --n_folds 5 --n_outer_folds 5
# Detach: Ctrl+b, d

# Session 3: YOLO11 on GPUs 4,5
tmux new -s yolo11_train
cd cropandweed-dataset/cnw/utilities/yolo_training_pipeline
conda activate cnw_env
python main.py --model yolo11s --data_path ./pipeline_dataset_CropOrWeed2 --gpu_ids "4,5" --nested_cv --n_trials 20 --epochs 50 --n_folds 5 --n_outer_folds 5
# Detach: Ctrl+b, d
```

---

## Monitoring and Results

### Real-Time Monitoring

**View live training logs:**

```bash
# Attach to the running session
tmux attach -t yolo_v5

# Or tail the log file
tail -f outputs/logs/yolov5su_*.log
```

**Monitor GPU usage:**

```bash
# Real-time GPU monitoring
watch -n 1 nvidia-smi
```

### Output Directory Structure

After training completes, check the `outputs/` directory:

```
outputs/
├── logs/
│   └── yolov5su_20240115_143022.log          # Training logs
│
├── yolov5su_CropOrWeed2_20240115_143022_results.json  # Metrics (JSON)
├── yolov5su_CropOrWeed2_20240115_143022_hardware.json # Hardware stats
│
└── visualizations/
    ├── yolov5su_CropOrWeed2_*_map_metrics.png        # Performance plots
    ├── yolov5su_CropOrWeed2_*_hyperparameters.png    # Hyperparameter analysis
    ├── yolov5su_CropOrWeed2_*_hardware_usage.png     # GPU/CPU usage
    └── yolov5su_CropOrWeed2_*_summary_report.png     # Complete summary
```

### Understanding Results

**Key metrics** (from `metrics.py`):

- **mAP50-95**: Mean Average Precision (IoU 0.5:0.95) - Primary metric
- **mAP50**: Mean Average Precision at IoU 0.5
- **Mean Test Score**: Average performance across CV folds
- **Std Test Score**: Performance variance across folds

**Visualizations** (from `visualization.py`):

1. **Performance Metrics**: Cross-validation scores per fold
2. **Hyperparameter Analysis**: Optuna optimization progress
3. **Hardware Usage**: GPU/CPU utilization over time
4. **Summary Report**: Overall training statistics

---

## Troubleshooting

### Common Issues

**Issue: CUDA Out of Memory**
```bash
# Solution: Bottleneck on memory
delete the runs directory, unused models, and outputs, and clear temp files and caches.
```

**Issue: Dataset Not Found**
```bash
# Verify paths are absolute or relative to current directory
ls -la /home/anad0001/cropandweed-dataset/cnw/utilities/pipeline_dataset_CropOrWeed2
```

**Issue: tmux Session Lost**
```bash
# List all sessions
tmux ls

# Re-attach
tmux attach -t yolo_v5
```

**Issue: Environment Not Activated**
```bash
# Always activate before training
conda activate cnw_env
which python  # Should show conda env path
```

---

## Best Practices

1. **Always use tmux** for long-running training sessions
2. **Monitor GPU usage** to avoid over-allocation
3. **Start with small `n_trials`** (e.g., 20) for quick testing
4. **Use absolute paths** for dataset directories
5. **Check logs regularly** for errors or warnings
6. **Keep `random_state` consistent** for reproducibility
7. **Save important results** to external storage

---

## Citation

If you use this dataset/pipeline, please cite:

```bibtex
@InProceedings{Steininger_2023_WACV,
    author    = {Steininger, Daniel and Trondl, Andreas and Croonen, Gerardus and Simon, Julia and Widhalm, Verena},
    title     = {The CropAndWeed Dataset: A Multi-Modal Learning Approach for Efficient Crop and Weed Manipulation},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2023},
    pages     = {3729-3738}
}
```

---

## Additional Resources

- **Main Repository**: [CropAndWeed Dataset](https://github.com/Bibliophiles/cropandweed-dataset)
- **Paper**: [WACV 2023](https://openaccess.thecvf.com/content/WACV2023/papers/Steininger_The_CropAndWeed_Dataset_A_Multi-Modal_Learning_Approach_for_Efficient_Crop_WACV_2023_paper.pdf)
- **Supplementary**: [Additional Materials](https://openaccess.thecvf.com/content/WACV2023/supplemental/Steininger_The_CropAndWeed_Dataset_WACV_2023_supplemental.pdf)
- **Ultralytics YOLO**: [Documentation](https://docs.ultralytics.com)
- **Optuna**: [Hyperparameter Optimization](https://optuna.org)

---

## Support

For issues or questions:
- Open an issue on [GitHub](https://github.com/Bibliophiles/cropandweed-dataset/issues)
- Check existing documentation
- Review code comments for implementation details

---

**Happy Training! 🌱🚀**
