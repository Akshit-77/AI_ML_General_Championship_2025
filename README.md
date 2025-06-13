# Bird Species Classification Project

A machine learning project for classifying 200 different bird species using deep learning techniques. This project includes data augmentation tools and a Swin Transformer-based classifier.

## HUGGING FACE INTERFACE LINK - https://huggingface.co/spaces/Akshit-77/Bird-Classifier

## Project Overview

This project tackles the challenge of bird species identification using computer vision. With a dataset containing images of 200 different bird species, we implement:

1. **Data Augmentation** - Increase dataset size and diversity to improve model performance
2. **Swin Transformer Classification** - Use state-of-the-art vision transformer for accurate bird identification

The project is designed to be straightforward and practical, focusing on getting good results without unnecessary complexity.

## Files Description

- `birds-augments.py` - Data augmentation script to expand your training dataset
- `swin-transformer.py` - Main training script using Swin Transformer architecture


## Dataset Structure

Your dataset should be organized like this:
```
dataset/
├── train/
│   ├── 001.Black_footed_Albatross/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── 002.Laysan_Albatross/
│   └── ...
└── test/
    ├── test_image1.jpg
    ├── test_image2.jpg
    └── ...
```

## Requirements

Install the required packages:

```bash
pip install torch torchvision tqdm pillow pandas timm numpy
```

Or if you have a `requirements.txt`:
```bash
pip install -r requirements.txt
```

### System Requirements
- Python 3.7+
- CUDA-capable GPU (recommended for training)
- At least 8GB RAM

## Quick Start

### Step 1: Data Augmentation (Optional but Recommended)

Run the augmentation script to increase your dataset size:

```bash
python birds-augments.py
```

**What it does:**
- Creates 4x more training images using various transformations
- Maintains balanced train/test ratios
- Generates zip files for easy sharing

**Key features:**
- Simple 4x augmentation with flips, rotations, and translations
- Smart balanced augmentation that maintains dataset distribution
- Automatic zip file creation

### Step 2: Train the Model

Run the main training script:

```bash
python swin-transformer.py
```

**What it does:**
- Loads a pretrained Swin Transformer model
- Gradually unfreezes layers for optimal training
- Automatically finds the best configuration
- Generates predictions for test images

## Customization

### Modifying Paths

Update these variables in the scripts to match your setup:

In `birds-augments.py`:
```python
input_dir = "path/to/your/train/folder"
output_dir = "path/to/output/folder"
```

In `swin-transformer.py`:
```python
train_folder = 'path/to/train/folder'
test_folder = 'path/to/test/folder'
```

### Adjusting Training Parameters

In `swin-transformer.py`, you can modify:
- `batch_size=64` - Reduce if you get out of memory errors
- `layer_options = [1, 2, 4, 8, 12]` - Try different unfreezing strategies
- `epochs=6` - Increase for longer training per stage

### Augmentation Options

In `birds-augments.py`:
- Change `multiply_by=2` to create different dataset sizes
- Modify transforms in the `transforms_to_use` list
- Adjust the number of augmentation versions

## Expected Results

- **Training time**: 2-4 hours on a good GPU
- **Accuracy**: Typically 85-95% depending on dataset quality
- **Memory usage**: 6-8GB GPU memory with default settings

## Output Files

After running the scripts, you'll get:

From augmentation:
- `birds_4x_augmented.zip` - Simple 4x augmented dataset
- `birds_balanced_augmented.zip` - Balanced augmented dataset

From training:
- `predictions.csv` - Detailed predictions with confidence scores
- `submission.csv` - Competition-ready submission file
- `best_model.pth` - Trained model weights

## Troubleshooting

### Common Issues

**Out of memory error:**
- Reduce batch_size from 64 to 32 or 16
- Close other applications using GPU memory

**Slow training:**
- Make sure you're using GPU (check "Using: cuda" message)
- Reduce number of workers if CPU is bottleneck

**File not found errors:**
- Double-check your folder paths
- Ensure images are in supported formats (.jpg, .jpeg, .png)

**Poor accuracy:**
- Try different layer unfreezing strategies
- Increase training epochs
- Check if your dataset is balanced


## Model Architecture

The project uses **Swin Transformer Base** (`swin_base_patch4_window7_224`):
- Pre-trained on ImageNet
- 224x224 input images
- Hierarchical feature learning
- Window-based attention mechanism

### Why Swin Transformer?

- **Better than CNNs** for image classification
- **Efficient** compared to standard vision transformers
- **Good transfer learning** from ImageNet
- **State-of-the-art** results on many vision tasks



## Need Help?

If you run into issues:
1. Check the troubleshooting section above
2. Verify your environment setup
3. Make sure your dataset structure matches the expected format
4. Try with a smaller subset of data first
