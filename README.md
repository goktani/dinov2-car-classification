```markdown
# DINOv2 Car Image Classification

Fine-tuning DINOv2 on a car image dataset for classification using both local scripts and Google Colab.

> **Note:**  
> Current results do not meet the desired accuracy. Used batch size: 32, epochs: 20.

## Project Structure

```
dinov2/
│
├── dataset.py                 # Data preparation script
├── model.py                   # DINOv2 model setup
├── inference.py               # Inference on images
├── train.py                   # Training script
├── Dinov2_version1.ipynb      # Colab notebook
├── labels.json                # Class labels
├── dinov2info.jpeg            # DINOv2 comparison image
├── dinov2_finetuned_car5.pth  # Model weights
├── DINOv2- Görsel Temsil...pdf # DINOv2 info (Turkish)
│
├── test_images/               # Test images (sample)
├── test_imagesv0/             # Additional test images
└── split_dataset/
    └── test/
    └── train/
    └── val/
        └── 000, 001, 002...
```

## Installation

Clone the repository and install required packages:
```bash
git clone https://github.com/goktani/dinov2-car-classification.git
cd dinov2-car-classification/dinov2
pip install torch torchvision
```

## Usage

### Training
```bash
python train.py
```
_Edit batch size and epochs in `train.py` if needed._

### Inference
```bash
python inference.py
```

### Google Colab

Use the notebook [`Dinov2_version1.ipynb`](Dinov2_version1.ipynb) for GPU-supported training and testing on Colab.

## Limitations

- Model does not reach expected accuracy with current settings.
- Training was computationally intensive; Colab was used instead of local GPU.

## Resources

- `dinov2info.jpeg`: DINOv2 model version comparison
- `DINOv2- Görsel Temsil...pdf`: Turkish intro to DINOv2
