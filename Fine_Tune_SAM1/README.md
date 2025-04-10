# Fine-Tune SAM1

A toolkit for fine-tuning the Segment Anything Model (SAM) on custom datasets and performing inference with the fine-tuned model.

## Project Overview

This project provides tools and scripts for fine-tuning Meta AI's Segment Anything Model (SAM) on custom datasets. SAM is a state-of-the-art foundation model for image segmentation tasks that can be adapted to specific domains through fine-tuning.

## Project Structure

```
Fine_Tune_SAM1/
├── notebook/                  # Jupyter notebooks
│   └── finetune.ipynb         # Main fine-tuning notebook
├── data/                      # Place your training data here
│   ├── images/                # Training images
│   └── masks/                 # Binary masks for training
├── models/                    # Saved models directory
│   └── fine_tuned_sam/        # Fine-tuned SAM checkpoints
├── utils/                     # Utility scripts
└── README.md                  # This file
```

## Features

- Fine-tune SAM on custom image-mask datasets
- Support for different SAM model sizes (ViT-B, ViT-L, ViT-H)
- Image patching support for large images
- Various data augmentation techniques
- Inference tools for segmenting new images
- Evaluation metrics for model performance

## Requirements

- Python 3.8+
- PyTorch 1.13+
- Transformers
- Segment Anything (SAM)
- MONAI (for loss functions)
- Datasets (for data preparation)
- Patchify (for handling large images)

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/fine-tune-sam.git
   cd fine-tune-sam/Fine_Tune_SAM1
   ```

2. Install dependencies:
   ```bash
   pip install git+https://github.com/facebookresearch/segment-anything.git
   pip install git+https://github.com/huggingface/transformers.git
   pip install datasets monai patchify
   ```

## Usage

### Fine-tuning

1. Prepare your dataset:
   - Place your images in `data/images/`
   - Place corresponding binary masks in `data/masks/`

2. Use the notebook:
   - Open `notebook/finetune.ipynb`
   - Follow the step-by-step instructions to configure and run fine-tuning

3. Alternatively, use the Python script:
   ```bash
   python finetune.py --config config.yaml
   ```

### Inference

1. Load your fine-tuned model:
   ```python
   from transformers import SamModel, SamProcessor
   
   model = SamModel.from_pretrained("./models/fine_tuned_sam")
   processor = SamProcessor.from_pretrained("./models/fine_tuned_sam")
   ```

2. Perform inference on new images:
   ```python
   import torch
   from PIL import Image
   
   image = Image.open("path/to/your/image.jpg")
   inputs = processor(image, return_tensors="pt")
   
   with torch.no_grad():
       outputs = model(**inputs)
       
   # Process outputs to get segmentation masks
   masks = processor.post_process_masks(
       outputs.pred_masks.cpu(),
       inputs["original_sizes"].cpu(),
       inputs["reshaped_input_sizes"].cpu()
   )
   ```

## Training Parameters

You can customize the following parameters for fine-tuning:
- Learning rate
- Batch size
- Number of epochs
- Loss function
- Data augmentation strategies
- Model size

## Acknowledgements

This project has been adapted from [NielsRogge's Transformers Tutorials](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/SAM/Fine_tune_SAM_(segment_anything)_on_a_custom_dataset.ipynb) and modified to work with custom binary mask datasets.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
