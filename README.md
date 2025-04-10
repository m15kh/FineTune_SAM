# Fine-Tune SAM
This repository provides tools and scripts for fine-tuning the Segment Anything Model (SAM). SAM is a state-of-the-art model for image segmentation tasks.

## Models
This repository contains two fine-tuned versions of SAM:
- **SAM1**: Complete fine-tuned model optimized for mitochondria segmentation
- **SAM2**: Work in progress - enhanced version with additional capabilities (not yet complete)

## Features
- Fine-tune SAM on custom datasets.
- Easily configurable pipeline for training and evaluation.
- Support for various datasets and augmentation techniques.
- Multiple model variants with different specializations.

## Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/fine-tune-sam.git
   cd fine-tune-sam
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Prepare your dataset:
   - Ensure your dataset is in the required format.
   - Update the configuration file with the dataset path.

## Usage
### Training
Run the training pipeline:
```bash
python src/pipeline.py --train --config config.yaml --model sam1
```

To continue training SAM2 (work in progress):
```bash
python src/pipeline.py --train --config config.yaml --model sam2
```

### Inference
Run inference using the fine-tuned model:
```bash
python src/inference.py --model sam1 --input_path /path/to/image --output_dir /path/to/output
```

### Configuration
Modify the `config.yaml` file to customize training parameters such as learning rate, batch size, and number of epochs.

## Model Checkpoints
- SAM1: `/checkpoints/mito_model_checkpoint.pth`
- SAM2: Coming soon

## Examples
Example usage of the fine-tuned models can be found in the `examples` directory.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License
