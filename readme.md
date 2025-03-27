# Fine-Tune SAM

This repository provides tools and scripts for fine-tuning the Segment Anything Model (SAM). SAM is a state-of-the-art model for image segmentation tasks.

## Features

- Fine-tune SAM on custom datasets.
- Easily configurable pipeline for training and evaluation.
- Support for various datasets and augmentation techniques.

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
python pipeline.py --train --config config.yaml
```

### Evaluation

Evaluate the fine-tuned model:
```bash
python pipeline.py --eval --config config.yaml
```

### Configuration

Modify the `config.yaml` file to customize training parameters such as learning rate, batch size, and number of epochs.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
