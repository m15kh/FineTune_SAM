import numpy as np
import tifffile
import sys
import os
from SmartAITool.core import cprint
from datasets import Dataset
from PIL import Image
from transformers import SamProcessor

#local
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(project_path)
from models.finetune.preprocess import patches, filter_empty_masks, show_and_save_comparison
from models.finetune.trainer import create_dataloader, inspect_batch, train_model, SAMDataset

def main():
    # load data and masks
    large_images = tifffile.imread("/home/ubuntu/m15kh/own/fine-tune-sam/data/training.tif")
    large_masks = tifffile.imread("/home/ubuntu/m15kh/own/fine-tune-sam/data/training_groundtruth.tif")

    # patchify large images and masks
    patch_size = 256
    step = 256
    patch_large_img = patches(large_images, patch_size, step)
    patch_large_mask = patches(large_masks, patch_size, step, normalize_masks=True)

    # Filter out patches with empty masks
    filtered_img_patches, filtered_mask_patches = filter_empty_masks(patch_large_img, patch_large_mask)

    # Convert the NumPy arrays to Pillow images and store them in a dictionary
    dataset_dict = {
        "image": [Image.fromarray(img) for img in filtered_img_patches],
        "label": [Image.fromarray(mask) for mask in filtered_mask_patches],
    }

    # Create the dataset using the datasets.Dataset class
    dataset = Dataset.from_dict(dataset_dict)
    
    cprint(f"Dataset created with {dataset} samples", "green")
    
    # Initialize the SAM processor
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    
    # Create SAM dataset with the processor
    train_dataset = SAMDataset(dataset, processor)
    cprint(f"SAM dataset created with {len(train_dataset)} samples", "green")
    
    # Create a DataLoader instance for the training dataset
    train_dataloader = create_dataloader(train_dataset, batch_size=2)
    
    # Inspect the batch
    inspect_batch(train_dataloader)
    
    # Train the model
    model_save_path = "/home/ubuntu/m15kh/own/fine-tune-sam/checkpoints/mito_model_checkpoint.pth"
    model = train_model(
        train_dataloader, 
        num_epochs=1, 
        lr=1e-5, 
        save_path=model_save_path
    )

if __name__ == "__main__":
    main()
    cprint("code finished successfully", 'magenta')
