import torch
from torch.utils.data import DataLoader
from transformers import SamModel
from torch.optim import Adam
import monai
from tqdm import tqdm
from statistics import mean
import os
from SmartAITool.core import cprint

from torch.utils.data import Dataset
import numpy as np

def get_bounding_box(ground_truth_mask):
    """
    Get the bounding box coordinates from a binary mask.
    Returns: [x_min, y_min, x_max, y_max]
    """
    # Find rows and columns with non-zero values
    y_indices, x_indices = np.where(ground_truth_mask > 0)
    
    # Get bounding box coordinates
    if len(y_indices) > 0 and len(x_indices) > 0:
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        return [x_min, y_min, x_max, y_max]
    else:
        # Return a default small box in the middle if mask is empty
        h, w = ground_truth_mask.shape
        return [w//2-10, h//2-10, w//2+10, h//2+10]

class SAMDataset(Dataset):
    """
    This class is used to create a dataset that serves input images and masks.
    It takes a dataset and a processor as input and overrides the __len__ and __getitem__ methods of the Dataset class.
    """
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"]
        ground_truth_mask = np.array(item["label"])

        # get bounding box prompt
        prompt = get_bounding_box(ground_truth_mask)

        # prepare image and prompt for the model
        inputs = self.processor(image, input_boxes=[[prompt]], return_tensors="pt")

        # remove batch dimension which the processor adds by default
        inputs = {k:v.squeeze(0) for k,v in inputs.items()}

        # add ground truth segmentation
        inputs["ground_truth_mask"] = ground_truth_mask

        return inputs


def initialize_model(device="cuda"):
    """Initialize and prepare the SAM model for fine-tuning."""
    model = SamModel.from_pretrained("facebook/sam-vit-base")
    
    # Make sure we only compute gradients for mask decoder
    for name, param in model.named_parameters():
        if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
            param.requires_grad_(False)
    
    model.to(device)
    return model

def create_dataloader(train_dataset, batch_size=2):
    """Create a DataLoader for training."""
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    return train_dataloader

def inspect_batch(dataloader):
    """Inspect shapes of items in a batch."""
    batch = next(iter(dataloader))
    for k, v in batch.items():
        print(k, v.shape)
    print("Ground truth mask shape:", batch["ground_truth_mask"].shape)

def train_model(train_dataloader, num_epochs=1, lr=1e-5, save_path=None):
    """Train the SAM model on the provided dataloader."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = initialize_model(device)
    
    # Initialize the optimizer and the loss function
    optimizer = Adam(model.mask_decoder.parameters(), lr=lr, weight_decay=0)
    # Using DiceCELoss, but can also try DiceFocalLoss or FocalLoss
    seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    
    model.train()
    for epoch in range(num_epochs):
        epoch_losses = []
        for batch in tqdm(train_dataloader):
            # forward pass
            outputs = model(
                pixel_values=batch["pixel_values"].to(device),
                input_boxes=batch["input_boxes"].to(device),
                multimask_output=False
            )
            
            # compute loss
            predicted_masks = outputs.pred_masks.squeeze(1)
            ground_truth_masks = batch["ground_truth_mask"].float().to(device)
            loss = seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1))
            
            # backward pass (compute gradients of parameters w.r.t. loss)
            optimizer.zero_grad()
            loss.backward()
            
            # optimize
            optimizer.step()
            epoch_losses.append(loss.item())
            
        print(f'EPOCH: {epoch}')
        print(f'Mean loss: {mean(epoch_losses)}')
    
    # Save the model's state dictionary to a file if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        cprint(f"Model saved to {save_path}", "green")
    
    return model
