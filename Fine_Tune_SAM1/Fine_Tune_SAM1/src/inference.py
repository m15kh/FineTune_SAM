import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import tifffile
from patchify import patchify, unpatchify
from transformers import SamModel, SamConfig, SamProcessor
import random

def get_bounding_box(mask):
    """
    Get the bounding box coordinates from a binary mask.
    Returns [x_min, y_min, x_max, y_max]
    """
    # Find the bounding box of the mask
    y_indices, x_indices = np.where(mask > 0)
    
    if len(y_indices) > 0 and len(x_indices) > 0:
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        # Add a small margin
        x_min = max(0, x_min - 5)
        y_min = max(0, y_min - 5)
        x_max = min(mask.shape[1] - 1, x_max + 5)
        y_max = min(mask.shape[0] - 1, y_max + 5)
        return [x_min, y_min, x_max, y_max]
    else:
        # Return default box if mask is empty
        return [0, 0, mask.shape[1] - 1, mask.shape[0] - 1]

def load_model(model_checkpoint_path, device=None):
    """
    Load SAM model from a checkpoint file.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load the model configuration and processor
    model_config = SamConfig.from_pretrained("facebook/sam-vit-base")
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    
    # Create an instance of the model architecture with the loaded configuration
    model = SamModel(config=model_config)
    
    # Update the model by loading the weights from saved file
    model.load_state_dict(torch.load(model_checkpoint_path, map_location=device))
    
    # Set the device
    model = model.to(device)
    model.eval()
    
    return model, processor, device

def predict_single_image(model, processor, image, prompt=None, input_points=None, device=None):
    """
    Make predictions on a single image using either bounding box or point prompts.
    
    Args:
        model: The loaded SAM model
        processor: SAM processor
        image: Input image (PIL Image or numpy array)
        prompt: Bounding box in format [x_min, y_min, x_max, y_max]
        input_points: Tensor of shape (1, 1, N, 2) with N points
        device: Device to run inference on
    
    Returns:
        pred_mask: Binary prediction mask
        pred_prob: Probability map
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
    # Prepare inputs based on available prompts
    if prompt is not None:
        inputs = processor(image, input_boxes=[[prompt]], return_tensors="pt")
    elif input_points is not None:
        inputs = processor(image, input_points=input_points, return_tensors="pt")
    else:
        inputs = processor(image, return_tensors="pt")
    
    # Move inputs to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Inference
    with torch.no_grad():
        outputs = model(**inputs, multimask_output=False)
    
    # Process output
    pred_prob = torch.sigmoid(outputs.pred_masks.squeeze(1))
    pred_prob = pred_prob.cpu().numpy().squeeze()
    pred_mask = (pred_prob > 0.5).astype(np.uint8)
    
    return pred_mask, pred_prob

def create_point_grid(patch_size, grid_size=10):
    """
    Create a grid of points to use as prompts.
    Returns a tensor of shape (1, 1, grid_size*grid_size, 2)
    """
    # Generate the grid points
    x = np.linspace(0, patch_size-1, grid_size)
    y = np.linspace(0, patch_size-1, grid_size)
    xv, yv = np.meshgrid(x, y)
    
    # Convert to list format
    points = [[[int(x), int(y)] for x, y in zip(x_row, y_row)] for x_row, y_row in zip(xv, yv)]
    input_points = torch.tensor(points).view(1, 1, grid_size*grid_size, 2)
    
    return input_points

def process_large_image(model, processor, large_image, patch_size=256, step=256, device=None):
    """
    Process a large image by dividing it into patches, making predictions,
    and then reassembling the patches.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Extract patches
    patches = patchify(large_image, (patch_size, patch_size), step=step)
    
    # Create point grid for prompts
    input_points = create_point_grid(patch_size)
    
    # Initialize arrays for predictions
    h, w = patches.shape[0], patches.shape[1]
    patch_predictions = np.zeros((h, w, patch_size, patch_size), dtype=np.float32)
    
    # Process each patch
    for i in range(h):
        for j in range(w):
            patch = patches[i, j]
            patch_image = Image.fromarray(patch)
            
            # Make prediction
            _, patch_prob = predict_single_image(
                model, processor, patch_image, 
                input_points=input_points, device=device
            )
            
            # Store prediction
            patch_predictions[i, j] = patch_prob
    
    # Unpatchify to get full prediction
    full_probability = unpatchify(patch_predictions, large_image.shape)
    full_prediction = (full_probability > 0.5).astype(np.uint8)
    
    return full_prediction, full_probability

def visualize_prediction(image, mask=None, prob=None, title=None, save_path='output.png'):
    """
    Visualize input image and model predictions and save to file.
    
    Args:
        image: Input image
        mask: Prediction mask
        prob: Probability map
        title: Plot title
        save_path: Path to save the visualization
    """
    n_plots = 1 + (mask is not None) + (prob is not None)
    fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 5))
    
    if n_plots == 1:
        axes = [axes]
    
    # Plot the image
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title("Image")
    
    plot_idx = 1
    
    # Plot probability map if available
    if prob is not None:
        axes[plot_idx].imshow(prob)
        axes[plot_idx].set_title("Probability Map")
        plot_idx += 1
        
    # Plot mask if available
    if mask is not None:
        axes[plot_idx].imshow(mask, cmap='gray')
        axes[plot_idx].set_title("Prediction")
    
    # Hide axis ticks and labels
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    
    if title:
        fig.suptitle(title)
    
    # Save the figure instead of showing it
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Visualization saved to {save_path}")

def inference_on_dataset(model, processor, dataset, idx=None, device=None, save_path='dataset_result.png'):
    """
    Run inference on a sample from a dataset and compare with ground truth.
    
    Args:
        model: The SAM model
        processor: The SAM processor
        dataset: Dataset containing images and labels
        idx: Index of sample to use (random if None)
        device: Device to run inference on
        save_path: Path to save the visualization
    
    Returns:
        test_image: The input image
        pred_mask: Predicted binary mask
        pred_prob: Prediction probability map
        ground_truth_mask: Ground truth segmentation mask
    """
    if idx is None:
        idx = random.randint(0, len(dataset)-1)
    
    # Load image
    test_image = dataset[idx]["image"]
    ground_truth_mask = np.array(dataset[idx]["label"])
    
    # Get box prompt based on ground truth segmentation map
    prompt = get_bounding_box(ground_truth_mask)
    
    # Run prediction
    pred_mask, pred_prob = predict_single_image(
        model, processor, test_image, prompt=prompt, device=device
    )
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot original image
    axes[0].imshow(np.array(test_image), cmap='gray')
    axes[0].set_title("Image")
    
    # Plot prediction mask
    axes[1].imshow(pred_mask, cmap='gray')
    axes[1].set_title("Prediction")
    
    # Plot probability map
    axes[2].imshow(pred_prob)
    axes[2].set_title("Probability Map")
    
    # Hide axis ticks and labels
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Save the figure instead of showing it
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Dataset inference result saved to {save_path}")
    
    return test_image, pred_mask, pred_prob, ground_truth_mask

def save_prediction_outputs(mask, prob=None, mask_path='mask.png', prob_path=None):
    """
    Save prediction outputs to files.
    
    Args:
        mask: Binary prediction mask
        prob: Probability map
        mask_path: Path to save the mask
        prob_path: Path to save the probability map (if provided)
    """
    # Save the mask as an image
    mask_img = Image.fromarray((mask * 255).astype(np.uint8))
    mask_img.save(mask_path)
    print(f"Mask saved to {mask_path}")
    
    # Save probability map if provided
    if prob is not None and prob_path is not None:
        # Scale to 0-255 for saving as an image
        prob_img = Image.fromarray((prob * 255).astype(np.uint8))
        prob_img.save(prob_path)
        print(f"Probability map saved to {prob_path}")

def main():
    """
    Main function to demonstrate usage of the SAM inference pipeline.
    """
    # Set the path to your model
    model_path = "/home/ubuntu/m15kh/own/fine-tune-sam/checkpoints/mito_model_checkpoint.pth"
    output_dir = "/home/ubuntu/m15kh/own/fine-tune-sam/Fine_Tune_SAM1/output"
    
    # Load the model
    model, processor, device = load_model(model_path)
    
    print(f"Model loaded successfully on {device}")
    print("Use the functions in this module to run inference on your images.")
    
    # Example usage with single image
    image = Image.open("path/to/image.png")
    mask, prob = predict_single_image(model, processor, image, device=device)
    visualize_prediction(np.array(image), mask, prob, save_path=f"{output_dir}/visualization.png")
    save_prediction_outputs(mask, prob, mask_path=f"{output_dir}/mask.png", prob_path=f"{output_dir}/probability.png")
    
    # Example usage with large image
    large_image = tifffile.imread("path/to/large_image.tif")
    full_mask, full_prob = process_large_image(model, processor, large_image, device=device)
    visualize_prediction(large_image, full_mask, full_prob, save_path=f"{output_dir}/large_image_result.png")
    save_prediction_outputs(full_mask, full_prob, mask_path=f"{output_dir}/large_mask.png", prob_path=f"{output_dir}/large_prob.png")

if __name__ == "__main__":
    main()
