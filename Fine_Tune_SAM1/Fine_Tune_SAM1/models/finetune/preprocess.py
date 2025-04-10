import numpy as np
from patchify import patchify

def patches(data, patch_size, step, normalize_masks=False):
    """
    Preprocess large data (images or masks) into smaller patches.

    Args:
        data (numpy.ndarray): Array of large images or masks.
        patch_size (int): Size of the patches (height and width).
        step (int): Step size for patch extraction.
        normalize_masks (bool): Whether to normalize mask values to 0 and 1.

    Returns:
        numpy.ndarray: Array of patches.
    """
    all_patches = []
    for img_idx in range(data.shape[0]):
        large_data = data[img_idx]
        patches = patchify(large_data, (patch_size, patch_size), step=step)

        for i in range(patches.shape[0]):
            for j in range(patches.shape[1]):
                single_patch = patches[i, j, :, :]
                if normalize_masks:
                    single_patch = (single_patch / 255.).astype(np.uint8)
                all_patches.append(single_patch)

    return np.array(all_patches)

def filter_empty_masks(images, masks):
    """
    Filter out image-mask pairs where the mask is empty (all zeros).
    
    Args:
        images (numpy.ndarray): Array of images.
        masks (numpy.ndarray): Array of masks corresponding to the images.
        
    Returns:
        tuple: Filtered images and masks as numpy arrays.
    """
    # Create a list to store the indices of non-empty masks
    valid_indices = [i for i, mask in enumerate(masks) if mask.max() != 0]
    
    # Filter the image and mask arrays to keep only the non-empty pairs
    filtered_images = images[valid_indices]
    filtered_masks = masks[valid_indices]
    
    return filtered_images, filtered_masks


import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

def show_and_save_comparison(image, mask, output_path, filename='showdata.png'):
    """
    Save a side-by-side comparison of an image and its mask.
    
    Args:
        image: PIL Image or numpy array - The input image
        mask: PIL Image or numpy array - The mask image
        output_path: str - The directory path to save the output image
        filename: str - The filename for the saved image
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Convert PIL images to numpy arrays if necessary
    if isinstance(image, Image.Image):
        image = np.array(image)
    if isinstance(mask, Image.Image):
        mask = np.array(mask)
    
    # Create a figure with two subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # Plot the image on the left
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title("Image")
    
    # Plot the mask on the right
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title("Mask")
    
    # Hide axis ticks and labels
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    
    # Save the figure
    plt.tight_layout()
    save_path = os.path.join(output_path, filename)
    plt.savefig(save_path)
    plt.close()
    
    return save_path

