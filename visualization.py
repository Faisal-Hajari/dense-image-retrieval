import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2
import os
from PIL import Image, ImageDraw, ImageFont
import torch
from typing import Tuple, List
import math

def visualize_attention_retrieval(retrieval_model, image: Image.Image, save_dir: str, image_name: str):
    """
    Visualize the complete attention-based retrieval process and save all intermediate steps.
    
    Args:
        retrieval_model: InvAttenRetrival instance
        image: PIL Image to process
        save_dir: Directory to save visualizations
        image_name: Base name for saved files
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Save original image
    orig_W, orig_H = image.size
    image.save(os.path.join(save_dir, f"{image_name}_1_original.jpg"))
    
    # Preprocess to 384×384
    x = retrieval_model.preprocess(image).unsqueeze(0).to(retrieval_model.device)
    
    # Get attention map
    with torch.no_grad():
        tokens, _ = retrieval_model.clip.visual.trunk.forward_intermediates(x)
        full_feature, attn = retrieval_model.wrap_attn_pool(retrieval_model.clip.visual.trunk.attn_pool, tokens)
        attn_np = attn.detach().cpu().numpy()
    
    # Build mask at 384×384
    mask = retrieval_model.create_mask(attn_np, 384, 384, retrieval_model.threshold)
    
    # Create and save heatmap visualization
    create_heatmap_overlay(image, attn_np, save_dir, f"{image_name}_2_heatmap")
    
    # Create and save binary mask overlay
    create_binary_mask_overlay(image, mask, save_dir, f"{image_name}_2b_binary_mask")
    
    # Sliding window in 384×384 coords
    boxes384 = retrieval_model.sliding_windows(mask, retrieval_model.kernel_size, retrieval_model.threshold, retrieval_model.stride)
    
    # Rescale boxes back to original image size
    scale_x = orig_W / 384
    scale_y = orig_H / 384
    
    boxes = []
    for x1, y1, x2, y2 in boxes384:
        boxes.append((
            int(x1 * scale_x),
            int(y1 * scale_y),
            int(x2 * scale_x),
            int(y2 * scale_y),
        ))
    
    boxes = np.array(boxes, dtype=int)
    
    # Save boxes before KMeans
    draw_boxes_on_image_clean(image, boxes, save_dir, f"{image_name}_3_boxes_before_kmeans")
    
    # Apply KMeans clustering if needed
    if len(boxes) == 0:
        # Fallback: use center crop
        center_x, center_y = orig_W // 2, orig_H // 2
        crop_size = min(orig_W, orig_H) // 3
        merged_boxes = np.array([[
            max(0, center_x - crop_size),
            max(0, center_y - crop_size), 
            min(orig_W, center_x + crop_size),
            min(orig_H, center_y + crop_size)
        ]])
    elif retrieval_model.n_crops < len(boxes):
        from sklearn.cluster import KMeans
        k_means = KMeans(n_clusters=retrieval_model.n_crops, random_state=42)
        k_means.fit(boxes)
        labels = k_means.labels_
        merged_boxes = []
        for i in range(retrieval_model.n_crops):
            cluster_boxes = boxes[labels == i]
            if len(cluster_boxes) == 0:
                continue
            x_min = np.min(cluster_boxes[:, 0])
            y_min = np.min(cluster_boxes[:, 1])
            x_max = np.max(cluster_boxes[:, 2])
            y_max = np.max(cluster_boxes[:, 3])
            merged_boxes.append([x_min, y_min, x_max, y_max])
        merged_boxes = np.array(merged_boxes)
    else:
        merged_boxes = boxes
    
    # Save boxes after KMeans
    draw_boxes_on_image_clean(image, merged_boxes, save_dir, f"{image_name}_4_boxes_after_kmeans")
    
    # Extract and save crops
    crops = []
    for i, box in enumerate(merged_boxes):
        xmin, ymin, xmax, ymax = box
        if xmax <= xmin or ymax <= ymin:
            continue
        cropped_image = image.crop((xmin, ymin, xmax, ymax))
        crops.append(cropped_image)
        # Save individual crop
        cropped_image.save(os.path.join(save_dir, f"{image_name}_5_crop_{i:02d}.jpg"))
    
    # Create crops grid visualization
    create_crops_grid_clean(image, crops, merged_boxes, save_dir, f"{image_name}_5_all_crops")
    
    print(f"Saved visualizations to {save_dir}: original, heatmap, binary mask, {len(boxes)} boxes before clustering, {len(merged_boxes)} boxes after clustering, and {len(crops)} crops")
    return crops

def create_heatmap_overlay(image: Image.Image, attn_np: np.ndarray, save_dir: str, filename: str):
    """Create and save attention heatmap overlaid directly on the original image."""
    B, heads, Q, S = attn_np.shape
    g = int(math.sqrt(S))
    
    # Average attention across all heads
    avg_attn = np.mean(attn_np[0, :, 0], axis=0).reshape(g, g)
    
    # Normalize to 0-1
    avg_attn = (avg_attn - avg_attn.min()) / (avg_attn.max() - avg_attn.min() + 1e-8)
    
    # Resize to original image size
    orig_W, orig_H = image.size
    heatmap = cv2.resize(avg_attn, (orig_W, orig_H), interpolation=cv2.INTER_LINEAR)
    
    # Create figure with just the overlay
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(image)
    ax.imshow(heatmap, cmap='hot', alpha=0.5)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{filename}.jpg"), dpi=150, bbox_inches='tight', pad_inches=0)
    plt.close()

def create_binary_mask_overlay(image: Image.Image, mask: np.ndarray, save_dir: str, filename: str):
    """Create and save binary mask overlaid on the original image."""
    # Resize mask to original image size
    orig_W, orig_H = image.size
    mask_resized = cv2.resize(mask.astype(np.uint8), (orig_W, orig_H), interpolation=cv2.INTER_NEAREST)
    
    # Create figure with just the overlay
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(image)
    
    # Create a colored mask - red for regions that will be processed (mask == 1)
    # The mask indicates areas where sliding windows will be applied
    colored_mask = np.zeros((orig_H, orig_W, 4))  # RGBA
    colored_mask[:, :, 0] = mask_resized  # Red channel
    colored_mask[:, :, 3] = mask_resized * 0.4  # Alpha channel (40% opacity)
    
    ax.imshow(colored_mask, alpha=1.0)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{filename}.jpg"), dpi=150, bbox_inches='tight', pad_inches=0)
    plt.close()

def draw_boxes_on_image_clean(image: Image.Image, boxes: np.ndarray, save_dir: str, filename: str):
    """Draw bounding boxes on image without any labels or numbers."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(image)
    
    # Draw boxes without numbers
    colors = plt.cm.Set3(np.linspace(0, 1, len(boxes)))
    for i, (box, color) in enumerate(zip(boxes, colors)):
        if len(box) == 4:
            xmin, ymin, xmax, ymax = box
            width = xmax - xmin
            height = ymax - ymin
            rect = patches.Rectangle((xmin, ymin), width, height, 
                                   linewidth=2, edgecolor=color, facecolor='none', alpha=0.8)
            ax.add_patch(rect)
    
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{filename}.jpg"), dpi=150, bbox_inches='tight', pad_inches=0)
    plt.close()

def create_crops_grid_clean(original_image: Image.Image, crops: List[Image.Image], boxes: np.ndarray, 
                           save_dir: str, filename: str):
    """Create a grid showing the original image and all crops without titles."""
    n_crops = len(crops)
    
    # Calculate grid size - ensure at least 1 column for original image
    grid_cols = max(1, min(4, n_crops + 1))  # +1 for original image
    total_items = n_crops + 1  # crops + original
    grid_rows = (total_items + grid_cols - 1) // grid_cols
    
    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(4*grid_cols, 4*grid_rows))
    
    # Handle different subplot configurations
    if grid_rows == 1 and grid_cols == 1:
        axes = np.array([[axes]])  # Make it 2D
    elif grid_rows == 1:
        axes = axes.reshape(1, -1)  # Make it 2D
    elif grid_cols == 1:
        axes = axes.reshape(-1, 1)  # Make it 2D
    # else: already 2D when grid_rows > 1 and grid_cols > 1
    
    # Show original image in first position
    axes[0, 0].imshow(original_image)
    axes[0, 0].axis('off')
    
    # Show crops
    item_idx = 1  # Start after original image
    for i, crop in enumerate(crops):
        row = item_idx // grid_cols
        col = item_idx % grid_cols
        
        if row < grid_rows and col < grid_cols:
            axes[row, col].imshow(crop)
            axes[row, col].axis('off')
        item_idx += 1
    
    # Turn off remaining empty axes
    for item_idx in range(len(crops) + 1, grid_rows * grid_cols):
        row = item_idx // grid_cols
        col = item_idx % grid_cols
        if row < grid_rows and col < grid_cols:
            axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{filename}.jpg"), dpi=150, bbox_inches='tight', pad_inches=0)
    plt.close()

# Example usage function
def test_visualization():
    """Test the visualization with a sample image."""
    from models.retrivals import InvAttenRetrival
    from torchvision.datasets import CocoCaptions
    
    # Initialize model
    retrieval_model = InvAttenRetrival(
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        n_crops=5,
        include_full_image=True, 
    )
    
    # Load a sample image (replace with your path)
    coco_root = "data/val2017"
    coco_ann_file = "annotations/captions_val2017.json"
    dataset = CocoCaptions(root=coco_root, annFile=coco_ann_file)
    
    # Get first image
    image, captions = dataset[4444]
    # image = Image.open("image.png").convert("RGB")  # Replace with actual image path if needed
    # Create visualization
    save_dir = "visualizations"
    visualize_attention_retrieval(retrieval_model, image, save_dir, "sample_image_001")

if __name__ == "__main__":
    test_visualization()