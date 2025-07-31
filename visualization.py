import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2
import os
from PIL import Image, ImageDraw, ImageFont
import torch
from typing import Tuple, List
import math
import textwrap
from sklearn.metrics.pairwise import cosine_similarity

def visualize_attention_retrieval(retrieval_model, image: Image.Image, save_dir: str, image_name: str):
    """
    Visualize the complete attention-based retrieval process and save selected outputs.
    
    Args:
        retrieval_model: InvAttenRetrival instance
        image: PIL Image to process
        save_dir: Directory to save visualizations
        image_name: Base name for saved files
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Save original image
    orig_W, orig_H = image.size
    image.save(os.path.join(save_dir, "original.jpg"))
    
    # Preprocess to 384×384
    x = retrieval_model.preprocess(image).unsqueeze(0).to(retrieval_model.device)
    
    # Get attention map
    with torch.no_grad():
        tokens, _ = retrieval_model.clip.visual.trunk.forward_intermediates(x)
        full_feature, attn = retrieval_model.wrap_attn_pool(retrieval_model.clip.visual.trunk.attn_pool, tokens)
        attn_np = attn.detach().cpu().numpy()
    
    # Build mask at 384×384
    mask = retrieval_model.create_mask(attn_np, 384, 384, retrieval_model.threshold)
    
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
    
    # Save boxes after KMeans (final merged boxes)
    draw_boxes_on_image_clean(image, merged_boxes, save_dir, "boxes_after_kmeans")
    
    # Extract crops
    crops = []
    for i, box in enumerate(merged_boxes):
        xmin, ymin, xmax, ymax = box
        if xmax <= xmin or ymax <= ymin:
            continue
        cropped_image = image.crop((xmin, ymin, xmax, ymax))
        crops.append(cropped_image)
    
    # Create and save crops grid visualization
    create_crops_grid_clean(image, crops, merged_boxes, save_dir, "all_crops")
    
    print(f"Saved visualizations to {save_dir}: original.jpg, boxes_after_kmeans.jpg, all_crops.jpg")
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
                                   linewidth=5, edgecolor=color, facecolor='none', alpha=0.8)
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

def create_similarity_bar_chart(retrieval_model, image: Image.Image, text_queries: List[str], 
                               save_dir: str, filename: str):
    """
    Create a bar chart comparing full image vs best crop similarity for multiple text queries.
    
    Args:
        retrieval_model: InvAttenRetrival instance (must have both full image and crops)
        image: PIL Image to process
        text_queries: List of text strings to compare against
        save_dir: Directory to save the chart
        filename: Name for the saved file (without extension)
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Get image embeddings (includes full image + crops)
    with torch.no_grad():
        image_features = retrieval_model.embed_image(image)  # Shape: [n_crops+1, embedding_dim] or [n_crops, embedding_dim]
        
        # Get text embeddings
        text_embeddings = retrieval_model.embed_texts(text_queries, batch_size=len(text_queries))
    
    # Convert to numpy for cosine similarity calculation
    image_features_np = image_features.numpy()
    text_embeddings_np = text_embeddings.numpy()
    
    # Calculate similarities
    similarity_matrix = cosine_similarity(text_embeddings_np, image_features_np)  # [n_texts, n_image_features]
    
    # Extract similarities
    if retrieval_model.include_full_image:
        # First feature is full image, rest are crops
        full_image_similarities = similarity_matrix[:, 0]  # [n_texts]
        crop_similarities = similarity_matrix[:, :]  # [n_texts, n_crops]
        # Get maximum similarity across all features (full image + crops)
        max_all_similarities = np.max(similarity_matrix, axis=1)  # [n_texts]
    else:
        # All features are crops, need to compute full image separately
        with torch.no_grad():
            # Get full image embedding
            x = retrieval_model.preprocess(image).unsqueeze(0).to(retrieval_model.device)
            tokens, _ = retrieval_model.clip.visual.trunk.forward_intermediates(x)
            full_feature, _ = retrieval_model.wrap_attn_pool(retrieval_model.clip.visual.trunk.attn_pool, tokens)
            full_image_features_np = full_feature.cpu().numpy()
        
        full_image_similarities = cosine_similarity(text_embeddings_np, full_image_features_np)[:, 0]
        max_all_similarities = np.max(similarity_matrix, axis=1)  # Max across crops only
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(max(8, len(text_queries) * 1.5), 7))
    
    x_pos = np.arange(len(text_queries))
    width = 0.35
    
    # Create bars
    bars1 = ax.bar(x_pos - width/2, full_image_similarities, width, 
                   label='SigLIP Similarity', alpha=0.8, color='skyblue')
    bars2 = ax.bar(x_pos + width/2, max_all_similarities, width,
                   label='Ours Similarity', alpha=0.8, color='lightcoral')
    
    # Customize chart
    ax.set_xlabel('Text Queries', fontsize=12)
    ax.set_ylabel('Cosine Similarity', fontsize=12)
    ax.set_title('Text-Image Similarity: SigLIP vs Ours', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    
    # Handle long text queries with multi-line wrapping
    wrapped_queries = []
    for query in text_queries:
        if len(query) > 20:  # Wrap if longer than 20 characters
            wrapped = textwrap.fill(query, width=20)
            wrapped_queries.append(wrapped)
        else:
            wrapped_queries.append(query)
    
    ax.set_xticklabels(wrapped_queries, fontsize=14, ha='center')
    
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)
    
    add_value_labels(bars1)
    add_value_labels(bars2)
    
    # Set y-axis limits to fixed range
    ax.set_ylim(-0.1, 0.15)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{filename}.jpg"), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Print results for debugging
    print(f"\nSimilarity Results for {len(text_queries)} queries:")
    print("-" * 50)
    for i, query in enumerate(text_queries):
        print(f"Query: '{query[:50]}{'...' if len(query) > 50 else ''}'")
        print(f"  SigLIP Similarity: {full_image_similarities[i]:.4f}")
        print(f"  Ours Similarity: {max_all_similarities[i]:.4f}")
        improvement = max_all_similarities[i] - full_image_similarities[i]
        print(f"  Improvement: {improvement:+.4f}")
        print()
    
    return {
        'full_image_similarities': full_image_similarities,
        'max_all_similarities': max_all_similarities,
        'text_queries': text_queries
    }

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
    
    # # Load a sample image (replace with your path)
    coco_root = "data/val2017"
    coco_ann_file = "annotations/captions_val2017.json"
    dataset = CocoCaptions(root=coco_root, annFile=coco_ann_file)

    
    #image_0
    image = Image.open(
        "tests/Screenshot 1447-01-13 at 7.20.54 PM.png"
        ).convert("RGB")  # Replace with actual image path if needed
    save_dir = "image_0"
 
    
    #image_1
    # image = Image.open(
    #     "tt/Images/Picture14-Enhanced.jpg"
    #     ).convert("RGB")  # Replace with actual image path if needed
    # save_dir = "image_1"
    
    
    #image_2
    # image, captions = dataset[4444]
    # save_dir = "image_2"
    
    visualize_attention_retrieval(retrieval_model, image, save_dir, "014")
    
    # Example text queries for similarity comparison
    text_queries = [
        "a man holding a bag over his shoulder", 
        "a group of women wearing a blue headscarf",
        "a parked truck",
    ]
    
    # Create similarity bar chart
    similarity_results = create_similarity_bar_chart(
        retrieval_model, 
        image, 
        text_queries, 
        save_dir, 
        "similarity_comparison"
    )
    
    print(f"Saved all visualizations to {save_dir}: original.jpg, boxes_after_kmeans.jpg, all_crops.jpg, similarity_comparison.jpg")

if __name__ == "__main__":
    test_visualization()