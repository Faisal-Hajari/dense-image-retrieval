from models.retrivals import OpenCLIPRetrival
from torchvision.datasets import CocoCaptions
import pandas as pd 
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


#Dataset
coco_root = "data/val2017"
coco_ann_file = "annotations/captions_val2017.json"
dataset = CocoCaptions(
    root=coco_root,
    annFile=coco_ann_file,
)


def calculate_recall_at_k_from_table(similarity_df, k_vals):
    """Calculate text-to-image recall@k from similarity table"""
    results = {}
    
    # Group by text_idx (each individual text query)
    grouped = similarity_df.groupby('text_idx')
    
    for k in k_vals:
        correct_retrievals = 0
        total_texts = len(grouped)
        
        for text_idx, group in grouped:
            # Sort by similarity score (descending) and take top-k
            top_k = group.nlargest(k, 'cosine_similarity')
            
            # Check if any of the top-k results are correct
            if top_k['is_correct_pair'].any():
                correct_retrievals += 1
        
        recall_at_k = correct_retrievals / total_texts
        results[f'R@{k}'] = recall_at_k
        print(f"Text-to-Image R@{k}: {100*recall_at_k:.2f}%")
    
    return results

def calculate_image_to_text_recall_at_k(similarity_df, k_vals):
    """Calculate image-to-text recall@k from similarity table"""
    results = {}
    
    # Group by image_idx (each individual image)
    grouped = similarity_df.groupby('image_idx')
    
    for k in k_vals:
        correct_retrievals = 0
        total_images = len(grouped)
        
        for image_idx, group in grouped:
            # Sort by similarity score (descending) and take top-k texts
            top_k = group.nlargest(k, 'cosine_similarity')
            
            # Check if any of the top-k texts are correct for this image
            if top_k['is_correct_pair'].any():
                correct_retrievals += 1
        
        recall_at_k = correct_retrievals / total_images
        results[f'I2T_R@{k}'] = recall_at_k
        print(f"Image-to-Text R@{k}: {100*recall_at_k:.2f}%")
    
    return results

def recall(retrival:OpenCLIPRetrival)->None:
    retrival.index_coco_dataset(dataset)
    images = pd.DataFrame(retrival.images)
    texts = pd.DataFrame(retrival.texts)  
    image_embeddings = np.stack(images['embedding'].values)  # Shape: [1000, 1152]
    text_embeddings = np.stack(texts['embedding'].values)    # Shape: [5000, 1152]

    # Compute similarity matrix
    similarity_matrix = cosine_similarity(text_embeddings, image_embeddings)

    # Direct array creation - fastest approach
    n_texts, n_images = len(texts), len(images)

    # Create flat indices
    text_indices = np.repeat(np.arange(n_texts), n_images)
    image_indices = np.tile(np.arange(n_images), n_texts)

    # Get data indices
    text_data_indices = texts['data_index'].iloc[text_indices].values
    image_data_indices = images['data_index'].iloc[image_indices].values

    # Create DataFrame directly
    similarity_df = pd.DataFrame({
        'text_data_index': text_data_indices,
        'image_data_index': image_data_indices,
        'text_idx': text_indices,
        'image_idx': image_indices,
        'cosine_similarity': similarity_matrix.flatten(),
        'is_correct_pair': text_data_indices == image_data_indices
    })
    recall_results = calculate_recall_at_k_from_table(similarity_df, k_vals=[1, 5, 10])
    i2t_recall_results = calculate_image_to_text_recall_at_k(similarity_df, k_vals=[1, 5, 10])
    return similarity_df, recall_results, i2t_recall_results

if __name__ == "__main__":
    retrival = OpenCLIPRetrival(
        model_name="ViT-SO400M-14-SigLIP-384",
        dataset="webli",
        device="cuda",
        batch_size=128
    )
    similarity_df, recall_results, i2t_recall_results = recall(retrival)
    print(recall_results)
    print(i2t_recall_results)