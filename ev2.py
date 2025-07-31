from typing import Any
from models.retrivals import ClipHuggingFaceRetrival, JinaRetrival, InvAttenRetrival, OpenCLIPRetrival, GridCroppingRetrival
import pandas as pd 
import numpy as np
import os
import json
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

#Global 
NEED_NORMALIZATION = True

class Kerpaty_cococ: 
    def __init__(self, images_path: str, single_caption: bool = True):
        self.images_path = images_path
        self.dataset = load_dataset("yerevann/coco-karpathy", split="test")
        self.root = self.images_path
        self.single_caption = single_caption
        
        
    def __getitem__(self, idx):
        image_path = os.path.join(
            self.images_path, 
            self.dataset[idx]['filename'].split("_")[-1]
        )
        image = Image.open(image_path)
        captions = self.dataset[idx]['sentences']
        
        # Handle the caption structure properly
        if self.single_caption: #this is used for Text-to-Image retrieval evaluation 
            captions = [captions[0]]
            
        return image, captions

    def __len__(self):
        return len(self.dataset)


def compute_similarity_max_crop(images_df, texts_df):
    """
    Compute similarity matrices for multi-crop retrieval (InvAttenRetrival).
    Takes maximum similarity across all crops for each image-text pair.
    
    Args:
        images_df: DataFrame with columns ['data_index', 'image', 'embedding']
                  Multiple rows per original image (crops)
        texts_df: DataFrame with columns ['data_index', 'text', 'embedding']
                 One row per text
    
    Returns:
        t2i_similarity: [n_texts, n_unique_images] matrix for text-to-image
    """
    print("Computing max-crop similarities...")
    
    # Get unique image data indices (original images)
    unique_image_indices = sorted(images_df['data_index'].unique())
    
    n_unique_images = len(unique_image_indices)
    n_texts = len(texts_df)
    
    print(f"Unique images: {n_unique_images}, Total texts: {n_texts}")
    
    # Get text embeddings and normalize
    text_embeddings = np.stack(texts_df['embedding'].values)
    if NEED_NORMALIZATION:
        text_embeddings = text_embeddings / np.linalg.norm(text_embeddings, axis=1, keepdims=True)
    
    # Initialize similarity matrices
    t2i_similarity = np.zeros((n_texts, n_unique_images))  # [n_texts, n_unique_images]
    
    # For each unique image, get all its crops and compute max similarity
    for img_idx, orig_img_id in enumerate(tqdm(unique_image_indices, desc="Computing max similarities")):
        # Get all crops for this original image
        crops_mask = images_df['data_index'] == orig_img_id
        crop_embeddings = np.stack(images_df[crops_mask]['embedding'].values)
        
        # Normalize crop embeddings
        crop_embeddings = crop_embeddings / np.linalg.norm(crop_embeddings, axis=1, keepdims=True)
        
        # Compute similarity between all texts and all crops of this image
        similarities = np.dot(text_embeddings, crop_embeddings.T)
        
        # Take maximum similarity across crops for each text
        max_similarities = np.max(similarities, axis=1)  # [n_texts]
        
        # Store in t2i matrix
        t2i_similarity[:, img_idx] = max_similarities
    print(f"Final similarity matrices: T2I {t2i_similarity.shape}")
    
    return t2i_similarity


def compute_retrieval_standard(a2b_sims, return_ranks=True):
    """
    Standard CLIP retrieval evaluation function
    Args:
        a2b_sims: Similarity matrix of shape [N, N] where N is number of samples
        return_ranks: Whether to return detailed rank information
    """
    npts = a2b_sims.shape[0]
    ranks = np.zeros(npts)
    top1 = np.zeros(npts)
    
    # Loop through each query
    for index in range(npts):
        # Get order of similarities (highest first)
        inds = np.argsort(a2b_sims[index])[::-1]
        # Find where the correct item (same index) is ranked
        where = np.where(inds == index)
        rank = where[0][0]
        ranks[index] = rank
        # Save the top1 result
        top1[index] = inds[0]
    
    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    r50 = 100.0 * len(np.where(ranks < 50)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    
    report_dict = {
        "r1": r1, "r5": r5, "r10": r10, "r50": r50, 
        "medr": medr, "meanr": meanr, "sum": r1 + r5 + r10
    }
    
    if return_ranks:
        return report_dict, (ranks, top1)
    else:
        return report_dict


def text2image_recall(retrival, save_path: str = None, single_caption: bool = True) -> dict[str, Any] | str:
    """
    Standard CLIP evaluation that matches published results.
    Handles both regular retrievals and InvAttenRetrival (with max similarity across crops).
    """
    # Use single caption for standard evaluation
    dataset = Kerpaty_cococ(images_path="coco_test_images", single_caption=single_caption)
    
    # Index dataset with retrieval model
    retrival.index_coco_dataset(dataset)
    images = pd.DataFrame(retrival.images)
    texts = pd.DataFrame(retrival.texts)  
    is_multi_crop = len(images) != len(texts)
    
    if is_multi_crop:
        print("Detected multi-crop retrieval")
        t2i_similarity = compute_similarity_max_crop(images, texts)
        print(f"Aggregated to: {t2i_similarity.shape[0]} images, {t2i_similarity.shape[0]} texts")
    else:
        print("Standard single-embedding retrieval")
        image_embeddings = np.stack(images['embedding'].values)
        text_embeddings = np.stack(texts['embedding'].values)
        
        # Normalize embeddings (important for CLIP)
        image_embeddings = image_embeddings / np.linalg.norm(image_embeddings, axis=1, keepdims=True)
        text_embeddings = text_embeddings / np.linalg.norm(text_embeddings, axis=1, keepdims=True)
        
        # Compute similarity matrices
        t2i_similarity = np.dot(image_embeddings, text_embeddings.T)  # Text to Image
    
    print("Computing Text-to-Image retrieval...")
    t2i_results, _ = compute_retrieval_standard(t2i_similarity)
    print(f"T2I Results: R@1: {t2i_results['r1']:.2f}%, R@5: {t2i_results['r5']:.2f}%, R@10: {t2i_results['r10']:.2f}%")
    
    # Save results if save_path is provided
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        
        images_for_saving = images.copy()
        images_for_saving = images_for_saving.drop('image', axis=1)
        
        # Save embeddings and metadata
        np.save(os.path.join(save_path, "t2i_similarity.npy"), t2i_similarity)
        
        images_for_saving.to_parquet(os.path.join(save_path, "images_metadata.parquet"), index=False)
        texts.to_parquet(os.path.join(save_path, "texts_metadata.parquet"), index=False)
        
        # Save results
        combined_results = {
            "text_to_image_recall": t2i_results,
            "dataset_stats": {
                "n_unique_images": t2i_similarity.shape[0],
                "n_texts": len(texts),
                "single_caption": single_caption,
                "is_multi_crop": is_multi_crop
            },
            "evaluation_type": "standard_clip_max_crop" if is_multi_crop else "standard_clip"
        }
        
        with open(os.path.join(save_path, "standard_clip_results.json"), 'w') as f:
            json.dump(combined_results, f, indent=2)
        
        print(f"Results saved to: {save_path}")
    
    return t2i_results


# def similarity_calculation(text_embeddings, image_embeddings):
#     """
#     Calculate similarity matrix using normalized dot products.
    
#     Args:
#         text_embeddings: numpy array of shape [n_texts, embed_dim]
#         image_embeddings: numpy array of shape [n_images, embed_dim] 
        
#     Returns:
#         similarity_matrix: numpy array of shape [n_texts, n_images]
#     """
#     # Normalize embeddings to unit vectors (L2 normalization)
#     text_embeddings_norm = text_embeddings / np.linalg.norm(text_embeddings, axis=1, keepdims=True)
#     image_embeddings_norm = image_embeddings / np.linalg.norm(image_embeddings, axis=1, keepdims=True)
    
#     # Compute similarity using dot product (equivalent to cosine similarity when normalized)
#     similarity_matrix = np.dot(text_embeddings_norm, image_embeddings_norm.T)
    
#     return similarity_matrix


# def recall_original(retrival: OpenCLIPRetrival, save_path: str = None) -> tuple:
#     """Your original evaluation approach for comparison"""
#     dataset = Kerpaty_cococ(images_path="coco_test_images", single_caption=False) 
#     retrival.index_coco_dataset(dataset)
#     images = pd.DataFrame(retrival.images)
#     texts = pd.DataFrame(retrival.texts)  
    
#     image_embeddings = np.stack(images['embedding'].values)
#     text_embeddings = np.stack(texts['embedding'].values)

#     # Replace cosine_similarity with custom dot product function
#     similarity_matrix = similarity_calculation(text_embeddings, image_embeddings)

#     n_texts, n_images = len(texts), len(images)
#     text_indices = np.repeat(np.arange(n_texts), n_images)
#     image_indices = np.tile(np.arange(n_images), n_texts)
#     text_data_indices = texts['data_index'].iloc[text_indices].values
#     image_data_indices = images['data_index'].iloc[image_indices].values

#     similarity_df = pd.DataFrame({
#         'text_data_index': text_data_indices,
#         'image_data_index': image_data_indices,
#         'text_idx': text_indices,
#         'image_idx': image_indices,
#         'cosine_similarity': similarity_matrix.flatten(),
#         'is_correct_pair': text_data_indices == image_data_indices
#     })
    
#     recall_results = calculate_recall_at_k_from_table(similarity_df, k_vals=[1, 5, 10])
#     i2t_recall_results = calculate_image_to_text_recall_at_k(similarity_df, k_vals=[1, 5, 10])
    
#     return similarity_df, recall_results, i2t_recall_results


# def calculate_recall_at_k_from_table(similarity_df, k_vals):
#     """Calculate text-to-image recall@k from similarity table"""
#     results = {}
#     grouped = similarity_df.groupby('text_idx')
    
#     for k in k_vals:
#         correct_retrievals = 0
#         total_texts = len(grouped)
        
#         for text_idx, group in grouped:
#             top_k = group.nlargest(k, 'cosine_similarity')
#             if top_k['is_correct_pair'].any():
#                 correct_retrievals += 1
        
#         recall_at_k = correct_retrievals / total_texts
#         results[f'R@{k}'] = recall_at_k
#         print(f"Text-to-Image R@{k}: {100*recall_at_k:.2f}%")
    
#     return results


# def calculate_image_to_text_recall_at_k(similarity_df, k_vals):
#     """Calculate image-to-text recall@k from similarity table"""
#     results = {}
#     grouped = similarity_df.groupby('image_idx')
    
#     for k in k_vals:
#         correct_retrievals = 0
#         total_images = len(grouped)
        
#         for image_idx, group in grouped:
#             top_k = group.nlargest(k, 'cosine_similarity')
#             if top_k['is_correct_pair'].any():
#                 correct_retrievals += 1
        
#         recall_at_k = correct_retrievals / total_images
#         results[f'I2T_R@{k}'] = recall_at_k
#         print(f"Image-to-Text R@{k}: {100*recall_at_k:.2f}%")
    
#     return results


if __name__ == "__main__":
    print("=" * 50)
    print("STANDARD CLIP EVALUATION (matches published results)")
    print("=" * 50)
    
    # Test different retrieval methods
    methods_to_test = [
        {
            "name": "CLIP",
            "retrieval": ClipHuggingFaceRetrival(
                model_name="openai/clip-vit-large-patch14",
                device="cuda:2",
                batch_size=128,
                prompt=""
            ),
            "save_path": "CLIP_huggingface"
        },
       {
           "name": "Jina CLIP v2",
           "retrieval": JinaRetrival(
               device="cuda:2",
               batch_size=128,
               prompt=""
           ),
           "save_path": "Jina"
       },
       {
           "name": "SigLIP",
           "retrieval": OpenCLIPRetrival(
                model_name="ViT-SO400M-14-SigLIP-384",
                dataset="webli",
                device="cuda:3",
                batch_size=128,
                prompt=""
            ),
            "save_path": "SigLIP"
       }, 
       {
           "name": "InvAtten",
           "retrieval": InvAttenRetrival(device="cuda:0"),
           "save_path": "InvAtten__"
       }, 
       {
        "name": "Grid Cropping", 
         "retrieval": GridCroppingRetrival(device="cuda:3", n_crops=5),
           "save_path": "Grid"  
       }
    ]
    
    all_results = {}
    
    for method_config in methods_to_test:
        print(f"\n{'='*60}")
        print(f"TESTING: {method_config['name']}")
        print(f"{'='*60}")
        
        try:
            t2i_results = text2image_recall(
                method_config["retrieval"], 
                save_path=method_config["save_path"], 
                single_caption=True
            )
            
            all_results[method_config['name']] = {
                't2i': t2i_results
            }
            
        except Exception as e:
            print(f"Error with {method_config['name']}: {e}")
            continue
    
    # Print comparison summary
    if len(all_results) > 1:
        print("\n" + "=" * 80)
        print("COMPARISON SUMMARY")
        print("=" * 80)
        print(f"{'Method':<20} {'T2I R@1':<10} {'T2I R@5':<10} {'T2I R@10':<10}")
        print("-" * 80)
        
        for method_name, results in all_results.items():
            t2i = results['t2i']
            print(f"{method_name:<20} "
                  f"{t2i['r1']:<10.2f} "
                  f"{t2i['r5']:<10.2f} "
                  f"{t2i['r10']:<10.2f}")
    
    print("\n" + "=" * 50)
    print("COMPARISON WITH ORIGINAL EVALUATION (Optional)")
    print("=" * 50)