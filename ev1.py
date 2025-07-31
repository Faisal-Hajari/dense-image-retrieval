from models.retrivals import OpenCLIPRetrival, ClipHuggingFaceRetrival, JinaRetrival
from torchvision.datasets import CocoCaptions
import pandas as pd 
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import json
from datasets import load_dataset
import requests
from PIL import Image
from io import BytesIO


def load_image_from_url(url):
    response = requests.get(url)
    response.raise_for_status()  # Raises an HTTPError for bad responses
    image = Image.open(BytesIO(response.content))
    return image


class Kerpathy_cococ: 
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
        if self.single_caption:
            # Use only the first caption
            if isinstance(captions[0], dict):
                captions = [captions[0]['raw']]
            else:
                captions = [captions[0]]  # captions[0] is already a string
        else:
            # Use all captions
            if isinstance(captions[0], dict):
                captions = [sent['raw'] for sent in captions]
            else:
                captions = captions  # captions is already a list of strings
            
        return image, captions

    def __len__(self):
        return len(self.dataset)


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


def recall_standard(retrival: OpenCLIPRetrival, save_path: str = None, single_caption: bool = True) -> tuple:
    """
    Standard CLIP evaluation that matches published results
    """
    # Use single caption for standard evaluation
    dataset = Kerpathy_cococ(images_path="coco_test_images", single_caption=single_caption)
    
    # Index dataset with retrieval model
    retrival.index_coco_dataset(dataset)
    images = pd.DataFrame(retrival.images)
    texts = pd.DataFrame(retrival.texts)  
    
    if len(images) != len(texts):
        raise ValueError(f"Unequal number of images ({len(images)}) and texts ({len(texts)})")
    
    # Get embeddings
    image_embeddings = np.stack(images['embedding'].values)
    text_embeddings = np.stack(texts['embedding'].values)
    
    # Normalize embeddings 
    image_embeddings = image_embeddings / np.linalg.norm(image_embeddings, axis=1, keepdims=True)
    text_embeddings = text_embeddings / np.linalg.norm(text_embeddings, axis=1, keepdims=True)
    
    # Compute similarity matrices
    i2t_similarity = np.dot(image_embeddings, text_embeddings.T)  # Image to Text
    t2i_similarity = np.dot(text_embeddings, image_embeddings.T)  # Text to Image
    
    i2t_results, _ = compute_retrieval_standard(i2t_similarity)
    # print(f"I2T Results: R@1: {i2t_results['r1']:.2f}%, R@5: {i2t_results['r5']:.2f}%, R@10: {i2t_results['r10']:.2f}%")
    
    t2i_results, _ = compute_retrieval_standard(t2i_similarity)
    # print(f"T2I Results: R@1: {t2i_results['r1']:.2f}%, R@5: {t2i_results['r5']:.2f}%, R@10: {t2i_results['r10']:.2f}%")
    
    # Save results if save_path is provided
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        
        images_for_saving = images.copy()
        images_for_saving = images_for_saving.drop('image', axis=1)
        
        # Save embeddings and metadata
        np.save(os.path.join(save_path, "image_embeddings.npy"), image_embeddings)
        np.save(os.path.join(save_path, "text_embeddings.npy"), text_embeddings)
        np.save(os.path.join(save_path, "i2t_similarity.npy"), i2t_similarity)
        np.save(os.path.join(save_path, "t2i_similarity.npy"), t2i_similarity)
        
        images_for_saving.to_parquet(os.path.join(save_path, "images_metadata.parquet"), index=False)
        texts.to_parquet(os.path.join(save_path, "texts_metadata.parquet"), index=False)
        
        # Save results
        combined_results = {
            "image_to_text_recall": i2t_results,
            "text_to_image_recall": t2i_results,
            "dataset_stats": {
                "n_images": len(images),
                "n_texts": len(texts),
                "single_caption": single_caption
            },
            "evaluation_type": "standard_clip"
        }
        
        with open(os.path.join(save_path, "standard_clip_results.json"), 'w') as f:
            json.dump(combined_results, f, indent=2)
        
        print(f"Results saved to: {save_path}")
    
    return i2t_results, t2i_results


# Keep your original function for comparison
# def recall_original(retrival: OpenCLIPRetrival, save_path: str = None) -> tuple:
#     """Your original evaluation approach for comparison"""
#     dataset = Kerpathy_cococ(images_path="coco_test_images", single_caption=False) 
#     retrival.index_coco_dataset(dataset)
#     images = pd.DataFrame(retrival.images)
#     texts = pd.DataFrame(retrival.texts)  
    
#     image_embeddings = np.stack(images['embedding'].values)
#     text_embeddings = np.stack(texts['embedding'].values)

#     similarity_matrix = cosine_similarity(text_embeddings, image_embeddings)

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


def calculate_recall_at_k_from_table(similarity_df, k_vals):
    """Calculate text-to-image recall@k from similarity table"""
    results = {}
    grouped = similarity_df.groupby('text_idx')
    
    for k in k_vals:
        correct_retrievals = 0
        total_texts = len(grouped)
        
        for text_idx, group in grouped:
            top_k = group.nlargest(k, 'cosine_similarity')
            if top_k['is_correct_pair'].any():
                correct_retrievals += 1
        
        recall_at_k = correct_retrievals / total_texts
        results[f'R@{k}'] = recall_at_k
        print(f"Text-to-Image R@{k}: {100*recall_at_k:.2f}%")
    
    return results


def calculate_image_to_text_recall_at_k(similarity_df, k_vals):
    """Calculate image-to-text recall@k from similarity table"""
    results = {}
    grouped = similarity_df.groupby('image_idx')
    
    for k in k_vals:
        correct_retrievals = 0
        total_images = len(grouped)
        
        for image_idx, group in grouped:
            top_k = group.nlargest(k, 'cosine_similarity')
            if top_k['is_correct_pair'].any():
                correct_retrievals += 1
        
        recall_at_k = correct_retrievals / total_images
        results[f'I2T_R@{k}'] = recall_at_k
        print(f"Image-to-Text R@{k}: {100*recall_at_k:.2f}%")
    
    return results


if __name__ == "__main__":
    print("=" * 50)
    print(" "*20, "EVALUATION")
    print("=" * 50)
    
    
    print(" "*20, "CLIP")
    retrival = ClipHuggingFaceRetrival(
        model_name="openai/clip-vit-large-patch14",
        device="cuda",
        batch_size=128,
        prompt=""  # No prompt for standard evaluation
    )
    
    i2t_results, t2i_results = recall_standard(
        retrival, 
        save_path="CLIP_standard_results", 
        single_caption=True
    )
    print(f"  I2T: R@1={i2t_results['r1']:.1f}%, R@5={i2t_results['r5']:.1f}%, R@10={i2t_results['r10']:.1f}%")
    print(f"  T2I: R@1={t2i_results['r1']:.1f}%, R@5={t2i_results['r5']:.1f}%, R@10={t2i_results['r10']:.1f}%")
    print("-" * 50)
    print(" "*20, "JINA")
    retrival = JinaRetrival(
        device="cuda",
        batch_size=128,
        prompt=""
    )
    i2t_results, t2i_results = recall_standard(
        retrival, 
        save_path="JINA_standard_results", 
        single_caption=True)
    print(f"  I2T: R@1={i2t_results['r1']:.1f}%, R@5={i2t_results['r5']:.1f}%, R@10={i2t_results['r10']:.1f}%")
    print(f"  T2I: R@1={t2i_results['r1']:.1f}%, R@5={t2i_results['r5']:.1f}%, R@10={t2i_results['r10']:.1f}%")
    print("-" * 50)
    print(" "*20, "SigLIP") 
    retrival = OpenCLIPRetrival(
        model_name="ViT-SO400M-14-SigLIP-384",
        dataset="webli",
        device="cuda",
        batch_size=128,
        prompt=""
    )
    i2t_results, t2i_results = recall_standard(
        retrival, 
        save_path="SigLIP_standard_results", 
        single_caption=True
    )
    print(f"  I2T: R@1={i2t_results['r1']:.1f}%, R@5={i2t_results['r5']:.1f}%, R@10={i2t_results['r10']:.1f}%")
    print(f"  T2I: R@1={t2i_results['r1']:.1f}%, R@5={t2i_results['r5']:.1f}%, R@10={t2i_results['r10']:.1f}%")
    print("-" * 50)
    
    
