from models.retrivals import OpenCLIPRetrival, InvAttenRetrival, ClipHuggingFaceRetrival
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


# def load_image_from_url(url):
#     response = requests.get(url)
#     response.raise_for_status()  # Raises an HTTPError for bad responses
#     image = Image.open(BytesIO(response.content))
#     return image


# class Kerpaty_cococ: 
#     def __init__(self, images_path:str):
#         self.images_path = images_path
#         self.dataset = load_dataset("yerevann/coco-karpathy", split="test")
#         self.root = self.images_path
        
        
#     def __getitem__(self, idx):
#         image_path = os.path.join(
#             self.images_path, 
#             self.dataset[idx]['filename'].split("_")[-1]
#         )
#         # image = load_image_from_url(self.dataset[idx]['url'])
#         image = Image.open(image_path)
#         captions = self.dataset[idx]['sentences']
#         return image, [captions[0]]

#Dataset
coco_root = "data/val2017"
coco_ann_file = "annotations/captions_val2017.json"
dataset = CocoCaptions(
    root=coco_root,
    annFile=coco_ann_file,
)


# dataset = Kerpaty_cococ(images_path="coco_test_images")


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

def recall(retrival: OpenCLIPRetrival, save_path: str = None) -> tuple:
    retrival.index_coco_dataset(dataset)
    images = pd.DataFrame(retrival.images)
    texts = pd.DataFrame(retrival.texts)  
    
    # # Convert PIL Images to file paths before saving
    # if save_path is not None:
    #     images_for_saving = images.copy()
    #     # Extract image paths from the dataset
    #     image_paths = []
    #     for idx in images['data_index']:
    #         # Get the image path from the COCO dataset
    #         img_info = dataset.coco.loadImgs(dataset.ids[idx])[0]
    #         img_path = os.path.join(dataset.root, img_info['file_name'])
    #         image_paths.append(img_path)
        
    #     images_for_saving['image_path'] = image_paths
    #     images_for_saving = images_for_saving.drop('image', axis=1)  # Remove PIL Image column
    
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
    
    # Save results if save_path is provided
    if save_path is not None: 
        images_for_saving = images.copy()
        images_for_saving = images_for_saving.drop('image', axis=1)  # Remove PIL Image column
        # Create directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)
        
        # Save similarity DataFrame as parquet (more efficient than CSV for large data)
        similarity_df.to_parquet(os.path.join(save_path, "similarity_matrix.parquet"), index=False)
        
        # Save images and texts metadata (with image paths instead of PIL Images)
        images_for_saving.to_parquet(os.path.join(save_path, "images_metadata.parquet"), index=False)
        texts.to_parquet(os.path.join(save_path, "texts_metadata.parquet"), index=False)
        
        # Save recall results as JSON
        combined_results = {
            "text_to_image_recall": recall_results,
            "image_to_text_recall": i2t_recall_results,
            "dataset_stats": {
                "n_images": n_images,
                "n_texts": n_texts,
                "total_similarity_pairs": len(similarity_df)
            }
        }
        
        with open(os.path.join(save_path, "recall_results.json"), 'w') as f:
            json.dump(combined_results, f, indent=2)
        
        print(f"Results saved to: {save_path}")
        print(f"- similarity_matrix.parquet: {len(similarity_df):,} rows")
        print(f"- images_metadata.parquet: {len(images_for_saving):,} rows (with image paths)") 
        print(f"- texts_metadata.parquet: {len(texts):,} rows")
        print(f"- recall_results.json: T2I and I2T metrics")
    
    return similarity_df, recall_results, i2t_recall_results

if __name__ == "__main__":
    # print("#### SigLIP ####")
    # retrival = OpenCLIPRetrival(
    #     model_name="ViT-SO400M-14-SigLIP-384",
    #     dataset="webli",
    #     device="cuda",
    #     batch_size=128
    # )
    # similarity_df, recall_results, i2t_recall_results = recall(retrival, save_path="results/siglip")
    print("#### CLIP ####")
    # retrival = OpenCLIPRetrival(
    #     model_name="ViT-L-14-336",
    #     dataset="openai",
    #     device="cuda",
    #     batch_size=128,
    #     # prompt="A photo of a "
    # )
    retrival = ClipHuggingFaceRetrival(
        model_name="openai/clip-vit-large-patch14",
        device="cuda",
        batch_size=128,
        prompt="A photo of "
    )
    similarity_df, recall_results, i2t_recall_results = recall(retrival, save_path="CLIP_eval_results") 
    # print("#### JINA ####")
    
    # print("#### InvAtten ####")
    # retrival = InvAttenRetrival()
    # similarity_df, recall_results, i2t_recall_results = recall(retrival, save_path="results/inv_atten")
    print("#### Done ####")