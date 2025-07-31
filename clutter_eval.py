import os
import json
import torch
from PIL import Image
from tqdm import tqdm
from collections import defaultdict
import numpy as np

# Import retrieval classes
from models.retrivals import ClipHuggingFaceRetrival, JinaRetrival, OpenCLIPRetrival, GridCroppingRetrival, InvAttenRetrival

class DataLoader:
    def __init__(self, json_path, image_folder):
        self.json_path = json_path
        self.image_folder = image_folder
        self.image_paths = []
        self.image_ids = []
        self.class_to_images = defaultdict(list)

    def load(self):
        with open(self.json_path, "r") as f:
            data = json.load(f)

        for item in data:
            filename = item["filename"]
            cocoid = int(filename.split(".")[0])
            path = os.path.join(self.image_folder, filename)
            if not os.path.exists(path):
                continue
            self.image_paths.append(path)
            self.image_ids.append(cocoid)
            for cls in item["rare_classes"]:
                self.class_to_images[cls].append(cocoid)

class ClutterDataset:
    """Adapter class to make DataLoader compatible with retrieval classes"""
    def __init__(self, image_paths, image_ids):
        self.image_paths = image_paths
        self.image_ids = image_ids
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        # Return empty list for texts since we only need images for this evaluation
        return image, ["test"]

class RetrievalModelAdapter:
    """Adapter to make retrieval classes compatible with the evaluation interface"""
    def __init__(self, retrieval_model, image_paths, image_ids):
        self.retrieval_model = retrieval_model
        self.image_features = None
        self.valid_image_ids = image_ids.copy()
        self.is_multi_crop = False
        self.image_id_to_crops = {}  # Maps image_id to list of crop embeddings
        
        # Index the dataset
        dataset = ClutterDataset(image_paths, image_ids)
        self._index_dataset(dataset)
    
    def _index_dataset(self, dataset):
        """Index the dataset and extract image embeddings"""
        # Use the retrieval model's own indexing method
        self.retrieval_model.index_coco_dataset(dataset)
        
        # Check if this is a multi-crop model by looking at the embeddings structure
        # Multi-crop models will have multiple embeddings per data_index
        embedding_counts_per_image = defaultdict(int)
        for data_idx in self.retrieval_model.images["data_index"]:
            embedding_counts_per_image[data_idx] += 1
        
        # If any image has more than 1 embedding, this is a multi-crop model
        max_embeddings_per_image = max(embedding_counts_per_image.values())
        self.is_multi_crop = max_embeddings_per_image > 1
        
        if self.is_multi_crop:
            print(f"Detected multi-crop model with up to {max_embeddings_per_image} embeddings per image")
            # Organize embeddings by original image data_index
            for i, (data_idx, embedding) in enumerate(zip(
                self.retrieval_model.images["data_index"], 
                self.retrieval_model.images["embedding"]
            )):
                image_id = dataset.image_ids[data_idx]
                if image_id not in self.image_id_to_crops:
                    self.image_id_to_crops[image_id] = []
                self.image_id_to_crops[image_id].append(embedding)
            
            print(f"Organized {len(self.image_id_to_crops)} images with variable crop counts")
        else:
            # Single embedding per image
            image_embeddings_list = self.retrieval_model.images["embedding"]
            self.image_features = torch.stack([torch.from_numpy(emb) for emb in image_embeddings_list])
            self.image_features = self.image_features / self.image_features.norm(dim=-1, keepdim=True)
        
    def encode_text(self, text):
        """Encode text using the retrieval model"""
        # Use the retrieval model's text embedding method
        text_embeddings = self.retrieval_model.embed_texts([f"a photo of a {text}"], batch_size=1)
        text_emb = text_embeddings[0:1]  # Keep batch dimension
        
        # Convert to tensor and normalize
        if isinstance(text_emb, np.ndarray):
            text_emb = torch.from_numpy(text_emb)
        
        return text_emb / text_emb.norm(p=2, dim=-1, keepdim=True)
    
    def compute_similarities_max_crop(self, text_emb):
        """Compute similarities for multi-crop models using max similarity across crops"""
        similarities = []
        
        for image_id in self.valid_image_ids:
            # Get all crop embeddings for this image (variable number)
            crop_embeddings_list = self.image_id_to_crops[image_id]
            
            # Stack crop embeddings: [num_crops_for_this_image, emb_dim]
            crop_embeddings = np.stack(crop_embeddings_list)
            
            # Convert to tensor and normalize
            crop_embeddings = torch.from_numpy(crop_embeddings)
            crop_embeddings = crop_embeddings / crop_embeddings.norm(dim=-1, keepdim=True)
            
            # Compute similarities between text and all crops for this image
            crop_sims = torch.matmul(text_emb, crop_embeddings.T).squeeze(0)  # [num_crops_for_this_image]
            
            # Take maximum similarity across crops
            if crop_sims.dim() == 0:  # Single crop case
                max_sim = crop_sims
            else:
                max_sim = torch.max(crop_sims)
            similarities.append(max_sim)
        
        return torch.stack(similarities)

def evaluate(model, class_to_images):
    r1 = r5 = r10 = 0
    total = 0
    for cls, gt_ids in tqdm(class_to_images.items(), desc="Evaluating"):
        text_emb = model.encode_text(cls)
        
        if model.is_multi_crop:
            # Use max similarity across crops
            sims = model.compute_similarities_max_crop(text_emb)
        else:
            # Standard similarity computation
            sims = torch.matmul(text_emb, model.image_features.T).squeeze(0)
        
        topk = sims.topk(10).indices.tolist()
        top_ids = [model.valid_image_ids[i] for i in topk]

        gt_set = set(gt_ids)
        def recall_at_k(k):
            hits = len(set(top_ids[:k]) & gt_set)
            denom = min(k, len(gt_set)) if len(gt_set) > 0 else 1
            return hits / denom

        r1 += recall_at_k(1)
        r5 += recall_at_k(5)
        r10 += recall_at_k(10)
        total += 1

    return r1 / total, r5 / total, r10 / total, total

def main():
    device = "cuda:0"
    image_folder = "data/val2017"
    json_path = "rare_classes_top_images.json"

    data = DataLoader(json_path, image_folder)
    data.load()

    # Create retrieval models
    retrieval_models = {
        "OpenAI CLIP (ViT-L/14)": ClipHuggingFaceRetrival(
            model_name="openai/clip-vit-large-patch14", 
            device=device, 
            batch_size=128
        ),
        "Jina CLIP (jina-clip-v2)": JinaRetrival(
            device=device, 
            batch_size=128
        ),
        "SigLIP (ViT-SO400M-14)": OpenCLIPRetrival(
            model_name="ViT-SO400M-14-SigLIP-384", 
            dataset="webli", 
            device=device, 
            batch_size=128
        ),
        "Grid-Crop": GridCroppingRetrival(device="cuda:3", n_crops=5),
        "InvAtten": InvAttenRetrival(device="cuda:0")
    }

    # Create model adapters
    models = {}
    for name, retrieval_model in retrieval_models.items():
        print(f"Initializing {name}...")
        models[name] = RetrievalModelAdapter(retrieval_model, data.image_paths, data.image_ids)

    results = {}
    for name, model in models.items():
        print(f"Evaluating {name}...")
        r1, r5, r10, total = evaluate(model, data.class_to_images)
        results[name] = (r1, r5, r10, total)

    print("\n===== Final Recall@K Comparison (Text → Image) =====")
    for name, (r1, r5, r10, total) in results.items():
        print(f"\n{name}")
        print(f"Recall@1  = {r1:.2%}")
        print(f"Recall@5  = {r5:.2%}")
        print(f"Recall@10 = {r10:.2%}")
        print(f"Total Classes Evaluated: {total}")

if __name__ == "__main__":
    main()