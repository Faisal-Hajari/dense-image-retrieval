import os
import json
import torch
from PIL import Image
from tqdm import tqdm
from collections import defaultdict
import clip
from transformers import AutoProcessor, AutoModel
from transformers import CLIPProcessor, CLIPModel as CLIPModel_hf
import open_clip

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

class CLIPModel:
    def __init__(self, model_name, device):
        self.device = device
        self.model = CLIPModel_hf.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        
        
        self.image_features = []
        self.valid_image_ids = []

    def encode_images(self, paths, ids):
        feats = []
        for path, cocoid in tqdm(zip(paths, ids), total=len(paths), desc="OpenAI CLIP"):
            img = Image.open(path).convert("RGB")
            img_tensor = self.processor(images=img, return_tensors="pt").to(self.device)
            with torch.no_grad():
                emb = self.model.get_image_features(**img_tensor)
                emb /= emb.norm(dim=-1, keepdim=True)
                feats.append(emb.cpu())
                self.valid_image_ids.append(cocoid)
        self.image_features = torch.cat(feats).to(self.device)

    def encode_text(self, text):
        inputs = self.processor(text=[f"a photo of a {text}"], return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            emb = self.model.get_text_features(**inputs)
            return emb / emb.norm(p=2, dim=-1, keepdim=True)

class JinaModel:
    def __init__(self, device):
        self.device = device
        self.processor = AutoProcessor.from_pretrained("jinaai/jina-clip-v2", trust_remote_code=True)
        self.model = AutoModel.from_pretrained("jinaai/jina-clip-v2", trust_remote_code=True).to(device)
        self.model.eval()
        self.image_features = []
        self.valid_image_ids = []

    def encode_images(self, paths, ids):
        feats = []
        for path, cocoid in tqdm(zip(paths, ids), total=len(paths), desc="Jina CLIP"):
            try:
                image = Image.open(path).convert("RGB")
                inputs = self.processor(images=image, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    emb = self.model.get_image_features(**inputs)
                    emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
                    feats.append(emb.cpu())
                    self.valid_image_ids.append(cocoid)
            except Exception as e:
                raise RuntimeError(f"Failed to process image {path}: {str(e)}")
        self.image_features = torch.cat(feats).to(self.device)

    def encode_text(self, text):
        inputs = self.processor(text=[f"a photo of a {text}"], return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            emb = self.model.get_text_features(**inputs)
            return emb / emb.norm(p=2, dim=-1, keepdim=True)

class SigLIPModel:
    def __init__(self, model_name, pretrained, device):
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name=model_name, pretrained=pretrained, device=device)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()
        self.image_features = []
        self.valid_image_ids = []

    def encode_images(self, paths, ids):
        feats = []
        for path, cocoid in tqdm(zip(paths, ids), total=len(paths), desc="SigLIP"):
            try:
                img = Image.open(path).convert("RGB")
                img_tensor = self.preprocess(img).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    feat = self.model.encode_image(img_tensor)
                    feat /= feat.norm(dim=-1, keepdim=True)
                    feats.append(feat.cpu())
                    self.valid_image_ids.append(cocoid)
            except Exception as e:
                print(f"Failed image {path}: {e}")
        self.image_features = torch.cat(feats).to(self.device)

    def encode_text(self, text):
        tokens = self.tokenizer([f"a photo of a {text}"]).to(self.device)
        with torch.no_grad():
            feat = self.model.encode_text(tokens)
            feat /= feat.norm(dim=-1, keepdim=True)
        return feat

def evaluate(model, class_to_images):
    r1 = r5 = r10 = 0
    total = 0
    for cls, gt_ids in tqdm(class_to_images.items(), desc="Evaluating"):
        text_emb = model.encode_text(cls)
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
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cuda:0"
    image_folder = "data/val2017"
    json_path = "rare_classes_top_images.json"

    data = DataLoader(json_path, image_folder)
    data.load()

    models = {
        "OpenAI CLIP (ViT-L/14)": CLIPModel("openai/clip-vit-large-patch14", device),
        "Jina CLIP (jina-clip-v2)": JinaModel(device),
        "SigLIP (ViT-SO400M-14)": SigLIPModel("ViT-SO400M-14-SigLIP-384", "webli", device)
    }

    results = {}
    for name, model in models.items():
        model.encode_images(data.image_paths, data.image_ids)
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