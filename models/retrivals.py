import transformers
import open_clip 
import torch 
from PIL import Image
from torchvision.datasets import CocoCaptions
from tqdm import tqdm 
import numpy as np
import math 
import cv2 
from torch.nn import functional as F
from sklearn.cluster import KMeans
from transformers import AutoModel
from transformers import CLIPProcessor, CLIPModel

class TextImageRetrieval:
    def index_images(self, image_paths:list[str])->None: 
        pass 
    
    def index_coco_dataset(self, coco_dataset:CocoCaptions)->None:
        pass        

class OpenCLIPRetrival(TextImageRetrieval):
    def __init__(self, model_name:str, dataset:str=None, device:str="cuda", batch_size:int=128, prompt:str="") :
        self.device = device
        if dataset is None:
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, device=device)
        else:
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=dataset, device=device)
            
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.batch_size = batch_size
        self.images = {"data_index":[], "image":[], "embedding":[]}
        self.texts = {"data_index":[], "text":[], "embedding":[]}
        self.prompt = prompt
        
    def index_coco_dataset(self, coco_dataset)->None:
        # we build the meta data first 
        for idx, (image, texts) in tqdm(enumerate(coco_dataset), desc="building tables"):
            self.images["data_index"].append(idx)
            self.images["image"].append(image)    
            for txt in texts: 
                self.texts["data_index"].append(idx)
                self.texts["text"].append(txt)
        
        # we build the embedding here so we can take advantage of batching 
        image_embeddings = self.embed_images(self.images["image"], self.batch_size)
        text_embeddings = self.embed_texts(self.texts["text"], self.batch_size)
        
        image_embeddings_list = [emb.numpy() for emb in image_embeddings]
        text_embeddings_list = [emb.numpy() for emb in text_embeddings]
        
        # we store the embeddings in the meta data 
        self.images["embedding"] = image_embeddings_list
        self.texts["embedding"] = text_embeddings_list
        
    @torch.no_grad()
    def embed_images(self, images:list[Image], batch_size:int=128)->torch.Tensor:
        embeddings = [] 
        for i in tqdm(range(0, len(images), batch_size), desc="Embedding images"):
            batch = images[i:i+batch_size]
            batch = [self.preprocess(image) for image in batch]
            batch = torch.stack(batch).to(self.device)  # [batch_size, 3, H, W]
            image_features = self.model.encode_image(batch)
            embeddings.append(image_features.cpu())
        return torch.cat(embeddings, dim=0) # [len(images), embedding dim]
    
    @torch.no_grad()
    def embed_texts(self, texts:list[str], batch_size:int=128)->torch.Tensor:
        embeddings = [] 
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding texts"):
            batch = texts[i:i+batch_size]
            inputs = self.tokenizer([self.prompt + text for text in batch]).to(self.device)
            text_features = self.model.encode_text(inputs)
            embeddings.append(text_features.cpu())
        return torch.cat(embeddings, dim=0)


class InvAttenRetrival(TextImageRetrieval):
    def __init__(self,
                device: torch.device = torch.device("cuda:0"), 
                kernel_size: tuple = (64,64),
                threshold: float = 0.4,
                stride: tuple = (16, 16),
                n_crops: int = 5, 
                include_full_image: bool = True):
        
        self.n_crops = n_crops
        self.include_full_image = include_full_image
        self.device = device
        self.clip, _, self.preprocess = open_clip.create_model_and_transforms("ViT-SO400M-14-SigLIP-384", pretrained="webli", device=device)
        self.clip.eval()
        self.kernel_size = kernel_size
        self.threshold = threshold 
        self.stride = stride
        self.tokenizer = open_clip.get_tokenizer("ViT-SO400M-14-SigLIP-384")
        self.batch_size = 512
        self.images = {"data_index":[], "image":[], "embedding":[]}
        self.texts = {"data_index":[], "text":[], "embedding":[]}
        
        
    def index_coco_dataset(self, coco_dataset: CocoCaptions) -> None:
        embedding_per_image = self.n_crops + 1 if self.include_full_image else self.n_crops
        for idx, (image, texts) in tqdm(enumerate(coco_dataset), desc="embedding images"):
            image_embeddings = self.embed_image(image)
            for embedding in image_embeddings:
                self.images["data_index"].append(idx)
                self.images["image"].append(image)
                self.images["embedding"].append(embedding.numpy())
            for txt in texts: 
                self.texts["data_index"].append(idx)
                self.texts["text"].append(txt)
        
        # we build the embedding here so we can take advantage of batching 
        text_embeddings = self.embed_texts(self.texts["text"], self.batch_size)
        
        text_embeddings_list = [emb.numpy() for emb in text_embeddings]
        
        # we store the embeddings in the meta data 
        self.texts["embedding"] = text_embeddings_list 
        
        
    @torch.no_grad()
    def embed_images(self, images: list[Image], batch_size: int = 128) -> torch.Tensor:
        embeddings = [] 
        for image in tqdm(images, desc="Embedding images"):             
            image_features = self.forward_images(image)
            embeddings.append(image_features.cpu().unsqueeze(0)) # [1, n_crops+1, embedding dim]
        return torch.cat(embeddings, dim=0) # [len(images), (n_crops+1), embedding dim]
    
    #thsi takes an image PIL adn returns a tensor of shape [n_crops+1, embedding dim]
    def embed_image(self, image): 
        image_features = self.forward_images(image)
        return image_features.cpu()
    
    
        
    @torch.no_grad()
    def embed_texts(self, texts:list[str], batch_size:int=128)->torch.Tensor:
        embeddings = [] 
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding texts"):
            batch = texts[i:i+batch_size]
            inputs = self.tokenizer(batch).to(self.device)
            text_features = self.clip.encode_text(inputs)
            embeddings.append(text_features.cpu())
        return torch.cat(embeddings, dim=0) # [len(texts), embedding dim]
    
    
    # Ignore theses :: 
    @torch.no_grad()
    def forward_images(self, image:Image) -> torch.Tensor:
        # save original size
        orig_W, orig_H = image.size

        # preprocess to 384×384
        x = self.preprocess(image).unsqueeze(0).to(self.device)
        # get attention map
        tokens,_ = self.clip.visual.trunk.forward_intermediates(x)
        full_feature, attn = self.wrap_attn_pool(self.clip.visual.trunk.attn_pool, tokens)
        attn_np = attn.cpu().numpy()

        # build mask at 384×384
        mask = self.create_mask(attn_np, 384, 384, self.threshold)

        # sliding window in 384×384 coords
        boxes384 = self.sliding_windows(mask, self.kernel_size, self.threshold, self.stride)

        # rescale boxes back to original image size
        scale_x = orig_W / 384
        scale_y = orig_H / 384

        boxes = []
        for x1,y1,x2,y2 in boxes384:
            boxes.append((
                int(x1 * scale_x),
                int(y1 * scale_y),
                int(x2 * scale_x),
                int(y2 * scale_y),
            ))

        boxes = np.array(boxes, dtype=int)

        # Check if no boxes found - fallback to full image only
        if len(boxes) == 0:
            return full_feature  # [1, embedding_dim]
        
        if self.n_crops < len(boxes):
            k_means = KMeans(n_clusters=self.n_crops)
            k_means.fit(boxes)
            labels = k_means.labels_
            merged_boxes = []
            for i in range(self.n_crops):
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

        crops = []
        for box in merged_boxes:
            xmin, ymin, xmax, ymax = box
            cropped_image = image.crop((xmin, ymin, xmax, ymax))
            crops.append(cropped_image)
        
        crops = [self.preprocess(crop) for crop in crops]
        crops = torch.stack(crops).to(self.device)  # [batch_size, 3, H, W]
        image_features = self.clip.encode_image(crops)
        
        if self.include_full_image:
            all_features = torch.cat([full_feature, image_features], dim=0)
            return all_features # [n_crops+1, embedding dim]
        else:
            return image_features  # [n_crops, embedding dim]
    
    def sliding_windows(self, mask, k, thresh, stride=None):
        h,w = mask.shape
        kh,kw = k
        sh,sw = stride or k
        boxes=[]
        for y in range(0,h-kh+1,sh):
            for x in range(0,w-kw+1,sw):
                if mask[y:y+kh,x:x+kw].mean()>=thresh:
                    boxes.append((x,y,x+kw,y+kh))
        return boxes
    
    
    def create_mask(self, attn: np.ndarray, H: int, W: int, thresh: float):
        B, heads, Q, S = attn.shape
        g = int(math.sqrt(S))
        combined = np.zeros((H,W), bool)

        for h in range(heads):
            heat = attn[0,h,0].reshape(g,g)
            heat = (heat - heat.min())/(heat.max()-heat.min()+1e-8)
            up = cv2.resize((heat*255).astype(np.uint8),
                            (W,H), interpolation=cv2.INTER_LINEAR)/255.0
            combined |= (up >= up.mean())

        return (~combined).astype(np.uint8)
    
    
    def wrap_attn_pool(self, head, x):
        B, N, C = x.shape

        if head.pos_embed is not None:
            x = x + head.pos_embed.unsqueeze(0).to(x.dtype)

        # build queries from learned latent
        q_latent = head.latent.expand(B, -1, -1)
        q = head.q(q_latent) \
            .reshape(B, head.latent_len, head.num_heads, head.head_dim) \
            .transpose(1, 2)   # [B, heads, queries, dim]

        # build keys & values
        kv = head.kv(x) \
                .reshape(B, N, 2, head.num_heads, head.head_dim) \
                .permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)

        # normalize
        q, k = head.q_norm(q), head.k_norm(k)

        # compute attention
        if head.fused_attn:
            x_out = F.scaled_dot_product_attention(q, k, v)
        else:
            q_scaled = q * head.scale
            attn = (q_scaled @ k.transpose(-2, -1)).softmax(dim=-1)
            x_out = attn @ v

        # project and residual
        x_out = (
            x_out.transpose(1, 2)
                .reshape(B, head.latent_len, C)
        )
        x_out = head.proj(x_out)
        x_out = head.proj_drop(x_out)
        x_out = x_out + head.mlp(head.norm(x_out))

        # optional pooling to final vector
        if head.pool == 'token':
            pooled = x_out[:, 0]
        elif head.pool == 'avg':
            pooled = x_out.mean(1)
        else:
            pooled = x_out

        # re‑compute raw attn map if fused path was used
        q_scaled = q * head.scale
        attn = (q_scaled @ k.transpose(-2, -1)).softmax(dim=-1)

        return pooled, attn  # attn: [B, heads, queries, seq_len]



class ClipHuggingFaceRetrival(TextImageRetrieval):
    def __init__(self, model_name:str="openai/clip-vit-large-patch14", device:str="cuda", batch_size:int=128, prompt:str="", truncate_dim=None):
        self.device = device
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        
        self.batch_size = batch_size
        self.images = {"data_index":[], "image":[], "embedding":[]}
        self.texts = {"data_index":[], "text":[], "embedding":[]}
        self.prompt = prompt
        self.truncate_dim = truncate_dim
        
    def index_coco_dataset(self, coco_dataset)->None:
        # we build the meta data first 
        for idx, (image, texts) in tqdm(enumerate(coco_dataset), desc="building tables"):
            self.images["data_index"].append(idx)
            self.images["image"].append(image)    
            for txt in texts: 
                self.texts["data_index"].append(idx)
                self.texts["text"].append(txt)
        
        # we build the embedding here so we can take advantage of batching 
        image_embeddings = self.embed_images(self.images["image"], self.batch_size)
        text_embeddings = self.embed_texts(self.texts["text"], self.batch_size)
        
        image_embeddings_list = [emb.numpy() for emb in image_embeddings]
        text_embeddings_list = [emb.numpy() for emb in text_embeddings]
        
        # we store the embeddings in the meta data 
        self.images["embedding"] = image_embeddings_list
        self.texts["embedding"] = text_embeddings_list
        
    @torch.no_grad()
    def embed_images(self, images:list[Image], batch_size:int=128)->torch.Tensor:
        embeddings = [] 
        for i in tqdm(range(0, len(images), batch_size), desc="Embedding images"):
            batch = images[i:i+batch_size]
            
            # Process images using HuggingFace processor
            inputs = self.processor(images=batch, return_tensors="pt", padding=False)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get image features
            image_features = self.model.get_image_features(**inputs)
            
            # Apply truncation if specified
            if self.truncate_dim is not None:
                image_features = image_features[:, :self.truncate_dim]
                
            # Normalize features (following CLIP convention)
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            
            embeddings.append(image_features.cpu())
        return torch.cat(embeddings, dim=0)
    
    @torch.no_grad()
    def embed_texts(self, texts:list[str], batch_size:int=128)->torch.Tensor:
        embeddings = [] 
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding texts"):
            batch = texts[i:i+batch_size]
            
            # Add prompt if specified
            if self.prompt:
                batch = [self.prompt + text for text in batch]
            
            # Process texts using HuggingFace processor
            inputs = self.processor(text=batch, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get text features
            text_features = self.model.get_text_features(**inputs)
            
            # Apply truncation if specified
            if self.truncate_dim is not None:
                text_features = text_features[:, :self.truncate_dim]
                
            # Normalize features (following CLIP convention)
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
            
            embeddings.append(text_features.cpu())
        return torch.cat(embeddings, dim=0)


class JinaRetrival(TextImageRetrieval):
    def __init__(self, device:str="cuda", batch_size:int=128, prompt:str="", truncate_dim = None) :
        self.device = device

        self.model = AutoModel.from_pretrained('jinaai/jina-clip-v2', trust_remote_code=True)
        self.model.to(device)
        self.model.eval()
        self.truncate_dim = truncate_dim        
        self.batch_size = batch_size
        self.images = {"data_index":[], "image":[], "embedding":[]}
        self.texts = {"data_index":[], "text":[], "embedding":[]}
        self.prompt = prompt
    
    
    def embed_texts(self, texts:list[str], batch_size:int=128)->torch.Tensor:
        embeddings = [] 
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding texts"):
            batch = texts[i:i+batch_size]
            embeddings.append(self.model.encode_text(batch, task='retrieval.query', truncate_dim=self.truncate_dim))
        return torch.cat(embeddings, dim=0)

    def embed_images(self, images:list[Image], batch_size:int=128)->torch.Tensor:
        embeddings = [] 
        for i in tqdm(range(0, len(images), batch_size), desc="Embedding images"):
            batch = images[i:i+batch_size]
            embeddings.append(self.model.encode_image(batch, truncate_dim=self.truncate_dim))
        return torch.cat(embeddings, dim=0)
    
    def index_coco_dataset(self, coco_dataset)->None:
        # we build the meta data first 
        for idx, (image, texts) in tqdm(enumerate(coco_dataset), desc="building tables"):
            self.images["data_index"].append(idx)
            self.images["image"].append(image)    
            for txt in texts: 
                self.texts["data_index"].append(idx)
                self.texts["text"].append(txt)
        
        # we build the embedding here so we can take advantage of batching 
        image_embeddings = self.embed_images(self.images["image"], self.batch_size)
        text_embeddings = self.embed_texts(self.texts["text"], self.batch_size)
        
        image_embeddings_list = [emb.numpy() for emb in image_embeddings]
        text_embeddings_list = [emb.numpy() for emb in text_embeddings]
        
        # we store the embeddings in the meta data 
        self.images["embedding"] = image_embeddings_list
        self.texts["embedding"] = text_embeddings_list