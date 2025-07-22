import transformers
import open_clip 
import torch 
from PIL import Image
from torchvision.datasets import CocoCaptions
from tqdm import tqdm 


class TextImageRetrieval:
    def index_images(self, image_paths:list[str])->None: 
        pass 
    
    def index_coco_dataset(self, coco_dataset:CocoCaptions)->None:
        pass        

class OpenCLIPRetrival(TextImageRetrieval):
    def __init__(self, model_name:str, dataset:str=None, device:str="cuda", batch_size:int=128):
        self.device = device
        if dataset is None:
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, device=device)
        else:
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=dataset, device=device)
            
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.batch_size = batch_size
        self.images = {"data_index":[], "image":[], "embedding":[]}
        self.texts = {"data_index":[], "text":[], "embedding":[]}
        
        
    def index_coco_dataset(self, coco_dataset)->None:
        # we build the meta data first 
        print(type(coco_dataset))
        print("##")
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
            inputs = self.tokenizer(batch).to(self.device)
            text_features = self.model.encode_text(inputs)
            embeddings.append(text_features.cpu())
        return torch.cat(embeddings, dim=0)