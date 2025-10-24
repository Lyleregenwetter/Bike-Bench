import abc
import time
from typing import List, Optional
import numpy as np
import torch
from PIL import Image
# from torchvision import transforms
from transformers import CLIPTokenizer, CLIPImageProcessor, CLIPModel
from tqdm import trange


# _DEVICE = "cuda"
# _MODEL_ID = "openai/clip-vit-base-patch32"
# _CLIP_PROCESSOR = CLIPProcessor.from_pretrained(_MODEL_ID)
# _TOKENIZER = CLIPTokenizerFast.from_pretrained(_MODEL_ID)
# _MODEL = CLIPModel.from_pretrained(_MODEL_ID)  # .to(device)


# class ClipEmbeddingCalculator(metaclass=abc.ABCMeta):
#     @abc.abstractmethod
#     def from_text(self, text: str) -> np.ndarray:
#         pass

#     @abc.abstractmethod
#     def from_image_path(self, image_path: str) -> np.ndarray:
#         pass


# class ClipEmbeddingCalculatorImpl(ClipEmbeddingCalculator):
#     def from_text(self, text: str) -> np.ndarray:
#         embedding_tensor = _MODEL.get_text_features(**_TOKENIZER(text, return_tensors="pt"))
#         return embedding_tensor

#     def from_image_path(self, image_path: str) -> np.ndarray:
#         return self.from_image(Image.open(image_path))

#     def from_image(self, img: Image.Image):
#         start = time.time()

#         img = img.convert("RGB")
#         width, height = img.size
#         img = img.resize((width // 2, height // 2))
#         result = Image.new(img.mode, (1300, 1300), (255, 255, 255))
#         result.paste(img, img.getbbox())
#         image = np.asarray(result)
#         img_processed = _CLIP_PROCESSOR(text=None, images=image, return_tensors='pt')['pixel_values']  # .to(device)
#         embedding_tensor = _MODEL.get_image_features(img_processed)

#         print(f"Obtaining embeddings took {time.time() - start}")
#         return embedding_tensor

#     def from_image_tensor(self, image_tensor: torch.Tensor) -> np.ndarray:
#         img = image_tensor.cpu().clone()
#         img = transforms.Resize((1300, 1300))(img)
#         result = Image.new(img.mode, (1300, 1300), (255, 255, 255))
#         result.paste(img, img.getbbox())
#         image = np.asarray(result)
#         img_processed = _CLIP_PROCESSOR(text=None, images=image, return_tensors='pt')['pixel_values']
#         embedding_tensor = _MODEL.get_image_features(img_processed)
#         return embedding_tensor




class ClipEmbeddingCalculator:
    """
    A batch-friendly CLIP embedding class for text and images.
    """
    def __init__(self, device: str = "cuda", batch_size: Optional[int] = None):
        model_id = "openai/clip-vit-base-patch32"
        self.device     = device
        self.batch_size = batch_size

        # text tokenizer (fast Rust version)
        self.tokenizer       = CLIPTokenizer.from_pretrained(model_id, use_fast=True)
        # new image processor (replaces the deprecated FeatureExtractor)
        self.image_processor = CLIPImageProcessor.from_pretrained(model_id)

        self.model = CLIPModel.from_pretrained(model_id).to(self.device)
        self.model.eval()

    def embed_texts(self, texts: List[str]) -> torch.Tensor:
        all_feats = []
        bs = self.batch_size or len(texts)
        for i in range(0, len(texts), bs):
            chunk = texts[i : i + bs]
            enc   = self.tokenizer(
                        chunk,
                        padding=True,
                        truncation=True,
                        return_tensors="pt"
                    )
            inputs = {k: v.to(self.device) for k,v in enc.items()}
            with torch.no_grad():
                feats = self.model.get_text_features(**inputs)
            all_feats.append(feats)
        return torch.cat(all_feats, dim=0)

    def embed_images(self, images: torch.Tensor) -> torch.Tensor:
        all_feats = []
        n  = images.shape[0]
        bs = self.batch_size or n
        for i in range(0, n, bs):
            chunk = images[i : i + bs]
            enc   = self.image_processor(
                        images=chunk, 
                        return_tensors="pt"
                    )
            inputs = {k: v.to(self.device) for k,v in enc.items()}
            with torch.no_grad():
                feats = self.model.get_image_features(**inputs)
            all_feats.append(feats)
        return torch.cat(all_feats, dim=0)




# def get_augmented_views_gpu(images_tensor):
#     transform = transforms.RandomApply([
#                                         transforms.RandomHorizontalFlip(),
#                                         transforms.RandomAdjustSharpness(0.2), 
#                                         transforms.RandomAdjustSharpness(2), 
#                                         transforms.RandomPerspective(fill=(0, 0, 0)),
#                                         transforms.RandomRotation(degrees = 45, fill= (0, 0, 0)), 
#                                        #  transforms.ColorJitter(brightness=0.1, contrast = 0.1, saturation=0.1, hue=0.0),
#                                        ],p=1)
#     res = transform(images_tensor.cuda()).cpu()
#     return res


