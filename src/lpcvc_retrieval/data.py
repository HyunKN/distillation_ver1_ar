from __future__ import annotations
import os, json
from collections import defaultdict
from typing import List, Tuple, Dict, Any, Optional

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from transformers import CLIPTokenizer

def build_tokenizer() -> CLIPTokenizer:
    return CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

def _img_transform_train(augment: bool = True) -> transforms.Compose:
    """
    Training transform with optional augmentation.
    
    Augmentations improve model generalization by creating varied views of images:
    - RandomResizedCrop: Random crop and resize, forces model to recognize partial objects
    - RandomHorizontalFlip: Mirror images, doubles effective dataset size
    - ColorJitter: Random brightness/contrast changes, improves lighting robustness
    """
    if augment:
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),  # [0,1]
        ])
    else:
        return _img_transform_eval()

def _img_transform_eval() -> transforms.Compose:
    """Evaluation transform: deterministic resize and center crop."""
    return transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),  # [0,1]
    ])

# Legacy alias for backward compatibility
def _img_transform() -> transforms.Compose:
    return _img_transform_eval()

class JsonlRetrievalDataset(Dataset):
    """
    JSONL format (each line):
      {"image":"train2017/0000.jpg","captions":["a ...", "..."]}
    image paths are relative to image_root.
    """
    def __init__(self, image_root: str, jsonl_path: str, tokenizer: CLIPTokenizer, 
                 max_caps_per_image: int = 1, is_train: bool = False, augment: bool = True):
        self.image_root = image_root
        self.jsonl_path = jsonl_path
        self.tokenizer = tokenizer
        self.max_caps_per_image = max(1, int(max_caps_per_image))
        self.is_train = is_train
        
        # Use augmentation only for training
        self.tf = _img_transform_train(augment) if is_train else _img_transform_eval()

        self.samples: List[Tuple[str, List[str]]] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                img_rel = obj["image"]
                caps = obj.get("captions", [])
                if not isinstance(caps, list):
                    caps = [str(caps)]
                if len(caps) == 0:
                    continue
                self.samples.append((img_rel, caps))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_rel, caps = self.samples[idx]
        
        # Random caption selection during training for variety
        if self.is_train and len(caps) > 1:
            import random
            pool = caps if self.max_caps_per_image >= len(caps) else random.sample(caps, k=self.max_caps_per_image)
            cap = random.choice(pool)
        else:
            cap = caps[0]

        img_path = os.path.join(self.image_root, img_rel)
        image = Image.open(img_path).convert("RGB")
        x = self.tf(image).to(torch.float32)

        tok = self.tokenizer(
            cap,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        input_ids = tok["input_ids"][0].to(torch.int32)  # [77], competition expects int32
        return x, input_ids, cap

class CocoCaptionsRetrievalDataset(Dataset):
    """
    Reads COCO captions_*.json directly.
    Requires:
      coco_root/
        train2017/
        val2017/
        annotations/captions_train2017.json
    """
    def __init__(self,
                 coco_root: str,
                 split: str,
                 captions_json_rel: str,
                 tokenizer: CLIPTokenizer,
                 max_caps_per_image: int = 1,
                 is_train: bool = False,
                 augment: bool = True):
        self.coco_root = coco_root
        self.split = split  # 'train2017' or 'val2017'
        self.captions_json_path = os.path.join(coco_root, captions_json_rel)
        self.tokenizer = tokenizer
        self.max_caps_per_image = max(1, int(max_caps_per_image))
        self.is_train = is_train
        
        # Use augmentation only for training
        self.tf = _img_transform_train(augment) if is_train else _img_transform_eval()

        with open(self.captions_json_path, "r", encoding="utf-8") as f:
            coco = json.load(f)

        id2file = {img["id"]: img["file_name"] for img in coco.get("images", [])}
        caps = defaultdict(list)
        for ann in coco.get("annotations", []):
            caps[ann["image_id"]].append(ann.get("caption", ""))

        self.samples: List[Tuple[str, List[str]]] = []
        for image_id, file_name in id2file.items():
            c = caps.get(image_id, [])
            c = [x for x in c if isinstance(x, str) and x.strip()]
            if not c:
                continue
            img_rel = f"{self.split}/{file_name}"
            self.samples.append((img_rel, c))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_rel, caps = self.samples[idx]
        
        # Random caption selection during training for variety
        if self.is_train and len(caps) > 1:
            import random
            pool = caps if self.max_caps_per_image >= len(caps) else random.sample(caps, k=self.max_caps_per_image)
            cap = random.choice(pool)
        else:
            cap = caps[0]
            
        img_path = os.path.join(self.coco_root, img_rel)
        image = Image.open(img_path).convert("RGB")
        x = self.tf(image).to(torch.float32)

        tok = self.tokenizer(
            cap,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        input_ids = tok["input_ids"][0].to(torch.int32)
        return x, input_ids, cap

def make_datasets(cfg, tokenizer: CLIPTokenizer):
    mode = str(cfg.data.get("mode", "jsonl")).lower()
    max_caps = int(cfg.data.get("max_captions_per_image", 1))
    train_augment = bool(cfg.data.get("train_augment", True))
    
    if mode == "coco":
        coco_root = cfg.data.coco_root
        train_ds = CocoCaptionsRetrievalDataset(
            coco_root=coco_root,
            split=str(cfg.data.get("train_split", "train2017")),
            captions_json_rel=str(cfg.data.get("train_captions_json", "annotations/captions_train2017.json")),
            tokenizer=tokenizer,
            max_caps_per_image=max_caps,
            is_train=True,
            augment=train_augment,
        )
        val_ds = CocoCaptionsRetrievalDataset(
            coco_root=coco_root,
            split=str(cfg.data.get("val_split", "val2017")),
            captions_json_rel=str(cfg.data.get("val_captions_json", "annotations/captions_val2017.json")),
            tokenizer=tokenizer,
            max_caps_per_image=max_caps,
            is_train=False,
            augment=False,
        )
        return train_ds, val_ds

    # default jsonl mode
    train_ds = JsonlRetrievalDataset(
        image_root=str(cfg.data.get("image_root", ".")),
        jsonl_path=str(cfg.data.get("train_jsonl", "train.jsonl")),
        tokenizer=tokenizer,
        max_caps_per_image=max_caps,
        is_train=True,
        augment=train_augment,
    )
    val_ds = JsonlRetrievalDataset(
        image_root=str(cfg.data.get("image_root", ".")),
        jsonl_path=str(cfg.data.get("val_jsonl", "val.jsonl")),
        tokenizer=tokenizer,
        max_caps_per_image=max_caps,
        is_train=False,
        augment=False,
    )
    return train_ds, val_ds

def collate_fn(batch):
    imgs = torch.stack([b[0] for b in batch], dim=0).float()
    toks = torch.stack([b[1] for b in batch], dim=0)
    caps = [b[2] for b in batch]
    return imgs, toks, caps
