from __future__ import annotations

import json
import os
import random
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
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

        # Train: (img_rel, image_id, captions)
        # Eval:  (img_rel, image_id, caption, ann_id)
        self.samples: List[Tuple[Any, ...]] = []
        image_counter = 0
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                img_rel = obj["image"]
                caps = obj.get("captions", [])
                if not isinstance(caps, list):
                    caps = [str(caps)]
                caps = [c for c in caps if isinstance(c, str) and c.strip()]
                if len(caps) == 0:
                    continue

                try:
                    image_id = int(os.path.basename(img_rel).split(".")[0])
                except (ValueError, IndexError):
                    image_id = image_counter
                image_counter += 1

                if self.is_train:
                    self.samples.append((img_rel, image_id, caps))
                else:
                    for ann_id, cap in enumerate(caps):
                        self.samples.append((img_rel, image_id, cap, ann_id))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        item = self.samples[idx]

        if self.is_train:
            img_rel, image_id, caps = item

            if len(caps) > 1:
                pool = caps if self.max_caps_per_image >= len(caps) else random.sample(caps, k=self.max_caps_per_image)
                cap = random.choice(pool)
            else:
                cap = caps[0]
            ann_id = None
        else:
            img_rel, image_id, cap, ann_id = item
            
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
        input_ids = tok["input_ids"][0].to(torch.int32)

        # [수정] 학습/평가 모두에서 meta 딕셔너리를 생성하여 반환합니다.
        meta = {
            "image_id": image_id,
            "ann_id": ann_id,
            "caption": cap,
            "img_rel": img_rel,
        }
        # 기존: return x, input_ids, cap
        # 변경: 아래와 같이 return하여 loss 계산 시 image_id를 쓸 수 있게 합니다.
        return x, input_ids, meta


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

        self.samples = []

        if self.is_train:
            # Train: image 단위로 samples 생성 (캡션들을 리스트로 묶음)
            caps = defaultdict(list)
            for ann in coco.get("annotations", []):
                caps[ann["image_id"]].append(ann.get("caption", ""))

            for image_id, file_name in id2file.items():
                c = caps.get(image_id, [])
                c = [x for x in c if isinstance(x, str) and x.strip()]
                if not c:
                    continue
                img_rel = f"{self.split}/{file_name}"
                # [수정] image_id를 튜플에 포함
                self.samples.append((img_rel, int(image_id), c))
        else:
            # Eval: 캡션 단위로 samples 생성 (image_id 포함)
            for ann in coco.get("annotations", []):
                image_id = ann.get("image_id", None)
                caption = ann.get("caption", "")
                ann_id = ann.get("id", None)

                if image_id is None:
                    continue
                if not (isinstance(caption, str) and caption.strip()):
                    continue

                file_name = id2file.get(image_id, None)
                if file_name is None:
                    continue

                img_rel = f"{self.split}/{file_name}"
                self.samples.append((img_rel, int(image_id), caption, int(ann_id) if ann_id is not None else None))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        item = self.samples[idx]

        if self.is_train:
            img_rel, image_id, caps = item

            if self.is_train and len(caps) > 1:
                pool = caps if self.max_caps_per_image >= len(caps) else random.sample(caps, k=self.max_caps_per_image)
                cap = random.choice(pool)
            else:
                cap = caps[0]

            ann_id = None
        else:
            img_rel, image_id, cap, ann_id = item

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

        meta = {
            "image_id": image_id,   # train이면 None일 수 있음
            "ann_id": ann_id,
            "caption": cap,
            "img_rel": img_rel,
        }
        return x, input_ids, meta


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
    metas = [b[2] for b in batch]   
    return imgs, toks, metas
