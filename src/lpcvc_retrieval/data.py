from __future__ import annotations

import json
import os
import random
import hashlib
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import CLIPTokenizer


def build_tokenizer() -> CLIPTokenizer:
    return CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")


def _clean_source(source: Any) -> str:
    if not isinstance(source, str):
        return "unknown"
    source = source.strip()
    return source if source else "unknown"


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
        self._forced_caption_indices = None
        
        # Use augmentation only for training
        self.tf = _img_transform_train(augment) if is_train else _img_transform_eval()

        # Train: (img_rel, image_id, captions, source)
        # Eval:  (img_rel, image_id, caption, ann_id, source)
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

                source = _clean_source(obj.get("source"))

                if self.is_train:
                    self.samples.append((img_rel, image_id, caps, source))
                else:
                    for ann_id, cap in enumerate(caps):
                        self.samples.append((img_rel, image_id, cap, ann_id, source))

    def set_forced_caption_indices(self, caption_indices):
        """Force deterministic caption selection per sample index (offline distillation mode)."""
        if caption_indices is None:
            self._forced_caption_indices = None
            return
        if torch.is_tensor(caption_indices):
            caption_indices = caption_indices.tolist()
        self._forced_caption_indices = [int(x) for x in caption_indices]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        item = self.samples[idx]

        if self.is_train:
            img_rel, image_id, caps, source = item
            forced = self._forced_caption_indices

            if forced is not None and idx < len(forced):
                cap_idx = int(forced[idx])
                if cap_idx < 0:
                    cap_idx = 0
                if cap_idx >= len(caps):
                    cap_idx = len(caps) - 1
                cap = caps[cap_idx]
            else:
                if len(caps) > 1:
                    if self.max_caps_per_image >= len(caps):
                        pool_idx = list(range(len(caps)))
                    else:
                        pool_idx = random.sample(range(len(caps)), k=self.max_caps_per_image)
                    cap_idx = random.choice(pool_idx)
                    cap = caps[cap_idx]
                else:
                    cap_idx = 0
                    cap = caps[0]
            ann_id = None
        else:
            img_rel, image_id, cap, ann_id, source = item
            cap_idx = None
            
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
            "caption_idx": cap_idx,
            "img_rel": img_rel,
            "source": source,
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
        self._forced_caption_indices = None
        
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
                self.samples.append((img_rel, int(image_id), c, "coco"))
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
                self.samples.append(
                    (img_rel, int(image_id), caption, int(ann_id) if ann_id is not None else None, "coco")
                )

    def set_forced_caption_indices(self, caption_indices):
        """Force deterministic caption selection per sample index (offline distillation mode)."""
        if caption_indices is None:
            self._forced_caption_indices = None
            return
        if torch.is_tensor(caption_indices):
            caption_indices = caption_indices.tolist()
        self._forced_caption_indices = [int(x) for x in caption_indices]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        item = self.samples[idx]

        if self.is_train:
            img_rel, image_id, caps, source = item
            forced = self._forced_caption_indices

            if forced is not None and idx < len(forced):
                cap_idx = int(forced[idx])
                if cap_idx < 0:
                    cap_idx = 0
                if cap_idx >= len(caps):
                    cap_idx = len(caps) - 1
                cap = caps[cap_idx]
            else:
                if self.is_train and len(caps) > 1:
                    if self.max_caps_per_image >= len(caps):
                        pool_idx = list(range(len(caps)))
                    else:
                        pool_idx = random.sample(range(len(caps)), k=self.max_caps_per_image)
                    cap_idx = random.choice(pool_idx)
                    cap = caps[cap_idx]
                else:
                    cap_idx = 0
                    cap = caps[0]

            ann_id = None
        else:
            img_rel, image_id, cap, ann_id, source = item
            cap_idx = None

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
            "caption_idx": cap_idx,
            "img_rel": img_rel,
            "source": source,
        }
        return x, input_ids, meta


class OfflineFeatureDataset(Dataset):
    """
    Wrapper that loads pre-extracted Teacher embeddings alongside the base dataset.
    """
    def __init__(self, base_dataset: Dataset, feature_dir: str, split: str = "train"):
        self.base_dataset = base_dataset
        self.feature_dir = feature_dir
        self.split = split

        # Fingerprint current dataset order/content to detect stale or shuffled offline features.
        self.current_dataset_fingerprint = None
        if hasattr(self.base_dataset, "samples"):
            hasher = hashlib.sha1()
            for s in self.base_dataset.samples:
                # train: (img_rel, image_id, caps), eval: (img_rel, image_id, caption, ann_id)
                try:
                    img_rel = str(s[0])
                    image_id = int(s[1])
                except Exception:
                    img_rel = str(s[0]) if len(s) > 0 else ""
                    image_id = -1
                hasher.update(f"{image_id}|{img_rel}\n".encode("utf-8"))
            self.current_dataset_fingerprint = hasher.hexdigest()
        
        # Load all teacher features
        self.teacher_features = []
        t_idx = 0
        while True:
            path = os.path.join(feature_dir, f"teacher_{t_idx}_{split}.pt")
            if not os.path.exists(path):
                break
            data = torch.load(path, map_location="cpu", weights_only=True)
            self.teacher_features.append(data)
            print(f"[Offline] Loaded {path} | img: {data['img_embs'].shape}, txt: {data['txt_embs'].shape}")
            t_idx += 1
        
        if len(self.teacher_features) == 0:
            raise FileNotFoundError(
                f"[Offline] No teacher feature files found in {feature_dir} for split '{split}'. "
                f"Run 'python scripts/extract_features.py' first."
            )
        
        # Validate size match
        expected_len = len(self.base_dataset)
        shared_caption_indices = None
        shared_fingerprint = None
        for t_idx, tf in enumerate(self.teacher_features):
            actual_len = tf["img_embs"].shape[0]
            if actual_len != expected_len:
                raise ValueError(
                    f"[Offline] Size mismatch: dataset has {expected_len} samples "
                    f"but teacher_{t_idx}_{split}.pt has {actual_len}. "
                    f"Re-run extract_features.py."
                )
            if tf["img_embs"].shape != tf["txt_embs"].shape:
                raise ValueError(
                    f"[Offline] Dimension mismatch in teacher_{t_idx}_{split}.pt: "
                    f"img_embs={tf['img_embs'].shape}, txt_embs={tf['txt_embs'].shape}"
                )
            if "sample_count" in tf and int(tf["sample_count"]) != expected_len:
                raise ValueError(
                    f"[Offline] sample_count mismatch in teacher_{t_idx}_{split}.pt: "
                    f"{int(tf['sample_count'])} != {expected_len}"
                )
            if "dataset_fingerprint" in tf:
                fp = str(tf["dataset_fingerprint"])
                if shared_fingerprint is None:
                    shared_fingerprint = fp
                elif shared_fingerprint != fp:
                    raise ValueError(
                        f"[Offline] dataset_fingerprint mismatch across teachers: "
                        f"{shared_fingerprint} vs {fp}"
                    )
                if (
                    self.current_dataset_fingerprint is not None
                    and fp != self.current_dataset_fingerprint
                ):
                    raise ValueError(
                        "[Offline] dataset_fingerprint mismatch between current dataset and "
                        f"feature file teacher_{t_idx}_{split}.pt: {self.current_dataset_fingerprint} vs {fp}. "
                        "Dataset order/content changed after extraction. Re-run scripts/extract_features.py."
                    )
            if "caption_indices" in tf:
                ci = tf["caption_indices"].to(torch.long)
                if ci.shape[0] != expected_len:
                    raise ValueError(
                        f"[Offline] caption_indices length mismatch in teacher_{t_idx}_{split}.pt: "
                        f"{ci.shape[0]} != {expected_len}"
                    )
                if shared_caption_indices is None:
                    shared_caption_indices = ci
                elif not torch.equal(shared_caption_indices, ci):
                    raise ValueError(
                        f"[Offline] caption_indices mismatch across teachers in split '{split}'."
                    )
        
        print(f"[Offline] {len(self.teacher_features)} teacher(s) loaded for '{split}' ({expected_len} samples)")
        if shared_fingerprint is not None:
            print(f"[Offline] dataset_fingerprint={shared_fingerprint}")

        # In train split, force the same caption_idx that was used during feature extraction.
        if split == "train" and shared_caption_indices is not None:
            if hasattr(self.base_dataset, "set_forced_caption_indices"):
                self.base_dataset.set_forced_caption_indices(shared_caption_indices)
                print(f"[Offline] Forced caption indices injected for split '{split}'.")
            else:
                print(
                    "[Offline] Warning: base dataset does not support forced caption indices; "
                    "offline text distillation may be noisy."
                )
        elif split == "train":
            print(
                "[Offline] Warning: caption_indices not found in feature files. "
                "Re-run scripts/extract_features.py to avoid caption mismatch noise."
            )
    
    def __len__(self) -> int:
        return len(self.base_dataset)
    
    def __getitem__(self, idx: int):
        base_item = self.base_dataset[idx]
        
        teacher_embs = []
        for tf in self.teacher_features:
            teacher_embs.append((
                tf["img_embs"][idx],
                tf["txt_embs"][idx],
            ))
        
        return (*base_item, teacher_embs)


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
    else:
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
    
    distill_cfg = cfg.get("distill", {})
    offline_feature_dir = distill_cfg.get("offline_feature_dir", None) if distill_cfg else None
    
    if offline_feature_dir and os.path.isdir(str(offline_feature_dir)):
        offline_feature_dir = str(offline_feature_dir)
        print(f"[Offline] Wrapping train dataset with pre-extracted features from: {offline_feature_dir}")
        train_ds = OfflineFeatureDataset(train_ds, offline_feature_dir, split="train")
    
    return train_ds, val_ds


def collate_fn(batch):
    imgs = torch.stack([b[0] for b in batch], dim=0).float()
    toks = torch.stack([b[1] for b in batch], dim=0)
    metas = [b[2] for b in batch]
    
    if len(batch[0]) > 3:
        num_teachers = len(batch[0][3])
        teacher_batch = []
        for t_idx in range(num_teachers):
            t_imgs = torch.stack([b[3][t_idx][0] for b in batch], dim=0)
            t_txts = torch.stack([b[3][t_idx][1] for b in batch], dim=0)
            teacher_batch.append((t_imgs, t_txts))
        return imgs, toks, metas, teacher_batch
    
    return imgs, toks, metas
