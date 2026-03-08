# Third-Party Licenses and Notices

This repository's source code is licensed under Apache License 2.0 (see `LICENSE`).

Third-party models, weights, datasets, and frameworks used by this project are governed by their own licenses or model-card terms. You must follow the original upstream terms when downloading, using, redistributing, or publishing outputs.

## 1) Current Default Models and Frameworks

| Component | Source | License / Terms (check official page) |
|---|---|---|
| MobileNetV4 Hybrid Large image student | [timm/mobilenetv4_hybrid_large.e600_r384_in1k](https://huggingface.co/timm/mobilenetv4_hybrid_large.e600_r384_in1k) | Apache-2.0 |
| EVA-CLIP text student | [QuanSun/EVA-CLIP](https://huggingface.co/QuanSun/EVA-CLIP) | MIT |
| SigLIP2 teacher | [timm/ViT-gopt-16-SigLIP2-256](https://huggingface.co/timm/ViT-gopt-16-SigLIP2-256) | Apache-2.0 |
| PE-Core teacher | [facebook/PE-Core-G14-448](https://huggingface.co/facebook/PE-Core-G14-448) | Apache-2.0 |
| timm | [huggingface/pytorch-image-models](https://github.com/huggingface/pytorch-image-models) | Apache-2.0 |
| OpenCLIP | [mlfoundations/open_clip](https://github.com/mlfoundations/open_clip) | MIT |

## 2) Datasets

| Dataset | Source | License / Terms |
|---|---|---|
| COCO 2017 | [cocodataset.org](https://cocodataset.org/) | Follow COCO terms and underlying image licenses/attribution requirements |
| Flickr30k | [Kaggle mirror](https://www.kaggle.com/datasets/adityajn105/flickr30k) | Follow dataset provider terms and original image licenses |
| Open Images V7 | [Google Open Images](https://storage.googleapis.com/openimages/web/index.html) | Follow Open Images terms and attribution requirements |
| WIT | [google-research-datasets/wit](https://github.com/google-research-datasets/wit) | Follow WIT terms and source image/content licenses |

## 3) Practical Notes

- Do not assume all artifacts are Apache-2.0 just because this repository is Apache-2.0.
- Before release or commercial use, re-check each upstream model card, repository LICENSE, and dataset terms.
- If required, include attribution and notices in papers, demos, benchmarks, or product documentation.
- If you switch student or teacher models in another branch, update this file together with `README.md` and `docs/PROJECT_GUIDE.md`.

This file is a practical engineering notice and is not legal advice.
