# Third-Party Licenses and Notices

This repository's source code is licensed under MIT (see `LICENSE`).

Third-party models, weights, datasets, and frameworks used by this project are governed by their own licenses/terms. You must follow the original terms when downloading, using, redistributing, or publishing outputs.

## 1) Models and Frameworks

| Component | Source | License / Terms (check official page) |
|---|---|---|
| MobileCLIP2 (student) code | https://github.com/apple/ml-mobileclip | MIT (repository code) |
| MobileCLIP2 weights | https://huggingface.co/apple/MobileCLIP2-S0 | Apple model card terms (Apple ML Research TOU/license in model card) |
| SigLIP2 teacher | https://huggingface.co/timm/ViT-gopt-16-SigLIP2-256 | Apache-2.0 (model card) |
| PE-Core teacher | https://huggingface.co/facebook/PE-Core-G14-448 | Apache-2.0 (model card) |
| OpenCLIP | https://github.com/mlfoundations/open_clip | MIT |

## 2) Datasets

| Dataset | Source | License / Terms |
|---|---|---|
| COCO 2017 | https://cocodataset.org/ | Follow COCO terms and underlying image licenses/attribution requirements |
| Flickr30k | https://www.kaggle.com/datasets/adityajn105/flickr30k | Follow dataset provider terms and original image licenses |
| Open Images V7 | https://storage.googleapis.com/openimages/web/index.html | Follow Open Images terms and attribution requirements |
| WIT | https://github.com/google-research-datasets/wit | Follow WIT terms and source image/content licenses |

## 3) Usage Notes

- Do not assume all artifacts are MIT just because this repository is MIT.
- Before release or commercial use, re-check each model card, repository LICENSE, and dataset terms.
- If required, include attribution and notices in your paper, demo, or product documentation.

This file is a practical notice for engineering use and is not legal advice.
