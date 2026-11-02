# ASL Alphabet Recognition with a Vision Transformer

Fine-tunes **ViT-Base/16** (ImageNet-21k pretrained, 85.8M parameters) to recognise all **29 American Sign Language classes**: A–Z, `space`, `del` and `nothing`. Two-phase training reaches **96.7% validation accuracy on 8,700 held-out images**, up from 73.7% with only the classifier head trained.

## Results

Numbers come from the Hugging Face `Trainer` logs saved in [`notebooks/asl_vit.ipynb`](notebooks/asl_vit.ipynb). Training ran on a single Colab T4.

| Phase | Trainable parameters | Epochs run | Validation accuracy |
|---|---|---|---|
| 1. Classifier head only (backbone frozen) | 22,301 | 5 (33 min) | 61.3% → **73.7%** |
| 2. Head + last 2 transformer blocks | 14,198,045 | 4 of 10 (34 min) | 90.7% → 94.3% → 96.0% → **96.7%** |

Validation loss fell from 2.18 to 0.84 over the two phases. The phase-2 run was stopped at epoch 4.9 of 10 while accuracy was still rising, so 96.7% is the last completed evaluation, not a converged ceiling. The notebook's later evaluation cells (per-class report, confusion matrix) were not executed in this run. Run `scripts/evaluate.py` to produce them.

## Data

The [ASL Alphabet dataset](https://www.kaggle.com/datasets/debashishsau/aslamerican-sign-language-aplhabet-dataset) on Kaggle (CC0) holds **223,074 images** across 29 classes, mostly 7,000–8,500 per class (`nothing` has 3,030).

* **Sampling:** 1,500 images per class, giving 43,500 images, with a sorted, fixed label mapping so class indices never depend on filesystem order.
* **Split:** stratified 80/20, giving **34,800 training and 8,700 validation images** (seed 42).

## Method

| Setting | Value | Why |
|---|---|---|
| Backbone | `google/vit-base-patch16-224-in21k` | Strong ImageNet-21k features; 16×16 patch attention |
| Phase 1 | Head only, LR 5e-4, 5 epochs, warm-up 100 steps | Fit the new 29-class head without disturbing the backbone |
| Phase 2 | Last 2 blocks + head, LR 2e-5, up to 10 epochs, warm-up 200 steps | Adapt high-level features at a low LR to avoid catastrophic forgetting |
| Effective batch | 64 (32×2, then 16×4 with gradient accumulation) | Stable gradients within T4 memory |
| Schedule | Cosine with warm-up | Smooth convergence |
| Regularisation | Label smoothing 0.1, weight decay 0.05 then 0.01, early stopping (patience 2) | Avoids over-confident, overfit predictions |
| Precision | fp16 mixed precision | About 2× faster on T4 |
| Augmentation | Random resized crop (0.85–1.0), mild horizontal flip (p=0.3), ±10° rotation, 5% translation, colour jitter | Robustness to hand position and lighting. Flip is kept mild because some signs are not symmetric |

## Quick start

```bash
pip install -r requirements.txt
kaggle datasets download -d debashishsau/aslamerican-sign-language-aplhabet-dataset -p data --unzip
python scripts/train.py --train-dir data/ASL_Alphabet_Dataset/asl_alphabet_train --out artifacts/
python scripts/evaluate.py --model-dir artifacts/asl_vit_hf_model --split artifacts/val_split.csv --out eval/
python scripts/predict.py --model-dir artifacts/asl_vit_hf_model --image my_hand.jpg --topk 5
pytest                                   # data-pipeline tests (pandas + scikit-learn only)
```

`train.py` saves the exact train and validation splits as CSVs, `label_mapping.json`, and a Hugging Face model folder that loads with `from_pretrained`. `evaluate.py` writes `metrics.json` (accuracy, weighted precision, recall and F1, per-class report) and a row-normalised confusion matrix.

## Repository layout

```
notebooks/asl_vit.ipynb      end-to-end Colab run with saved outputs (the run of record)
src/asl_vit/data.py          indexing, fixed label mapping, per-class sampling, stratified split
src/asl_vit/dataset.py       PyTorch dataset and augmentation pipelines
scripts/train.py             two-phase fine-tuning with the Hugging Face Trainer
scripts/evaluate.py          metrics JSON and confusion matrix on a saved split
scripts/predict.py           top-k letter prediction for images
tests/test_data.py           unit tests for the data pipeline
```

## Limitations

* The validation images come from the same source as training (similar signers, backgrounds and lighting). Expect lower accuracy on webcam footage of new signers until the model is evaluated and fine-tuned on such data.
* Static letters only. J and Z are motion signs in ASL and are represented here by single frames.
* Single-image classification: there is no hand detector or temporal smoothing. Add a hand-cropping stage (for example MediaPipe Hands) for live video.

## License

MIT for this code. The dataset is CC0, and the ViT weights keep their Hugging Face model license.
