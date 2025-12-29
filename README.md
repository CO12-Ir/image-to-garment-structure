# Image-to-Garment-Structure

Multi-label image classification for **garment structure decomposition**.

This project predicts **collar, sleeve, and hem attributes** from a single clothing image,
designed as the first stage of an image-to-garment modeling pipeline.

---

## Overview

- **Task**: Multi-label classification
- **Input**: RGB clothing image
- **Output**: Structured garment attributes
- **Backbone**: ResNet-18 (ImageNet pretrained)
- **Loss**: BCEWithLogitsLoss (class-balanced)

---

## Label Groups

- Collars: v-neck, round neck, shirt collar, turtleneck, hoodie hood, etc.
- Sleeves: sleeveless, short / long sleeve, puffy variants
- Hems: above belly, hip length, knee length, calf length

Each image may activate multiple labels.

---

## Repository Structure

.
├── dataset.py
├── model.py
├── main.py
└── README.md


---

## Dataset Format

Raw data is not included.

Expected format:

image.jpg
image.jpg.txt # comma-separated structure tags


---

## Training

```bash
python main.py
```

Includes:
Train / validation split
Class imbalance handling
Early stopping
Per-label F1 evaluation

Single-image inference logic is included in `model.py` for prototyping purposes.


## Notes
This repository focuses on structural semantics rather than fashion aesthetics,
and is intended as a modular component for downstream garment modeling systems.