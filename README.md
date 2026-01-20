# ChronoFace - Aging Module (Part 2)

**Author:** Ouriel Mimoun  
**Project:** ChronoFace - Analysis and Temporal Synthesis of Faces  
**Technology:** CycleGAN (PyTorch)

---

## 📋 Description
This module implements the **aging and rejuvenation synthesis** functionality of the ChronoFace project. It uses a **CycleGAN** (Generative Adversarial Network) architecture to perform realistic age transformations on facial images.

Unlike traditional approaches, this model was **trained from scratch** on the UTKFace dataset, learning to dissociate facial structure (identity) from age attributes.

### Key Features:
* **Aging:** Transformation of a "Young" face to "Old". 
* **Rejuvenation:** Transformation of an "Old" face to "Young". 
* **Identity Preservation:** Uses *Cycle Consistency Loss* to ensure the person remains recognizable.

---

## 🛠️ Installation

### Prerequisites
* Python 3.8+
* PyTorch (CUDA support recommended)
* Libraries listed in `requirements.txt`

### Installing Dependencies
```bash
pip install -r requirements.txt
```

---
