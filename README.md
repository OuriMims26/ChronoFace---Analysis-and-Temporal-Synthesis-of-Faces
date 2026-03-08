# ChronoFace - Aging Simulation Module (Part 2) 👴👶

**Authors:** Ouriel Mimoun & David Ceylon  
**Project:** ChronoFace - Analysis and Temporal Synthesis of Faces  
**Core Tech:** CycleGAN (PyTorch)

---

## 📋 Overview
This module implements the **aging and rejuvenation synthesis** functionality of the ChronoFace project. It leverages a **CycleGAN** (Generative Adversarial Network) architecture to perform realistic, unpaired image-to-image translation on facial images.

Unlike traditional supervised approaches, this model was **trained from scratch** on the UTKFace dataset, effectively learning to dissociate facial structure (identity) from temporal age attributes.

### Key Features:
* **👴 Aging:** Realistic transformation from a "Young" domain to an "Old" domain. 
* **👶 Rejuvenation:** Inverse transformation from "Old" to "Young". 
* **Identity Preservation:** Uses *Cycle Consistency Loss* to ensure the subject remains recognizable after the transformation.

---

## 🧠 Model Architecture
The system utilizes a standard CycleGAN architecture, which allows for learning transformations between two domains without requiring paired training data.

### Generator (ResNet-based)
- **Initial Convolution**: 7x7 filters with reflection padding and Instance Normalization.
- **Downsampling**: Two convolution layers with stride 2.
- **Residual Blocks**: 9 residual blocks to maintain global feature consistency.
- **Upsampling**: Two upsampling layers with scale factor 2 (transposed convolutions/upsampling).
- **Output Layer**: Tanh activation producing pixels in the normalized `[-1, 1]` range.

---

## 📂 Required Files
To run the simulations, the following weight files must be present in the project root:
- **`best_G_AB.pth`**: Generator weights for **Aging** (Young -> Old).
- **`G_BA_45.pth`**: Generator weights for **Rejuvenation** (Old -> Young).

📥 **Download all weights here**: [Google Drive - ChronoFace Weights](https://drive.google.com/drive/folders/10_EO8AE7Fqk7aJcVDLWeXDiSA9fClN15?usp=sharing) (Please place them in the project root folder).

---

## 🛠️ Installation & Setup

### 1. Prerequisites
- Python 3.9+
- A virtual environment (recommended)
- PyTorch (The GAN module defaults to **CPU** for maximum compatibility cross-platform).

### 2. Environment Setup
```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Mac/Linux
# .venv\Scripts\activate   # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run
The Aging module is fully integrated into the ChronoFace dashboard:

```bash
streamlit run app.py
```

1. Upload an image.
2. Select the effect in the sidebar/options: **👴 Aging** or **👶 Rejuvenation**.
3. Click on **"2. Transform ✨"**.
4. Compare the original and transformed results side-by-side.

---

## 📊 Technical Specifications
- **Input Resolution**: Fixed at 256x256 pixels.
- **Normalization**: Images are normalized to `[-1, 1]` range for GAN processing.
- **Training**: Trained on UTKFace using unpaired domain translation.
