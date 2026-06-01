# Tourniquet Hemorrhage Control Classifier

Multimodal deep learning system for tourniquet effectiveness assessment. Fuses computer vision (placement classification) and force sensor time-series (tightness adequacy) to predict hemorrhage control outcomes.

## Architecture
- **Branch 1 (Vision):** Fine-tuned MobileNetV3-Small — classifies tourniquet placement from images
- **Branch 2 (Force):** 1D-CNN on flex sensor time-series — classifies tightness adequacy
- **Fusion head:** Concatenated embeddings → FC layers → binary verdict (controlled / not controlled)

## Dataset
- 240+ labeled image+sensor sample pairs across 4 conditions (correct/incorrect placement × adequate/inadequate force)
- Augmented to 1000+ effective samples

## Results
*In progress*

## Setup
```bash
git clone https://github.com/YOURUSERNAME/tourniquet-hemorrhage-control
cd tourniquet-hemorrhage-control
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Stack
Python · PyTorch · OpenCV · scikit-learn · Arduino (flex sensor)