# Lung Cancer Classification Using CNN–Conformer and Grad-CAM

## Overview

This project focuses on developing a deep learning-based approach for lung cancer classification using Chest CT scan images. It implements a custom CNN–Conformer architecture that combines Convolutional Neural Networks (CNN) for extracting local image features and Transformer-based self-attention for capturing global contextual information.

The model classifies CT scan images into four categories and uses Grad-CAM (Gradient-weighted Class Activation Mapping) to visualize important image regions that contribute to the model's predictions.

## Objectives

- Develop a deep learning model for multi-class lung cancer classification.
- Enhance CT scan images to improve image quality and feature representation.
- Combine CNN and Transformer architectures for local and global feature extraction.
- Evaluate model performance using multiple classification metrics.
- Apply Grad-CAM to interpret and visualize model predictions.

## Dataset

**Dataset:** Chest CT-Scan Images

The dataset contains four categories:

1. Adenocarcinoma
2. Large Cell Carcinoma
3. Squamous Cell Carcinoma
4. Normal

**Dataset Source:** [Kaggle – Chest CT-Scan Images](https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images)

## Methodology

### 1. Image Preprocessing and Enhancement
- Window-level intensity adjustment.
- Median filtering for noise reduction.
- Lung region segmentation using Otsu thresholding and morphological operations.
- Multi-scale CLAHE for contrast enhancement.
- Gaussian sharpening and intensity normalization.

### 2. Data Augmentation
- Random rotation.
- Translation.
- Scaling.
- Horizontal flipping.
- Brightness variation.

### 3. CNN–Conformer Architecture

The proposed custom architecture combines two branches:

**CNN Branch**
- Extracts local features from CT scan images.
- Uses convolutional layers and residual bottleneck blocks.

**Transformer Branch**
- Captures global contextual information.
- Uses multi-head self-attention and Transformer blocks.

**Feature Coupling**
- Facilitates information exchange between CNN feature maps and Transformer tokens.

The extracted features are combined and passed through fully connected layers for four-class classification.

### 4. Model Training
- Optimizer: AdamW
- Learning Rate: 0.0003
- Batch Size: 16
- Maximum Epochs: 150
- Loss Function: Weighted Cross Entropy with Label Smoothing
- Learning Rate Scheduler: Cosine Annealing
- Early Stopping: Enabled

### 5. Model Evaluation

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Specificity
- AUC-ROC
- Confusion Matrix
- Class-wise ROC Curves

### 6. Model Explainability using Grad-CAM

Grad-CAM is implemented to highlight image regions that influence the CNN branch's predictions.

It provides:
- Original CT scan images.
- Grad-CAM heatmaps.
- Heatmap overlays.
- Predicted class probabilities.

## Technologies Used

- Python
- PyTorch
- Torchvision
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- SciPy
- PIL

## Model Configuration

| Parameter | Value |
|---|---|
| Input Image Size | 224 × 224 |
| Number of Classes | 4 |
| CNN Dimension | 128 |
| Transformer Dimension | 192 |
| Attention Heads | 6 |
| Transformer Blocks | 4 |
| Dropout | 0.3 |
| Optimizer | AdamW |

## Results

The model performance is evaluated on a held-out test split using classification metrics and visual evaluation techniques.

**Evaluation metrics:**
- Overall Accuracy: To be updated
- Macro Precision: To be updated
- Macro Recall: To be updated
- Macro F1-Score: To be updated

Visual evaluation includes:
- Training and validation loss curves.
- Training and validation accuracy curves.
- Confusion matrix.
- Class-wise ROC curves.
- Grad-CAM visualizations.

*Replace the placeholders with your actual experimental results after running the notebook.*

## How to Run the Project

### Requirements

Install the required libraries:

```bash
pip install torch torchvision
pip install numpy matplotlib seaborn
pip install opencv-python scikit-learn scipy pillow tqdm
```

### Execution Steps

1. Clone this repository:

```bash
git clone https://github.com/YOUR-USERNAME/Lung-Cancer-Classification-CNN-Conformer.git
```

2. Open the notebook `cnn-conformer.ipynb` using Kaggle or Jupyter Notebook.

3. Download the Chest CT-Scan Images dataset from Kaggle.

4. If using Kaggle, add the dataset through **Add Input**.

5. Enable GPU acceleration (T4 or P100 recommended).

6. Run all notebook cells.

7. View the classification results, evaluation plots, and Grad-CAM visualizations.

## Project Structure

```text
Lung-Cancer-Classification-CNN-Conformer/
│
├── cnn-conformer.ipynb
├── README.md
├── requirements.txt
│
└── outputs/
    ├── training_history.png
    ├── confusion_matrix.png
    ├── roc_curves.png
    ├── grad_cam.png
    └── lung_cancer_conformer.pth
```

*The outputs folder contains generated results and model weights after execution. Add these files to the repository only if you choose to publish them.*

## Future Enhancements

- Experiment with different CNN and Transformer configurations.
- Improve model generalization using larger and more diverse datasets.
- Develop a web-based interface for image classification.
- Explore additional explainable AI techniques.
- Validate the model using independent external datasets.

## Disclaimer

This project is developed for academic and research purposes only. The model is experimental and is not intended to replace professional medical diagnosis or clinical decision-making.

## Author

**Shri Vardhine C**

M.Tech – Computer Science and Engineering
