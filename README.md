
# Brain Tumor Detection

This project implements a convolutional neural network (CNN) for detecting brain tumors using image data. It leverages advanced deep learning techniques, including transfer learning and data augmentation, to enhance prediction accuracy. The goal is to improve the reliability of automated tumor detection for medical imaging.

---

## About the Dataset

### Domain:
- **Medical Imaging**: Focused on detecting brain tumors using MRI scans.
- **Purpose**: Enhance early diagnosis of brain tumors to aid in timely treatment.

### Dataset Overview:
- **Data Type**: MRI images.
- **Task**: Binary classification (Tumor/No Tumor).
- **Attributes**:
  - **Input**: Grayscale images resized to 128x128 pixels.
  - **Labels**: Two classes—Tumor and No Tumor.

### Dataset Statistics:
The dataset used for this study consists of **7,023 MRI images** sourced from three different datasets: **Figshare**, **SARTAJ**, and **Br35H**. These images are categorized into four distinct classes:
1. **Glioma**: Tumors arising from glial cells in the brain or spine.
2. **Meningioma**: Tumors originating in the meninges, the protective layers surrounding the brain and spinal cord.
3. **Pituitary**: Tumors located in the pituitary gland at the base of the brain.
4. **No Tumor**: MRI scans without any abnormalities.

For this study:
- **Training Images**: 5,620
- **Testing Images**: 1,403
  
## Dataset Samples

Below are some representative samples from the dataset, showcasing different tumor types and no tumor cases:

![Dataset Samples](https://github.com/radwanzoubi1/BrainTumorDetection/blob/main/Brain_Tumor_sample.JPG)

### Data Source:

- Brain Tumor MRI Dataset:
https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
---

## Project Objectives

1. **Deep Learning Model Implementation**:
   - Developed multiple CNN architectures using TensorFlow and Keras.
   - Integrated transfer learning with pre-trained ResNet models.

2. **Model Optimization**:
   - Applied data augmentation techniques like rotation, flipping, and zoom.
   - Experimented with hyperparameters such as learning rate, batch size, and activation functions.

3. **Evaluation and Comparison**:
   - Evaluated the performance of CNNs using precision, recall, F1-score, and accuracy.
   - Compared CNN results with a standard multi-layer perceptron (MLP).

4. **Visualization**:
   - Plotted training and validation loss/accuracy curves.
   - Displayed Grad-CAM visualizations to highlight regions influencing model predictions.

---

## Key Questions Explored

1. **How effective are CNNs for medical image classification?**
   - Explored the ability of CNNs to learn from MRI data for binary classification.

2. **What impact do data augmentation and transfer learning have?**
   - Analyzed their role in improving model performance and generalization.

3. **How does the CNN compare to simpler models?**
   - Evaluated the superiority of CNNs over traditional MLPs in handling image data.

---

### Data Preprocessing:
1. **Resizing**: All images resized to 128x128 for consistency.
2. **Normalization**: Pixel values normalized to the range [0, 1].
3. **Augmentation**: Used rotation, zoom, horizontal/vertical flip, and brightness adjustments.
4. **Splitting**: Stratified train-test split (80/20) to ensure balanced class distribution.

---

## Results

### Model Performance Metrics:
- **CNN (Best Model)**:
  - **Accuracy**: 89%
  - **Precision**: 87%
  - **Recall**: 91%
  - **F1-Score**: 89%
- **MLP**:
  - **Accuracy**: 78%
  - **Precision**: 75%
  - **Recall**: 76%
  - **F1-Score**: 75%

### Key Observations:
- Transfer learning with ResNet significantly improved accuracy and recall.
- Data augmentation enhanced model robustness to variations in input data.
- CNNs outperformed MLPs in all evaluation metrics, particularly in recall.

---

## Tools and Libraries Used

- **Python**: Programming language.
- **Jupyter Notebook**: Interactive environment for experimentation.
- **TensorFlow/Keras**: Deep learning framework.
- **OpenCV**: For image preprocessing.
- **Matplotlib/Seaborn**: Data visualization tools.

---

## Project Structure

```plaintext
.
├── BrainTumorDetection.ipynb   # Main Jupyter Notebook
├── README.md                   # Project documentation
├── Brain_Tumor_sample.JPG      # Dataset sampiles
```

---

## How to Run

### 1. Clone the Repository:
```bash
git clone https://github.com/radwanzoubi1/Brain-Tumor-Detection.git
cd Brain-Tumor-Detection
```

### 2. Install Dependencies:
```bash
pip install numpy pandas matplotlib opencv-python tensorflow
```

### 3. Run the Notebook:
- **Open Jupyter Notebook**:
```bash
jupyter notebook BrainTumorDetection.ipynb
```
- Execute all cells to reproduce the analysis.

---

## Future Work

- Explore multi-class classification for different tumor types.
- Integrate with real-time MRI systems for clinical use.
- Optimize the model further with more extensive datasets.


