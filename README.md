# Face Mask Detection

## Overview
This project detects whether a person is wearing a face mask or not using deep learning. It utilizes a dataset from Kaggle and employs image classification techniques to build an effective mask detection model.

## Dataset
- **Source**: [Face Mask Dataset](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset)
- **Categories**:
  - `with_mask`
  - `without_mask`
- The dataset consists of labeled images of faces with and without masks.

## Dependencies
To run this project, install the required dependencies:
```bash
pip install numpy pandas matplotlib opencv-python tensorflow keras scikit-learn
```

## Setup & Execution
1. Clone or download the repository.
2. Download the dataset from Kaggle and extract it.
3. Run the Jupyter Notebook:
   ```bash
   jupyter notebook Face_Mask_Detection.ipynb
   ```
4. Follow the notebook cells to preprocess data, train the model, and evaluate performance.

## Model Architecture
- **Preprocessing**:
  - Image resizing and normalization.
  - Splitting into training and testing sets.
- **Deep Learning Model**:
  - Uses **Convolutional Neural Networks (CNNs)** for classification.
  - Applies data augmentation for better generalization.
- **Loss Function**: Categorical Crossentropy.
- **Optimizer**: Adam.

## Results
- The model predicts whether a person is wearing a mask with high accuracy.
- Performance is evaluated using metrics such as accuracy and loss curves.

## Future Improvements
- Use transfer learning (e.g., MobileNet, VGG16) for better accuracy.
- Deploy the model as a **real-time mask detection system** using OpenCV.
- Implement a web application for live face mask detection.
