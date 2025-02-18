# Titano-N1

Titano-N1 is a personal AI project focused on learning and experimenting with image recognition using deep learning. It is built with TensorFlow and ResNet50V2, providing tools for training, dataset management, and image classification.

## About This Project
This project was created as a hands-on learning experience in AI and machine learning. It is not intended for production use but serves as a foundation for exploring deep learning concepts, data preprocessing, and model training.

## Features
- Train an image classification model using TensorFlow and ResNet50V2.
- Automate image dataset creation with a built-in downloader.
- Predict image classes with confidence scores.

## Installation
Ensure you have Python 3.8+ and install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### **1. Download Dataset**
Run the script to download images for training:
```bash
python download.py
```

### **2. Train the Model**
```bash
python train.py
```

### **3. Predict an Image**
```bash
python predict.py
```
You will be prompted to enter an image path, and the model will classify it.

## File Overview
- `train.py` - Trains the model using a dataset.
- `predict.py` - Runs image classification on an input image.
- `download.py` - Downloads images for dataset creation.

## Disclaimer
This project is purely for educational purposes and experimentation in AI. It is not optimized for professional or large-scale applications.

## License
This project is open-source under the MIT License.
