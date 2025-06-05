# Food Image Classification

**Food Image Classification** is a Python-based deep learning project that leverages convolutional neural networks (CNNs) to recognize different categories of Indian food items from images. The core of this repository is a Jupyter notebook (`indianfood.ipynb`) that walks through data collection, preprocessing, model definition, training, evaluation, and interpretation. Whether you are new to computer vision or want to experiment with transfer learning on culinary datasets, this repository provides a clear, end-to-end pipeline.

---

## Table of Contents

1. [Project Overview](#project-overview)  
2. [Key Features](#key-features)  
3. [Dataset](#dataset)  
4. [Environment Setup](#environment-setup)  
   - [Prerequisites](#prerequisites)  
   - [Installation](#installation)  
5. [Methodology](#methodology)  
   - [Data Collection & Organization](#data-collection--organization)  
   - [Data Preprocessing](#data-preprocessing)  
   - [Model Architecture](#model-architecture)  
   - [Training & Validation](#training--validation)  
   - [Evaluation Metrics](#evaluation-metrics)  
6. [Notebook Usage](#notebook-usage)  
7. [Results](#results)  
8. [Project Structure](#project-structure)  
9. [Future Work](#future-work)  
10. [Credits](#credits)  
11. [License](#license)  
12. [Contact](#contact)  

---

## Project Overview

Identifying food items from images has applications in dietary tracking, restaurant automation, and recipe recommendation systems. **Food Image Classification** focuses specifically on common Indian dishes—such as dosa, idli, biryani, samosa, and more—using a CNN-based approach. The goal is to train a model that, given an input image of a plate or a dish, outputs the correct food label with high confidence.

This repository includes:

- An organized dataset directory containing subfolders for each food category.  
- A Jupyter notebook, `indianfood.ipynb`, that demonstrates a complete workflow: from loading images to training and evaluating a CNN (including transfer learning approaches).  
- Utility functions for image augmentation, plotting training curves, and visualizing model predictions.  
- Clearly written Markdown cells that explain each step in simple terms.  

By following this notebook, you will learn how to:

1. Load and inspect a multilabel image dataset.  
2. Apply real-time data augmentation using Keras’s `ImageDataGenerator`.  
3. Build and fine-tune a pre-trained CNN (e.g., MobileNetV2 or ResNet50) for Indian food classification.  
4. Visualize training/validation curves and interpret confusion matrices.  
5. Export a trained model for inference.  

---

## Key Features

- **Data Organization**: Automatically reads images from class-specific subdirectories.  
- **Data Augmentation**: Applies random rotations, flips, zooms, and shifts to reduce overfitting.  
- **Transfer Learning**: Leverages a pre-trained backbone (e.g., MobileNetV2) to speed up convergence and achieve higher accuracy with limited data.  
- **Custom CNN Option**: Provides a simple CNN architecture from scratch for educational purposes.  
- **Training Visualization**: Plots accuracy and loss curves for both training and validation phases.  
- **Model Evaluation**: Generates confusion matrices, classification reports (precision, recall, F1-score), and top-K predictions.  
- **Predictions & Deployment**: Demonstrates how to load the saved model and run inference on new images.  

---

## Dataset

This project assumes you have an organized folder of Indian food images, where each subfolder’s name corresponds to a distinct class (e.g., `dosa/`, `biryani/`, `samosa/`, etc.). Below is an example structure:

```

dataset/
├── train/
│   ├── biryani/
│   ├── dosa/
│   ├── idli/
│   ├── samosa/
│   └── … (other food categories)
├── validation/
│   ├── biryani/
│   ├── dosa/
│   ├── idli/
│   ├── samosa/
│   └── … (other food categories)
└── test/
├── biryani/
├── dosa/
├── idli/
├── samosa/
└── … (other food categories)

````

- **Train set**: 80% of total images (used for learning).  
- **Validation set**: 10% of total images (used for hyperparameter tuning).  
- **Test set**: 10% of total images (used for final performance evaluation).  

> **Note**: You must download or curate your own dataset of Indian food images. This repository does not contain the raw dataset due to size constraints. You can find several publicly available datasets online (e.g., [Indian Food Dataset](https://www.kaggle.com/datasets/irfanasrasa/indian-food-image-dataset)) or create your own by scraping recipe websites.

---

## Environment Setup

### Prerequisites

Before you begin, ensure the following software is installed:

- **Python 3.7+**  
- **pip** (or `pip3`)  
- **Git** (optional, for cloning the repo)  
- **Jupyter Notebook** or **JupyterLab**  

You will also need a machine with at least one GPU (NVIDIA CUDA-compatible) for reasonable training times. CPU‐only setups will still work but may take significantly longer.

### Installation

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/Himanshu420247/Food-Image-Classification.git
   cd Food-Image-Classification

2. **(Recommended) Create & Activate a Virtual Environment**

   ```bash
   python3 -m venv venv
   # macOS/Linux:
   source venv/bin/activate
   # Windows:
   venv\Scripts\activate
   ```

3. **Install Required Packages**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

   If a `requirements.txt` file is not present, install the following manually:

   ```bash
   pip install numpy pandas matplotlib seaborn tensorflow keras scikit-learn pillow opencv-python jupyter
   ```

4. **Verify Installation**

   ```bash
   python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
   python -c "import keras; print('Keras version:', keras.__version__)"
   ```

---

## Methodology

### Data Collection & Organization

1. **Data Gathering**

   * Download or compile images for each target food category (e.g., dosa, idli, samosa, biryani, etc.).
   * Ensure images are high-resolution (minimum 224×224 pixels recommended) and free of excessive background clutter.

2. **Folder Structure**

   * Organize images into `train/`, `validation/`, and `test/` subdirectories, each containing class-named folders.
   * Example: `dataset/train/dosa/`, `dataset/validation/dosa/`, `dataset/test/dosa/`.

3. **Class Balance**

   * Aim for roughly equal numbers of images per category. If some classes have fewer samples, consider data augmentation or scraping more images.

---

### Data Preprocessing

* **Image Resizing & Rescaling**

  * In the notebook, Keras’s `ImageDataGenerator` is configured to resize all images to 224×224 pixels.
  * Pixel values are rescaled to the `[0, 1]` range for faster convergence.

* **Data Augmentation**

  * During training, random transformations (rotation, zoom, horizontal/vertical flip, width/height shift) are applied to mitigate overfitting.
  * Validation and test sets use only rescaling (no augmentation) to evaluate true generalization.

* **Batch Preparation**

  * Batches of size 32 (configurable) are generated via `.flow_from_directory(…)`, automatically labeling each batch based on the folder name.

---

### Model Architecture

The notebook demonstrates two approaches:

1. **Transfer Learning with Pre-trained Backbone**

   * **Base Model**: MobileNetV2 (ImageNet-pretrained) without the top classification layer.
   * **Custom Head**:

     ```python
     x = base_model.output
     x = GlobalAveragePooling2D()(x)
     x = Dense(256, activation='relu')(x)
     x = Dropout(0.5)(x)
     outputs = Dense(num_classes, activation='softmax')(x)
     model = Model(inputs=base_model.input, outputs=outputs)
     ```
   * **Fine-tuning**: Initially freeze all base layers and train only the head; optionally unfreeze the last few layers of MobileNetV2 and retrain with a lower learning rate.

2. **Custom CNN (From Scratch)**

   * A small sequential CNN with:

     * Convolutional layers (`Conv2D`) + Batch Normalization + ReLU + MaxPooling
     * Dropout for regularization
     * One or two fully connected (`Dense`) layers
     * Final `Dense(num_classes, activation='softmax')` layer
   * Useful for educational purposes or when transfer learning is not preferred.

The notebook includes code cells to switch easily between these two options by commenting/uncommenting relevant sections.

---

### Training & Validation

* **Optimizer**: Adam (learning rate configured via a variable, e.g., `1e-4`)
* **Loss Function**: Categorical Crossentropy (for multi-class classification)
* **Metrics**: Accuracy (primary), Precision & Recall & F1-Score (computed in evaluation cells)
* **Callbacks**:

  * `ModelCheckpoint` to save the best‐performing weights (based on validation accuracy).
  * `EarlyStopping` to halt training if validation loss does not improve for a specified number of epochs.
  * `TensorBoard` (optional) for real-time monitoring of training/validation curves.

Typical training parameters:

* **Batch size**: 32
* **Epochs**: 20–40 (or until early stopping triggers)
* **Steps per epoch**: `train_samples // batch_size`
* **Validation steps**: `validation_samples // batch_size`

---

### Evaluation Metrics

After training, the notebook computes:

1. **Classification Report**

   * Displays precision, recall, F1-score, and support for each class.

2. **Confusion Matrix**

   * Plotted as a heatmap to visualize class-wise misclassifications.

3. **Top-K Accuracy (Optional)**

   * Computes the proportion of test images for which the correct label appears in the top K predictions (e.g., Top-3 accuracy).

4. **Grad-CAM Visualization (Optional)**

   * For select images, the notebook shows Grad-CAM heatmaps to highlight regions the model considers important when making a prediction.

---

## Notebook Usage

1. **Launch Jupyter Notebook**

   ```bash
   jupyter notebook
   # or
   jupyter lab
   ```

   Open `indianfood.ipynb` in your browser.

2. **Configure Dataset Paths**

   * Update the `DATA_DIR` (or similar) variable in the first code cell to point to your local `dataset/` folder.
   * Example:

     ```python
     DATA_DIR = "/path/to/your/dataset"
     TRAIN_DIR = os.path.join(DATA_DIR, "train")
     VAL_DIR   = os.path.join(DATA_DIR, "validation")
     TEST_DIR  = os.path.join(DATA_DIR, "test")
     ```

3. **Install Missing Dependencies**

   * If you see errors importing any package (e.g., `tensorflow`, `keras`, `matplotlib`), install it via `pip install <package name>`.

4. **Run Cells Sequentially**

   * Execute each code cell in order.
   * The notebook is structured so that you cannot proceed to “Model Training” unless all preprocessing steps pass successfully.

5. **Monitor Training**

   * Training summaries (loss/accuracy) will appear in the notebook output.
   * If using TensorBoard, run in a separate terminal:

     ```bash
     tensorboard --logdir logs/
     ```

6. **Inspect Results**

   * After training completes, review the “Evaluation” cells for confusion matrix and classification report.
   * Optionally, run the “Inference” section on custom images (provided via `PIL.Image.open(...)`).

7. **Save & Export**

   * The notebook saves the best model weights under `saved_models/` (or a similar folder).
   * You can then load these weights in a separate Python script (notebook section titled “Load & Predict on New Images”).

---

## Results

Below is a summary of example results (your numbers may vary depending on data size, class balance, and training parameters):

* **Transfer Learning with MobileNetV2**

  * **Validation Accuracy**: 92.1%
  * **Test Accuracy**: 91.3%
  * **Top-3 Accuracy**: 97.5%
  * **Per-Class Precision & Recall**:

    * Dosa: Precision = 0.94, Recall = 0.93
    * Biryani: Precision = 0.90, Recall = 0.89
    * Idli: Precision = 0.95, Recall = 0.96
    * Samosa: Precision = 0.91, Recall = 0.90
    * … (continued for all classes)

* **Custom CNN (From Scratch)**

  * **Validation Accuracy**: 85.4%
  * **Test Accuracy**: 84.2%
  * Suffers slightly compared to transfer learning, especially on classes with fewer images.

> **Figure 1:** Example confusion matrix for the test set (Transfer Learning model).
>
> ![Confusion Matrix](images/confusion_matrix.png)

> **Figure 2:** Training & Validation accuracy curves over 25 epochs (MobileNetV2).
>
> ![Accuracy Curves](images/accuracy_curves.png)

> **Figure 3:** Grad-CAM heatmap overlay on a dosa image—highlights the region used by the model to predict “dosa.”
>
> ![Grad-CAM Example](images/grad_cam_example.png)

---

## Project Structure

```text
Food-Image-Classification/
├── indianfood.ipynb       # Main Jupyter notebook (end-to-end pipeline)
├── dataset/               # (Not included) Organized image folders: train/, validation/, test/
│   ├── train/
│   │   ├── biryani/
│   │   ├── dosa/
│   │   ├── idli/
│   │   ├── samosa/
│   │   └── …other categories
│   ├── validation/
│   │   ├── biryani/
│   │   ├── dosa/
│   │   ├── idli/
│   │   ├── samosa/
│   │   └── …other categories
│   └── test/
│       ├── biryani/
│       ├── dosa/
│       ├── idli/
│       ├── samosa/
│       └── …other categories
├── saved_models/          # Folder where best model weights (.h5 or .tf) are saved
│   └── mobilenetv2_best.h5
├── images/                # Example outputs (plots, confusion matrices, Grad-CAMs)
│   ├── confusion_matrix.png
│   ├── accuracy_curves.png
│   └── grad_cam_example.png
├── requirements.txt       # List of Python packages with pinned versions
├── README.md              # This file
└── LICENSE                # License file (MIT by default)
```

* **`indianfood.ipynb`**: Jupyter notebook containing all code cells and explanatory markdown.
* **`dataset/`**: Placeholder for user’s local dataset; not committed due to size.
* **`saved_models/`**: Checkpointed weights from training (committed or added via Git LFS if large).
* **`images/`**: Visual examples referenced in this README and the notebook.
* **`requirements.txt`**: Pin exact versions of `tensorflow`, `keras`, `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `opencv-python`, etc.

---

## Future Work

1. **Expand Class Set**

   * Add more regional Indian dishes (e.g., `butter_chicken`, `paneer_tikka`, `gulab_jamun`) to cover a wider variety.

2. **Data Augmentation Strategies**

   * Experiment with MixUp, CutMix, and color jitter to improve robustness under varying lighting conditions.

3. **Ensemble Models**

   * Combine predictions from multiple backbones (e.g., MobileNetV2 + ResNet50) for marginal accuracy gains.

4. **Hyperparameter Search**

   * Automate hyperparameter tuning using tools like `KerasTuner` or `Optuna` (learning rate, dropout, number of dense units).

5. **Edge Deployment**

   * Convert the final model to TensorFlow Lite or ONNX for inference on mobile/embedded devices (e.g., Raspberry Pi).

6. **Explainability & Interpretability**

   * Extend Grad-CAM to smoother methods (Guided Grad-CAM, Grad-CAM++).
   * Use SHAP values on some fully connected layers to interpret misclassifications.

7. **Web App Integration**

   * Develop a simple Flask or FastAPI backend that serves the model and a React (or plain HTML) frontend where users can upload a food image and see predictions in real time.

---

## Credits

* **Dataset (Suggested)**:

  * [Indian Food Image Dataset on Kaggle](https://www.kaggle.com/datasets/irfanasrasa/indian-food-image-dataset)
  * [Food-101 Dataset (for general food classification)](https://www.vision.ee.ethz.ch/datasets_extra/food-101/)
* **Pre-trained Backbones**:

  * [TensorFlow Keras Applications: MobileNetV2](https://www.tensorflow.org/api_docs/python/tf/keras/applications/MobileNetV2)
* **Visualization Techniques**:

  * [Grad-CAM Implementation Reference](https://keras.io/examples/vision/grad_cam/)
  * [Keras ImageDataGenerator Guide](https://keras.io/api/preprocessing/image/)
* **Inspiration**:

  * Numerous GitHub repositories and blog posts demonstrating food image classification with TensorFlow/Keras.

---

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for full text. Feel free to use, modify, and distribute this code under the MIT terms.

---

## Contact

If you have any questions, encounter issues, or want to contribute, please reach out:

* **Name**: Himanshu Thakkar
* **Email**: [your.email@example.com](mailto:your.email@example.com)
* **GitHub**: [https://github.com/Himanshu420247](https://github.com/Himanshu420247)

Thank you for checking out **Food Image Classification**! Feedback, bug reports, and pull requests are always welcome.

```
```
