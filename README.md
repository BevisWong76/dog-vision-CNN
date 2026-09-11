# Dog Vision — Dog Breed Classification with EfficientNetV2 & ONNX

## Overview
Dog Vision CNN is a lightweight, end-to-end computer vision web application designed to classify dog breeds from uploaded images in real time. Built on a fine-tuned EfficientNetV2B3 architecture trained on the Stanford Dogs dataset, the core model reaches high-accuracy predictions across 120 distinct breeds. To optimize performance for cloud deployment, the model was converted to the ONNX (Open Neural Network Exchange) format for execution via ONNX Runtime, drastically reducing memory overhead and inference latency while serving predictions through a clean Streamlit user interface.


> For in-depth details on Exploratory Data Analysis (EDA), Model Selection and Tuning, and Results, please refer to the [Full Technical Report](dog_vision_report.md).

---

## Demo

![Streamlit App Demo](./assets/demo.png)

**Try the Live Interactive Web App:** [Dog Breed Classifier on Streamlit Cloud](https://dog-vision-cnn-ts2zeabpjnpgre2qivmnxg.streamlit.app)

---

## Key Features & Highlights

* **High-Accuracy Fine-Tuned Model:** Utilizes a two-stage transfer learning pipeline with EfficientNetV2B3 to achieve over 93% validation accuracy across 120 dog breeds.

* **Lightweight ONNX Inference Engine:** Deploys an optimized ONNX model instead of heavy deep learning frameworks, dropping system RAM consumption to <50MB for ultra-fast predictions.

* **Interactive Streamlit Web App:** Features a responsive user interface allowing instant image uploads, confidence score metrics, top-3 prediction breakdowns, and visual probability displays.

---

## Key Results

<p align="center">
  <img src="plots/evaluation/01_training_history_curves.png" width="90%">
</p>

* **High Classification Performance:** Achieved ~**93.8% validation accuracy** and reduced cross-entropy validation loss to **~0.18** across 120 breeds.
* **Effective Two-Stage Fine-Tuning:** 
  * **Stage 1 (Feature Extraction, Epochs 1–8):** The frozen EfficientNetV2B3 backbone rapidly stabilized, reaching ~92% training accuracy without overfitting.
  * **Stage 2 (Fine-Tuning, Epochs 9–14):** Unfreezing top layers successfully eliminated the initial training/validation gap, converging training and validation metrics smoothly.

<p align="center">
  <img src="plots/evaluation/02_top_bottom_breeds_performance.png" width="90%">
</p>

* **Top-Performing Breeds (100% Recall):** Breeds with distinct visual features—such as *Irish Setter, Pembroke, Groenendael, English Springer, Gordon Setter, English Setter, German Short-Haired Pointer, Komondor, Pug,* and *Leonberg*—achieved **perfect 1.0 recall** on the test set.
* **Challenging Breeds:** Visual similarities caused lower accuracy (~65%–70% recall) in closely related pairs or highly variable breeds, such as *Border Collie*, *Eskimo Dog*, *Staffordshire Bull Terrier*, and *Siberian Husky*.

---

## Tech Stack

* **Language:** Python `3.12.3`
* **Data Processing & Analysis:** TensorFlow Datasets, Pandas, NumPy
* **Machine Learning:** TensorFlow, Scikit-Learn
* **Visualization:** Matplotlib, Seaborn
* **Model Persistence:** tf2onnx
* **Web Framework:** Streamlit, ONNX Runtime

---

## Project Structure

```text
dog-vision-CNN/
├── assets/                  # Images for README
├── dog_vision.ipynb         # Complete Transfer Learning pipeline
├── dog_vision_report.md     # Comprehensive technical report
├── app.py                   # Interactive Streamlit web application
├── requirements.txt         # Python dependencies for the web app
├── requirements-dev.txt     # Python dependencies for the notebook
└── README.md                # Project documentation
```

The execution pipeline automatically generates and manages the following runtime directories:

```text
├── data/               # Dataset downloaded from TFDS
├── models/             # Stores trained model files
├── results/            # Stores the model classification report
└── plots/              # Generated visualizations
    ├── eda/            # Exploratory Data Analysis plots
    └── evaluation/     # Visualizations of model evaluation
```

---

## How to Run

First, clone the repository and navigate into the project directory:
```bash
git clone https://github.com/BevisWong76/dog-vision-CNN.git
cd dog-vision-CNN
```

### Option 1: Docker Compose (Recommended for Windows & Linux with NVIDIA GPU)

Ideal for environments with NVIDIA GPUs. It uses NVIDIA's official TensorFlow container (`nvcr.io/nvidia/tensorflow:25.02-tf2-py3`), ensuring zero CUDA driver setup on the host machine and instant reproducibility.

**Prerequisites:** [Docker Desktop / Engine](https://www.docker.com/) and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

1. Start the containerized Jupyter Lab server:
   ```bash
   docker compose up -d
   ```
2. Access Jupyter Lab in your browser at `http://localhost:8888`.

---

### Option 2: Native Virtual Environment (Recommended for macOS & CPU/Native Linux)

Best for Apple Silicon Macs (M-series) to leverage native Metal/MPS acceleration, or Linux/Windows setups running lightweight CPU inference.

> Note: We recommend [`uv`](https://github.com/astral-sh/uv) for ultra-fast dependency management.

1. **Create virtual environment with Python 3.12 via uv:**
   ```bash
   pip install uv
   uv venv --python 3.12
   ```

2. **Activate the environment:**
   * **Windows (PowerShell):** `.venv\Scripts\Activate.ps1`
   * **Windows (CMD):** `.venv\Scripts\activate.bat`
   * **macOS / Linux:** `source .venv/bin/activate`

3. **Install dependencies:**

   * **For Web App Inference only (Lightweight ONNX Runtime):**
     ```bash
     uv pip install -r requirements.txt
     ```
   * **For Model Training & Notebook Exploration (Full Dev Suite):**
     ```bash
     uv pip install -r requirements-dev.txt
     ```

4. **Launch Application / Notebook:**
   * **Run Streamlit Web App:**
     ```bash
     streamlit run app.py
     ```
   * **Launch Jupyter Lab:**
     ```bash
     jupyter lab dog_vision.ipynb
     ```