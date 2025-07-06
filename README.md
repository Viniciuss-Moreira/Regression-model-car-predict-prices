[🇧🇷 Ver em Português](./README.pt-br.md)
---
#  Regression Model: A Deep Learning Car Price Predictor

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=for-the-badge&logo=matplotlib&logoColor=white)](https://matplotlib.org/)
[![Gradio](https://img.shields.io/badge/Gradio-4.29-FF7622?style=for-the-badge&logo=gradio&logoColor=white)](https://www.gradio.app/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-yellow?style=for-the-badge)](https://huggingface.co/spaces)

**Live Application Link:** [**➡️ Click Here to Try the Live Demo**](https://vinimoreira-regression-prices-cars-b4-it.hf.space)
---
* **[🔬 View the Full Training Notebook on GitHub](./notebooks/notebook.ipynb)** * **[🚀 Open and Run Interactively in Google Colab](https://colab.research.google.com/drive/1r--GE8Np_mnvnmFqMAOKteCkH2Pe0HwE?usp=sharing)**
---
This project showcases a complete Machine Learning workflow, from data analysis to fine-tuning and deploying a deep learning model for tabular regression. It's an interactive application to predict the selling price of used cars in the Brazilian market.

![App Demo](./img/demo.gif)
---

## 📖 Project Description

This project introduces FipeFinder AI, a machine learning application designed to predict the selling price of used cars in the Brazilian market. Unlike static price tables, this model leverages a large, real-world dataset of over 550,000 vehicle listings to capture the complex nuances and relationships that determine a car's true market value.

The core of the project is a fine-tuned regression model, demonstrating an advanced approach to tabular data regression. The entire process, from data exploration to the final interactive web application, is documented here.

---

## 🛠️ Project Workflow & Architecture

The project follows a classic and rigorous Data Science lifecycle:
1.  **Data Analysis & Feature Engineering:** Deep statistical analysis to clean data, handle missing values, and engineer new high-value features (`age`, `km_per_year`, etc.).
2.  **Model Engineering & Fine-Tuning:** A pre-trained Portuguese model, **BERTimbau**, was fine-tuned for regression using a custom-built training loop in PyTorch, accelerated by a local NVIDIA GPU.
3.  **Evaluation:** The final model was evaluated on a hold-out test set of over 110,000 samples, achieving a Mean Absolute Error (MAE) of approximately R$ 1,721.
4.  **Deployment:** The trained model and other artifacts were packaged into an interactive web application using Gradio and deployed to Hugging Face Spaces.

---

## 🚀 Tech Stack

* **Data Science & ML:** Pandas, NumPy, Scikit-learn, PyTorch, Hugging Face Transformers
* **UI & Deployment:** Gradio, Hugging Face Spaces, Git LFS

---

## ⚙️ How to Run Locally

This project has two main components: the final Gradio application and the training notebook.

### 1. Running the Final Application (Gradio Demo) Locally

These instructions are for running the pre-trained model in the interactive Gradio interface on your local machine.

1.  **Prerequisites:** Python 3.11+, Git, and Git LFS.
2.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/Viniciuss-Moreira/Regression-model-car-predict-prices.git](https://github.com/Viniciuss-Moreira/Regression-model-car-predict-prices.git)
    cd Regression-model-car-predict-prices
    ```
3.  **Set Up the Virtual Environment:**
    ```bash
    python -m venv .venv
    # On Windows:
    .\.venv\Scripts\activate
    # On Mac/Linux:
    source .venv/bin/activate
    ```
4.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
5.  **Run the Application:**
    ```bash
    python -m frontend.app
    ```
    This will launch the Gradio interface on a local URL (e.g., `http://127.0.0.1:7860`).

### 2. Replicating the Model Training

The entire process of data analysis, feature engineering, and model training is documented in the Jupyter Notebook. To replicate it, you will need a machine with a CUDA-enabled NVIDIA GPU and a correctly configured Python/CUDA environment.

## 🔮 Future Improvements

* **Hyperparameter Tuning:** Use a systematic approach like Optuna or Ray Tune to find the optimal combination of learning rate, batch size, and network architecture.
* **Explainable AI (XAI):** Implement techniques like SHAP to understand *which* features are most influential in the model's predictions.
* **Advanced Feature Engineering:** Incorporate external data, such as economic indicators from the time of sale, to potentially improve prediction accuracy.
