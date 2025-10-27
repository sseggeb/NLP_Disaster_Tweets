# NLP Disaster Tweet Classification

## 🚀 Project Overview

This project implements a Natural Language Processing (NLP) solution to the Kaggle competition **"Natural Language Processing with Disaster Tweets."** The objective is to determine whether a given tweet is about a **real disaster** (target=1) or not (target=0).

The core solution utilizes a deep learning approach: a **Bidirectional Long Short-Term Memory (Bi-LSTM)** network integrated with contextual features to improve predictive performance over text alone.

## 🛠️ Methodology and Architecture

### 1. Feature Engineering & Preprocessing

Robust data cleaning was critical due to the noisy nature of social media data.

* **Text Cleaning:** Removal of URLs, HTML, mentions (@), and hashtags (#); normalization (lowercasing, punctuation removal); and **Lemmatization** for root word standardization.
* **Keyword Aggregation:** The noisy `keyword` column was cleaned, decoded, and aggregated, keeping only the **Top 15** most frequent keywords and grouping the rest into an 'OTHER\_KEYWORD' category.
* **Location Standardization:** The highly varied `location` column was cleaned, standardized (e.g., 'NY' $\rightarrow$ 'new york city'), and aggregated, retaining the **Top 100** most frequent locations and grouping the rest into an 'OTHER\_LOCATION' category. Missing values were imputed as 'UNKNOWN'.

### 2. Model Architecture: Multi-Input Bi-LSTM

The final model is a **two-branch neural network** built using Keras/TensorFlow:

| Branch | Input | Processing | Rationale |
| :--- | :--- | :--- | :--- |
| **Text Branch** | Cleaned Text Sequence | **Keras Embedding Layer** (100-dim, learned during training) $\rightarrow$ **Bidirectional LSTM (64 units)** $\rightarrow$ **Dropout (0.5)** | Effectively captures both forward and backward textual context and long-term dependencies. |
| **Categorical Branch** | One-Hot Encoded Features | **Dense Layer** (32 units) $\rightarrow$ **Dropout (0.5)** | Incorporates powerful external signals (`keyword`, `location`) that the text alone may miss. |

The outputs of both branches are **concatenated** and passed through a final Dense layer with a **Sigmoid** activation for binary classification.

## 📈 Performance and Results

| Metric | Training Value | Validation Value |
| :--- | :--- | :--- |
| **Accuracy** | 86.0% | **80.0%** |
| **F1 Score** | 82.7% | **75.0%** |
| **Precision** | 87.9% | 80.4% |
| **Recall** | 78.0% | 70.3% |
| **Best Epoch** | 6 | - |

**Hyperparameter Optimization:** Training utilized the **Adam optimizer** with a low learning rate ($1e^{-4}$) and aggressive regularization (**Dropout $0.5$)** to prevent overfitting. **Early Stopping** was implemented with patience on the validation loss to ensure optimal model selection.

## ⚙️ How to Run the Notebook

### Prerequisites

* Python 3.x
* Jupyter Notebook or JupyterLab
* Required libraries: `pandas`, `numpy`, `tensorflow`/`keras`, `scikit-learn`, `nltk` (with `punkt`, `stopwords`, `wordnet` downloaded).

### Setup

1.  **Clone this repository** (or download the files).
2.  **Ensure data is present:** Place the Kaggle competition files (`train.csv` and `test.csv`) in the same directory as the notebook.
3.  **Run all cells:** Open the `notebook (2).ipynb` file and execute all cells in sequence.

The final cell will generate a **`submission.csv`** file containing the predictions on the test set, ready for upload to Kaggle.
