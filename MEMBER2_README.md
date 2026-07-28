# Member 2 Documentation: EDA, Linear SVM, and CNN Models

**Author:** Member 2 (Thiranji - CIT-24-01-0266)  
**Branch:** `feature/member2-eda-svm-cnn`  
**Task Scope:** Exploratory Data Analysis (EDA), Baseline Linear Support Vector Machine (Linear SVM), and Deep Learning Convolutional Neural Network (CNN).

---

## 1. Project Overview & Assigned Tasks

Member 2 is responsible for:
1. **Exploratory Data Analysis (EDA)** on the arXiv academic paper abstracts dataset.
2. **Baseline ML Model**: Linear Support Vector Machine (SVM) utilizing TF-IDF feature extraction (Unigram & Bigram).
3. **Deep Learning Model**: 1D Convolutional Neural Network (CNN) implemented via TensorFlow / Keras.
4. **Evaluation & Comparison**: Performance analysis between traditional feature engineering (TF-IDF + SVM) vs deep learning text representation (Embedding + CNN).

---

## 2. Dataset Summary

- **Dataset File:** `arxiv_15000_balanced.csv`
- **Total Records:** 15,000 instances
- **Target Categories (6 classes):**
  - Computer Science (`cs`)
  - Physics (`physics` / `astro-ph`)
  - Mathematics (`math`)
  - Statistics (`stat`)
  - Quantitative Biology (`q-bio`)
  - Quantitative Finance (`q-fin`)
- **Train / Test Split:** 80% Training (12,000 samples), 20% Testing (3,000 samples).

---

## 3. Exploratory Data Analysis (EDA) Highlights

- **Balanced Class Distribution:** Equal representation of ~2,500 samples per class to eliminate class imbalance bias.
- **Abstract Length Distribution:** Average sequence length of paper abstracts is ~150–250 words.
- **Top Common Words:** Identified top 20 domain-specific terms across academic paper abstracts post stop-word removal.

---

## 4. Model Architectures & Hyperparameters

### 4.1 Linear Support Vector Machine (Linear SVM)
- **Feature Extractor:** `TfidfVectorizer`
  - `ngram_range`: `(1, 2)` (Unigrams & Bigrams)
  - `max_features`: 30,000
  - `stop_words`: English standard stop words
- **Classifier:** `LinearSVC(C=1.0, random_state=42)`
- **Execution Speed:** Fast training (~2.5 seconds)

### 4.2 Convolutional Neural Network (CNN)
- **Tokenizer / Vocabulary:** Vocabulary size = 30,000 words, OOV Token = `<OOV>`
- **Sequence Padding:** `max_len` = 300 tokens (`padding='post'`, `truncating='post'`)
- **Architecture Pipeline:**
  1. `Embedding` layer (Input dim: 30,000, Output dim: 128, Input length: 300)
  2. `Conv1D` layer (Filters: 128, Kernel size: 5, Activation: `relu`)
  3. `GlobalMaxPooling1D` layer
  4. `Dense` layer (64 units, Activation: `relu`)
  5. `Dropout` layer (Rate: 0.5)
  6. `Dense` Output layer (6 units, Activation: `softmax`)
- **Optimizer & Loss:** Adam optimizer, `sparse_categorical_crossentropy` loss.
- **Training Setup:** 5 Epochs, batch size 64.

---

## 5. Experimental Results & Performance Comparison

| Model | Architecture / Method | Test Accuracy | Strengths |
| :--- | :--- | :--- | :--- |
| **Linear SVM** | TF-IDF (1,2-gram) + LinearSVC | **90.13%** | Highest accuracy, fast training, robust on text classification |
| **CNN** | Keras Embedding + Conv1D + GlobalMaxPool | **87.06%** | End-to-end word sequence feature learning |

> **Key Finding:** Linear SVM with TF-IDF unigram+bigram features outperformed 1D CNN by **3.07%**, demonstrating that n-gram frequency features are exceptionally strong baselines for domain-specific text classification.

---

## 6. How to Reproduce Member 2 Results

1. Ensure the dataset `data/processed/arxiv_15000_balanced.csv` is present.
2. Open the main Jupyter Notebook:
   ```bash
   jupyter notebook notebooks/member2_svm_cnn.ipynb
   ```
3. Run all cells sequentially. Output models will automatically be saved to `models/`.
