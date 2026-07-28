# Member 01 Contribution & Documentation - ResearchScope AI

> 🌐 **Live Web Application Demo:** [https://researchscope-ai.streamlit.app/](https://researchscope-ai.streamlit.app/)

**Student Name:** S.A.Kavindu Oshadha Perera  
**Student ID:** CIT-24-01-0476 (Member 01)  
**Branch:** `feature/member1-preprocessing-logistic-lstm`  
**Project Title:** ResearchScope AI - Predicting Research Paper Subject Categories Using NLP  
**Group:** Group 20 (ROT NLP Solutions)  

---

## 📌 Individual NLP Pipeline (Member 01)

As specified in the Project Proposal Submission (Section 3 - Member 01), Member 01 is responsible for the following core NLP pipeline components:

1. **Step 1: Data Loading & Initial Inspection**
   - Loaded dataset `data/processed/arxiv_15000_balanced.csv` (15,000 balanced records, 2,500 per class across 6 categories).
   - Inspected missing values, data shapes, and label distributions.

2. **Step 2: Text Cleaning**
   - Removed URLs, HTML tags, unwanted special symbols, numbers, and extra whitespaces using regular expressions (`re`).

3. **Step 3: Lowercasing**
   - Standardized all titles and abstracts to lowercase for consistent vocabulary mapping.

4. **Step 4: Tokenization**
   - Tokenized research paper text into individual word tokens using NLTK `word_tokenize`.

5. **Step 5: Stop-word Removal**
   - Filtered out uninformative English stop-words using NLTK `stopwords`.

6. **Step 6: Lemmatization**
   - Reduced tokens to their base dictionary forms using NLTK `WordNetLemmatizer`.

7. **Step 7: Feature Extraction & Model Development**
   - **Machine Learning (ML Model):** Logistic Regression trained with TF-IDF n-gram feature extraction.
   - **Deep Learning (DL Model):** Long Short-Term Memory (LSTM) network trained using Keras Tokenizer sequence padding and spatial dropout.

---

## 🤖 Assigned ML & DL Models Summary

| Model Type | Model Name | Feature Representation | Key Parameters / Architecture | Test Accuracy |
| :--- | :--- | :--- | :--- | :---: |
| **ML Model** | **Logistic Regression** | TF-IDF Vectorizer (ngram 1-2, max 5,000 features) | `C=1.0`, `max_iter=1000`, Multi-class Multinomial | **89.33%** |
| **DL Model** | **LSTM Neural Network** | Keras Tokenizer (num_words=10000, maxlen=200) | Embedding(128d) -> SpatialDropout1D(0.2) -> LSTM(128) -> Dense(64) -> Softmax(6) | **85.17%** |

---

## 📁 Member 01 File Structure & Deliverables

```
├── notebooks/
│   ├── member1_logistic_lstm.ipynb         # Main Jupyter Notebook (Step-by-step Pipeline)
│   └── member1_logistic_lstm_BACKUP.ipynb  # Backup Notebook
├── src/
│   └── preprocessing.py                    # Modularized text preprocessing helper script
├── models/
│   ├── member1_logistic_regression.pkl     # Saved Logistic Regression model
│   ├── member1_tfidf_vectorizer.pkl        # Saved TF-IDF vectorizer artifact
│   ├── member1_label_encoder.pkl           # Saved Label Encoder
│   ├── member1_lstm_model.h5               # Saved Keras LSTM trained weights
│   └── member1_lstm_tokenizer.pkl          # Saved Keras Tokenizer artifact
└── screenshots/member1/                    # Verification screenshots for Member 01
    ├── 1member1_python_tensorflow_setup.png
    ├── 2member1_dataset_loaded.png
    ├── 3member1_missing_values.png
    ├── 4member1_class_distribution.png
    ├── 5member1_text_cleaning.png
    ├── 6member1_train_test_split.png
    ├── 7member1_logistic_regression_result.png
    ├── 8member1_logistic_regression_confusion_matrix.png
    ├── 9member1_lstm_input_preparation.png
    ├── 10member1_lstm_model_summary.png
    ├── 11member1_lstm_training.png
    ├── 12member1_lstm_accuracy.png
    ├── 13member1_lstm_classification_report.png
    ├── 14member1_model_comparison.png
    └── 15member1_saved_models_folder.png
```

---

## 🛠️ How to Execute Member 01 Notebook

1. Ensure the Python virtual environment is activated:
   ```bash
   .venv\Scripts\activate
   ```
2. Open Jupyter Notebook / VS Code:
   ```bash
   jupyter notebook notebooks/member1_logistic_lstm.ipynb
   ```
3. Run all cells sequentially to reproduce Member 01's preprocessing, ML model training, DL model training, and artifact exports.
