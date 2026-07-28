# Member 01 Contribution & Documentation

**Student Name:** S.A.Kavindu Oshadha Perera  
**Student ID:** CIT-24-01-0476 (Member 01)  
**Branch:** `feature/member1-preprocessing-logistic-lstm`  
**Assigned Tasks:** Text Preprocessing Pipeline, Logistic Regression (ML), LSTM (DL)

---

## 📌 Individual NLP Pipeline Implementation

In accordance with Section 3 of the submitted Project Proposal, Member 01 implemented the core text preprocessing and model training pipeline:

1. **Step 1: Data Loading & Inspection**
   - Loaded `data/processed/arxiv_15000_balanced.csv` (15,000 balanced records across 6 categories).
   - Inspected missing values, data types, and class distribution.

2. **Step 2: Text Cleaning**
   - Removed URLs, HTML tags, punctuation noise, non-alphabetic characters, and extra whitespaces.

3. **Step 3: Lowercasing**
   - Standardized all input text to lowercase.

4. **Step 4: Tokenization**
   - Tokenized text into individual words using NLTK `word_tokenize`.

5. **Step 5: Stop-word Removal**
   - Filtered out uninformative English stop-words.

6. **Step 6: Lemmatization**
   - Reduced tokens to base root words using NLTK `WordNetLemmatizer`.

7. **Step 7: Feature Extraction & Model Development**
   - **Machine Learning (ML):** Logistic Regression trained with TF-IDF n-gram feature extraction.
   - **Deep Learning (DL):** Keras LSTM network trained with sequence tokenization and padding.

---

## 🤖 Member 01 Models & Performance

| Model Type | Model Name | Feature Method | Architecture / Parameters | Test Accuracy |
| :--- | :--- | :--- | :--- | :---: |
| **Machine Learning** | **Logistic Regression** | TF-IDF Vectorizer (ngram 1-2) | Multi-class Multinomial, `max_iter=1000` | **89.33%** |
| **Deep Learning** | **LSTM Neural Network** | Keras Tokenizer (num_words=10000, maxlen=200) | Embedding(128d) -> SpatialDropout1D -> LSTM(128) -> Dense(64) -> Softmax(6) | **85.17%** |

---

## 📁 Member 01 Saved Artifacts & Deliverables

- **Main Notebook:** `notebooks/member1_logistic_lstm.ipynb`
- **Helper Script:** `src/preprocessing.py`
- **Saved Model Artifacts:**
  - `models/member1_logistic_regression.pkl`
  - `models/member1_tfidf_vectorizer.pkl`
  - `models/member1_label_encoder.pkl`
  - `models/member1_lstm_model.h5`
  - `models/member1_lstm_tokenizer.pkl`
- **Verification Screenshots:** `screenshots/member1/` (16 stage screenshots)

---

## 🛠️ Execution

To run Member 01's notebook:
```bash
.venv\Scripts\activate
jupyter notebook notebooks/member1_logistic_lstm.ipynb
```
