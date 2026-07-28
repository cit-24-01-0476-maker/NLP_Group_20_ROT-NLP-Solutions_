# ResearchScope AI: Predicting Research Paper Subject Categories Using NLP

A Natural Language Processing (NLP) text classification project developed for academic evaluation by **Group 20 (ROT NLP Solutions)**.

---

## 👥 Group Members & Work Allocation

| Student Name | Student ID | Designated Models & Responsibilities | Branch |
| :--- | :--- | :--- | :--- |
| **Member 1 (Oshadha)** | CIT-24-01-0476 | Preprocessing Pipeline, Logistic Regression (ML), LSTM (DL) | `feature/member1-preprocessing-logistic-lstm` |
| **Member 2 (Thiranji)** | CIT-24-01-0266 | EDA, Support Vector Machine (ML), 1D CNN (DL) | `feature/member2-eda-svm-cnn` |
| **Member 3 (Ravindu)** | CIT-24-01-0447 | XGBoost (ML), DistilBERT (DL), Streamlit Web App Integration | `feature/member3-xgboost-bert-app` |

---

## 📌 Project Overview & Problem Statement

Academic research databases process thousands of paper submissions daily. Manually categorizing papers into domain-specific subjects based on abstracts is time-consuming. **ResearchScope AI** automates this classification pipeline by predicting the subject domain of a paper given its **Title** and **Abstract**.

### Target Subject Categories:
1. **Computer Science**
2. **Mathematics**
3. **Physics**
4. **Statistics**
5. **Quantitative Biology**
6. **Quantitative Finance**

---

## 📊 Dataset Details

- **Source**: arXiv Research Paper Dataset (Cornell University / Kaggle)
- **Dataset File**: `data/processed/arxiv_15000_balanced.csv`
- **Volume**: 15,000 balanced records (2,500 samples per class)
- **Primary Fields**: `title`, `abstract`, `main_category`

---

## 🤖 Model Comparison & Evaluation Summary

| Member | Model Name | Model Type | Feature Representation | Test Accuracy |
| :--- | :--- | :--- | :--- | :---: |
| **Member 1** | **Logistic Regression** | Machine Learning | TF-IDF Vectorizer (ngram 1-2) | **89.33%** |
| **Member 2** | **Support Vector Machine (SVM)** | Machine Learning | TF-IDF Vectorizer | **88.67%** |
| **Member 3** | **XGBoost Classifier** | Machine Learning | TF-IDF Vectorizer | **86.90%** |
| **Member 3** | **DistilBERT Transformer** | Deep Learning / Transformer | HuggingFace Pretrained Tokenizer | **86.67%** |
| **Member 2** | **1D CNN** | Deep Learning | Tokenizer + Embedding Layer | **86.40%** |
| **Member 1** | **LSTM Neural Network** | Deep Learning | Tokenizer + Sequence Padding | **85.17%** |

---

## 📁 Repository Structure

```
NLP_Group_20_ROT-NLP-Solutions_/
├── app/
│   ├── app.py                      # Main Streamlit Web Application
│   └── admin_panel.py              # Application Admin Panel
├── data/
│   └── processed/
│       └── arxiv_15000_balanced.csv # Cleaned balanced dataset (15k records)
├── models/                         # Trained model artifacts (.pkl, .h5, distilbert)
├── notebooks/
│   ├── member1_logistic_lstm.ipynb
│   ├── member2_svm_cnn.ipynb
│   └── member3_xgboost_bert.ipynb
├── src/
│   └── preprocessing.py            # Modularized text cleaning pipeline
├── screenshots/                    # Verification screenshots for project stages
├── reports/                        # Project submission reports & PDF documentation
├── MEMBER1_README.md
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🚀 How to Run the Project Locally

1. **Clone the repository**:
   ```bash
   git clone https://github.com/cit-24-01-0476-maker/NLP_Group_20_ROT-NLP-Solutions_.git
   cd NLP_Group_20_ROT-NLP-Solutions_
   ```

2. **Set up virtual environment**:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. **Install required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the Streamlit Web Application**:
   ```bash
   streamlit run app/app.py
   ```
   Open `http://localhost:8501` in your browser.