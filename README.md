# ResearchScope AI: Academic Research Paper Subject Category Classifier

An NLP-powered scientific research paper topic classification system built for **CCS3356 - Natural Language Processing (NLP)**.

---

## 👥 Group Members (Group 20)

| Member | Responsibility & Contribution | Progress |
| :--- | :--- | :---: |
| **Member 1 (Oshadha)** | Core web application design, text preprocessing modularization, TF-IDF Baseline, Logistic Regression, LSTM Deep Learning model, and Ensemble Experiments (V3/V4). | 100% |
| **Member 2 (Thiranji)** | Support Vector Machine (SVM) classifier, 1D Convolutional Neural Network (CNN) text model, validation metrics evaluation, and integration. | 100% |
| **Member 3 (Ravindu)** | XGBoost classifier (with TF-IDF), DistilBERT Sequence classification transformer pipeline, and evaluation reporting. | 100% |

---

## 📝 Problem Statement

In academic research databases (such as arXiv), thousands of papers are uploaded daily. Manually sorting these documents into appropriate fields is time-consuming and error-prone. This project delivers an automated, transparent, and explainable NLP pipeline to classify scientific research papers into one of six core subject categories based on their **Title** and **Abstract**:
1. **Computer Science (CS)**
2. **Mathematics (Math)**
3. **Physics (Phys)**
4. **Statistics (Stat)**
5. **Quantitative Biology (Q-Bio)**
6. **Quantitative Finance (Q-Fin)**

---

## 📊 Dataset Information

* **Filename**: `arxiv_15000_balanced.csv`
* **Size**: 15,000 records
* **Distribution**: Perfectly balanced dataset with 2,500 samples per class.
* **Train/Test Split**: 82% training (12,000 records), 18% testing (3,000 records).
* **Data Sources**: arXiv Public Dataset.

---

## ⚙️ Setup Instructions

To set up and run this application locally, ensure you have Python (version 3.10+) installed.

1. **Clone the repository**:
   ```bash
   git clone https://github.com/NLP_Group_20.git
   cd NLP_Group_20
   ```

2. **Set up virtual environment**:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 How to Run the Project

Launch the Streamlit web application using:
```bash
.venv\Scripts\python -m streamlit run app/app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🤖 Model Summary

Our architecture strictly complies with the requirement that no two members use the same models:

### Member 1:
* **Logistic Regression (ML)**: Uses TF-IDF features. Baseline classifier.
* **LSTM (DL)**: Standard tokenized word index sequence representation with Keras LSTM layers.

### Member 2:
* **SVM (ML)**: Support Vector Machine with TF-IDF features.
* **CNN (DL)**: 1D CNN for text sequences using word embeddings.

### Member 3:
* **XGBoost (ML)**: Extreme Gradient Boosting tree-based classifier with TF-IDF.
* **DistilBERT (DL / Transformer)**: Fine-tuned DistilBERT transformer sequence classifier.

---

## 📈 Results Summary

The performance of each member's models evaluated on the test split:

| Model | Owner | Type | Feature Extraction | Accuracy |
| :--- | :--- | :--- | :--- | :---: |
| **Advanced Ensemble V4** | Member 1 | Ensemble ML | Title-Weighted TF-IDF Soft Voting | **91.07%** |
| **Advanced Ensemble V3** | Member 1 | Ensemble ML | Stable Soft Voting TF-IDF Ensemble | **90.80%** |
| **Logistic Regression** | Member 1 | Machine Learning | TF-IDF Vectorizer (ngram 1,2) | **89.33%** |
| **SVM Model** | Member 2 | Machine Learning | TF-IDF Vectorizer | **88.67%** |
| **XGBoost Model** | Member 3 | Machine Learning | TF-IDF Vectorizer (ngram 1,2) | **86.90%** |
| **DistilBERT Model** | Member 3 | Transformer | HuggingFace Pretrained Tokenizer | **86.67%** |
| **CNN Model** | Member 2 | Deep Learning | Tokenizer + Embedding | **86.40%** |
| **LSTM Model** | Member 1 | Deep Learning | Tokenizer + Padding Sequence | **85.17%** |