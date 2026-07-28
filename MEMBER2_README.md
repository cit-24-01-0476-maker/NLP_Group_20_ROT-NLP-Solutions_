# Member 2 Documentation: Proposal-Aligned NLP Pipeline (EDA, Linear SVM, CNN)

**Author:** Member 2 (Thiranji - CIT-24-01-0266)  
**Group:** Group 20 (ROT NLP Solutions)  
**Branch:** `feature/member2-eda-svm-cnn`  

---

## Assigned Tasks (From Project Proposal Section 3 - Member 02)

| Proposal Step | Task Name | Description & Implementation Details |
| :--- | :--- | :--- |
| **Step 1** | **Dataset Exploration** | Checked dataset size (15,000 records), number of classes (6 classes), class balance (2,500 records per class), and inspected sample abstracts. |
| **Step 2** | **Handling Missing & Duplicate Data** | Verified no null values in titles or abstracts, dropped missing categories, and ensured data quality. |
| **Step 3** | **Category Mapping** | Mapped detailed arXiv categories (e.g., `cs.DS`, `astro-ph`, `math.ST`) into 6 main subject categories: *Computer Science, Mathematics, Physics, Statistics, Quantitative Biology, Quantitative Finance*. |
| **Step 4** | **Exploratory Data Analysis (EDA)** | Generated visualizations for class distribution, abstract length (word count), and top 20 most frequent terms. |
| **Step 5** | **Text Preprocessing** | Applied text cleaning (removing URLs/HTML tags/special characters), lowercasing, tokenization, and stop-word removal. |
| **Step 6** | **TF-IDF & Linear SVM Model** | Built `TfidfVectorizer` (unigrams & bigrams, max 30,000 features) and trained `LinearSVC`. **Accuracy: 90.13%**. |
| **Step 7** | **Sequence Prep, CNN & Keyword Extraction** | Tokenized text (vocab 30,000, max length 300), built Keras 1D CNN (`Embedding -> Conv1D -> GlobalMaxPool -> Dense -> Softmax`). **Accuracy: 87.06%**. Extracted important domain keywords and confusion matrix analysis. |

---

## Member 2 Model Performance Summary

| Model | Technique / Architecture | Test Accuracy | Role in Project |
| :--- | :--- | :--- | :--- |
| **Linear SVM** | TF-IDF (1,2-gram) + `LinearSVC` | **90.13%** | Primary ML Baseline Model for Member 2 |
| **CNN** | Keras Embedding + Conv1D + GlobalMaxPool | **87.06%** | Primary DL Model for Member 2 |

---

## Saved Model Artifacts

- `models/member2_svm_model.pkl`
- `models/member2_cnn_model.h5`
- `models/member2_tokenizer.pkl`
- `models/member2_label_encoder.pkl`

---

## How to Run Member 2 Notebook

```bash
# 1. Checkout Member 2 branch
git checkout feature/member2-eda-svm-cnn

# 2. Run Member 2 Jupyter Notebook
jupyter notebook notebooks/member2_svm_cnn.ipynb
```
