# NLP_Group_20-
"A Natural Language Processing (NLP) pipeline to predict research paper subject categories using abstracts and titles. Includes ML/DL models (Logistic Regression, SVM, XGBoost, LSTM, CNN, BERT) and a Streamlit web application."

## Member 1 Progress

Member 1 implemented the preprocessing pipeline, TF-IDF feature extraction, Logistic Regression model, and LSTM model.

### Dataset
- Dataset: arXiv balanced subset
- Records: 15,000
- Classes: 6
- Train set: 12,000 records
- Test set: 3,000 records

### ML Model
- Model: Logistic Regression
- Features: TF-IDF
- Accuracy: 89.33%
- Macro F1-score: 0.89

### DL Model
- Model: LSTM
- Input: Tokenized and padded text sequences
- Evaluation: Accuracy, precision, recall, and F1-score

### Saved Files
- member1_logistic_regression.pkl
- member1_tfidf_vectorizer.pkl
- member1_label_encoder.pkl
- member1_lstm_model.h5
- member1_lstm_tokenizer.pkl