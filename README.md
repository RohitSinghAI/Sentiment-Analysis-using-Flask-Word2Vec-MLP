# Sentiment Analysis using Flask, Word2Vec & MLP

A Machine Learning based Sentiment Analysis web application built using **Python, Flask, Word2Vec, and MLP Classifier**.

The application takes text input from the user, preprocesses the text, converts it into a numerical sentence vector using a trained Word2Vec model, and predicts whether the sentiment is **Positive** or **Negative**.

---

## 🚀 Features

- Text preprocessing using custom NLP functions
- Word2Vec-based text embeddings
- Sentence vector generation
- MLP Classifier for sentiment prediction
- Flask web application
- Simple and user-friendly HTML interface
- Real-time Positive/Negative sentiment prediction

---

## 🛠️ Technologies Used

- Python
- Flask
- NumPy
- Pandas
- Scikit-learn
- Gensim
- Word2Vec
- Joblib
- HTML
- CSS

---

## 📂 Project Structure

```text
Sentiment-Analysis-using-Flask-Word2Vec-MLP/
│
├── app.py
├── mlp_model.pkl
├── random_forest_model.pkl
├── ada_boost_model.pkl
├── Word_2_Vec.model
├── run_command.txt
│
├── templates/
│   └── index.html
│
├── utils/
│   ├── preprocess.py
│   └── sentence_vector_creation.py
│
├── .gitignore
└── README.md