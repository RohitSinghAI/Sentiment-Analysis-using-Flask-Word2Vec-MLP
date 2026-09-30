from flask import Flask, render_template, request
from utils.preprocess import preprocess_text
from utils.sentence_vector_creation import get_sentence_vector
import joblib
import numpy as np
from gensim.models import Word2Vec
import pickle
app = Flask(__name__)

# Load the Word2Vec model and classifier
word2vec_model =  Word2Vec.load("Word_2_Vec.model")
classifier = joblib.load('mlp_model.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    sentiment = None
    if request.method == 'POST':
        # Get the input text from the form
        unseen_text = request.form['text']
        
        # Preprocess the text
        tokens = preprocess_text(unseen_text)
        
        # Convert to sentence vector
        sentence_vector = get_sentence_vector(tokens, word2vec_model)
        
        # Reshape for prediction
        sentence_vector = np.reshape(sentence_vector, (1, -1))
        
        # Predict sentiment
        prediction = classifier.predict(sentence_vector)
        sentiment = 'Positive' if prediction[0] == 1 else 'Negative'
    
    return render_template('index.html', sentiment=sentiment)

if __name__ == "__main__":
    app.run(debug=True)
