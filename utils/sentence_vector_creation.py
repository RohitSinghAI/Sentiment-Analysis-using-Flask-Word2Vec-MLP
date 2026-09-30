import numpy as np
from gensim.models import Word2Vec

def get_sentence_vector(tokens, model):
    """
    Convert a list of tokens into a sentence vector using the provided Word2Vec model.
    """
    # Get word vectors for each token in the sentence
    word_vectors = [model.wv[word] for word in tokens if word in model.wv]
    
    # If no valid words, return a zero vector
    if len(word_vectors) == 0:
        return np.zeros(model.vector_size)
    
    # Average the word vectors to get the sentence vector
    return np.mean(word_vectors, axis=0)
