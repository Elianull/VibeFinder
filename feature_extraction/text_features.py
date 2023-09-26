from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer()

def extract_text_features(text):
    """
    Function to extract features from text data.

    Parameters:
        text (str): The raw text from which to extract features.

    Returns:
        vector (array): The extracted feature vector.
    """
  
    # Fitting and transforming the text
    tfidf_matrix = vectorizer.fit_transform([text])
    
    # Converting to array and flattening to get a simple 1-D array
    vector = tfidf_matrix.toarray().flatten()
    
    return vector
