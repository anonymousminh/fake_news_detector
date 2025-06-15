import re
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# nltk.download('punkt_tab')
# nltk.download('stopwords')
# nltk.download('wordnet')


def clean_text(text: str) -> str:
    """
    Preprocess the input text by removing special characters and extra spaces.
    
    Args:
        text (str): The input text to preprocess.
        
    Returns:
        str: The preprocessed text.
    """
    # Convert to lowercase
    text = text.lower()
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove URLs
    text = re.sub(r'http[s]?://\S+', '', text)

    # Remove special characters except for alphanumeric characters, spaces, and basic punctuation
    text = re.sub(r'[^a-zA-Z0-9\s.,!?\'\"-]', '', text)
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def tokenize_and_remove_stopwords(text: str) -> list:
    """
    Tokenize the input text and remove stopwords.
    
    Args:
        text (str): The input text to tokenize and remove stopwords from.
        
    Returns:
        list: A list of tokens with stopwords removed.
    """   
    # Clean the text first
    text = clean_text(text)

    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Get English stopwords
    stop_words = set(stopwords.words('english'))
    
    # Remove stopwords
    filtered_tokens = [word for word in tokens if word not in stop_words]
    
    return filtered_tokens
    

def process_tokens(tokens: list, use_stemming: bool, use_lemmatization: bool) -> list:
    """
    Apply stemming or lemmatization to tokens.
    
    Args:
        tokens (list): List of tokens.
        use_stemming (bool): Whether to apply stemming.
        use_lemmatization (bool): Whether to apply lemmatization.
        
    Returns:
        list: List of processed tokens.
    """
    if use_stemming:
        ps = PorterStemmer()
        tokens = [ps.stem(token) for token in tokens]
    
    if use_lemmatization:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return tokens


def preprocess_text(text, remove_stopwords=True, use_stemming=False, use_lemmatization=True):
    """
    Complete text preprocessing pipeline.
    """
    # Clean text
    cleaned_text = clean_text(text)
    
    # Tokenize
    tokens = word_tokenize(cleaned_text)
    
    # Remove stopwords if enabled
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
    
    # Apply stemming or lemmatization using the helper function
    tokens = process_tokens(tokens, use_stemming, use_lemmatization)

    # Join tokens back into text
    processed_text = ' '.join(tokens)
    
    return processed_text

def vectorize_texts(texts, max_features=5000):
    """
    Convert a list of texts into a TF-IDF features.
    
    Args:
        texts (list): List of texts to vectorize.
        max_features (int): Maximum number of features to extract.
        
    Returns:
        TfidfVectorizer: Fitted vectorizer.
        array: Transformed text data.
    """
    vectorizer = TfidfVectorizer(max_features=max_features, min_df=2, max_df=0.85)

    # Fit and transform the texts
    X = vectorizer.fit_transform(texts)

    return X, vectorizer

# Test your function
texts = [
    "This is a sample document about cats and dogs.",
    "Another document discussing dogs and pets.",
    "A third document about politics and economics.",
    "Yet another document about pets and animals."
]

# Preprocess each text
preprocessed_texts = [preprocess_text(text) for text in texts]
print("Preprocessed texts:")
for text in preprocessed_texts:
    print(f"- {text}")

# Vectorize
X, vectorizer = vectorize_texts(preprocessed_texts)
print(f"\nFeature matrix shape: {X.shape}")
print(f"Feature names (first 10): {vectorizer.get_feature_names_out()[:10]}")