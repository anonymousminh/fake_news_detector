import re
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer

nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

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
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    
    # Clean the text first
    text = clean_text(text)

    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Get English stopwords
    stop_words = set(stopwords.words('english'))
    
    # Remove stopwords
    filtered_tokens = [word for word in tokens if word not in stop_words]
    
    return filtered_tokens
    

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
    
    # Apply stemming or lemmatization
    tokens = preprocess_tokens(tokens, use_stemming, use_lemmatization)

    # Join tokens back into text
    processed_text = ' '.join(tokens)
    
    return processed_text

# Test your function
sample_text = "The runner was running faster than all the other runners in the race!"
processed = preprocess_text(sample_text)
print(f"Original: {sample_text}")
print(f"Processed: {processed}")