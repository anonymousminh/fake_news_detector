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

def process_tokens(tokens, use_stemming=False, use_lemmatization=True):
    """
    Process tokens with stemming or lemmatization.
    """
    if use_stemming:
        stemmer = PorterStemmer()
        processed_tokens = [stemmer.stem(token) for token in tokens]
    elif use_lemmatization:
        lemmatizer = WordNetLemmatizer()
        processed_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    else:
        processed_tokens = tokens
    
    return processed_tokens

# Test your function
tokens = ["running", "jumps", "better", "studies", "studying", "cats", "dogs"]
print(f"Original tokens: {tokens}")
print(f"Stemmed tokens: {process_tokens(tokens, use_stemming=True, use_lemmatization=False)}")
print(f"Lemmatized tokens: {process_tokens(tokens, use_stemming=False, use_lemmatization=True)}")