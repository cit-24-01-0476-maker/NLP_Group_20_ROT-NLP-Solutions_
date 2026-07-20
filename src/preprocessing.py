import re

def clean_text(text):
    """
    Cleans raw text by converting to lowercase, removing URLs,
    removing non-alphabetic characters, and collapsing whitespace.
    """
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
