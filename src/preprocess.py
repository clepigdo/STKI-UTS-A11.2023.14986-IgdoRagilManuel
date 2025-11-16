import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("Downloading NLTK stopwords...")
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt')


factory = StemmerFactory()
stemmer = factory.create_stemmer()

list_stopwords_dasar = stopwords.words('indonesian')

custom_stopwords = [
    'yg', 'utk', 'dgn', 'jg', 'dg', 'sbb', 'yakni', 'wfo', 'wfh', 
    'di', 'dan', 'atau', 'untuk', 'adalah', 'merupakan', 'pada', 'ke', 'dari',
    'dengan', 'yang', 'ini', 'itu', 'tersebut', 'info'
]


STOPWORDS_SET = set(list_stopwords_dasar)
STOPWORDS_SET.update(custom_stopwords)


def clean(text):
    """1. Case Folding & Normalisasi (Menghapus Angka/Tanda Baca)."""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    return text.strip()
    
def tokenize(text):
    """2. Tokenisasi."""
    return word_tokenize(text)

def remove_stopwords(tokens):
    """3. Stopword Removal & Filter Panjang Kata."""
    return [word for word in tokens if word not in STOPWORDS_SET and len(word) > 2] 

def stem(tokens):
    """4. Stemming."""
    return [stemmer.stem(word) for word in tokens]


def preprocess_text(text):
    """
    Fungsi utama untuk membersihkan teks (Menggabungkan 4 langkah terpisah).
    """
    cleaned_text = clean(text)
    tokens = tokenize(cleaned_text)
    stopped_tokens = remove_stopwords(tokens)
    stemmed_tokens = stem(stopped_tokens)
    
    # Hapus token kosong jika ada
    final_tokens = [token for token in stemmed_tokens if token]
    
    return final_tokens

if __name__ == '__main__':
    print("--- MENJALANKAN TEST PREPROCESS.PY ---")
    sample_text = "Info Magang (Internship) Web Developer. Lokasi: WFO di Semarang Tengah. Syarat skill: PHP, Gaji nego."
    
    # Demo 4 langkah terpisah
    cleaned = clean(sample_text)
    tokenized = tokenize(cleaned)
    stopped = remove_stopwords(tokenized)
    stemmed = stem(stopped)

    print(f"\n1. Teks Asli: {sample_text}")
    print(f"2. HASIL clean() & tokenize(): {tokenized}")
    print(f"3. HASIL remove_stopwords(): {stopped}")
    print(f"4. HASIL preprocess_text() (Final): {stemmed}")