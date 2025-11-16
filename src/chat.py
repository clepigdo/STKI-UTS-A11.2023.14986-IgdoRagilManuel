import sys
import os
import re
import glob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
import nltk

# --- SETUP (Sama seperti search.py) ---
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

factory = StemmerFactory()
stemmer = factory.create_stemmer()
list_stopwords_dasar = stopwords.words('indonesian')
STOPWORDS_SET = set(list_stopwords_dasar)
STOPWORDS_SET.update(['yg', 'utk', 'dgn', 'jg', 'dg', 'wfo', 'wfh'])

# --- FUNGSI PREPROCESSING ---
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    tokens = text.split()
    tokens_without_stopwords = [
        word for word in tokens
        if word not in STOPWORDS_SET and len(word) > 2
    ]
    stemmed_tokens = [stemmer.stem(word) for word in tokens_without_stopwords]
    return [token for token in stemmed_tokens if token]

def load_data(root_dir):
    """Memuat data ASLI (untuk snippet) dan data PROSES (untuk VSM)"""
    
    # 1. Muat Data Proses (untuk VSM)
    processed_dir = os.path.join(root_dir, 'data', 'processed')
    file_paths_proc = sorted(glob.glob(os.path.join(processed_dir, 'processed_doc*.txt')))
    doc_names = [os.path.basename(f) for f in file_paths_proc]
    processed_corpus = [open(f, 'r', encoding='utf-8').read() for f in file_paths_proc]

    # 2. Muat Data Asli (untuk snippet)
    original_dir = os.path.join(root_dir, 'data')
    original_docs = {}
    for f_proc in file_paths_proc:
        # Ubah 'processed_doc01...' menjadi 'doc01...'
        original_name = os.path.basename(f_proc).replace('processed_', '')
        original_path = os.path.join(original_dir, original_name)
        
        try:
            original_docs[os.path.basename(f_proc)] = open(original_path, 'r', encoding='utf-8').read()
        except FileNotFoundError:
            original_docs[os.path.basename(f_proc)] = "Teks asli tidak ditemukan."
            
    if not doc_names:
        print(f"Error: Tidak ada file di '{processed_dir}'.")
        sys.exit(1)
        
    return doc_names, processed_corpus, original_docs

def generate_template_response(results, original_docs):
    """Langkah 3b: Generator template-based (gabungkan kalimat kunci)"""
    
    if not results:
        return "Maaf, saya tidak menemukan dokumen yang relevan."

    response = "Berikut adalah hasil teratas yang saya temukan:\n\n"
    
    for res in results:
        doc_name = res['doc']
        score = res['score']
        
        # Ambil kalimat kunci (snippet) dari dokumen asli
        original_text = original_docs.get(doc_name, "")
        snippet = original_text.split('\n')[0] # Ambil baris pertama sebagai "kalimat kunci"
        snippet = snippet[:150] + "..." if len(snippet) > 150 else snippet

        response += f"--- Dokumen: {doc_name} (Skor: {score:.4f}) ---\n"
        response += f"\"{snippet}\"\n\n"
        
    return response

# --- MAIN CHAT INTERFACE ---

def main():
    print("Mempersiapkan mesin VSM...")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(current_dir, '..'))
    
    doc_names, processed_corpus, original_docs = load_data(root_dir)
    
    # Bangun Vektorizer & Matriks TF-IDF 
    vectorizer = TfidfVectorizer()
    tfidf_matrix_docs = vectorizer.fit_transform(processed_corpus)
    
    print("Sistem Temu Kembali Informasi (VSM) siap.")
    print("Ketik 'exit' untuk keluar.")
    
    while True:
        query = input("\nMasukkan kueri Anda: ")
        if query.lower() == 'exit':
            break
        
        # 1. Preprocess Kueri
        clean_query = ' '.join(preprocess_text(query))
        
        if not clean_query:
            print("Kueri tidak valid setelah preprocessing.")
            continue

        # 2. Representasi & Similarity (Langkah 3a)
        query_vector = vectorizer.transform([clean_query])
        cosine_scores = cosine_similarity(query_vector, tfidf_matrix_docs).flatten()
        
        # 3. Ambil Top-k (k=3)
        k = 3
        top_k_indices = cosine_scores.argsort()[-k:][::-1]
        
        results = []
        for index in top_k_indices:
            score = cosine_scores[index]
            if score > 0.0:
                results.append({
                    "doc": doc_names[index],
                    "score": score
                })
        
        # 4. Generate Respon (Langkah 3b)
        response = generate_template_response(results, original_docs)
        print(response)

if __name__ == "__main__":
    main()