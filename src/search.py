import argparse
import glob
import os
import re
import sys
from collections import defaultdict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
import nltk

# --- SETUP (Diambil dari Notebook Anda) ---
# Download NLTK data (jika belum)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Setup Stemmer Sastrawi
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Setup Stopwords NLTK
list_stopwords_dasar = stopwords.words('indonesian')
STOPWORDS_SET = set(list_stopwords_dasar)
STOPWORDS_SET.update(['yg', 'utk', 'dgn', 'jg', 'dg', 'wfo', 'wfh'])

# --- FUNGSI INTI (Diambil dari Notebook Anda) ---

def preprocess_text(text):
    """Fungsi preprocessing lengkap dari Sel 7 notebook Anda."""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    tokens = text.split() # Gunakan split() sederhana untuk CLI
    tokens_without_stopwords = [
        word for word in tokens
        if word not in STOPWORDS_SET and len(word) > 2
    ]
    stemmed_tokens = [stemmer.stem(word) for word in tokens_without_stopwords]
    return [token for token in stemmed_tokens if token]

def load_processed_data(root_dir):
    """Memuat dokumen yang SUDAH bersih dari data/processed/"""
    processed_dir = os.path.join(root_dir, 'data', 'processed')
    file_paths = sorted(glob.glob(os.path.join(processed_dir, 'processed_doc*.txt')))
    
    doc_names = [os.path.basename(f) for f in file_paths]
    processed_corpus = []
    
    for file in file_paths:
        with open(file, 'r', encoding='utf-8') as f:
            processed_corpus.append(f.read())
            
    if not doc_names:
        print(f"Error: Tidak ada file di '{processed_dir}'.")
        print("Pastikan Anda sudah menjalankan Soal 02 dan menyimpan hasilnya.")
        sys.exit(1)
        
    return doc_names, processed_corpus

# --- LOGIKA MODEL (Langkah 2) ---

def run_boolean_search(query, doc_names, processed_corpus):
    """Menjalankan pencarian Boolean (dari Sel 12 & 13)"""
    
    # 1. Bangun Inverted Index (on-the-fly)
    inverted_index = defaultdict(set)
    for i, text in enumerate(processed_corpus):
        tokens = text.split()
        for token in tokens:
            inverted_index[token].add(doc_names[i])

    # 2. Preprocess Kueri
    query_tokens = query.split()
    clean_terms = [stemmer.stem(t.lower()) for t in query_tokens if t.upper() not in ['AND', 'OR', 'NOT']]
    
    # 3. Parse Kueri (dari Sel 13)
    try:
        current_result_set = inverted_index.get(clean_terms[0], set())
        i = 1
        while i < len(query_tokens):
            operator = query_tokens[i].upper()
            next_term = clean_terms[i//2] # Ambil term bersih berikutnya
            next_term_postings = inverted_index.get(next_term, set())
            
            if operator == 'AND':
                current_result_set = current_result_set.intersection(next_term_postings)
            elif operator == 'OR':
                current_result_set = current_result_set.union(next_term_postings)
            elif operator == 'NOT':
                current_result_set = current_result_set.difference(next_term_postings)
            i += 2
        
        return list(current_result_set)
    
    except Exception as e:
        print(f"Error parsing boolean query: {e}")
        return []

def run_vsm_search(query, k, doc_names, processed_corpus):
    """Menjalankan pencarian VSM (dari Sel 15 & 16)"""
    
    # 1. Bangun Vektorizer & Matriks TF-IDF (Model A: Standar)
    vectorizer = TfidfVectorizer()
    tfidf_matrix_docs = vectorizer.fit_transform(processed_corpus)
    
    # 2. Preprocess Kueri (Gunakan fungsi lengkap)
    clean_query = ' '.join(preprocess_text(query))
    if not clean_query:
        return []

    # 3. Representasi Kueri
    query_vector = vectorizer.transform([clean_query])
    
    # 4. Hitung Cosine Similarity
    cosine_scores = cosine_similarity(query_vector, tfidf_matrix_docs).flatten()
    
    # 5. Ambil Top-k
    top_k_indices = cosine_scores.argsort()[-k:][::-1]
    
    results = []
    for index in top_k_indices:
        score = cosine_scores[index]
        if score > 0.0:
            # Dapatkan top-terms (explain singkat)
            feature_names = vectorizer.get_feature_names_out()
            doc_vector = tfidf_matrix_docs[index]
            top_terms_indices = doc_vector.indices[doc_vector.data.argsort()[-3:]][::-1]
            top_terms = [feature_names[i] for i in top_terms_indices]
            
            results.append({
                "doc": doc_names[index],
                "score": score,
                "explain": f"Top terms: {', '.join(top_terms)}"
            })
    return results

# --- MAIN ORCHESTRATOR ---

def main():
    parser = argparse.ArgumentParser(description="STKI Search Engine Orchestrator")
    parser.add_argument('--model', required=True, choices=['boolean', 'vsm'], help="Model yang digunakan.")
    parser.add_argument('--k', type=int, default=3, help="Jumlah hasil (untuk VSM).")
    parser.add_argument('--query', required=True, help="Kueri pencarian.")
    
    args = parser.parse_args()
    
    # Tentukan ROOT_DIR (asumsi search.py ada di root atau src/)
    # Sesuaikan ini jika perlu
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(current_dir, '..'))
    
    # Muat data yang sudah diproses
    doc_names, processed_corpus = load_processed_data(root_dir)
    
    print(f"--- Menjalankan Model: {args.model.upper()} ---")
    print(f"Kueri: {args.query}\n")
    
    if args.model == 'boolean':
        results = run_boolean_search(args.query, doc_names, processed_corpus)
        print("Hasil ditemukan:")
        if not results:
            print("Tidak ada dokumen yang cocok.")
        for doc in results:
            print(f"- {doc}")
            
    elif args.model == 'vsm':
        results = run_vsm_search(args.query, args.k, doc_names, processed_corpus)
        print(f"Hasil Top-{args.k}:")
        if not results:
            print("Tidak ada dokumen yang relevan.")
        for res in results:
            print(f"- Dokumen: {res['doc']}")
            print(f"  Skor: {res['score']:.4f}")
            print(f"  Explain: {res['explain']}\n")

if __name__ == "__main__":
    main()