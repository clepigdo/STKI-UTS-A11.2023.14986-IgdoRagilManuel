

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np



def build_vsm_model(processed_corpus_text):
    """
    Membangun model VSM dari daftar teks yang sudah diproses.
    Menerima: list of strings (e.g., ["kata satu", "kata dua"])
    
    Nama parameter 'processed_corpus_text' sengaja dibuat
    agar sama dengan yang dikirim dari main.py
    """
    
    # 1. Buat Vectorizer
    
    vectorizer = TfidfVectorizer()
    
    # 2. Buat TF-IDF Matrix
    
    
    tfidf_matrix_docs = vectorizer.fit_transform(processed_corpus_text)
    
    print("Model VSM (TfidfVectorizer) berhasil dibangun di vsm_ir.py.")
    
    # Kembalikan model yang sudah 'di-fit'
    return (vectorizer, tfidf_matrix_docs)


def search_vsm(query_text, vsm_model, doc_names, preprocessed_docs, k=5):
    """
    Mencari kueri di model VSM.
    Parameter di sini HARUS sinkron dengan panggilan di app/main.py
    """
    
    # 1. Unpack model
    vectorizer, tfidf_matrix_docs = vsm_model
    
    # 2. Preprocess kueri PENGGUNA (string mentah)
    
    try:
        from src.preprocess import preprocess_text
    except ImportError:
        print("ERROR: search_vsm tidak bisa impor preprocess_text")
        return []
        
    clean_tokens = preprocess_text(query_text)
    clean_query = ' '.join(clean_tokens)
    
    if not clean_query:
        print("Kueri VSM kosong setelah preprocessing.")
        return []
    
    # 3. Ubah kueri bersih menjadi Vektor TF-IDF
    query_vector = vectorizer.transform([clean_query])
    
    # 4. Hitung Cosine Similarity
    cosine_scores = cosine_similarity(query_vector, tfidf_matrix_docs).flatten()
    
    # 5. Ambil Top-k
    top_k_indices = cosine_scores.argsort()[-k:][::-1]
    
    # 6. Format hasil
    results = []
    for index in top_k_indices:
        score = cosine_scores[index]
        if score > 0.0: # Hanya tampilkan jika relevan
            results.append((doc_names[index], score))
            
    return results