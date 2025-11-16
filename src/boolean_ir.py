# --- src/boolean_ir.py (VERSI KETAT) ---

from src.preprocess import preprocess_text
from scipy.sparse import csr_matrix
import pandas as pd
import numpy as np

def build_inverted_index(docs_preprocessed_map, doc_names):
    """
    Membangun Inverted Index (dict: term -> set of doc_names).
    """
    inverted_index = {}
    
    for doc_name, tokens in docs_preprocessed_map.items():
        for term in tokens:
            if term not in inverted_index:
                inverted_index[term] = set()
            inverted_index[term].add(doc_name)
            
    print("Indeks Boolean (Inverted Index) berhasil dibangun.")
    return inverted_index

def build_incidence_matrix(docs_preprocessed_map, doc_names):
    """
    Membangun Incidence Matrix (Sparse Matrix) untuk Soal 03.2.a.
    """
    vocabulary = sorted(list(set(term for tokens in docs_preprocessed_map.values() for term in tokens)))
    term_to_index = {term: i for i, term in enumerate(vocabulary)}
    
    rows, cols, data = [], [], []
    
    for doc_index, doc_name in enumerate(doc_names):
        tokens = docs_preprocessed_map.get(doc_name, [])
        unique_tokens = set(tokens)
        
        for term in unique_tokens:
            if term in term_to_index:
                rows.append(term_to_index[term])
                cols.append(doc_index)
                data.append(1) 

    incidence_matrix = csr_matrix(
        (data, (rows, cols)),
        shape=(len(vocabulary), len(doc_names))
    )
    
    metadata = {
        'vocabulary': vocabulary,
        'doc_names': doc_names,
        'term_to_index': term_to_index
    }
    
    print("Incidence Matrix (CSR) berhasil dibangun.")
    return incidence_matrix, metadata


def get_postings(term, inverted_index):
    """
    Helper function untuk mengambil postings, menggunakan preprocess_text.
    """
    clean_tokens = preprocess_text(term)
    
    if not clean_tokens:
        return set() 
        
    clean_term = clean_tokens[0]
    
    return inverted_index.get(clean_term, set())

def parse_boolean_query(query, inverted_index):
    """
    Parser Boolean Ketat: Hanya memproses Term [OPERATOR Term]...
    Jika operator tidak ada atau tidak dikenal, akan mengembalikan set kosong.
    """
    query_tokens = query.split()
    num_tokens = len(query_tokens)
    
    if num_tokens == 0:
        return set()
    
    # Kasus kueri tunggal (misal: "magang")
    if num_tokens == 1:
        return get_postings(query_tokens[0], inverted_index)
    
    # --- Perbaikan Logika Strictness ---
    # Jika jumlah token GENAP (misal 2, 4, 6) DAN operator di tengah tidak ada, harus gagal.
    # Kita mulai dengan term pertama
    current_result_set = get_postings(query_tokens[0], inverted_index)
    
    i = 1
    while i < num_tokens:
        
        operator = query_tokens[i].upper()
        
        # 1. Cek Operator yang Tidak Dikenal
        if operator not in ['AND', 'OR', 'NOT']:
            # GAGAL: Token di posisi 'i' seharusnya adalah operator, tapi malah 'website'
            return set() 
            
        # 2. Cek Jika Kueri Berakhir dengan Operator
        if i + 1 >= num_tokens:
            return set() # Gagal: Kueri berakhir dengan operator (misal: "magang AND")
            
        next_term = query_tokens[i+1]
        next_term_postings = get_postings(next_term, inverted_index)
        
        # Lakukan Operasi Boolean
        if operator == 'AND':
            current_result_set = current_result_set.intersection(next_term_postings)
        elif operator == 'OR':
            current_result_set = current_result_set.union(next_term_postings)
        elif operator == 'NOT':
            current_result_set = current_result_set.difference(next_term_postings)
            
        i += 2 # Lompat melewati operator dan term berikutnya
            
    return current_result_set