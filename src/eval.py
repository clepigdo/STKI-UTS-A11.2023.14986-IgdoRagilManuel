import numpy as np

# ---
# METRIK UNTUK BOOLEAN (Soal 03)
# ---

def calculate_precision_recall_f1(retrieved_set, relevant_set):
    """
    Menghitung Precision, Recall, dan F1-Score (sesuai Soal 03 & 05).
    
    Args:
        retrieved_set (set): Himpunan dokumen yang ditemukan oleh sistem.
        relevant_set (set): Himpunan dokumen yang benar (gold standard).
    
    Returns:
        tuple: (precision, recall, f1_score)
    """
    if not retrieved_set and not relevant_set:
        return 1.0, 1.0, 1.0 # Kasus ideal jika tidak ada yg dicari dan tidak ada yg ditemukan

    true_positives_set = retrieved_set.intersection(relevant_set)
    
    # Hitung TP, FP, FN
    tp = len(true_positives_set)
    fp = len(retrieved_set - relevant_set)
    fn = len(relevant_set - retrieved_set)
    
    # Precision
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    # Recall
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # F1-Score
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1_score

# ---
# METRIK UNTUK VSM / RANKING (Soal 04 & 05)
# ---

def calculate_precision_at_k(retrieved_ranked_list, relevant_set, k):
    """
    Menghitung Precision@k (P@k) (Sesuai Soal 04).
    
    Args:
        retrieved_ranked_list (list): Daftar dokumen HASIL VSM (sudah terurut).
        relevant_set (set): Himpunan dokumen yang benar (gold standard).
        k (int): Batas ranking (misal: 3 atau 5).
    
    Returns:
        float: Nilai P@k
    """
    top_k_retrieved = retrieved_ranked_list[:k]
    
    true_positives_at_k = 0
    for doc in top_k_retrieved:
        if doc in relevant_set:
            true_positives_at_k += 1
            
    precision_at_k = true_positives_at_k / k if k > 0 else 0.0
    return precision_at_k

# ---
# METRIK UNTUK MEAN AVERAGE PRECISION (MAP) (Soal 04 & 05)
# ---

def calculate_average_precision_at_k(retrieved_ranked_list, relevant_set, k):
    """
    Menghitung Average Precision (AP@k) untuk SATU kueri.
    Ini adalah dasar dari MAP.
    
    Args:
        retrieved_ranked_list (list): Daftar dokumen HASIL VSM (terurut).
        relevant_set (set): Himpunan dokumen yang benar (gold standard).
        k (int): Batas ranking.
    
    Returns:
        float: Nilai AP@k
    """
    if not relevant_set:
        return 0.0 # Tidak ada yang relevan, skor 0

    top_k_retrieved = retrieved_ranked_list[:k]
    
    precision_sum = 0.0
    true_positives_count = 0
    
    for i, doc in enumerate(top_k_retrieved):
        rank = i + 1
        if doc in relevant_set:
            true_positives_count += 1
            precision_at_i = true_positives_count / rank
            precision_sum += precision_at_i
            
    if true_positives_count == 0:
        return 0.0
        
    average_precision = precision_sum / len(relevant_set) # Dibagi total relevan, BUKAN k
    return average_precision

def calculate_map_at_k(all_queries_results, k):
    """
    Menghitung Mean Average Precision (MAP@k) untuk BANYAK kueri.
    
    Args:
        all_queries_results (list of dicts): 
            Contoh: [
                {'query': 'q1', 'retrieved': ['doc1', 'doc2'], 'relevant': {'doc1'}},
                {'query': 'q2', 'retrieved': ['doc3', 'doc1'], 'relevant': {'doc3', 'doc1'}}
            ]
        k (int): Batas ranking.
        
    Returns:
        float: Nilai MAP@k
    """
    ap_sum = 0.0
    if not all_queries_results:
        return 0.0
        
    for result in all_queries_results:
        ap = calculate_average_precision_at_k(
            result['retrieved'], 
            result['relevant'], 
            k
        )
        ap_sum += ap
        
    return ap_sum / len(all_queries_results)

# ---
# METRIK UNTUK nDCG (Soal 04 & 05)
# ---

def calculate_dcg_at_k(retrieved_ranked_list, relevant_set, k):
    """
    Menghitung Discounted Cumulative Gain (DCG@k).
    (Menggunakan relevance score biner: 1 jika relevan, 0 jika tidak)
    """
    dcg = 0.0
    top_k_retrieved = retrieved_ranked_list[:k]
    
    for i, doc in enumerate(top_k_retrieved):
        rank = i + 1
        relevance = 1 if doc in relevant_set else 0
        
        # Rumus DCG: relevance / log2(rank + 1)
        # (rank 1 pakai log2(2), rank 2 pakai log2(3), dst.)
        dcg += relevance / np.log2(rank + 1) 
        
    return dcg

def calculate_idcg_at_k(relevant_set, k):
    """
    Menghitung Ideal Discounted Cumulative Gain (IDCG@k).
    Ini adalah skor DCG maksimum yang mungkin didapat.
    """
    # Skor ideal terjadi jika semua dokumen relevan ada di peringkat teratas.
    num_relevant = len(relevant_set)
    
    idcg = 0.0
    # Kita hanya peduli sampai k, atau sampai jumlah relevan habis
    for i in range(min(k, num_relevant)):
        rank = i + 1
        idcg += 1 / np.log2(rank + 1) # Relevance selalu 1
        
    return idcg

def calculate_ndcg_at_k(retrieved_ranked_list, relevant_set, k):
    """
    Menghitung Normalized Discounted Cumulative Gain (nDCG@k).
    
    Args:
        retrieved_ranked_list (list): Daftar dokumen HASIL VSM (terurut).
        relevant_set (set): Himpunan dokumen yang benar (gold standard).
        k (int): Batas ranking.
        
    Returns:
        float: Nilai nDCG@k
    """
    dcg_at_k = calculate_dcg_at_k(retrieved_ranked_list, relevant_set, k)
    idcg_at_k = calculate_idcg_at_k(relevant_set, k)
    
    if idcg_at_k == 0:
        return 0.0 # Tidak ada dokumen relevan
        
    ndcg = dcg_at_k / idcg_at_k
    return ndcg

def calculate_mean_ndcg_at_k(all_queries_results, k):
    """
    Menghitung Rata-rata nDCG@k untuk BANYAK kueri.
    """
    ndcg_sum = 0.0
    if not all_queries_results:
        return 0.0
        
    for result in all_queries_results:
        ndcg = calculate_ndcg_at_k(
            result['retrieved'],
            result['relevant'],
            k
        )
        ndcg_sum += ndcg
    
    return ndcg_sum / len(all_queries_results)