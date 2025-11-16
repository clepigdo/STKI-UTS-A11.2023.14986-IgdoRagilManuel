import streamlit as st
import time
import sys
import os
import glob
import re # Diperlukan untuk helper snippet
import streamlit as st
import nltk

# Fungsi ini akan di-cache, hanya jalan sekali saat deploy
@st.cache_resource
def download_nltk_data():
    """Unduh resource NLTK yang diperlukan."""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab')
    print("Data NLTK sudah siap.")

download_nltk_data()


try:
    current_file_path = os.path.abspath(__file__)
except NameError:
    current_file_path = os.path.abspath(os.getcwd())

project_root = os.path.dirname(current_file_path) # Ini adalah folder 'app'
parent_root = os.path.dirname(project_root)      # Ini adalah folder 'stki-uts-...'

if parent_root not in sys.path:
    sys.path.insert(0, parent_root)

# --- 2. Import dari Modul 'src' ---
try:
    from src.preprocess import preprocess_text
    from src.boolean_ir import build_inverted_index, parse_boolean_query
    from src.vsm_ir import build_vsm_model, search_vsm
except ImportError as e:
    st.error(f"Gagal mengimpor modul 'src'. Pastikan folder 'src' ada di sebelah folder 'app'. Error: {e}")
    st.stop()


# --- 3. Import NLTK untuk Kalimat ---
import nltk
from nltk.tokenize import sent_tokenize
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


st.set_page_config(
    page_title="🚀 Mesin Pencari STKI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- FUNGSI HELPER ---
def load_documents_from_folder(data_folder):
    """
    Membaca semua file .txt dari folder data.
    """
    data_path = os.path.join(parent_root, data_folder, "*.txt")
    doc_files = glob.glob(data_path)
    
    docs_raw_map = {}
    doc_names = []
    
    if not doc_files:
        st.error(f"Tidak ada file .txt yang ditemukan di path: {data_path}. Pastikan folder 'data' ada di sebelah folder 'app' dan 'src'.")
        raise FileNotFoundError(f"Tidak ada file .txt yang ditemukan di path: {data_path}")
        
    for file_path in doc_files:
        doc_name = os.path.basename(file_path)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                docs_raw_map[doc_name] = f.read()
                doc_names.append(doc_name)
        except Exception as e:
            print(f"Gagal membaca {file_path} dengan utf-8: {e}. Mencoba 'latin-1'...")
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    docs_raw_map[doc_name] = f.read()
                    doc_names.append(doc_name)
            except Exception as e2:
                print(f"Gagal membaca {file_path} dengan encoding apapun: {e2}")
            
    print(f"Berhasil memuat {len(doc_names)} dokumen.")
    return docs_raw_map, doc_names

def create_snippet(text, query_terms):
    """
    Membuat snippet (kalimat kunci) sesuai Soal 05.3.b.
    Ini akan menyorot kata kunci kueri dalam kalimat terbaik.
    """
    try:
        sentences = sent_tokenize(text)
    except Exception as e:
        print(f"NLTK sent_tokenize gagal: {e}. Menggunakan fallback.")
        sentences = text.split('.') 
        
    best_sentence = sentences[0] if sentences else text[:150] # Fallback kalimat pertama
    max_hits = 0

    query_terms_lower = {term.lower() for term in query_terms}

    # Loop untuk menemukan kalimat terbaik
    for sentence in sentences:
        hits = 0
        sentence_lower = sentence.lower()
        for term in query_terms_lower:
            if term in sentence_lower:
                hits += 1
        if hits > max_hits:
            max_hits = hits
            best_sentence = sentence 
    
    # Jika tidak ada kalimat yang cocok, ambil kalimat pertama
    if max_hits == 0:
        best_sentence = sentences[0] if sentences else text[:150]


    for term in query_terms:
        try:
            best_sentence = re.sub(f"({re.escape(term)})", r"**\1**", best_sentence, flags=re.IGNORECASE)
        except re.error:
            pass 
    
    
    if len(best_sentence) > 250:
        best_sentence = "..." + best_sentence[:250] + "..."
        
    return best_sentence


# --- FUNGSI LOAD MODEL ---
@st.cache_resource(show_spinner="🧙‍♂️ Merapal mantra TF-IDF... Harap sabar, ya!")
def load_all_models_from_src(data_folder='data'):
    """
    Wrapper untuk memuat data dan membangun model dari 'src'.
    """
    try:
        docs_raw_map, doc_names = load_documents_from_folder(data_folder)
        
        docs_preprocessed_map = {} 
        docs_preprocessed_list = [] 
        
        print("Memulai preprocessing...")
        for name in doc_names:
            raw_text = docs_raw_map[name]
            tokens = preprocess_text(raw_text) # Menggunakan fungsi dari src
            
            docs_preprocessed_map[name] = tokens
            docs_preprocessed_list.append(" ".join(tokens))
            
            if not tokens:
                print(f"PERINGATAN: Dokumen '{name}' menjadi kosong setelah preprocessing.")

        print("Preprocessing selesai.")

        if not any(docs_preprocessed_list):
            print("KESALAHAN FATAL: Semua dokumen kosong setelah preprocessing.")
            raise ValueError(
                
            )

        print("Membangun model VSM...")
        vsm_model = build_vsm_model(docs_preprocessed_list)
        print("Model VSM berhasil dibangun.")
        
        print("Membangun Indeks Boolean...")
        boolean_index = build_inverted_index(docs_preprocessed_map, doc_names)
        print("Indeks Boolean berhasil dibangun.")
        
        return docs_raw_map, doc_names, vsm_model, boolean_index, docs_preprocessed_list, None
    
    except Exception as e:
        print(f"Error di load_all_models_from_src: {e}")
        import traceback
        traceback.print_exc() # Cetak traceback lengkap ke konsol
        return None, None, None, None, None, str(e) 

# --- INISIALISASI APLIKASI ---
with st.spinner('✨ Menyulap data menjadi informasi... Hampir siap! ✨'):
    docs_map, names, vsm_model, boolean_index, docs_vsm_list, error_msg = load_all_models_from_src(data_folder='data')

if error_msg:
    st.error(f"💥 OH TIDAK! Terjadi kesalahan fatal saat memuat model:\n\n{error_msg}")
    st.stop() 

# --- Disini Saya Menggunakan Inline Custom CSS ---
st.markdown(
    """
    <style>
   
    .stApp {
        background-color: #0E1117; 
        color: #e0e0e0;
    }
    .css-1d3f8gv { 
        background-color: #1a1a2e; 
    }
    .stTextInput>div>div>input {
        background-color: #262730;
        color: #e0e0e0;
        border-radius: 8px;
        border: 1px solid #3e4451;
    }
    .stButton>button {
        background-color: #6a0572; 
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
        font-weight: bold;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #9200a3;
    }
    h1 {
        color: #f7b32b; 
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
    }
    h3 {
        color: #e76f51;
    }
    .stAlert { border-radius: 8px; }
    

    .result-card {
        background-color: #1a1a2e;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #6a0572;
        margin-bottom: 15px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
        transition: box-shadow 0.3s ease-in-out;
    }
    .result-card:hover {
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.5);
    }
    .result-title a {
        font-size: 22px;
        color: #87CEEB; 
        text-decoration: none;
        font-weight: 600;
    }
    .result-url {
        font-size: 14px;
        color: #90ee90; 
        margin-bottom: 8px;
    }
    .result-snippet {
        font-size: 16px;
        color: #e0e0e0;
        line-height: 1.6;
    }
    .result-snippet strong, .result-snippet b {
        color: #f7b32b; 
        font-weight: bold;
    }
    .result-footer {
        font-size: 13px;
        color: #bdbdbd;
        margin-top: 15px;
        border-top: 1px solid #3e4451;
        padding-top: 10px;
        text-align: right;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- SIDEBAR (NAVIGASI & OPSI) ---
with st.sidebar:
    st.title("🕹️ Panel Kontrol STKI")
    st.caption("A11.2023.14986 - Igdo Ragil Manuel")
    
    st.markdown("### 1. Pilih Model Pencarian")
    search_mode = st.radio(
        "Pilih Model",
        ("✨ VSM (Ranking & Relevansi)", "🎯 Boolean (Pencarian Tepat)"),
        index=0
    )
    
    st.markdown("---")
    
    if search_mode.startswith("✨ VSM"):
        st.markdown("### 2. Atur Parameter VSM")
        top_k = st.slider(
            "Atur Top-K", 
            min_value=1, max_value=20, value=5, 
            help="Pilih jumlah dokumen teratas"
        )
    else:
        st.markdown("### 2. Petunjuk Model Boolean")
        st.info(
            """
            Gunakan operator:
            - `AND`: `admin AND semarang`
            - `OR`: `magang OR internship`
            - `NOT`: `designer NOT freelance`
            """
        )
    
    st.markdown("---")
    st.caption(f"📚 Total Dokumen Terindeks: **{len(docs_map)}**")


# --- HALAMAN UTAMA (MAIN CONTENT) ---

# Judul Utama
st.markdown(
    
    unsafe_allow_html=True
)

st.write("")

# --- FORM PENCARIAN ---
with st.form(key='search_form', clear_on_submit=False):
    col1, col2 = st.columns([4, 1])
    
    with col1:
        query = st.text_input(
            label="Apa yang sedang kamu cari hari ini?",
            placeholder="Contoh: 'magang digital marketing' atau 'admin AND semarang'",
            label_visibility="collapsed",
            key="search_query"
        )
    
    with col2:
        submit_button = st.form_submit_button(label='🚀 Tembak Kueri!', use_container_width=True)

# --- LOGIKA PENCARIAN & MENAMPILKAN HASIL ---
if submit_button:
    if not query.strip():
        st.warning("🧐 Eh, kuerinya masih kosong! Coba ketik sesuatu.")
        st.balloons()
    else:
        st.markdown("---")
        st.header(f"✨ Hasil Pencarian untuk: '{query}'")
        
        start_time = time.time() 
        
        # --- MODE 1: VSM ---
        if search_mode.startswith("✨ VSM"):
            with st.spinner(f"🚀 Menganalisis matriks TF-IDF untuk '{query}'..."):
                
                try:
                    # Panggil search_vsm dari src/vsm_ir.py
                    results_scores = search_vsm(
                        query_text=query,
                        vsm_model=vsm_model,
                        doc_names=names,
                        preprocessed_docs=docs_vsm_list, 
                        k=top_k
                    )
                except Exception as e:
                    st.error(f"💥 Terjadi error saat menjalankan `search_vsm`: {e}")
                    results_scores = []
                
                # Buat 'results' dengan snippet
                results = []
                query_terms = list(set(preprocess_text(query)))
                
                for doc_name, score in results_scores:
                    full_text = docs_map[doc_name]
                    # Buat snippet 
                    snippet = create_snippet(full_text, query_terms) 
                    results.append((doc_name, score, snippet))
            
            duration = time.time() - start_time
            
            if not results:
                 st.error("😔 Aduh! Tidak ada dokumen VSM yang relevan. Coba kueri lain.")
                 st.snow()
            else:
                st.success(f"✅ Misi Selesai! Ditemukan **{len(results)}** dokumen relevan dalam **{duration:.4f} detik**.")
                st.balloons()
                
                for i, (doc_name, score, snippet) in enumerate(results):
                    judul_bersih = doc_name.replace(".txt", "").replace("doc", "").replace("_", " ").strip().title()
                    
                    # --- TAMPILAN CARD (VSM) ---
                    st.markdown(
                        f"""
                        <div class="result-card">
                            <div class="result-title"><a href="#">{i+1}. {judul_bersih}</a></div>
                            <div class="result-url">🔗 {doc_name}</div>
                            <div class="result-snippet">{snippet}</div>
                            <div class="result-footer">
                                <b>SKOR COSINE (Soal 04): {score:.4f}</b>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    st.write("") 

        # --- MODE 2: BOOLEAN ---
        else: # Mode Boolean
            st.subheader(f"🎯 Hasil Boolean untuk: '{query}'")
            
            try:
                with st.spinner(f"🕵️‍♀️ Memeriksa Inverted Index untuk '{query}'..."):
                    results_set = parse_boolean_query(query, boolean_index)
                    results = sorted(list(results_set)) # Urutkan A-Z
                
                duration = time.time() - start_time
                
                if not results:
                    st.warning("⚠️ Tidak ada dokumen yang cocok 100% (Boolean). Coba 'OR'?")
                    st.snow()
                else:
                    st.success(f"🎉 Ditemukan **{len(results)}** dokumen yang cocok dalam **{duration:.4f} detik**.")
                    st.balloons()
                    
                    query_terms = list(set(preprocess_text(query.replace("AND", " ").replace("OR", " ").replace("NOT", " "))))

                    for i, doc_name in enumerate(results):
                        full_text = docs_map[doc_name]
                        judul_bersih = doc_name.replace(".txt", "").replace("doc", "").replace("_", " ").strip().title()
                        snippet = create_snippet(full_text, query_terms) # snippet untuk Boolean

                        # --- TAMPILAN CARD (BOOLEAN) ---
                        st.markdown(
                            f"""
                            <div class="result-card" style="border-left-color: #008080;"> 
                                <div class="result-title"><a href="#">{i+1}. {judul_bersih}</a></div>
                                <div class="result-url">🔗 {doc_name}</div>
                                <div class="result-snippet">{snippet}</div>
                                <div class="result-footer" style="color: #00C49A;">
                                    <b>✅ Kecocokan Tepat (Boolean)</b>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        st.write("") 
                        
            except Exception as e:
                 st.error(f"💥 Ups, ada yang salah dengan mantra Booleanmu. Pastikan sintaks (AND, OR, NOT) sudah benar.\nError: {e}")
else:
   
    st.info("Selamat datang! Silakan pilih model dan masukkan kueri di atas untuk memulai pencarian. 🚀")