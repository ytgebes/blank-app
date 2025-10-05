import streamlit as st
import json
import io
import time
import pandas as pd
import requests
from bs4 import BeautifulSoup
import PyPDF2
from functools import lru_cache
from streamlit_extras.let_it_rain import rain
from streamlit_extras.mention import mention
import google.generativeai as genai

# --- CONFIGURATION ---
MODEL_NAME = "gemini-1.5-flash"  # Corrected model name for better compatibility
st.set_page_config(page_title="Houston! We have a Problem!", layout="wide")

# --- GEMINI AI SETUP ---
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
except Exception as e:
    st.error(f"Error configuring Gemini AI: {e}")
    st.stop()

# --- INITIALIZE SESSION STATE ---
if 'summary_dict' not in st.session_state:
    st.session_state.summary_dict = {}
if 'lang' not in st.session_state:
    st.session_state.lang = "English"

# --- STYLING (No changes needed here) ---
st.markdown("""
    <style>
    /* Custom Nav button container for the top-left */
    .nav-container-ai {
        display: flex;
        justify-content: flex-start;
        padding-top: 3rem; 
        padding-bottom: 0rem;
    }
    .nav-button-ai a {
        background-color: #6A1B9A; /* Purple color */
        color: white; 
        padding: 10px 20px;
        border-radius: 8px; 
        text-decoration: none; 
        font-weight: bold;
        transition: background-color 0.3s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    .nav-button-ai a:hover { 
        background-color: #4F0A7B; /* Darker purple on hover */
    }
    /* HIDE STREAMLIT'S DEFAULT NAVIGATION (Sidebar hamburger menu) */
    [data-testid="stSidebar"] { display: none; }
    
    /* Push content to the top */
    .block-container { padding-top: 1rem !important; }
    
    /* Ensure no residual custom nav container is active */
    .nav-container { display: none; } 

    /* Main Theme */
    h1, h3 { text-align: center; }
    h1 { font-size: 4.5em !important; padding-bottom: 0.5rem; color: #000000; }
    h3 { color: #333333; }
    input[type="text"] {
        color: #000000 !important; background-color: #F0F2F6 !important;
        border: 1px solid #CCCCCC !important; border-radius: 8px; padding: 14px;
    }
    
    /* Result Card Styling (Full-Width) */
    .result-card {
        background-color: #FAFAFA; 
        padding: 1.5rem; 
        border-radius: 10px;
        margin-bottom: 1.5rem; /* More space between cards for UX */
        border: 1px solid #E0E0E0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Title Styling */
    .result-card .stMarkdown strong { 
        font-size: 1.15em; 
        display: block;
        margin-bottom: 10px; 
    }

    /* Consistent Purple Link Color */
    a { color: #6A1B9A; text-decoration: none; font-weight: bold; }
    a:hover { text-decoration: underline; }
    
    /* Summary Container (The inner block for summary text) */
    .summary-display {
        margin-top: 1rem;
        padding-top: 1rem;
        border-top: 1px dashed #CCC;
    }
    
    /* BUTTON: Full-width button now replaced with auto-width for single column */
    .stButton>button {
        border-radius: 8px; 
        width: auto; /* Auto width based on content */
        min-width: 200px; 
        background-color: #E6E0FF;
        color: #4F2083; 
        border: 1px solid #C5B3FF; 
        font-weight: bold;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover { background-color: #D6C9FF; border: 1px solid #B098FF; }
    
    /* Ensure Markdown headers in the summary are readable */
    .summary-display h3 {
        text-align: left !important;
        color: #4F2083;
        margin-top: 15px;
        margin-bottom: 5px;
        font-size: 1.3em;
    }
    </style>
""", unsafe_allow_html=True)

# --- TRANSLATIONS ---
TRANSLATIONS = {
    "English": {
        "page_title": "Houston! We have a Problem!",
        "header": 'Houston! We Have A<span style="color: #6A1B9A;"> Problem!</span>',
        "subheader": "Search, Discover, and Summarize NASA's Bioscience Publications",
        "search_placeholder": "Search publications... TELL US MORE!",
        "results_found": "Found {count} matching publications:",
        "no_results": "No matching publications found.",
        "summarize_button": "🔬 Gather & Summarize",
        "spinner_text": "Accessing and summarizing: {title}...",
        "summary_failed": "❌ Failed to Summarize:",
        "summary_error": "Error fetching/summarizing content: {error}",
        "assistant_ai_button": "Assistant AI 💬"
    },
    "Español": {
        "page_title": "¡Houston tenemos un problema!",
        "header": '¡Houston! Tenemos un<span style="color: #6A1B9A;"> Problema!</span>',
        "subheader": "Busque, Descubra y Resuma las Publicaciones de Biociencia de la NASA",
        "search_placeholder": "Buscar publicaciones... ¡CUÉNTANOS MÁS!",
        "results_found": "Se encontraron {count} publicaciones coincidentes:",
        "no_results": "No se encontraron publicaciones que coincidan.",
        "summarize_button": "🔬 Recopilar y Resumir",
        "spinner_text": "Accediendo y resumiendo: {title}...",
        "summary_failed": "❌ Falló al Resumir:",
        "summary_error": "Error al obtener/resumir el contenido: {error}",
        "assistant_ai_button": "Asistente IA 💬"
    },
    "Français": {
        "page_title": "Houston ! Nous avons un problème !",
        "header": 'Houston ! Nous avons un<span style="color: #6A1B9A;"> Problème !</span>',
        "subheader": "Recherchez, Découvrez et Résumez les Publications de biosciences de la NASA",
        "search_placeholder": "Rechercher des publications... DITES-NOUS EN PLUS !",
        "results_found": "{count} publications correspondantes trouvées :",
        "no_results": "Aucune publication correspondante trouvée.",
        "summarize_button": "🔬 Rassembler et Résumer",
        "spinner_text": "Accès et résumé de : {title}...",
        "summary_failed": "❌ Échec du Résumé :",
        "summary_error": "Erreur lors de la récupération/résumé du contenu : {error}",
        "assistant_ai_button": "Assistant IA 💬"
    }
    # Add other languages here following the same structure...
}

# --- HELPER FUNCTIONS ---
@st.cache_data
def load_data(file_path): 
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"File not found: {file_path}. Please ensure 'SB_publication_PMC.csv' is in the directory.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

@lru_cache(maxsize=128)
def fetch_url_text(url: str):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=20)
        r.raise_for_status()
    except requests.exceptions.RequestException as e: 
        return f"ERROR_FETCH: {e}"
    
    content_type = r.headers.get("Content-Type", "").lower()
    
    if "pdf" in content_type or url.lower().endswith(".pdf"):
        try:
            with io.BytesIO(r.content) as f:
                reader = PyPDF2.PdfReader(f)
                return "\n".join(p.extract_text() for p in reader.pages if p.extract_text())
        except Exception as e: 
            return f"ERROR_PDF_PARSE: {e}"
    else:
        try:
            soup = BeautifulSoup(r.text, "html.parser")
            for tag in soup(['script', 'style']): tag.decompose()
            # Truncate content for Gemini model context limit
            return " ".join(soup.body.get_text(separator=" ", strip=True).split())[:25000]
        except Exception as e: 
            return f"ERROR_HTML_PARSE: {e}"

def summarize_text_with_gemini(text: str):
    if not text or text.startswith("ERROR"): 
        return f"Could not summarize due to a content error: {text.split(': ')[-1]}"

    prompt = (f"Summarize this NASA bioscience paper. Output in clean Markdown with a level 3 heading (###) titled 'Key Findings' (using bullet points) and a level 3 heading (###) titled 'Overview Summary' (using a paragraph).\n\nContent:\n{text}")
    
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e: 
        return f"ERROR_GEMINI: {e}"

# --- MAIN PAGE FUNCTION ---
def search_page():
    # --- LANGUAGE SELECTION ---
    # Use columns to place the language selector on the right
    _, col2 = st.columns([0.85, 0.15]) # Adjust ratio as needed
    with col2:
        selected_lang = st.selectbox(
            "Select Language",
            options=list(TRANSLATIONS.keys()),
            index=list(TRANSLATIONS.keys()).index(st.session_state.lang),
            label_visibility="collapsed"
        )
    
    if selected_lang != st.session_state.lang:
        st.session_state.lang = selected_lang
        st.rerun()

    # Load the correct set of translations
    T = TRANSLATIONS[st.session_state.lang]

    # Set the translated page title
    st.set_page_config(page_title=T["page_title"], layout="wide")

    # Custom HTML Button for Assistant AI (using the translated text)
    st.markdown(
        f'<div class="nav-container-ai"><div class="nav-button-ai"><a href="/Assistant_AI" target="_self">{T["assistant_ai_button"]}</a></div></div>',
        unsafe_allow_html=True
    )
        
    # --- UI Header (using translated text) ---
    df = load_data("SB_publication_PMC.csv")
    st.markdown(f'<h1>{T["header"]}</h1>', unsafe_allow_html=True)
    st.markdown(f'<h3>{T["subheader"]}</h3>')

    search_query = st.text_input(
        "Search publications...", 
        placeholder=T["search_placeholder"], 
        label_visibility="collapsed"
    )
    
    # --- Search Logic ---
    if search_query:
        mask = df["Title"].astype(str).str.contains(search_query, case=False, na=False)
        results_df = df[mask].reset_index(drop=True)
        st.markdown("---")
        st.subheader(T["results_found"].format(count=len(results_df)))
        
        if results_df.empty:
            st.warning(T["no_results"])
        else:
            if 'summary_dict' not in st.session_state:
                st.session_state.summary_dict = {}
            
            for idx, row in results_df.iterrows():
                summary_key = f"summary_{idx}"
                
                with st.container():
                    st.markdown('<div class="result-card">', unsafe_allow_html=True)
                    
                    st.markdown(f"**Title:** <a href='{row['Link']}' target='_blank'>{row['Title']}</a>", unsafe_allow_html=True)
                    
                    if st.button(T["summarize_button"], key=f"btn_summarize_{idx}"):
                        spinner_text = T["spinner_text"].format(title=row['Title'])
                        with st.spinner(spinner_text):
                            try:
                                text = fetch_url_text(row['Link'])
                                summary = summarize_text_with_gemini(text)
                                st.session_state.summary_dict[summary_key] = summary
                            except Exception as e:
                                st.session_state.summary_dict[summary_key] = f"CRITICAL_ERROR: {e}"
                        st.rerun()

                    if summary_key in st.session_state.summary_dict:
                        summary_content = st.session_state.summary_dict[summary_key]
                        st.markdown('<div class="summary-display">', unsafe_allow_html=True)
                        
                        if summary_content.startswith("ERROR") or summary_content.startswith("CRITICAL_ERROR"):
                            st.markdown(f'**{T["summary_failed"]}** *{row["Title"]}*', unsafe_allow_html=True)
                            error_message = T["summary_error"].format(error=summary_content)
                            st.error(error_message)
                        else:
                            st.markdown(summary_content)
                            
                        st.markdown('</div>', unsafe_allow_html=True)
                            
                    st.markdown("</div>", unsafe_allow_html=True)

# --- APP NAVIGATION ---
# This setup uses Streamlit's new multipage app feature.
# Ensure you have a 'pages' directory with 'Assistant_AI.py' in it.
pg = st.navigation([
    st.Page(search_page, title="Simplified Knowledge 🔍"),
    st.Page("pages/Assistant_AI.py", title="Assistant AI 💬", icon="💬"),
])

pg.run()
