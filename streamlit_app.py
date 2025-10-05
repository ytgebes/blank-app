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

MODEL_NAME = "gemini-1.5-flash"

# --- INITIAL SETUP & CONFIGURATION ---
st.set_page_config(page_title="Simplified Knowledge", layout="wide")

try:
    # Check if the API key is set before configuring
    if st.secrets["GEMINI_API_KEY"]:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    else:
        st.error("GEMINI_API_KEY not found in secrets.")
        st.stop()
except Exception as e:
    st.error(f"Error configuring Gemini AI: {e}")
    st.stop()

# --- LANGUAGE DATA & TRANSLATION FUNCTIONS ---

# A clean, consistent dictionary for languages.
LANGUAGES = {
    "English": "en",
    "Español": "es",
    "Français": "fr",
    "Deutsch": "de",
    "Italiano": "it",
    "Türkçe": "tr",
    "日本語": "ja",
    "한국어": "ko",
    "हिन्दी": "hi",
}

# This function holds all UI text.
# In a real app, you might load this from JSON files per language.
@st.cache_data(show_spinner=False)
def get_translated_ui_strings(language: str):
    """
    Returns a dictionary of UI strings for the selected language.
    Includes fallbacks to English for any missing translations.
    """
    # Base English strings - the single source of truth for all keys
    UI_STRINGS_EN = {
        "title": "Simplified Knowledge",
        "description": "A dynamic dashboard that summarizes NASA bioscience publications and explores impacts and results.",
        "search_label": "Search publications...",
        "results_header": "Found {count} matching publications:",
        "no_results": "No matching publications found.",
        "summarize_button": "🔬 Gather & Summarize",
        "pdf_upload_header": "Upload PDFs to Summarize",
        "pdf_success": "✅ {count} PDF(s) uploaded and summarized",
        "pdf_summary_title": "📄 Summary: {name}",
        "assistant_ai_button": "Assistant AI 💬",
        "search_page_title": "Publication Search 🔍"
    }

    # Spanish Translations
    UI_STRINGS_ES = {
        "title": "Conocimiento Simplificado",
        "description": "Un panel dinámico que resume las publicaciones de biociencia de la NASA y explora sus impactos y resultados.",
        "search_label": "Buscar publicaciones...",
        "results_header": "Se encontraron {count} publicaciones coincidentes:",
        "no_results": "No se encontraron publicaciones coincidentes.",
        "summarize_button": "🔬 Recopilar y Resumir",
        "pdf_upload_header": "Subir PDFs para Resumir",
        "pdf_success": "✅ {count} PDF(s) subidos y resumidos",
        "pdf_summary_title": "📄 Resumen: {name}",
        "assistant_ai_button": "Asistente IA 💬",
        "search_page_title": "Búsqueda de Publicaciones 🔍"
    }
    
    # French Translations
    UI_STRINGS_FR = {
        "title": "Savoir Simplifié",
        "description": "Un tableau de bord dynamique qui résume les publications de biosciences de la NASA et explore les impacts et les résultats.",
        "search_label": "Rechercher des publications...",
        "results_header": "{count} publications correspondantes trouvées :",
        "no_results": "Aucune publication correspondante trouvée.",
        "summarize_button": "🔬 Rassembler et Résumer",
        "pdf_upload_header": "Télécharger des PDF à Résumer",
        "pdf_success": "✅ {count} PDF(s) téléchargé(s) et résumé(s)",
        "pdf_summary_title": "📄 Résumé : {name}",
        "assistant_ai_button": "Assistant IA 💬",
        "search_page_title": "Recherche de Publications 🔍"
    }

    # Logic to return the correct dictionary
    if language == "Español":
        # Use English as a base and update with Spanish translations
        return {**UI_STRINGS_EN, **UI_STRINGS_ES}
    if language == "Français":
        return {**UI_STRINGS_EN, **UI_STRINGS_FR}
    
    # Default to English
    return UI_STRINGS_EN

# --- INITIALIZE SESSION STATE ---
# This block runs only once when the app starts.
if 'current_lang' not in st.session_state:
    st.session_state.current_lang = "English" # Default language
if 'translated_strings' not in st.session_state:
    # Initialize with default English strings by calling the function
    st.session_state.translated_strings = get_translated_ui_strings("English")
if 'summary_dict' not in st.session_state:
    st.session_state.summary_dict = {}

# --- STYLING ---
st.markdown("""
<style>
/* ABSOLUTE POSITIONING FOR LANGUAGE DROPDOWN */
.language-dropdown-container {
    position: absolute;
    top: 1.5rem;
    right: 2rem;
    z-index: 1000;
}
/* HIDE STREAMLIT'S DEFAULT hamburger menu and header */
[data-testid="stSidebarNav"] { display: none; }
[data-testid="stHeader"] { visibility: hidden; }

/* Main Theme */
.block-container { padding-top: 2rem !important; }
h1 { font-size: 3.5em !important; text-align: center; }
h3 { color: #333333; text-align: center; }

/* Result Card Styling */
.result-card {
    background-color: #FAFAFA;
    padding: 1.5rem;
    border-radius: 10px;
    margin-bottom: 1.5rem;
    border: 1px solid #E0E0E0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}
.result-card .stMarkdown strong {
    font-size: 1.15em;
    display: block;
    margin-bottom: 10px;
}
a { color: #6A1B9A; text-decoration: none; font-weight: bold; }
a:hover { text-decoration: underline; }
.summary-display {
    margin-top: 1rem;
    padding-top: 1rem;
    border-top: 1px dashed #CCC;
}
</style>
""", unsafe_allow_html=True)


# --- LANGUAGE SELECTOR (TOP-RIGHT CORNER) ---
with st.container():
    st.markdown('<div class="language-dropdown-container">', unsafe_allow_html=True)
    
    # Find the index of the current language to set the default for the selectbox
    lang_options = list(LANGUAGES.keys())
    current_lang_index = lang_options.index(st.session_state.current_lang)

    # The visible language selector
    selected_language = st.selectbox(
        "Language",
        options=lang_options,
        index=current_lang_index,
        label_visibility="collapsed" # Hides the label "Language"
    )
    
    st.markdown('</div>', unsafe_allow_html=True)

# --- CORE LOGIC TO UPDATE LANGUAGE ---
# If the user has chosen a new language from the dropdown...
if selected_language != st.session_state.current_lang:
    # ...update the session state.
    st.session_state.current_lang = selected_language
    st.session_state.translated_strings = get_translated_ui_strings(selected_language)
    # ...and rerun the app to apply the changes everywhere.
    st.rerun()


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
    # This function remains unchanged
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
            return " ".join(soup.body.get_text(separator=" ", strip=True).split())[:25000]
        except Exception as e:
            return f"ERROR_HTML_PARSE: {e}"

def summarize_text_with_gemini(text: str):
    # This function remains unchanged
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
    # Load current translation from session state
    t = st.session_state.translated_strings

    # --- Main Page Content ---
    # UI Header using translated strings
    st.markdown(f'<h1>{t["title"].split()[0]} <span style="color: #6A1B9A;">{t["title"].split()[-1]}</span></h1>', unsafe_allow_html=True)
    st.markdown(f"### {t['description']}")

    search_query = st.text_input(t["search_label"], placeholder="e.g., microgravity, radiation, Artemis...", label_visibility="collapsed")
    
    df = load_data("SB_publication_PMC.csv")
    
    # --- PDF Summaries Display ---
    with st.sidebar:
        st.header("Settings ⚙️")
        st.markdown(f"<h3 style='text-align: left; margin-top: 20px;'>{t['pdf_upload_header']}</h3>", unsafe_allow_html=True)
        uploaded_files = st.file_uploader(label="Upload PDFs", type=["pdf"], accept_multiple_files=True, label_visibility="collapsed")
    
    if uploaded_files:
        st.success(t['pdf_success'].format(count=len(uploaded_files)))
        st.markdown("---")
        for uploaded_file in uploaded_files:
            summary_key = f"pdf_summary_{uploaded_file.name}"
            
            if summary_key not in st.session_state.summary_dict:
                pdf_bytes = io.BytesIO(uploaded_file.read())
                pdf_reader = PyPDF2.PdfReader(pdf_bytes)
                text = "".join([p.extract_text() or "" for p in pdf_reader.pages])
                
                with st.spinner(f"Summarizing: {uploaded_file.name} ..."):
                    summary = summarize_text_with_gemini(text)
                    st.session_state.summary_dict[summary_key] = summary
            
            st.markdown(f"### {t['pdf_summary_title'].format(name=uploaded_file.name)}")
            st.write(st.session_state.summary_dict[summary_key])
        st.markdown("---")

    # --- Search Logic ---
    if search_query:
        mask = df['Title'].astype(str).str.contains(search_query, case=False, na=False)
        results_df = df[mask].reset_index(drop=True)
        st.markdown("---")
        st.subheader(t['results_header'].format(count=len(results_df)))
        
        if results_df.empty:
            st.warning(t['no_results'])
        else:
            for idx, row in results_df.iterrows():
                summary_key = f"summary_{idx}_{row['PMCID']}"
                
                with st.container():
                    st.markdown('<div class="result-card">', unsafe_allow_html=True)
                    
                    title = row['Title']
                    link = row['Link']
                    
                    st.markdown(f"**Title:** <a href='{link}' target='_blank'>{title}</a>", unsafe_allow_html=True)
                    
                    if st.button(t["summarize_button"], key=f"btn_summarize_{idx}"):
                        with st.spinner(f"Accessing and summarizing: {title}..."):
                            try:
                                text = fetch_url_text(link)
                                summary = summarize_text_with_gemini(text)
                                st.session_state.summary_dict[summary_key] = summary
                            except Exception as e:
                                st.session_state.summary_dict[summary_key] = f"CRITICAL_ERROR: {e}"
                        st.rerun()

                    if summary_key in st.session_state.summary_dict:
                        summary_content = st.session_state.summary_dict[summary_key]
                        
                        st.markdown('<div class="summary-display">', unsafe_allow_html=True)
                        if summary_content.startswith("ERROR") or summary_content.startswith("CRITICAL_ERROR"):
                            st.error(f"Error fetching/summarizing content: {summary_content}")
                        else:
                            st.markdown(summary_content)
                        st.markdown('</div>', unsafe_allow_html=True)
                            
                    st.markdown("</div>", unsafe_allow_html=True)

# --- STREAMLIT PAGE NAVIGATION ---
# Using st.Page for multi-page apps is the modern approach
pg = st.navigation([
    st.Page(search_page, title=st.session_state.translated_strings["search_page_title"], default=True),
    st.Page("pages/Assistant_AI.py", title=st.session_state.translated_strings["assistant_ai_button"]),
])

pg.run()
