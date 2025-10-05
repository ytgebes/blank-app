import streamlit as st
import io
import pandas as pd
import requests
from bs4 import BeautifulSoup
import PyPDF2
from functools import lru_cache
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

# --- LANGUAGE & TRANSLATION SETUP ---

# A clean, simple dictionary mapping display names to language codes.
LANGUAGES = {
    "English": "en",
    "Español": "es",
    "Français": "fr",
    "Deutsch": "de",
    "Italiano": "it",
    "Türkçe": "tr",
    "日本語": "ja",
    "한국어": "ko",
    "हिन्दी": "hi"
}

# Master dictionary for all UI strings. This is cleaner than a long if/elif chain.
ALL_TRANSLATIONS = {
    "en": {
        "title": "Simplified Knowledge",
        "description": "A dynamic dashboard that summarizes NASA bioscience publications and explores impacts and results.",
        "search_label": "Search publications...",
        "results_header": "Found {count} matching publications:",
        "no_results": "No matching publications found.",
        "summarize_button": "🔬 Gather & Summarize",
        "pdf_upload_header": "Upload PDFs to Summarize",
        "sidebar_settings_header": "Settings ⚙️",
        "pdf_success": "✅ {count} PDF(s) uploaded and summarized",
        "pdf_summary_title": "📄 Summary: {name}",
        "nav_search": "Search Publications 🔍",
        "nav_assistant": "Assistant AI 💬"
    },
    "es": {
        "title": "Conocimiento Simplificado",
        "description": "Un panel dinámico que resume las publicaciones de biociencia de la NASA y explora sus impactos y resultados.",
        "search_label": "Buscar publicaciones...",
        "results_header": "Se encontraron {count} publicaciones coincidentes:",
        "no_results": "No se encontraron publicaciones coincidentes.",
        "summarize_button": "🔬 Recopilar y Resumir",
        "pdf_upload_header": "Subir PDFs para Resumir",
        "sidebar_settings_header": "Ajustes ⚙️",
        "pdf_success": "✅ {count} PDF(s) subidos y resumidos",
        "pdf_summary_title": "📄 Resumen: {name}",
        "nav_search": "Buscar Publicaciones 🔍",
        "nav_assistant": "Asistente IA 💬"
    }
    # Add dictionaries for other languages (fr, de, etc.) here
}

# --- INITIALIZE SESSION STATE ---
if 'current_lang' not in st.session_state:
    st.session_state.current_lang = "English"  # Default language
if 'translated_strings' not in st.session_state:
    # Initialize with default English strings from the master dictionary
    st.session_state.translated_strings = ALL_TRANSLATIONS["en"]
if 'summary_dict' not in st.session_state:
    st.session_state.summary_dict = {}

# --- CORE FUNCTIONS ---

def update_language():
    """Callback function to update UI strings when language is changed."""
    selected_lang_name = st.session_state.language_selector
    lang_code = LANGUAGES.get(selected_lang_name, "en") # Default to 'en'

    # Update session state
    st.session_state.current_lang = selected_lang_name
    # Fallback to English if a translation isn't available in ALL_TRANSLATIONS
    st.session_state.translated_strings = ALL_TRANSLATIONS.get(lang_code, ALL_TRANSLATIONS["en"])

@st.cache_data(show_spinner=False)
def translate_list_via_gemini(text_list: list, target_lang: str):
    """Placeholder for translating dataframe columns."""
    if target_lang == "English":
        return text_list
    # In a real app, you would call the Gemini API here.
    # prompt = f"Translate the following list of scientific column headers into {target_lang}. Return only the translated list, comma-separated:\n{', '.join(text_list)}"
    # return model.generate_content(prompt).text.split(', ')
    return [f"Translated_{item}" for item in text_list] # Mock translation for demonstration

# --- STYLING ---
st.markdown("""
<style>
    /* Main Theme */
    .block-container { padding-top: 1rem !important; }
    h1, h3 { text-align: center; }
    h1 { font-size: 4.5em !important; padding-bottom: 0.5rem; color: #000000; }
    h3 { color: #333333; }
    a { color: #6A1B9A; text-decoration: none; font-weight: bold; }
    a:hover { text-decoration: underline; }

    /* Floating Language Dropdown */
    .language-dropdown-column {
        position: absolute;
        top: 20px;
        right: 20px;
        z-index: 100;
        width: 130px;
    }
    .language-dropdown-column label { display: none !important; }
    .language-dropdown-column .stSelectbox > div[data-baseweb="select"] {
        background-color: #F8F7FF;
        border: 1px solid #C5B3FF;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        font-weight: 600;
        color: #4F2083;
    }

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
    .summary-display {
        margin-top: 1rem;
        padding-top: 1rem;
        border-top: 1px dashed #CCC;
    }
    [data-testid="stSidebar"] { display: none; }
</style>
""", unsafe_allow_html=True)


# --- FLOATING LANGUAGE SELECTOR (The single source of truth) ---
_, col_language = st.columns([10, 1])
with col_language:
    st.markdown('<div class="language-dropdown-column">', unsafe_allow_html=True)
    st.selectbox(
        label="L",  # Minimal label, hidden by CSS
        options=list(LANGUAGES.keys()),
        index=list(LANGUAGES.keys()).index(st.session_state.current_lang),
        key="language_selector",
        on_change=update_language # This callback triggers the translation
    )
    st.markdown('</div>', unsafe_allow_html=True)


# --- HELPER FUNCTIONS (Data Loading & Summarization) ---
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
    # Load current translation from session state
    t = st.session_state.translated_strings

    # --- PDF UPLOAD & SIDEBAR ---
    with st.sidebar:
        st.markdown(f"<h3 style='margin: 0; padding: 0;'>{t['sidebar_settings_header']}</h3>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='margin: 20px 0 0 0; padding: 0;'>{t['pdf_upload_header']}</h3>", unsafe_allow_html=True)
        uploaded_files = st.file_uploader(label=".", type=["pdf"], accept_multiple_files=True, label_visibility="collapsed")

        if uploaded_files:
            st.success(t['pdf_success'].format(count=len(uploaded_files)))

    # --- Main Page Content ---
    title_parts = t["title"].split()
    st.markdown(f'<h1>{title_parts[0]} <span style="color: #6A1B9A;">{" ".join(title_parts[1:])}</span></h1>', unsafe_allow_html=True)
    st.markdown(f"### {t['description']}")

    search_query = st.text_input(t["search_label"], placeholder="e.g., microgravity, radiation, Artemis...", label_visibility="collapsed")

    df = load_data("SB_publication_PMC.csv")
    original_cols = list(df.columns) # Save original columns for internal logic

    # --- Translate Dataset Columns (if language is not English) ---
    if st.session_state.current_lang != "English":
        with st.spinner("Translating dataset columns..."):
            translated_cols = translate_list_via_gemini(original_cols, st.session_state.current_lang)
            df.columns = translated_cols

    # --- PDF Summaries Display ---
    if uploaded_files:
        st.markdown("---")
        for uploaded_file in uploaded_files:
            summary_key = f"pdf_summary_{uploaded_file.name}"
            if summary_key not in st.session_state.summary_dict:
                with st.spinner(f"Summarizing: {uploaded_file.name} ..."):
                    pdf_bytes = io.BytesIO(uploaded_file.read())
                    pdf_reader = PyPDF2.PdfReader(pdf_bytes)
                    text = "".join([p.extract_text() or "" for p in pdf_reader.pages])
                    summary = summarize_text_with_gemini(text)
                    st.session_state.summary_dict[summary_key] = summary

            st.markdown(f"### {t['pdf_summary_title'].format(name=uploaded_file.name)}")
            st.write(st.session_state.summary_dict[summary_key])
        st.markdown("---")

    # --- Search Logic & Results Display ---
    if search_query:
        # Search using the original, untranslated 'Title' column for reliability
        mask = load_data("SB_publication_PMC.csv")['Title'].astype(str).str.contains(search_query, case=False, na=False)
        results_df = df[mask.values].reset_index(drop=True)

        st.markdown("---")
        st.subheader(t['results_header'].format(count=len(results_df)))

        if results_df.empty:
            st.warning(t['no_results'])
        else:
            # Find the current names of the 'Title' and 'Link' columns for display
            title_col_name = df.columns[original_cols.index('Title')]
            link_col_name = df.columns[original_cols.index('Link')]

            for idx, row in results_df.iterrows():
                summary_key = f"summary_{idx}_{row[title_col_name]}"
                with st.container():
                    st.markdown('<div class="result-card">', unsafe_allow_html=True)
                    st.markdown(f"**{row[title_col_name]}**", unsafe_allow_html=True)

                    if st.button(t["summarize_button"], key=f"btn_summarize_{idx}"):
                        with st.spinner(f"Accessing and summarizing..."):
                            # Fetch URL using the original, untranslated dataframe
                            original_row = load_data("SB_publication_PMC.csv")[mask].iloc[idx]
                            text = fetch_url_text(original_row['Link'])
                            summary = summarize_text_with_gemini(text)
                            st.session_state.summary_dict[summary_key] = summary
                            st.rerun()

                    if summary_key in st.session_state.summary_dict:
                        summary_content = st.session_state.summary_dict[summary_key]
                        st.markdown('<div class="summary-display">', unsafe_allow_html=True)
                        if summary_content.startswith("ERROR"):
                            st.error(f"Failed to summarize: {summary_content}")
                        else:
                            st.markdown(summary_content)
                        st.markdown('</div>', unsafe_allow_html=True)

                    st.markdown(f"Source: <a href='{original_row['Link']}' target='_blank'>Read Publication</a>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

# --- STREAMLIT PAGE NAVIGATION ---
pg = st.navigation([
    st.Page(search_page, title=st.session_state.translated_strings["nav_search"], icon="🔍"),
    st.Page("pages/Assistant_AI.py", title=st.session_state.translated_strings["nav_assistant"], icon="💬"),
])
pg.run()
