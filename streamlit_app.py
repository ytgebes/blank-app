import streamlit as st
import json
import io
import pandas as pd
import requests
from bs4 import BeautifulSoup
import PyPDF2
from functools import lru_cache
import google.generativeai as genai

# --- 1. CONFIGURATION & SETUP ---

# Set page layout
st.set_page_config(page_title="Simplified Knowledge", layout="wide")

# Configure Gemini AI (handles missing key gracefully)
GEMINI_AVAILABLE = True
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    MODEL_NAME = "gemini-1.5-flash"
except (KeyError, AttributeError):
    GEMINI_AVAILABLE = False
    # This warning will only show if the user tries to use a Gemini-dependent feature
    # st.warning("Gemini API key not found. Translation and summarization are disabled.")


# --- 2. INITIALIZE SESSION STATE ---
if 'current_lang' not in st.session_state:
    st.session_state.current_lang = 'English'
if 'summary_dict' not in st.session_state:
    st.session_state.summary_dict = {}
if 'translated_ui_cache' not in st.session_state:
    st.session_state.translated_ui_cache = {}

# --- 3. STYLING (CSS) ---
st.markdown("""
    <style>
    /* ------------------ CORE LAYOUT & THEME ------------------ */
    /* Push content down slightly to not overlap with floating elements */
    .block-container {
        padding-top: 3.5rem !important;
    }
    h1, h2, h3 {
        text-align: center;
    }
    /* Main title from st.title() */
    h1 {
        font-size: 2.5em !important;
        padding-bottom: 0.5rem;
        color: #333;
    }
    /* Main content area H2 title */
    .main-content h2 {
        font-size: 4.5em !important;
        font-weight: bold;
        padding-bottom: 0.5rem;
        color: #000000;
    }
    .main-content h3 {
        color: #333333;
        font-weight: normal;
    }
    .main-content input[type="text"] {
        color: #000000 !important;
        background-color: #F0F2F6 !important;
        border: 1px solid #CCCCCC !important;
        border-radius: 8px;
        padding: 14px;
        text-align: center; /* Center placeholder text */
    }
    /* ------------------ LANGUAGE SELECTOR (TOP-RIGHT) ------------------ */
    .language-selector-container {
        position: fixed;
        top: 0.8rem;
        right: 1.5rem;
        z-index: 1000; /* Ensure it's above other content */
        width: 150px; /* Give it a fixed width */
    }
    /* Hide Streamlit's default hamburger menu */
    [data-testid="stSidebar"] {
        display: none;
    }
    /* ------------------ ASSISTANT AI BUTTON (TOP-LEFT) ------------------ */
    .nav-button-ai {
        background-color: #6A1B9A; /* Purple color */
        color: white;
        padding: 10px 20px;
        border-radius: 8px;
        text-decoration: none;
        font-weight: bold;
        transition: background-color 0.3s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        display: inline-block; /* Make it behave like a button */
        margin-bottom: 2rem; /* Space below the button */
    }
    .nav-button-ai:hover {
        background-color: #4F0A7B; /* Darker purple on hover */
        color: white;
        text-decoration: none;
    }
    /* ------------------ SEARCH RESULTS STYLING ------------------ */
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
    .result-card a {
        color: #6A1B9A;
        text-decoration: none;
        font-weight: bold;
    }
    .result-card a:hover {
        text-decoration: underline;
    }
    .summary-display {
        margin-top: 1rem;
        padding-top: 1rem;
        border-top: 1px dashed #CCC;
    }
    .stButton>button {
        border-radius: 8px;
        width: auto;
        min-width: 200px;
        background-color: #E6E0FF;
        color: #4F2083;
        border: 1px solid #C5B3FF;
        font-weight: bold;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #D6C9FF;
        border: 1px solid #B098FF;
    }
    </style>
""", unsafe_allow_html=True)

# --- 4. LANGUAGE DEFINITIONS & TRANSLATION FUNCTIONS ---

# Language options
LANGUAGES = {
    "English": {"label": "English (English)", "code": "en"},
    "Français": {"label": "Français (French)", "code": "fr"},
    "Español": {"label": "Español (Spanish)", "code": "es"},
    "Deutsch": {"label": "Deutsch (German)", "code": "de"},
    "Türkçe": {"label": "Türkçe (Turkish)", "code": "tr"},
    "日本語": {"label": "日本語 (Japanese)", "code": "ja"},
    "Русский": {"label": "Русский (Russian)", "code": "ru"},
    "हिन्दी": {"label": "हिन्दी (Hindi)", "code": "hi"},
}

# UI strings in English (the default)
UI_STRINGS_EN = {
    "app_title": "Your Application Title Here",
    "main_title": "Simplified Knowledge",
    "main_subtitle": "A dynamic dashboard that summarizes NASA bioscience publications and explores impacts and results.",
    "search_placeholder": "e.g., microgravity, radiation, Artemis...",
    "assistant_ai_button": "Assistant AI 💬",
    "gather_button": "🔬 Gather & Summarize",
    "found_label": "Found {n} matching publications:",
    "no_matches": "No matching publications found.",
    "summary_fail": "❌ Failed to Summarize:",
    "error_fetching": "Error fetching/summarizing content:",
    "gemini_unavailable": "Gemini features (translation, summarization) are unavailable. Please configure your API key."
}

@st.cache_data(show_spinner=False)
def translate_list_via_gemini(items_tuple: tuple, target_language: str):
    """Translates a tuple of strings using Gemini and returns a list."""
    if target_language == 'English' or not GEMINI_AVAILABLE:
        return list(items_tuple)

    prompt = (
        "You are a translation assistant. Translate the following list of UI text strings into the target language. "
        "Return ONLY a JSON array of the translated strings in the exact same order. Do not include any other text, commentary, or code formatting.\n"
        f"Target language: {target_language}\n"
        f"List: {json.dumps(list(items_tuple), ensure_ascii=False)}\n"
    )
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(prompt)
        # Clean up response to find the JSON array
        txt = response.text.strip()
        start = txt.find('[')
        end = txt.rfind(']')
        if start != -1 and end != -1:
            json_part = txt[start:end+1]
            translated = json.loads(json_part)
            if isinstance(translated, list) and len(translated) == len(items_tuple):
                return translated
    except Exception as e:
        st.error(f"Translation API error: {e}")

    # Fallback to original if translation fails
    return list(items_tuple)

def get_ui_strings():
    """Gets the UI strings in the currently selected language, handling caching."""
    lang = st.session_state.current_lang
    if lang == 'English' or not GEMINI_AVAILABLE:
        return UI_STRINGS_EN

    # Return from cache if available
    if lang in st.session_state.translated_ui_cache:
        return st.session_state.translated_ui_cache[lang]

    # If not in cache, translate and store
    with st.spinner(f"Translating UI to {lang}..."):
        keys = list(UI_STRINGS_EN.keys())
        values_tuple = tuple(UI_STRINGS_EN.values())
        translated_values = translate_list_via_gemini(values_tuple, lang)
        
        # Check if translation was successful before caching
        if len(translated_values) == len(keys):
            translated_ui = dict(zip(keys, translated_values))
            st.session_state.translated_ui_cache[lang] = translated_ui
            return translated_ui
        else:
            # If translation failed, return English as a fallback
            return UI_STRINGS_EN


# --- 5. CORE HELPER FUNCTIONS ---

@st.cache_data
def load_data(file_path):
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"Data file not found: {file_path}. Please ensure it's in the same directory.")
        st.stop()

@lru_cache(maxsize=128)
def fetch_url_text(url: str):
    """Fetches and parses text content from a URL (HTML or PDF)."""
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
            for tag in soup(['script', 'style', 'header', 'footer', 'nav']):
                tag.decompose()
            return " ".join(soup.body.get_text(separator=" ", strip=True).split())[:25000]
        except Exception as e:
            return f"ERROR_HTML_PARSE: {e}"

def summarize_text_with_gemini(text: str):
    """Summarizes text using Gemini into a Markdown format."""
    if not GEMINI_AVAILABLE:
        return UI_STRINGS_EN["gemini_unavailable"]
    if not text or text.startswith("ERROR"):
        return f"Could not summarize due to a content error: {text.split(': ')[-1]}"

    prompt = (f"Summarize this NASA bioscience paper. Output in clean Markdown with a level 3 heading (###) titled 'Key Findings' "
              f"(using bullet points) and a level 3 heading (###) titled 'Overview Summary' (using a paragraph).\n\nContent:\n{text}")
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"ERROR_GEMINI: {e}"

# --- 6. MAIN APPLICATION ---

def run_app():
    """The main function to run the Streamlit app page."""
    
    # Get the translated UI strings
    ui = get_ui_strings()

    # --- Header Elements ---
    st.title(ui['app_title'])
    
    # Assistant AI Button (Top-Left)
    st.markdown(
        f'<a href="/Assistant_AI" target="_self" class="nav-button-ai">{ui["assistant_ai_button"]}</a>',
        unsafe_allow_html=True
    )
    
    st.markdown("---") # Visual separator

    # --- Main Content Area ---
    with st.container():
        st.markdown('<div class="main-content">', unsafe_allow_html=True)
        st.markdown(f"<h2>{ui['main_title']}</h2>", unsafe_allow_html=True)
        st.markdown(f"<h3>{ui['main_subtitle']}</h3>")

        # Search Input
        search_query = st.text_input(
            "search",
            placeholder=ui['search_placeholder'],
            label_visibility="collapsed"
        )
        st.markdown('</div>', unsafe_allow_html=True)

    # --- Search Logic & Results ---
    if search_query:
        df = load_data("SB_publication_PMC.csv")
        mask = df["Title"].astype(str).str.contains(search_query, case=False, na=False)
        results_df = df[mask].reset_index(drop=True)

        st.markdown("---")
        st.subheader(ui['found_label'].format(n=len(results_df)))

        if results_df.empty:
            st.warning(ui['no_matches'])
        else:
            # Display each result in its own card
            for idx, row in results_df.iterrows():
                summary_key = f"summary_{idx}_{row['PMCID']}" # More unique key
                
                with st.container():
                    st.markdown('<div class="result-card">', unsafe_allow_html=True)
                    
                    # Title with link
                    st.markdown(f"**Title:** <a href='{row['Link']}' target='_blank'>{row['Title']}</a>", unsafe_allow_html=True)

                    # Summarize Button
                    if st.button(ui['gather_button'], key=f"btn_{summary_key}"):
                        if not GEMINI_AVAILABLE:
                            st.error(ui["gemini_unavailable"])
                        else:
                            with st.spinner(f"Accessing and summarizing: {row['Title'][:50]}..."):
                                text = fetch_url_text(row['Link'])
                                summary = summarize_text_with_gemini(text)
                                st.session_state.summary_dict[summary_key] = summary
                                st.rerun()

                    # Display Summary if it exists in session state
                    if summary_key in st.session_state.summary_dict:
                        summary_content = st.session_state.summary_dict[summary_key]
                        st.markdown('<div class="summary-display">', unsafe_allow_html=True)
                        if "ERROR" in summary_content or "unavailable" in summary_content:
                            st.markdown(f"**{ui['summary_fail']}** *{row['Title']}*")
                            st.error(f"{ui['error_fetching']} {summary_content}")
                        else:
                            st.markdown(summary_content, unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.markdown("</div>", unsafe_allow_html=True)

# --- 7. SCRIPT EXECUTION ---

if __name__ == "__main__":
    # --- Language Selector (Rendered first to float on top) ---
    st.markdown('<div class="language-selector-container">', unsafe_allow_html=True)
    selected_lang = st.selectbox(
        label="Language",
        options=LANGUAGES.keys(),
        index=list(LANGUAGES.keys()).index(st.session_state.current_lang),
        label_visibility="collapsed"
    )
    if selected_lang != st.session_state.current_lang:
        st.session_state.current_lang = selected_lang
        st.rerun() # Rerun to apply the new language immediately
    st.markdown('</div>', unsafe_allow_html=True)

    # Run the main application
    run_app()
