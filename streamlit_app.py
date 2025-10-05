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
import streamlit as st
from streamlit_extras.mention import mention
import io
import google.generativeai as genai

# --- CONFIGURATION ---
MODEL_NAME = "gemini-1.5-flash-latest"
st.set_page_config(page_title="Houston! We have a problem!", layout="wide")

# --- GEMINI AI SETUP ---
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
except Exception as e:
    st.error(f"Error configuring Gemini AI: {e}")
    st.stop()

# --- INITIALIZE SESSION STATE ---
if 'summary_dict' not in st.session_state:
    st.session_state.summary_dict = {}
if 'selected_language' not in st.session_state:
    st.session_state.selected_language = "English"  # Default language

# --- STYLING (No changes here) ---
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

# --- LANGUAGE DATA & TRANSLATION LOGIC ---

LANGUAGES = {
    "English": {"label": "English (English)", "code": "en"},
    "Türkçe": {"label": "Türkçe (Turkish)", "code": "tr"},
    "Français": {"label": "Français (French)", "code": "fr"},
    "Español": {"label": "Español (Spanish)", "code": "es"},
    "Deutsch": {"label": "Deutsch (German)", "code": "de"},
    "Italiano": {"label": "Italiano (Italian)", "code": "it"},
    "Русский": {"label": "Русский (Russian)", "code": "ru"},
    "日本語": {"label": "日本語 (Japanese)", "code": "ja"},
    "한국어": {"label": "한국어 (Korean)", "code": "ko"},
    "हिन्दी": {"label": "हिन्दी (Hindi)", "code": "hi"},
    "العربية": {"label": "العربية (Arabic)", "code": "ar"},
    # Add any other languages from your original list here
}

# 🟢 UNCOMMENTED AND EXPANDED: Base UI strings in English for translation
UI_STRINGS_EN = {
    "title_part1": "Houston! We Have A",
    "title_part2": "Problem!",
    "subtitle": "Search, Discover, and Summarize NASA's Bioscience Publications",
    "search_placeholder": "TELL US MORE!",
    "search_label": "Search publications...",
    "results_found": "Found {count} matching publications:",
    "no_results": "No matching publications found.",
    "button_summarize": "🔬 Gather & Summarize",
    "spinner_text": "Accessing and summarizing: {title}...",
    "summary_failed": "❌ Failed to Summarize:",
    "summary_error": "Error fetching/summarizing content: {error}",
    "title_label": "Title:",
    "assistant_ai_button": "Assistant AI 💬"
}

# 🟢 NEW: Function to get translated strings, cached for performance
@st.cache_data
def get_translated_strings(target_lang_code: str):
    """Translates the UI_STRINGS_EN dictionary to the target language using Gemini."""
    if target_lang_code == "en":
        return UI_STRINGS_EN

    try:
        model = genai.GenerativeModel(MODEL_NAME)
        prompt = (
            f"Translate the values of the following JSON object from English to the language with code '{target_lang_code}'. "
            "Respond ONLY with the translated JSON object. Keep the keys exactly the same. "
            "Do not include any explanations or markdown formatting like ```json ... ```.\n\n"
            f"{json.dumps(UI_STRINGS_EN)}"
        )
        response = model.generate_content(prompt)
        
        # Clean up the response to ensure it's valid JSON
        cleaned_response = response.text.strip().replace("```json", "").replace("```", "")
        translated_dict = json.loads(cleaned_response)

        # Basic validation to ensure all keys are present
        if all(key in translated_dict for key in UI_STRINGS_EN):
            return translated_dict
        else:
            return UI_STRINGS_EN # Fallback to English on failure
            
    except Exception as e:
        st.error(f"Language translation failed: {e}")
        return UI_STRINGS_EN # Fallback to English on any error

# 🟢 NEW: Function to translate dataframe columns
@st.cache_data
def translate_columns(columns: list, target_lang_code: str):
    """Translates a list of column names."""
    if target_lang_code == 'en':
        return columns
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        prompt = (
            f"Translate the following list of table column headers from English to the language with code '{target_lang_code}'. "
            "Respond ONLY with a JSON array of the translated strings in the same order. "
            "For example, for ['Title', 'Author'], you might return ['Titre', 'Auteur']. "
            "Do not add any other text.\n\n"
            f"{json.dumps(columns)}"
        )
        response = model.generate_content(prompt)
        # Clean up the response to ensure it's valid JSON
        cleaned_response = response.text.strip().replace("```json", "").replace("```", "")
        return json.loads(cleaned_response)
    except Exception as e:
        st.warning(f"Could not translate column names: {e}")
        return columns # Fallback to original columns on error

# --- HELPER FUNCTIONS (Mostly unchanged) ---
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
    
    # --- 🟢 NEW: Language Selection ---
    # Create a mapping from the full label to the language key (e.g., "English")
    lang_label_map = {v["label"]: k for k, v in LANGUAGES.items()}
    
    # Get the current language's full label to set the index of the selectbox
    current_lang_label = LANGUAGES[st.session_state.selected_language]["label"]
    
    # Create the selectbox in a column to control its width
    _, col2 = st.columns([3, 1]) # Pushes the selector to the right
    with col2:
        selected_lang_label = st.selectbox(
            label="Language",
            options=lang_label_map.keys(),
            index=list(lang_label_map.keys()).index(current_lang_label),
            label_visibility="collapsed"
        )

    # Update session state if the language has changed
    selected_language_key = lang_label_map[selected_lang_label]
    if st.session_state.selected_language != selected_language_key:
        st.session_state.selected_language = selected_language_key
        st.rerun() # Rerun the app to apply the new language immediately

    # Get the language code and the translated UI strings
    lang_code = LANGUAGES[st.session_state.selected_language]['code']
    ui_strings = get_translated_strings(lang_code)

    # --- Assistant AI Button (uses translated string) ---
    st.markdown(
        f'<div class="nav-container-ai"><div class="nav-button-ai"><a href="/Assistant_AI" target="_self">{ui_strings["assistant_ai_button"]}</a></div></div>',
        unsafe_allow_html=True
    )
        
    # --- UI Header (uses translated strings) ---
    df = load_data("SB_publication_PMC.csv")
    original_cols = list(df.columns) # Store original column names

    st.markdown(f'<h1>{ui_strings["title_part1"]}<span style="color: #6A1B9A;"> {ui_strings["title_part2"]}</span></h1>', unsafe_allow_html=True)
    st.markdown(f"<h3>{ui_strings['subtitle']}</h3>")

    search_query = st.text_input(
        ui_strings["search_label"], 
        placeholder=ui_strings["search_placeholder"], 
        label_visibility="collapsed"
    )
    
    # --- 🟢 TWEAKED: Translate dataset column names if language is not English ---
    if lang_code != "en":
        translated_cols = translate_columns(original_cols, lang_code)
        if len(translated_cols) == len(original_cols):
             df.columns = translated_cols
        # If translation fails, it will just use the original columns
    
    # --- Search Logic (uses translated strings) ---
    if search_query:
        # Search is performed on the original 'Title' column for consistency
        mask = df[original_cols[0]].astype(str).str.contains(search_query, case=False, na=False)
        results_df = df[mask].reset_index(drop=True)
        st.markdown("---")
        st.subheader(ui_strings["results_found"].format(count=len(results_df)))
        
        if results_df.empty:
            st.warning(ui_strings["no_results"])
        else:
            if 'summary_dict' not in st.session_state:
                st.session_state.summary_dict = {}
            
            # Display Loop
            for idx, row in results_df.iterrows():
                summary_key = f"summary_{idx}"
                
                with st.container():
                    st.markdown(f'<div class="result-card">', unsafe_allow_html=True)
                    
                    # Title (uses original 'Title' and 'Link' from the dataframe source)
                    title = row[original_cols[0]]
                    link = row[original_cols[1]]
                    st.markdown(f"**{ui_strings['title_label']}** <a href='{link}' target='_blank'>{title}</a>", unsafe_allow_html=True)
                    
                    # Button (uses translated string)
                    if st.button(ui_strings["button_summarize"], key=f"btn_summarize_{idx}"):
                        
                        spinner_text = ui_strings["spinner_text"].format(title=title)
                        with st.spinner(spinner_text):
                            try:
                                text = fetch_url_text(link)
                                summary = summarize_text_with_gemini(text)
                                st.session_state.summary_dict[summary_key] = summary
                            except Exception as e:
                                st.session_state.summary_dict[summary_key] = f"CRITICAL_ERROR: {e}"
                        
                        st.rerun()

                    # DISPLAY SUMMARY (uses translated strings)
                    if summary_key in st.session_state.summary_dict:
                        summary_content = st.session_state.summary_dict[summary_key]
                        
                        st.markdown('<div class="summary-display">', unsafe_allow_html=True)
                        
                        if summary_content.startswith("ERROR") or summary_content.startswith("CRITICAL_ERROR"):
                            st.markdown(f"**{ui_strings['summary_failed']}** *{title}*", unsafe_allow_html=True)
                            error_message = ui_strings['summary_error'].format(error=summary_content)
                            st.error(error_message)
                        else:
                            st.markdown(summary_content)
                            
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                    st.markdown("</div>", unsafe_allow_html=True) 

# --- RUN THE MAIN PAGE ---
search_page()
