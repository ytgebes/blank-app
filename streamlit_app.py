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

MODEL_NAME = "gemini-2.5-flash"

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

# --- INITIALIZE SESSION STATE ---
if 'summary_dict' not in st.session_state:
    st.session_state.summary_dict = {}
if 'current_lang' not in st.session_state:
    st.session_state.current_lang = "English" # Default language
if 'translated_strings' not in st.session_state:
    # Initialize with default English strings
    st.session_state.translated_strings = {
        "title": "Simplified Knowledge",
        "description": "A dynamic dashboard that summarizes NASA bioscience publications and explores impacts and results.",
        "ask_label": "Ask anything:",
        "response_label": "Response:",
        "about_us": "This dashboard explores NASA bioscience publications dynamically.",
        "translate_dataset_checkbox": "Translate dataset column names",
        "search_label": "Search publications...",
        "results_header": "Found {count} matching publications:",
        "no_results": "No matching publications found.",
        "summarize_button": "🔬 Gather & Summarize",
        "pdf_upload_header": "Upload PDFs to Summarize",
        "pdf_success": "✅ {count} PDF(s) uploaded and summarized",
        "pdf_summary_title": "📄 Summary: {name}"
    }

# Languages dictionary (already defined)
LANGUAGES = {
    "English": {"label": "English (English)", "code": "en"},
    "Türkçe": {"label": "Türkçe (Turkish)", "code": "tr"},
    "Français": {"label": "Français (French)", "code": "fr"},
    "Español": {"label": "Español (Spanish)", "code": "es"},
    # ... (other languages remain the same)
    "Deutsch": {"label": "Deutsch (German)", "code": "de"},
    "Italiano": {"label": "Italiano (Italian)", "code": "it"},
    "日本語": {"label": "日本語 (Japanese)", "code": "ja"},
    "한국어": {"label": "한국어 (Korean)", "code": "ko"},
    "हिन्दी": {"label": "हिन्दी (Hindi)", "code": "hi"},
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Italian": "it",
    "Japanese": "ja",
    "Mandarin (Simplified)": "zh-CN",
}


# --- PLACEHOLDER/SIMULATION FOR TRANSLATION ---
# NOTE: In a real app, this function would call the Gemini API to translate text.
# For simplicity and to avoid excessive API calls during development, this is a placeholder.
@st.cache_data(show_spinner=False)
def get_translated_ui_strings(target_lang: str):
    # This should hold ALL translated UI strings for ALL languages
    # For this example, we'll only provide English, but structure the code to look up translations
    UI_STRINGS_EN = st.session_state.translated_strings # Use the default session state strings as the English base

    # You would load/translate other languages here. For the demo, we only return English
    # unless you implement the full translation logic.
    if target_lang == "English":
        return UI_STRINGS_EN
    elif target_lang == "Español":
        return {
            "title": "Conocimiento Simplificado",
            "description": "Un panel dinámico que resume las publicaciones de biociencia de la NASA y explora sus impactos y resultados.",
            "search_label": "Buscar publicaciones...",
            "results_header": "Se encontraron {count} publicaciones coincidentes:",
            "no_results": "No se encontraron publicaciones coincidentes.",
            "summarize_button": "🔬 Recopilar y Resumir",
            "pdf_upload_header": "Subir PDFs para Resumir",
            "pdf_success": "✅ {count} PDF(s) subidos y resumidos",
            "pdf_summary_title": "📄 Resumen: {name}",
            # Fallback for keys not defined in Spanish
            **{k: v for k, v in UI_STRINGS_EN.items() if k not in ["title", "description", "search_label", "results_header", "no_results", "summarize_button", "pdf_upload_header", "pdf_success", "pdf_summary_title"]}
        }
    else:
        # Fallback for any unsupported language
        return UI_STRINGS_EN

# Placeholder for translating column names
def translate_list_via_gemini(text_list: list, target_lang: str):
    # This is where you'd call Gemini to translate column names.
    # We'll just return the original list to avoid breaking the app without a live API call.
    if target_lang == "English":
        return text_list
    # In a real app:
    # prompt = f"Translate the following list of scientific column headers into {target_lang}. Return only the translated list, comma-separated:\n{', '.join(text_list)}"
    # return model.generate_content(prompt).text.split(', ')
    return [f"Translated_{item}" for item in text_list] # Mock translation


# --- STYLING (The CSS is copied from the original code and remains the same) ---
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
    <style>
/* ABSOLUTE POSITIONING */
.language-dropdown-column {
    position: absolute;
    top: 30px; 
    right: 20px; 
    z-index: 100;
    width: 130px; /* Reduced width */
}

/* STYLING (White/Light Purple) */
.language-dropdown-column .stSelectbox {
    background-color: white; 
    border-radius: 8px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1); 
    border: 1px solid #C5B3FF; 
}

.language-dropdown-column label {
    display: none !important; 
}

.language-dropdown-column .stSelectbox .st-bd { 
    background-color: #F8F7FF; 
    color: #4F2083; 
    border: none;
    border-radius: 8px;
    padding: 6px 10px; /* Reduced padding */
    font-size: 14px; /* Reduced font size */
    font-weight: 600;
}

.language-dropdown-column .stSelectbox .st-bd:hover {
    background-color: #E6E0FF; 
}

.language-dropdown-column .stSelectbox [data-testid="stTriangle"] {
    color: #6A1B9A; 
}

[data-testid="stSidebar"] { display: none; }
</style>
""", unsafe_allow_html=True)

_, col_language = st.columns([10, 1])
with col_language:
    st.markdown('<div class="language-dropdown-column">', unsafe_allow_html=True)
    
    selected_language_name = st.selectbox(
        "L", # Use a minimal label, hidden by CSS
        list(LANGUAGES.keys()),
        index=0,
        key="language_selector"
    )
    
    st.markdown('</div>', unsafe_allow_html=True)

selected_language_code = LANGUAGES[selected_language_name]
    
st.markdown('</div>', unsafe_allow_html=True)

# Get the language code
selected_language_code = LANGUAGES[selected_language_name]

# --- Demonstration of Use (Main Content) ---

st.title("Your Application Title Here")
st.markdown("---")
st.write(f"The content below would be displayed in the selected language.")
st.info(f"Language Selector Status: **{selected_language_name}** (Code: **{selected_language_code}**)")

# Example Content
st.header("Main Content Area 📄")
st.write("This is the main body of your Streamlit application, which now flows beneath the floating language selector in the top-right corner.")

# --- HELPER FUNCTIONS (Copied from original) ---
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
    # Load current translation
    translated_strings = st.session_state.translated_strings

    # 1. Custom HTML Button for Assistant AI
    st.markdown(
        '<div class="nav-container-ai"><div class="nav-button-ai"><a href="/Assistant_AI" target="_self">Assistant AI 💬</a></div></div>',
        unsafe_allow_html=True
    )
    
    # --- Language and PDF Sidebar Setup ---
    with st.sidebar:
        st.markdown("<h3 style='margin: 0; padding: 0;'>Settings ⚙️</h3>", unsafe_allow_html=True)
        st.session_state.current_lang = st.selectbox(
            "Select Language:",
            options=list(LANGUAGES.keys()),
            index=list(LANGUAGES.keys()).index(st.session_state.current_lang),
            key="lang_selector",
            on_change=lambda: st.session_state.update(translated_strings=get_translated_ui_strings(st.session_state.current_lang))
        )
        
        # --- PDF UPLOAD LOGIC ---
        st.markdown(f"<h3 style='margin: 20px 0 0 0; padding: 0;'>{translated_strings['pdf_upload_header']}</h3>", unsafe_allow_html=True)
        uploaded_files = st.file_uploader(label="", type=["pdf"], accept_multiple_files=True)
        
        if uploaded_files:
            # Display success message in the sidebar
            st.success(translated_strings['pdf_success'].format(count=len(uploaded_files)))
            
    # --- Main Page Content ---

    # 2. UI Header using translated strings
    st.markdown(f'<h1>{translated_strings["title"].split()[0]} <span style="color: #6A1B9A;">{translated_strings["title"].split()[1]}</span></h1>', unsafe_allow_html=True)
    st.markdown(f"### {translated_strings['description']}")

    search_query = st.text_input(translated_strings["search_label"], placeholder="e.g., microgravity, radiation, Artemis...", label_visibility="collapsed")
    
    # Load and potentially translate data
    df = load_data("SB_publication_PMC.csv")
    
    # --- Translate Dataset Columns (as requested in the original commented logic) ---
    original_cols = list(df.columns)
    if st.session_state.current_lang != "English":
        with st.spinner("Translating dataset columns..."):
            translated_cols = translate_list_via_gemini(original_cols, st.session_state.current_lang)
            df.rename(columns=dict(zip(original_cols, translated_cols)), inplace=True)
    
    # --- PDF Summaries Display (outside of the sidebar) ---
    if uploaded_files:
        st.markdown("---")
        for uploaded_file in uploaded_files:
            # Check if this PDF has already been processed in the current session
            summary_key = f"pdf_summary_{uploaded_file.name}"
            
            if summary_key not in st.session_state.summary_dict:
                pdf_bytes = io.BytesIO(uploaded_file.read())
                pdf_reader = PyPDF2.PdfReader(pdf_bytes)
                text = "".join([p.extract_text() or "" for p in pdf_reader.pages])
                
                with st.spinner(f"Summarizing: {uploaded_file.name} ..."):
                    summary = summarize_text_with_gemini(text)
                    st.session_state.summary_dict[summary_key] = summary
            
            # Display the result
            st.markdown(f"### {translated_strings['pdf_summary_title'].format(name=uploaded_file.name)}")
            st.write(st.session_state.summary_dict[summary_key])
        st.markdown("---")


    # --- Search Logic ---
    if search_query:
        # Use the original (untranslated) 'Title' column for searching since the query is in the user's language
        # A more robust solution would translate the search query before searching the English titles.
        # However, using the original column names is safest for internal logic here.
        search_col_name = "Title" if st.session_state.current_lang == "English" else original_cols[df.columns.get_loc(df.columns[df.columns.str.contains('Title', case=False)].tolist()[0])]

        mask = df[search_col_name].astype(str).str.contains(search_query, case=False, na=False)
        results_df = df[mask].reset_index(drop=True)
        st.markdown("---")
        st.subheader(translated_strings['results_header'].format(count=len(results_df)))
        
        if results_df.empty:
            st.warning(translated_strings['no_results'])
        else:
            # SINGLE COLUMN DISPLAY LOOP
            for idx, row in results_df.iterrows():
                summary_key = f"summary_{idx}"
                
                with st.container():
                    st.markdown(f'<div class="result-card">', unsafe_allow_html=True)
                    
                    # Title (Using the potentially translated column name for display)
                    title_col_name = df.columns[df.columns.str.contains('Title', case=False)].tolist()[0]
                    link_col_name = df.columns[df.columns.str.contains('Link', case=False)].tolist()[0]

                    st.markdown(f"**{title_col_name}:** <a href='{row[link_col_name]}' target='_blank'>{row[title_col_name]}</a>", unsafe_allow_html=True)
                    
                    # Button
                    if st.button(translated_strings["summarize_button"], key=f"btn_summarize_{idx}"):
                        
                        # GENERATE SUMMARY IMMEDIATELY UPON CLICK
                        with st.spinner(f"Accessing and summarizing: {row[title_col_name]}..."):
                            try:
                                # Must use the ORIGINAL 'Link' column for fetching the URL
                                text = fetch_url_text(row[original_cols[2]]) # Assuming 'Link' is the 3rd column (index 2) based on typical structure
                                summary = summarize_text_with_gemini(text)
                                st.session_state.summary_dict[summary_key] = summary
                            except Exception as e:
                                st.session_state.summary_dict[summary_key] = f"CRITICAL_ERROR: {e}"
                        
                        st.rerun()

                    # DISPLAY SUMMARY IF IT EXISTS FOR THIS PUBLICATION
                    if summary_key in st.session_state.summary_dict:
                        summary_content = st.session_state.summary_dict[summary_key]
                        
                        st.markdown('<div class="summary-display">', unsafe_allow_html=True)
                        
                        if summary_content.startswith("ERROR") or summary_content.startswith("CRITICAL_ERROR"):
                            st.markdown(f"**❌ Failed to Summarize:** *{row[title_col_name]}*", unsafe_allow_html=True)
                            st.error(f"Error fetching/summarizing content: {summary_content}")
                        else:
                            # Display the summary without an extra box, just the clean markdown
                            st.markdown(summary_content)
                            
                        st.markdown('</div>', unsafe_allow_html=True)
                            
                    st.markdown("</div>", unsafe_allow_html=True) 

# --- STREAMLIT PAGE NAVIGATION ---
pg = st.navigation([
    st.Page(search_page, title=st.session_state.translated_strings["title"] + " 🔍"),
    st.Page("pages/Assistant_AI.py", title="Assistant AI 💬", icon="💬"),
])

pg.run()
