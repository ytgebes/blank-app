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

# Gemini Ai
st.set_page_config(page_title="Houston! We have a!", layout="wide")

try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    MODEL_NAME = "gemini-2.5-flash"
except Exception as e:
    st.error(f"Error configuring Gemini AI: {e}")
    st.stop()

# --- INITIALIZE SESSION STATE ---
if 'summary_dict' not in st.session_state:
    st.session_state.summary_dict = {}
if 'current_lang' not in st.session_state:
    st.session_state.current_lang = "English"
if 'translate_dataset' not in st.session_state:
    st.session_state.translate_dataset = False

# Everything with style / ux
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
    
    /* 🟢 FIX: Remove the hidden page link CSS to make the nav button visible */
    /* [data-testid="stPageLink"] { display: none; } */ 

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

# Languages
LANGUAGES = {
    "English": {"label": "English (English)", "code": "en"},
    "മലയാളം": {"label": "മലയാളം (Malayalam)", "code": "ml"},
    "Latviešu": {"label": "Latviešu (Latvian)", "code": "lv"},
    "Lietuvių": {"label": "Lietuvių (Lithuanian)", "code": "lt"},
    "Magyar": {"label": "Magyar (Hungarian)", "code": "hu"},
    "हिन्दी": {"label": "हिन्दी (Hindi)", "code": "hi"},
}

# UI strings, PLEASE KEEP UNCOMMENTED FOR NOW.
UI_STRINGS_EN = {
    "title": "Simplified Knowledge",
    "description": "A dynamic dashboard that summarizes NASA bioscience publications and explores impacts and results.",
    "ask_label": "Ask anything:",
    "response_label": "Response:",
    "about_us": "This dashboard explores NASA bioscience publications dynamically.",
    "translate_dataset_checkbox": "Translate dataset column names"
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
        
# Page
def search_page():
    # 🟢 FIX: Custom HTML Button for Assistant AI
    st.markdown(
        '<div class="nav-container-ai"><div class="nav-button-ai"><a href="/Assistant_AI" target="_self">Assistant AI 💬</a></div></div>',
        unsafe_allow_html=True
    )

    # Language selector - top right
    cols = st.columns([3, 1])
    with cols[1]:
        lang_choice = st.selectbox("🌐 Choose language", list(LANGUAGES.keys()), index=list(LANGUAGES.keys()).index(st.session_state.current_lang) if st.session_state.current_lang in LANGUAGES else 0)
        st.session_state.current_lang = lang_choice

    # Choose UI strings based on language - currently only English strings exist
    if st.session_state.current_lang == "English":
        UI_STRINGS = UI_STRINGS_EN
    else:
        # Fallback to English for any languages not yet translated
        UI_STRINGS = UI_STRINGS_EN

    # --- UI Header ---
    df = load_data("SB_publication_PMC.csv")
    # Keep your big header while also showing the simplified title and description from UI_STRINGS
    st.markdown('<h1>Houston! We Have A<span style="color: #6A1B9A;"> Problem!</span></h1>', unsafe_allow_html=True)
    st.markdown(f"### {UI_STRINGS['description']}")

    search_query = st.text_input("Search publications...", placeholder="TELL US MORE!", label_visibility="collapsed")
    
    # --- Search Logic ---
    if search_query:
        mask = df["Title"].astype(str).str.contains(search_query, case=False, na=False)
        results_df = df[mask].reset_index(drop=True)
        st.markdown("---")
        st.subheader(f"Found {len(results_df)} matching publications:")
        
        if results_df.empty:
            st.warning("No matching publications found.")
        else:
            # Clear all session state summary variables to ensure clean display
            if 'summary_dict' not in st.session_state:
                 st.session_state.summary_dict = {}
            
            # SINGLE COLUMN DISPLAY LOOP (Stable)
            for idx, row in results_df.iterrows():
                summary_key = f"summary_{idx}"
                
                with st.container():
                    st.markdown(f'<div class="result-card">', unsafe_allow_html=True)
                    
                    # Title
                    st.markdown(f"**Title:** <a href='{row['Link']}' target='_blank'>{row['Title']}</a>", unsafe_allow_html=True)
                    
                    # Button
                    if st.button("🔬 Gather & Summarize", key=f"btn_summarize_{idx}"):
                        
                        # GENERATE SUMMARY IMMEDIATELY UPON CLICK
                        with st.spinner(f"Accessing and summarizing: {row['Title']}..."):
                            try:
                                text = fetch_url_text(row['Link'])
                                summary = summarize_text_with_gemini(text)
                                st.session_state.summary_dict[summary_key] = summary
                            except Exception as e:
                                st.session_state.summary_dict[summary_key] = f"CRITICAL_ERROR: {e}"
                        
                        # Use rerun to ensure the display updates correctly across the whole page
                        st.rerun()

                    # DISPLAY SUMMARY IF IT EXISTS FOR THIS PUBLICATION
                    if summary_key in st.session_state.summary_dict:
                        summary_content = st.session_state.summary_dict[summary_key]
                        
                        st.markdown('<div class="summary-display">', unsafe_allow_html=True)
                        
                        if summary_content.startswith("ERROR") or summary_content.startswith("CRITICAL_ERROR"):
                            st.markdown(f"**❌ Failed to Summarize:** *{row['Title']}*", unsafe_allow_html=True)
                            st.error(f"Error fetching/summarizing content: {summary_content}")
                        else:
                            # Display the summary without an extra box, just the clean markdown
                            st.markdown(summary_content)
                            
                        st.markdown('</div>', unsafe_allow_html=True)
                            
                    st.markdown("</div>", unsafe_allow_html=True) 

# EVERYTHING commented below is for backup just in case something doesn't work DO NOT DELETE.
    # PDF upload
#st.sidebar.success(f"✅ {len(uploaded_files)} PDF(s) uploaded")
#for uploaded_file in uploaded_files:
        #pdf_bytes = io.BytesIO(uploaded_file.read())
        #pdf_reader = PyPDF2.PdfReader(pdf_bytes)
        #text = ""
        #for page in pdf_reader.pages:
            #text += page.extract_text() or ""

        # Summarize each PDF
        #with st.spinner(f"Summarizing: {uploaded_file.name} ..."):
            #summary = summarize_text_with_gemini(text)
#else:
    #st.sidebar.info("Upload one or more PDF files to get summaries, try again!.")

# THIS IS FOR UPLOADING PDF
#with st.sidebar:
  #  st.markdown("<h3 style='margin: 0; padding: 0;'>Upload PDFs to Summarize</h3>", unsafe_allow_html=True)
    #uploaded_files = st.file_uploader(label="", type=["pdf"], accept_multiple_files=True)

#if uploaded_files:
    #st.success(f"✅ {len(uploaded_files)} PDF(s) uploaded and summarized")
    #for uploaded_file in uploaded_files:
        #pdf_bytes = io.BytesIO(uploaded_file.read())
        #pdf_reader = PyPDF2.PdfReader(pdf_bytes)
        #text = "".join([p.extract_text() or "" for p in pdf_reader.pages])
        #with st.spinner(f"Summarizing: {uploaded_file.name} ..."):
            #summary = summarize_text_with_gemini(text)
        #st.markdown(f"### 📄 Summary: {uploaded_file.name}")
        #st.write(summary)

# Translate dataset
#original_cols = list(df.columns)

#if st.session_state.current_lang != "English":
    #translated_cols = translate_list_via_gemini(original_cols, st.session_state.current_lang)
    #df.rename(columns=dict(zip(original_cols, translated_cols)), inplace=True)

# --- Run main page ---
if __name__ == "__main__":
    search_page()
