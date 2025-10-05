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
    if st.secrets.get("GEMINI_API_KEY"):
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

# UI strings in English (from the block you supplied)
UI_STRINGS_EN = {
    "title": "Simplified Knowledge",
    "description": "A dynamic dashboard that summarizes NASA bioscience publications and explores impacts and results.",
    "upload_label": "Upload CSV data",
    "ask_label": "Ask anything:",
    "response_label": "Response:",
    "click_button": "Click here, nothing happens",
    "translate_dataset_checkbox": "Translate dataset column names (may take time)",
    "mention_label": "Official NASA Website",
    "button_response": "Hooray",
    "pdf_upload_header": "Upload PDFs to Summarize",
    "pdf_success": "✅ {count} PDF(s) uploaded and summarized",
    "pdf_summary_title": "📄 Summary: {name}",
    "search_label": "Search publications...",
    "results_header": "Found {count} matching publications:",
    "no_results": "No matching publications found.",
    "summarize_button": "🔬 Gather & Summarize"
}

if 'current_lang' not in st.session_state:
    st.session_state.current_lang = "English"  # Default language
if 'translations' not in st.session_state:
    st.session_state.translations = {"English": UI_STRINGS_EN.copy()}
if 'translated_strings' not in st.session_state:
    st.session_state.translated_strings = st.session_state.translations["English"]

# --- CLEANED LANGUAGES DICT (only touch related to translation feature) ---
# Note: replaced the problematic duplicate entries with a consistent mapping.
LANGUAGES = {
    "العربية": {"label": "العربية (Arabic)", "code": "ar"},
    "বাংলা": {"label": "বাংলা (Bengali)", "code": "bn"},
    "Čeština": {"label": "Čeština (Czech)", "code": "cs"},
    "Dansk": {"label": "Dansk (Danish)", "code": "da"},
    "Deutsch": {"label": "Deutsch (German)", "code": "de"},
    "English": {"label": "English (English)", "code": "en"},
    "Español": {"label": "Español (Spanish)", "code": "es"},
    "فارسی": {"label": "فارسی (Persian)", "code": "fa"},
    "Suomi": {"label": "Suomi (Finnish)", "code": "fi"},
    "Français": {"label": "Français (French)", "code": "fr"},
    "ગુજરાતી": {"label": "ગુજરાતી (Gujarati)", "code": "gu"},
    "हिन्दी": {"label": "हिन्दी (Hindi)", "code": "hi"},
    "Magyar": {"label": "Magyar (Hungarian)", "code": "hu"},
    "Bahasa Indonesia": {"label": "Bahasa Indonesia (Indonesian)", "code": "id"},
    "Italiano": {"label": "Italiano (Italian)", "code": "it"},
    "日本語": {"label": "日本語 (Japanese)", "code": "ja"},
    "ಕನ್ನಡ": {"label": "ಕನ್ನಡ (Kannada)", "code": "kn"},
    "한국어": {"label": "한국어 (Korean)", "code": "ko"},
    "Latviešu": {"label": "Latviešu (Latvian)", "code": "lv"},
    "Lietuvių": {"label": "Lietuvių (Lithuanian)", "code": "lt"},
    "മലയാളം": {"label": "മലയാളം (Malayalam)", "code": "ml"},
    "मराठी": {"label": "मराठी (Marathi)", "code": "mr"},
    "Nederlands": {"label": "Nederlands (Dutch)", "code": "nl"},
    "Norsk": {"label": "Norsk (Norwegian)", "code": "no"},
    "Polski": {"label": "Polski (Polish)", "code": "pl"},
    "Português": {"label": "Português (Portuguese)", "code": "pt"},
    "Română": {"label": "Română (Romanian)", "code": "ro"},
    "Русский": {"label": "Русский (Russian)", "code": "ru"},
    "සිංහල": {"label": "සිංහල (Sinhala)", "code": "si"},
    "Slovenčina": {"label": "Slovenčina (Slovak)", "code": "sk"},
    "Slovenščina": {"label": "Slovenščina (Slovenian)", "code": "sl"},
    "سنڌي": {"label": "سنڌي (Sindhi)", "code": "sd"},
    "Svenska": {"label": "Svenska (Swedish)", "code": "sv"},
    "தமிழ்": {"label": "தமிழ் (Tamil)", "code": "ta"},
    "తెలుగు": {"label": "తెలుగు (Telugu)", "code": "te"},
    "ภาษาไทย": {"label": "ภาษาไทย (Thai)", "code": "th"},
    "Türkçe": {"label": "Türkçe (Turkish)", "code": "tr"},
    "Українська": {"label": "Українська (Ukrainian)", "code": "uk"},
    "اردو": {"label": "اردو (Urdu)", "code": "ur"},
    "Tiếng Việt": {"label": "Tiếng Việt (Vietnamese)", "code": "vi"},
    "中文 (简体)": {"label": "中文 (Mandarin, Simplified)", "code": "zh-CN"},
    "中文 (繁體)": {"label": "中文 (Mandarin, Traditional)", "code": "zh-TW"},
    "IsiZulu": {"label": "IsiZulu (Zulu)", "code": "zu"},
    "Shqip": {"label": "Shqip (Albanian)", "code": "sq"},
    "Հայերեն": {"label": "Հայերեն (Armenian)", "code": "hy"},
    "বাংলা (বাংলাদেশ)": {"label": "বাংলা (Bangladeshi Bengali)", "code": "bn-BD"},
    "Bosanski": {"label": "Bosanski (Bosnian)", "code": "bs"},
    "ქართული": {"label": "ქართული (Georgian)", "code": "ka"},
    "አማርኛ": {"label": "አማርኛ (Amharic)", "code": "am"},
    "Melayu": {"label": "Melayu (Malay)", "code": "ms"},
    "မြန်မာစာ": {"label": "မြန်မာစာ (Burmese)", "code": "my"},
    "ਪੰਜਾਬੀ": {"label": "ਪੰਜਾਬੀ (Punjabi)", "code": "pa"},
    "Српски": {"label": "Српски (Serbian)", "code": "sr"},
}


# ----------------- TRANSLATION HELPERS -----------------
def extract_json_from_text(text: str):
    start = text.find('{')
    end = text.rfind('}')
    if start == -1 or end == -1:
        raise ValueError("No JSON object found in model output.")
    return json.loads(text[start:end+1])

def translate_dict_via_gemini(source_dict: dict, target_lang_name: str):
    """
    Calls Gemini to translate the VALUES of a JSON object and returns a dict
    with the same keys and translated values. If Gemini fails, raises an exception
    which will be handled by the caller.
    """
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        prompt = (
            f"Translate the VALUES of the following JSON object into {target_lang_name}.\n"
            "Return ONLY a JSON object with the same keys and translated values (no commentary).\n"
            f"Input JSON:\n{json.dumps(source_dict, ensure_ascii=False)}\n"
        )
        resp = model.generate_content(prompt)
        return extract_json_from_text(resp.text)
    except Exception as e:
        # Reraise to be handled by outer logic so we can fallback gracefully.
        raise

def translate_list_via_gemini(items: list, target_lang_name: str):
    """
    Calls Gemini to translate a list of short strings and returns a list of translated strings.
    If Gemini fails, raises an exception for the caller to handle.
    """
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        prompt = (
            f"Translate this list of short strings into {target_lang_name}. "
            f"Return a JSON array of translated strings in the same order.\n"
            f"Input: {json.dumps(items, ensure_ascii=False)}\n"
        )
        resp = model.generate_content(prompt)
        start = resp.text.find('[')
        end = resp.text.rfind(']')
        if start == -1 or end == -1:
            raise ValueError("No JSON array found in model output.")
        return json.loads(resp.text[start:end+1])
    except Exception as e:
        # Reraise so caller can fallback
        raise

def perform_translation(lang_choice: str):
    """
    Centralized function to translate UI strings into 'lang_choice'.
    Shows emoji rain and spinner for ~6 seconds, attempts Gemini translation,
    and falls back to English if anything fails.
    """
    # If already the same language, just return current strings
    if lang_choice == st.session_state.current_lang and lang_choice in st.session_state.translations:
        st.session_state.translated_strings = st.session_state.translations[lang_choice]
        return st.session_state.translated_strings

    # visual animation and spinner (approx 6 seconds)
    rain(emoji="⏳", font_size=54, falling_speed=5, animation_length=3)
    with st.spinner(f"Translating UI to {lang_choice}..."):
        # ensure the spinner + animation last long enough
        start_t = time.time()
        try:
            if lang_choice in st.session_state.translations:
                translated_strings = st.session_state.translations[lang_choice]
            else:
                # Attempt to call Gemini to translate the known English UI strings
                translated_strings = translate_dict_via_gemini(st.session_state.translations["English"], lang_choice)
                st.session_state.translations[lang_choice] = translated_strings

            st.session_state.current_lang = lang_choice
            st.session_state.translated_strings = translated_strings
        except Exception as e:
            # If anything fails, fallback to English and show warning
            st.warning(f"Translation failed — using English. ({str(e)})")
            st.session_state.current_lang = "English"
            st.session_state.translated_strings = st.session_state.translations["English"]

        # Guarantee ~6 seconds total for UX (if translation was very fast)
        elapsed = time.time() - start_t
        if elapsed < 6:
            time.sleep(6 - elapsed)

    return st.session_state.translated_strings

# ----------------- STYLING (unchanged) -----------------
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
    width: 220px; /* Slightly expanded to fit labels */
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

# ----------------- TOP-RIGHT LANGUAGE SELECTOR (replaces earlier partial code) -----------------
_, col_language = st.columns([10, 1])
with col_language:
    st.markdown('<div class="language-dropdown-column">', unsafe_allow_html=True)
    # Show label text via LANGUAGES mapping
    # Use keys of LANGUAGES as options and format_func to show label
    try:
        index_default = list(LANGUAGES.keys()).index(st.session_state.current_lang)
    except ValueError:
        index_default = 0

    lang_choice = st.selectbox(
        "L",  # minimal label hidden via CSS
        options=list(LANGUAGES.keys()),
        index=index_default,
        format_func=lambda x: LANGUAGES[x]["label"] if isinstance(LANGUAGES.get(x), dict) else str(x),
        key="language_selector",
    )
    st.markdown('</div>', unsafe_allow_html=True)

# Apply translation if needed (top selector)
perform_translation(lang_choice)
selected_language_code = LANGUAGES.get(st.session_state.current_lang, {}).get("code", "")

# --- Demonstration of Use (Main Content) ---
st.title("Your Application Title Here")
st.markdown("---")
st.write(f"The content below would be displayed in the selected language.")
st.info(f"Language Selector Status: **{st.session_state.current_lang}** (Code: **{selected_language_code}**)")

# Example Content
st.header("Main Content Area 📄")
st.write("This is the main body of your Streamlit application, which now flows beneath the floating language selector in the top-right corner.")

# --- HELPER FUNCTIONS (Copied from original, unchanged) ---
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

# --- MAIN PAGE FUNCTION (unchanged except using st.session_state.translated_strings) ---
def search_page():
    # Load current translation
    translated_strings = st.session_state.translated_strings

    # 1. Custom HTML Button for Assistant AI
    st.markdown(
        '<div class="nav-container-ai"><div class="nav-button-ai"><a href="/Assistant_AI" target="_self">Assistant AI 💬</a></div></div>',
        unsafe_allow_html=True
    )

    # --- Language and PDF Sidebar Setup ---
    # Sidebar language selector: keep it in sync with top selector
    def sidebar_lang_changed():
        # read the sent value and perform translation
        chosen = st.session_state.get("lang_selector", st.session_state.current_lang)
        perform_translation(chosen)

    with st.sidebar:
        st.markdown("<h3 style='margin: 0; padding: 0;'>Settings ⚙️</h3>", unsafe_allow_html=True)
        try:
            index_default_sidebar = list(LANGUAGES.keys()).index(st.session_state.current_lang)
        except ValueError:
            index_default_sidebar = 0

        st.session_state.current_lang = st.selectbox(
            "Select Language:",
            options=list(LANGUAGES.keys()),
            index=index_default_sidebar,
            key="lang_selector",
            on_change=sidebar_lang_changed
        )

        # --- PDF UPLOAD LOGIC ---
        st.markdown(f"<h3 style='margin: 20px 0 0 0; padding: 0;'>{translated_strings.get('pdf_upload_header', 'Upload PDFs to Summarize')}</h3>", unsafe_allow_html=True)
        uploaded_files = st.file_uploader(label="", type=["pdf"], accept_multiple_files=True)

        if uploaded_files:
            # Display success message in the sidebar
            st.success(translated_strings.get('pdf_success', "✅ {count} PDF(s) uploaded and summarized").format(count=len(uploaded_files)))

    # --- Main Page Content ---

    # 2. UI Header using translated strings
    # Keep title display logic simple and robust to missing strings
    title_full = translated_strings.get("title", "Simplified Knowledge")
    title_parts = title_full.split()
    if len(title_parts) >= 2:
        st.markdown(f'<h1>{title_parts[0]} <span style="color: #6A1B9A;">{" ".join(title_parts[1:])}</span></h1>', unsafe_allow_html=True)
    else:
        st.markdown(f'<h1>{title_full}</h1>', unsafe_allow_html=True)

    st.markdown(f"### {translated_strings.get('description', '')}")

    search_query = st.text_input(translated_strings.get("search_label", "Search publications..."), placeholder="e.g., microgravity, radiation, Artemis...", label_visibility="collapsed")

    # Load and potentially translate data
    df = load_data("SB_publication_PMC.csv")

    # --- Translate Dataset Columns (as requested) ---
    original_cols = list(df.columns)
    if st.session_state.current_lang != "English":
        with st.spinner("Translating dataset columns..."):
            try:
                # attempt to translate column names via Gemini; fallback to prefix if fails
                translated_cols = translate_list_via_gemini(original_cols, st.session_state.current_lang)
            except Exception:
                translated_cols = [f"Translated_{item}" for item in original_cols]
            df.rename(columns=dict(zip(original_cols, translated_cols)), inplace=True)

    # --- PDF Summaries Display (outside of the sidebar) ---
    if 'uploaded_files' in locals() and uploaded_files:
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
            st.markdown(f"### {translated_strings.get('pdf_summary_title', '📄 Summary: {name}').format(name=uploaded_file.name)}")
            st.write(st.session_state.summary_dict[summary_key])
        st.markdown("---")


    # --- Search Logic ---
    if search_query:
        # Use the original (untranslated) 'Title' column for searching if possible
        # Fallback: try to find any column containing 'Title' case-insensitive
        search_col_name = None
        if "Title" in original_cols:
            search_col_name = "Title"
        else:
            title_cols = [c for c in df.columns if 'title' in c.lower()]
            if title_cols:
                search_col_name = title_cols[0]
            else:
                # fallback to first column
                search_col_name = df.columns[0]

        mask = df[search_col_name].astype(str).str.contains(search_query, case=False, na=False)
        results_df = df[mask].reset_index(drop=True)
        st.markdown("---")
        st.subheader(translated_strings.get('results_header', "Found {count} matching publications:").format(count=len(results_df)))

        if results_df.empty:
            st.warning(translated_strings.get('no_results', "No matching publications found."))
        else:
            # SINGLE COLUMN DISPLAY LOOP
            for idx, row in results_df.iterrows():
                summary_key = f"summary_{idx}"

                with st.container():
                    st.markdown(f'<div class="result-card">', unsafe_allow_html=True)

                    # Title (Using the potentially translated column name for display)
                    title_col_name = df.columns[df.columns.str.contains('Title', case=False)].tolist()[0]
                    link_col_name_candidates = df.columns[df.columns.str.contains('Link', case=False)].tolist()
                    link_col_name = link_col_name_candidates[0] if link_col_name_candidates else df.columns[2] if len(df.columns) > 2 else df.columns[0]

                    st.markdown(f"**{title_col_name}:** <a href='{row[link_col_name]}' target='_blank'>{row[title_col_name]}</a>", unsafe_allow_html=True)

                    # Button
                    if st.button(translated_strings.get("summarize_button", "🔬 Gather & Summarize"), key=f"btn_summarize_{idx}"):

                        # GENERATE SUMMARY IMMEDIATELY UPON CLICK
                        with st.spinner(f"Accessing and summarizing: {row[title_col_name]}..."):
                            try:
                                # Must use the ORIGINAL 'Link' column for fetching the URL
                                text = fetch_url_text(row[original_cols[2]])  # as original code assumed index 2
                                summary = summarize_text_with_gemini(text)
                                st.session_state.summary_dict[summary_key] = summary
                            except Exception as e:
                                st.session_state.summary_dict[summary_key] = f"CRITICAL_ERROR: {e}"

                        st.experimental_rerun()

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

# --- STREAMLIT PAGE NAVIGATION (unchanged) ---
pg = st.navigation([
    st.Page(search_page, title=st.session_state.translated_strings.get("title", "Simplified Knowledge") + " 🔍"),
    st.Page("pages/Assistant_AI.py", title="Assistant AI 💬", icon="💬"),
])

pg.run()
