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

# --- Configuration ---
MODEL_NAME = "gemini-2.5-flash"
st.set_page_config(page_title="Houston! We Have A Problem!", layout="wide")

# --- Gemini setup (graceful) ---
GEMINI_AVAILABLE = True
try:
    genai.configure(api_key=st.secrets.get("GEMINI_API_KEY"))
except Exception as e:
    GEMINI_AVAILABLE = False
    st.warning("Gemini API key not configured in st.secrets['GEMINI_API_KEY']. Translation features will be disabled.")

# --- LANGUAGES ---
LANGUAGES = {
    "English": {"label": "English (English)", "code": "en"},
    "Türkçe": {"label": "Türkçe (Turkish)", "code": "tr"},
    "Français": {"label": "Français (French)", "code": "fr"},
    "Español": {"label": "Español (Spanish)", "code": "es"},
    "Deutsch": {"label": "Deutsch (German)", "code": "de"},
    "日本語": {"label": "日本語 (Japanese)", "code": "ja"},
}

# --- Defaults in session state ---
if 'current_lang' not in st.session_state:
    st.session_state.current_lang = 'English'
if 'summary_dict' not in st.session_state:
    st.session_state.summary_dict = {}
if 'translated_columns_cache' not in st.session_state:
    st.session_state.translated_columns_cache = {}
if 'translated_ui_cache' not in st.session_state:
    st.session_state.translated_ui_cache = {}

# --- Styling ---
st.markdown("""
    <style>
    /* Minimal styling kept from original */
    h1 { text-align: center; font-size: 3.2em; }
    .block-container { padding-top: 1rem !important; }
    .result-card { background-color: #FAFAFA; padding: 1.2rem; border-radius: 10px; margin-bottom: 1rem; border: 1px solid #E0E0E0; }
    .summary-display { margin-top: 1rem; padding-top: 1rem; border-top: 1px dashed #CCC; }
    .stButton>button { border-radius: 8px; min-width: 200px; }
    </style>
""", unsafe_allow_html=True)

# --- Helpers ---
@st.cache_data
def load_data(file_path: str):
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"File not found: {file_path}. Please ensure the CSV is in the app directory.")
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
    if "pdf" in content_type or url.lower().endswith('.pdf'):
        try:
            with io.BytesIO(r.content) as f:
                reader = PyPDF2.PdfReader(f)
                return "\n".join(p.extract_text() for p in reader.pages if p.extract_text())
        except Exception as e:
            return f"ERROR_PDF_PARSE: {e}"
    else:
        try:
            soup = BeautifulSoup(r.text, "html.parser")
            for tag in soup(['script', 'style']):
                tag.decompose()
            return " ".join(soup.body.get_text(separator=" ", strip=True).split())[:25000]
        except Exception as e:
            return f"ERROR_HTML_PARSE: {e}"


def summarize_text_with_gemini(text: str):
    if not text or text.startswith("ERROR"):
        return f"Could not summarize due to a content error: {text.split(': ')[-1]}"
    if not GEMINI_AVAILABLE:
        return "Summarization unavailable: Gemini API not configured."

    prompt = (
        "Summarize the following scientific text. Output strict JSON with two keys: 'key_findings' (list of short bullet strings) "
        "and 'overview' (one paragraph). Do not include any other text. Text:\n" + text
    )
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(prompt)
        # Model returns content string — try to parse JSON from it
        txt = response.text
        try:
            return json.loads(txt)
        except Exception:
            # If model didn't return JSON, return raw text as fallback
            return txt
    except Exception as e:
        return f"ERROR_GEMINI: {e}"


@st.cache_data
def translate_list_via_gemini(items_tuple: tuple, target_language: str):
    """
    Translate a tuple of strings into the target_language using Gemini.
    Returns a list of translated strings in the same order.
    """
    if target_language == 'English' or not GEMINI_AVAILABLE:
        return list(items_tuple)

    # Build a short prompt asking for a JSON array output
    prompt = (
        "You are a translation assistant. Translate the following list of strings into the target language. "
        "Return ONLY a JSON array of translated strings in the exact same order. Do not include any extra commentary.\n"
        f"Target language: {target_language}\nList: {json.dumps(list(items_tuple), ensure_ascii=False)}\n"
    )

    try:
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(prompt)
        txt = response.text.strip()
        # Attempt to find the first JSON array in the response
        start = txt.find('[')
        end = txt.rfind(']')
        if start != -1 and end != -1:
            json_part = txt[start:end+1]
            translated = json.loads(json_part)
            if isinstance(translated, list) and len(translated) == len(items_tuple):
                return translated
        # fallback: naive line-split
        lines = [l.strip(' "') for l in txt.splitlines() if l.strip()]
        if len(lines) >= len(items_tuple):
            return lines[:len(items_tuple)]
    except Exception as e:
        st.error(f"Translation error: {e}")

    # Final fallback: return original
    return list(items_tuple)


# --- UI Strings (default English) ---
UI_STRINGS_EN = {
    "title": "Houston! We Have A Problem!",
    "subtitle": "Search, Discover, and Summarize NASA's Bioscience Publications",
    "search_placeholder": "TELL US MORE!",
    "gather_button": "🔬 Gather & Summarize",
    "found_label": "Found {n} matching publications:",
    "no_matches": "No matching publications found.",
    "translate_dataset_checkbox": "Translate dataset column names",
}


def get_ui_strings():
    # If cached translation exists, return it
    if st.session_state.current_lang == 'English' or not GEMINI_AVAILABLE:
        return UI_STRINGS_EN
    cached = st.session_state.translated_ui_cache.get(st.session_state.current_lang)
    if cached:
        return cached

    # Otherwise translate the UI strings values
    keys = list(UI_STRINGS_EN.keys())
    values = tuple(UI_STRINGS_EN[k] for k in keys)

    try:
        # Show a small animated rain while translating
        with st.spinner(f"Translating UI to {st.session_state.current_lang}..."):
            try:
                rain(emoji="🌐", animation_length=1)
            except Exception:
                # try different signatures quietly
                try:
                    rain(emoji="🌐", font_size=40, falling_speed=5, animation_length=1)
                except Exception:
                    pass
            translated_values = translate_list_via_gemini(values, st.session_state.current_lang)
            translated_ui = {k: v for k, v in zip(keys, translated_values)}
            st.session_state.translated_ui_cache[st.session_state.current_lang] = translated_ui
            return translated_ui
    except Exception as e:
        st.error(f"UI translation failed: {e}")
        return UI_STRINGS_EN


# --- Main app ---

def search_page(df, ui_strings):
    st.markdown(f"<h1>{ui_strings['title']}</h1>", unsafe_allow_html=True)
    st.markdown(f"### {ui_strings['subtitle']}")

    search_query = st.text_input("Search publications...", placeholder=ui_strings['search_placeholder'], label_visibility="collapsed")

    translate_cols = st.checkbox(ui_strings['translate_dataset_checkbox'], value=(st.session_state.current_lang != 'English'))

    display_df = df.copy()
    if translate_cols and st.session_state.current_lang != 'English':
        original_cols = tuple(display_df.columns.tolist())
        cache_key = (original_cols, st.session_state.current_lang)
        if cache_key in st.session_state.translated_columns_cache:
            translated_cols = st.session_state.translated_columns_cache[cache_key]
        else:
            with st.spinner(f"Translating dataset column names to {st.session_state.current_lang}..."):
                try:
                    rain(emoji="🔁", animation_length=1)
                except Exception:
                    pass
                translated_cols = translate_list_via_gemini(original_cols, st.session_state.current_lang)
                st.session_state.translated_columns_cache[cache_key] = translated_cols
        display_df.columns = translated_cols

    if search_query:
        # Try to search on Title column — original English title column name probably 'Title'
        title_col_candidates = [c for c in df.columns if 'title' in c.lower()]
        if not title_col_candidates:
            st.error("No 'Title' column found in the dataset to perform search.")
            return
        title_col = title_col_candidates[0]
        mask = df[title_col].astype(str).str.contains(search_query, case=False, na=False)
        results_df = df[mask].reset_index(drop=True)
        st.markdown("---")
        st.subheader(ui_strings['found_label'].format(n=len(results_df)))

        if results_df.empty:
            st.warning(ui_strings['no_matches'])
        else:
            for idx, row in results_df.iterrows():
                summary_key = f"summary_{idx}"
                with st.container():
                    st.markdown(f'<div class="result-card">', unsafe_allow_html=True)
                    # Title (use english title column value)
                    title_val = row[title_col]
                    link_val = row.get('Link', '') if 'Link' in row else ''
                    if link_val:
                        st.markdown(f"**Title:** <a href='{link_val}' target='_blank'>{title_val}</a>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"**Title:** {title_val}")

                    if st.button(ui_strings['gather_button'], key=f"btn_summarize_{idx}"):
                        with st.spinner(f"Accessing and summarizing: {title_val}..."):
                            text = fetch_url_text(link_val) if link_val else ''
                            summary = summarize_text_with_gemini(text)
                            st.session_state.summary_dict[summary_key] = summary
                        st.experimental_rerun()

                    if summary_key in st.session_state.summary_dict:
                        summary_content = st.session_state.summary_dict[summary_key]
                        st.markdown('<div class="summary-display">', unsafe_allow_html=True)
                        if isinstance(summary_content, str) and (summary_content.startswith("ERROR") or summary_content.startswith("CRITICAL_ERROR")):
                            st.markdown(f"**❌ Failed to Summarize:** *{title_val}*")
                            st.error(f"Error fetching/summarizing content: {summary_content}")
                        else:
                            # If the summarizer returned JSON, nicely render it
                            if isinstance(summary_content, dict):
                                if 'key_findings' in summary_content:
                                    st.markdown('### Key Findings')
                                    for item in summary_content['key_findings']:
                                        st.write(f"- {item}")
                                if 'overview' in summary_content:
                                    st.markdown('### Overview Summary')
                                    st.write(summary_content['overview'])
                            else:
                                st.markdown(summary_content)
                        st.markdown('</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)


# --- App entry ---

def main():
    # Top row: language selector and simple page tabs
    cols = st.columns([1, 2, 1])
    with cols[0]:
        selected = st.selectbox("Language", list(LANGUAGES.keys()), index=list(LANGUAGES.keys()).index(st.session_state.current_lang))
        if selected != st.session_state.current_lang:
            st.session_state.current_lang = selected
            # Immediately translate small UI strings to reflect change
            # We intentionally do not force a full rerun here; UI will update on next widget render
    with cols[2]:
        # small place for mention component if needed
        try:
            mention('Welcome to the app!')
        except Exception:
            pass

    # Load dataset
    df = load_data("SB_publication_PMC.csv")

    # Get UI strings (possibly translated)
    ui_strings = get_ui_strings()

    # Render search page (single-page for simplicity)
    search_page(df, ui_strings)


if __name__ == '__main__':
    main()
