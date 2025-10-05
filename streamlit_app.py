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

# ----------------- Configure Gemini API -----------------
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-2.5-flash")

# ----------------- Supported Languages -----------------
LANGUAGES = {
    "English": {"label": "🇺🇸 English (English)", "code": "en"},
    "Türkçe": {"label": "🇹🇷 Türkçe (Turkish)", "code": "tr"},
    "Français": {"label": "🇫🇷 Français (French)", "code": "fr"},
    "Español": {"label": "🇪🇸 Español (Spanish)", "code": "es"},
    "Afrikaans": {"label": "🇿🇦 Afrikaans", "code": "af"},
    "العربية": {"label": "🇸🇦 العربية", "code": "ar"},
    "Tiếng Việt": {"label": "🇻🇳 Tiếng Việt", "code": "vi"},
    "isiXhosa": {"label": "🇿🇦 isiXhosa", "code": "xh"},
    "ייִדיש": {"label": "🇮🇱 ייִדיש", "code": "yi"},
    "Yorùbá": {"label": "🇳🇬 Yorùbá", "code": "yo"},
    "isiZulu": {"label": "🇿🇦 isiZulu", "code": "zu"},
}

# ----------------- UI Strings -----------------
UI_STRINGS_EN = {
    "title": "Simplified Knowledge",
    "description": "A dynamic dashboard that summarizes NASA bioscience publications and explores impacts and results.",
    "ask_label": "Ask anything:",
    "response_label": "Response:",
    "click_button": "Click here, nothing happens",
    "translate_dataset_checkbox": "Translate dataset column names (may take time)",
    "mention_label": "Official NASA Website",
    "button_response": "Hooray",
    "about_us": "This dashboard explores NASA bioscience publications dynamically."
}

# ----------------- Helper Functions -----------------
def extract_json_from_text(text):
    start = text.find('{')
    end = text.rfind('}')
    if start == -1 or end == -1:
        raise ValueError("No JSON object found in model output.")
    return json.loads(text[start:end+1])

def translate_dict_via_gemini(source_dict: dict, target_lang_name: str):
    prompt = (
        f"Translate the VALUES of the following JSON object into {target_lang_name}.\n"
        "Return ONLY a JSON object with the same keys and translated values (no commentary).\n"
        f"Input JSON:\n{json.dumps(source_dict, ensure_ascii=False)}\n"
    )
    resp = model.generate_content(prompt)
    return extract_json_from_text(resp.text)

def translate_list_via_gemini(items: list, target_lang_name: str):
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

@lru_cache(maxsize=256)
def fetch_url_text(url: str) -> str:
    """Download url and return extracted text (PDF or HTML). Cached in-memory."""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; NASA-App/1.0)"}
        r = requests.get(url, headers=headers, timeout=15)
        r.raise_for_status()
    except Exception as e:
        return f"ERROR_FETCH: {str(e)}"

    content_type = r.headers.get("Content-Type", "").lower()

# ----------------- Session State -----------------
if "current_lang" not in st.session_state:
    st.session_state.current_lang = "English"
if "translations" not in st.session_state:
    st.session_state.translations = {"English": UI_STRINGS_EN.copy()}

# ----------------- Page Config -----------------
st.set_page_config(page_title="NASA BioSpace Dashboard", layout="wide")

# ----------------- Sidebar -----------------
with st.sidebar:
    lang_choice = st.selectbox(
        "🌐 Choose language",
        options=list(LANGUAGES.keys()),
        format_func=lambda x: LANGUAGES[x]["label"],
        index=list(LANGUAGES.keys()).index(st.session_state.current_lang)
    )

    if lang_choice != st.session_state.current_lang:
        rain(emoji="⏳", font_size=54, falling_speed=5, animation_length=2)
        with st.spinner(f"Translating UI to {lang_choice}..."):
            try:
                if lang_choice in st.session_state.translations:
                    translated_strings = st.session_state.translations[lang_choice]
                else:
                    translated_strings = translate_dict_via_gemini(
                        st.session_state.translations["English"],
                        lang_choice
                    )
                    st.session_state.translations[lang_choice] = translated_strings
                st.session_state.current_lang = lang_choice
            except Exception as e:
                st.error("Translation failed — using English. Error: " + str(e))
                translated_strings = st.session_state.translations["English"]
                st.session_state.current_lang = "English"
    else:
        translated_strings = st.session_state.translations[st.session_state.current_lang]

# ----------------- Main UI -----------------
st.title(translated_strings["title"])
st.write(translated_strings["description"])

mention(
    label=translated_strings["mention_label"],
    icon="NASA International Space Apps Challenge",
    url="https://www.spaceappschallenge.org/"
)

# ----------------- Load CSV -----------------
df = pd.read_csv("SB_publication_PMC.csv")

# ----------------- Translate dataset -----------------
translate_dataset = st.checkbox(translated_strings["translate_dataset_checkbox"])
if translate_dataset and 'original_cols' in locals() and st.session_state.current_lang != "English":
    translated_cols = translate_list_via_gemini(original_cols, st.session_state.current_lang)
    df.rename(columns=dict(zip(original_cols, translated_cols)), inplace=True)

# ----------------- Gemini Input -----------------
user_input = st.text_input(translated_strings["ask_label"], key="gemini_input")
if user_input:
    with st.spinner("Generating..."):
        resp = model.generate_content(user_input)
    st.subheader(translated_strings["response_label"])
    st.write(resp.text)

# ----------------- Button -----------------
if st.button(translated_strings["click_button"]):
    st.write(translated_strings["button_response"])
