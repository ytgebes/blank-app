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
    "Afrikaans": {"label": "🇿🇦 Afrikaans (Afrikaans)", "code": "af"},
    "العربية": {"label": "🇸🇦 العربية (Arabic)", "code": "ar"},
    "Tiếng Việt": {"label": "🇻🇳 Tiếng Việt (Vietnamese)", "code": "vi"},
    "isiXhosa": {"label": "🇿🇦 isiXhosa (Xhosa)", "code": "xh"},
    "ייִדיש": {"label": "🇮🇱 ייִדיש (Yiddish)", "code": "yi"},
    "Yorùbá": {"label": "🇳🇬 Yorùbá (Yoruba)", "code": "yo"},
    "isiZulu": {"label": "🇿🇦 isiZulu (Zulu)", "code": "zu"},
}

# ----------------- UI Strings -----------------
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
    "about_us": "This dashboard explores NASA bioscience publications dynamically.",
}

MORE_INFO_TEXT = """
### What our website does:
Our website summarizes all the PDF publications provided to our AI. To start, you enter keywords. Then, our AI searches for those keywords across the selected PDF publications and simplifies and explains the content of each PDF.

**Q: How does the language feature work?**  
**A:** [Your answer here]

**Q: Can I enable dark mode?**  
**A:** Yes, click the three dots on the top right, go to settings, and click 'Use system setting' or 'Light'. Once clicked, select 'Dark'. If it turns dark, just click the cross mark on top.

**Q: What did we develop?**  
**A:** Our development addresses the challenge by providing a solution that simplifies the process and makes it more efficient for users to achieve their goals.

**Q: Why is it important?**  
**A:** [Your answer here]

**Q: Any other info we should know?**  
**A:** [Your answer here]

### About Us:
We are a group of innovators that are extremely eager to learn. Blah blah testing.
"""

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
def fetch_url_text(url):
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        r.raise_for_status()
    except Exception as e:
        return f"ERROR_FETCH: {str(e)}"
    content_type = r.headers.get("Content-Type", "").lower()
    if "pdf" in content_type or url.lower().endswith(".pdf"):
        try:
            pdf_bytes = io.BytesIO(r.content)
            reader = PyPDF2.PdfReader(pdf_bytes)
            text_parts = [p.extract_text() or "" for p in reader.pages]
            return "\n".join(text_parts)
        except Exception as e:
            return f"ERROR_PDF_PARSE: {str(e)}"
    else:
        try:
            soup = BeautifulSoup(r.text, "html.parser")
            paragraphs = [p.get_text(strip=True) for p in soup.find_all("p") if p.get_text(strip=True)]
            return "\n\n".join(paragraphs)[:20000] if paragraphs else "ERROR_EXTRACT: No text found"
        except Exception as e:
            return f"ERROR_HTML_PARSE: {str(e)}"

# ----------------- Session State -----------------
if "current_lang" not in st.session_state:
    st.session_state.current_lang = "English"
if "translations" not in st.session_state:
    st.session_state.translations = {"English": UI_STRINGS_EN.copy()}
if "page" not in st.session_state:
    st.session_state.page = "main"

# ----------------- Page Config -----------------
st.set_page_config(page_title="NASA BioSpace Dashboard", layout="wide")

# ----------------- Sidebar -----------------
with st.sidebar:
    # Page selection (Main vs More Info)
    page_option = st.radio(
        "Navigate",
        options=["Main", "More Info"],
        index=0 if st.session_state.page == "main" else 1,
        horizontal=False
    )
    st.session_state.page = "main" if page_option == "Main" else "more_info"

    # Language selection
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
                        st.session_state.translations["English"], lang_choice
                    )
                    st.session_state.translations[lang_choice] = translated_strings
                st.session_state.current_lang = lang_choice
            except Exception as e:
                st.error("Translation failed — using English. Error: " + str(e))
                translated_strings = st.session_state.translations["English"]
                st.session_state.current_lang = "English"
    else:
        translated_strings = st.session_state.translations[st.session_state.current_lang]

    # CSV upload
    if st.session_state.page == "main":
        uploaded_csv = st.file_uploader(translated_strings["upload_label"], type=["csv"])
        uploaded_pdfs = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

# ----------------- Main or More Info -----------------
if st.session_state.page == "more_info":
    st.markdown(MORE_INFO_TEXT)
else:
    # ----------------- Main UI -----------------
    st.title(translated_strings["title"])
    st.write(translated_strings["description"])

    mention(
        label=translated_strings["mention_label"],
        icon="NASA International Space Apps Challenge",
        url="https://www.spaceappschallenge.org/"
    )

    # ----------------- Load CSV -----------------
    if uploaded_csv:
        df = pd.read_csv(uploaded_csv)
        st.success(f"Loaded {len(df)} rows")
        st.dataframe(df.head())
        original_cols = df.columns.tolist()
    else:
        df = pd.DataFrame()
        original_cols = []

    # ----------------- Translate dataset -----------------
    translate_dataset = st.checkbox(translated_strings["translate_dataset_checkbox"])
    if translate_dataset and original_cols and st.session_state.current_lang != "English":
        translated_cols = translate_list_via_gemini(original_cols, st.session_state.current_lang)
        df.rename(columns=dict(zip(original_cols, translated_cols)), inplace=True)

    # ----------------- Extract PDFs -----------------
    if uploaded_pdfs:
        st.success(f"{len(uploaded_pdfs)} PDF(s) uploaded")
        for pdf_file in uploaded_pdfs:
            pdf_bytes = io.BytesIO(pdf_file.read())
            pdf_reader = PyPDF2.PdfReader(pdf_bytes)
            text = "".join([p.extract_text() or "" for p in pdf_reader.pages])
            st.write(f"Extracted {len(text)} characters from {pdf_file.name}")

    # ----------------- Search publications -----------------
    query = st.text_input("Enter keyword to search publications")
    if query and not df.empty:
        results = df[df["Title"].astype(str).str.contains(query, case=False, na=False)]
        st.subheader(f"Results: {len(results)} matching titles")
    else:
        results = pd.DataFrame(columns=df.columns)

    for idx, row in results.iterrows():
        title = row.get("Title", "No title")
        link = row.get("Link", "#")
        st.markdown(f"**[{title}]({link})**")

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
