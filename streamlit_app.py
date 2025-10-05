# --- LANGUAGE TRANSLATION FEATURE INTEGRATION ---
from streamlit_extras.let_it_rain import rain

# UI strings in English
UI_STRINGS_EN = {
    "title": "Simplified Knowledge",
    "description": "A dynamic dashboard that summarizes NASA bioscience publications and explores impacts and results.",
    "upload_label": "Upload CSV data",
    "ask_label": "Ask anything:",
    "response_label": "Response:",
    "click_button": "Click here, nothing happens",
    "translate_dataset_checkbox": "Translate dataset column names (may take time)",
    "mention_label": "Official NASA Website",
    "button_response": "Hooray"
}

# Helper function: extract JSON from model output
def extract_json_from_text(text):
    start = text.find('{')
    end = text.rfind('}')
    if start == -1 or end == -1:
        raise ValueError("No JSON object found in model output.")
    return json.loads(text[start:end+1])

# Helper function: translate dictionary via Gemini
def translate_dict_via_gemini(source_dict: dict, target_lang_name: str):
    prompt = (
        f"Translate the VALUES of the following JSON object into {target_lang_name}.\n"
        "Return ONLY a JSON object with the same keys and translated values (no commentary).\n"
        f"Input JSON:\n{json.dumps(source_dict, ensure_ascii=False)}\n"
    )
    resp = model.generate_content(prompt)
    return extract_json_from_text(resp.text)

# Helper function: translate list via Gemini
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

# Initialize session state for language
if "current_lang" not in st.session_state:
    st.session_state.current_lang = "English"
if "translations" not in st.session_state:
    st.session_state.translations = {"English": UI_STRINGS_EN.copy()}

# --- LANGUAGE SELECTOR ---
_, col_language = st.columns([10, 1])
with col_language:
    st.markdown('<div class="language-dropdown-column">', unsafe_allow_html=True)
    lang_choice = st.selectbox(
        "🌐 L",  # Minimal label, hidden via CSS
        options=list(LANGUAGES.keys()),
        index=list(LANGUAGES.keys()).index(st.session_state.current_lang),
        key="language_selector"
    )
    st.markdown('</div>', unsafe_allow_html=True)

# --- HANDLE LANGUAGE CHANGE ---
if lang_choice != st.session_state.current_lang:
    # Simulate ~6 sec loading with emoji rain
    rain(emoji="⏳", font_size=54, falling_speed=5, animation_length=6)
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

# --- UPDATE UI STRINGS IN SESSION ---
st.session_state.translated_strings.update(translated_strings)
