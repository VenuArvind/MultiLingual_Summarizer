import streamlit as st
import re
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM #summarizer
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# WHITESPACE_HANDLER = lambda k: re.sub('\s+', ' ', re.sub('\n+', ' ', k.strip()))

# Language codes dictionary
language_codes = {
    "Arabic": "ar_AR", "Czech": "cs_CZ", "German": "de_DE", "English": "en_XX",
    "Spanish": "es_XX", "Estonian": "et_EE", "Finnish": "fi_FI", "French": "fr_XX",
    "Gujarati": "gu_IN", "Hindi": "hi_IN", "Italian": "it_IT", "Japanese": "ja_XX",
    "Kazakh": "kk_KZ", "Korean": "ko_KR", "Lithuanian": "lt_LT", "Latvian": "lv_LV",
    "Burmese": "my_MM", "Nepali": "ne_NP", "Dutch": "nl_XX", "Romanian": "ro_RO",
    "Russian": "ru_RU", "Sinhala": "si_LK", "Turkish": "tr_TR", "Vietnamese": "vi_VN",
    "Chinese": "zh_CN", "Afrikaans": "af_ZA", "Azerbaijani": "az_AZ", "Bengali": "bn_IN",
    "Persian": "fa_IR", "Hebrew": "he_IL", "Croatian": "hr_HR", "Indonesian": "id_ID",
    "Georgian": "ka_GE", "Khmer": "km_KH", "Macedonian": "mk_MK", "Malayalam": "ml_IN",
    "Mongolian": "mn_MN", "Marathi": "mr_IN", "Polish": "pl_PL", "Pashto": "ps_AF",
    "Portuguese": "pt_XX", "Swedish": "sv_SE", "Swahili": "sw_KE", "Tamil": "ta_IN",
    "Telugu": "te_IN", "Thai": "th_TH", "Tagalog": "tl_XX", "Ukrainian": "uk_UA",
    "Urdu": "ur_PK", "Xhosa": "xh_ZA", "Galician": "gl_ES", "Slovene": "sl_SI"
}

st.title("Multilingual Article Summarizer")

# Load the model and tokenizer of summarizer
# model_name = "csebuetnlp/mT5_multilingual_XLSum"
# tokenizer = AutoTokenizer.from_pretrained(model_name,use_fast=False)
# model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Load model and tokenizer
model1 = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
tokenizer1 = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

def translate_text(text, src_lang, tgt_lang):
    tokenizer1.src_lang = src_lang
    encoded_text = tokenizer1(text, return_tensors="pt")
    tgt_lang_code = tokenizer1.lang_code_to_id[tgt_lang]
    generated_tokens = model1.generate(
        **encoded_text,
        forced_bos_token_id=tgt_lang_code
    )
    translated_text = tokenizer1.batch_decode(generated_tokens, skip_special_tokens=True)
    return translated_text[0]



st.sidebar.title("Language Selection")
selected_language = st.sidebar.selectbox("Select target language:", list(language_codes.keys()))

article_text = st.text_area("Enter the article text:")

def execute_prompt(prompt):
    response = model.generate_content(prompt)
    if response.parts:
        return response.text
    else:
        return "Unable to generate a response. Please try again with a different input."


model = genai.GenerativeModel(model_name="gemini-1.0-pro")

if st.button("Translate"):
    if article_text:
        tgt_lang_code = language_codes[selected_language]
        # summary= summarize_article(article_text)
        gem_prompt = "Summarize the following article in around 2 sentences at max: "+article_text
        gem_summary = execute_prompt(gem_prompt)
        translated_text = translate_text(gem_summary, "en_XX", tgt_lang_code)
        st.write("Translated text in " + selected_language + ":")
        st.write(translated_text)
    else:
        st.warning("Please enter valid text")







