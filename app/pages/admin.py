import sys
from pathlib import Path

import streamlit as st


APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from admin_panel import load_admin_config, render_admin_panel


st.set_page_config(
    page_title="ResearchScope AI Admin",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=Space+Grotesk:wght@500;600;700;800&display=swap');

* {
    font-family: 'Inter', sans-serif;
    box-sizing: border-box;
}

.stApp {
    background:
        radial-gradient(circle at 15% 15%, rgba(217,255,0,0.08), transparent 28%),
        radial-gradient(circle at 85% 20%, rgba(34,211,238,0.07), transparent 25%),
        linear-gradient(135deg, #030303 0%, #080808 48%, #000000 100%);
    color: #f8fafc;
}

header[data-testid="stHeader"] {
    background: transparent;
}

section[data-testid="stSidebar"] {
    display: none;
}

.main .block-container,
.block-container {
    max-width: 1280px !important;
    width: calc(100% - 28px) !important;
    padding: 1.1rem 0 2rem 0 !important;
    margin-left: auto !important;
    margin-right: auto !important;
}

h1, h2, h3, h4, p, label, span {
    color: #f8fafc;
}

[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea,
input,
textarea {
    background-color: #101010 !important;
    color: #f8fafc !important;
    -webkit-text-fill-color: #f8fafc !important;
    border-radius: 14px !important;
    border: 1px solid rgba(217,255,0,0.22) !important;
}

[data-testid="stSelectbox"] [data-baseweb="select"],
[data-testid="stSelectbox"] [data-baseweb="select"] > div {
    background-color: #090909 !important;
    color: #f8fafc !important;
    border-radius: 14px !important;
    border: 1px solid rgba(217,255,0,0.22) !important;
}

div.stButton > button:first-child,
div[data-testid="stFormSubmitButton"] button {
    min-height: 45px !important;
    border-radius: 999px !important;
    background: linear-gradient(135deg, #faffd7 0%, #ffffff 48%, #d9ff00 100%) !important;
    color: #050505 !important;
    -webkit-text-fill-color: #050505 !important;
    border: 1px solid rgba(217,255,0,0.50) !important;
    font-weight: 850 !important;
    text-transform: uppercase !important;
}

div[data-testid="stExpander"],
div[data-testid="stVerticalBlockBorderWrapper"] {
    border-color: rgba(217,255,0,0.16) !important;
    background: rgba(255,255,255,0.025) !important;
    border-radius: 18px !important;
}
</style>
""",
    unsafe_allow_html=True,
)


st.title("⚙️ ResearchScope AI Admin")
st.caption("Direct admin route. Main website menu එකේ Admin Panel එක පේන්නේ නෑ.")

config = load_admin_config()
render_admin_panel(config)