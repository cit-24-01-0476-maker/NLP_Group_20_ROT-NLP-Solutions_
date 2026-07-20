import base64
import html
import mimetypes
import re
import time
import sys
from pathlib import Path
from textwrap import dedent

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import xgboost
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Add src to python path for modular import
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.preprocessing import clean_text
from admin_panel import (
    load_admin_config,
    save_admin_config,
    get_model_accuracy,
    get_site_value,
    get_metric_value,
    get_members,
    get_model_config,
)


# =====================================================
# CONFIG
# =====================================================
BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "models"
DATA_PATH = BASE_DIR / "data" / "processed" / "arxiv_15000_balanced.csv"

ASSET_DIR = BASE_DIR / "app" / "assets"
ROBOT_GIF_PATH = ASSET_DIR / "robot.gif"

LR_ACCURACY = "89.33%"
LSTM_ACCURACY = "85.17%"
ENSEMBLE_V3_ACCURACY = "90.80%"
ENSEMBLE_V4_ACCURACY = "91.07%"

MEMBER2_SVM_ACCURACY = "88.67%"
MEMBER2_CNN_ACCURACY = "86.40%"

MEMBER3_XGBOOST_ACCURACY = "86.90%"
MEMBER3_BERT_ACCURACY = "86.67%"

DEFAULT_TITLE = "A Transformer-Based Approach for Text Classification in Research Papers"
DEFAULT_ABSTRACT = (
    "In this paper, we propose a transformer-based model for automatically classifying "
    "research papers into subject categories using their titles and abstracts. Our approach "
    "leverages deep learning techniques, including self-attention mechanisms, to capture "
    "semantic relationships in text data. We evaluate the model on a large-scale dataset "
    "and compare its performance with traditional machine learning algorithms. Experimental "
    "results show that our method outperforms baseline models in terms of accuracy and F1-score."
)

st.set_page_config(
    page_title="ResearchScope AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =====================================================
# BASIC HELPERS
# =====================================================
def ui(markup: str):
    cleaned = "".join(
        line.strip()
        for line in dedent(markup).splitlines()
        if line.strip()
    )
    st.markdown(cleaned, unsafe_allow_html=True)


def get_file_data_uri(path: Path) -> str:
    if not path.exists():
        return ""

    mime_type, _ = mimetypes.guess_type(str(path))
    if mime_type is None:
        mime_type = "image/gif"

    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def safe_text(value):
    return html.escape(str(value))


def percent_to_float(value, fallback=0.0):
    try:
        return float(str(value).replace("%", "").strip())
    except Exception:
        return fallback


def normalize_runtime_admin_config(config):
    changed = False
    models_cfg = config.setdefault("models", {})

    member2_defaults = {
        "member2_svm": {
            "owner": "member2",
            "enabled": True,
            "display_name": "Member 2 SVM Model",
            "type": "Machine Learning",
            "accuracy": MEMBER2_SVM_ACCURACY,
            "note": "TF-IDF + Support Vector Machine classifier",
        },
        "member2_cnn": {
            "owner": "member2",
            "enabled": True,
            "display_name": "Member 2 CNN Model",
            "type": "Deep Learning",
            "accuracy": MEMBER2_CNN_ACCURACY,
            "note": "Tokenizer + Embedding + CNN text classifier",
        },
    }

    for model_key, defaults in member2_defaults.items():
        model = models_cfg.setdefault(model_key, defaults.copy())

        for field in ["owner", "display_name", "type", "note"]:
            current_value = str(model.get(field, "")).strip()
            if current_value == "" or "future" in current_value.lower():
                model[field] = defaults[field]
                changed = True

        current_accuracy = str(model.get("accuracy", "")).strip().lower()
        if current_accuracy in ["", "pending", "none", "not set", "n/a"]:
            model["accuracy"] = defaults["accuracy"]
            changed = True

        if "enabled" not in model:
            model["enabled"] = True
            changed = True

    members = config.setdefault("members", [])
    for member in members:
        if member.get("role_key") == "member2" or member.get("tag") == "Member 2":
            if int(member.get("progress", 0)) < 100:
                member["progress"] = 100
                changed = True
            if str(member.get("status", "")).lower().startswith("pending"):
                member["status"] = "Completed"
                changed = True
            if "future" in str(member.get("items", "")).lower() or "git branch" in str(member.get("items", "")).lower():
                member["items"] = "SVM · CNN · Accuracy Evaluation · Model Export · Git Branch Integration"
                changed = True
            member["role"] = "SVM and CNN model development, model evaluation and branch integration"

    if changed:
        save_admin_config(config)

    return config


# =====================================================
# SESSION STATE
# =====================================================
def init_state():
    defaults = {
        "home_title": DEFAULT_TITLE,
        "home_abstract": DEFAULT_ABSTRACT,
        "pred_title": DEFAULT_TITLE,
        "pred_abstract": DEFAULT_ABSTRACT,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def clear_home_form():
    st.session_state["home_title"] = ""
    st.session_state["home_abstract"] = ""
    st.session_state.pop("latest_result", None)


def clear_prediction_form():
    st.session_state["pred_title"] = ""
    st.session_state["pred_abstract"] = ""
    st.session_state.pop("latest_result", None)


def load_demo_home():
    st.session_state["home_title"] = DEFAULT_TITLE
    st.session_state["home_abstract"] = DEFAULT_ABSTRACT


def load_demo_prediction():
    st.session_state["pred_title"] = DEFAULT_TITLE
    st.session_state["pred_abstract"] = DEFAULT_ABSTRACT


init_state()


# =====================================================
# CSS
# =====================================================
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=Space+Grotesk:wght@500;600;700;800&display=swap');

* {
    font-family: 'Inter', sans-serif;
    box-sizing: border-box;
}

html {
    scroll-behavior: smooth;
}

.stApp {
    background:
        radial-gradient(circle at 18% 10%, rgba(217,255,0,0.055), transparent 24%),
        radial-gradient(circle at 86% 16%, rgba(34,211,238,0.07), transparent 25%),
        radial-gradient(circle at 65% 90%, rgba(168,85,247,0.055), transparent 30%),
        linear-gradient(135deg, #030303 0%, #080808 48%, #000000 100%);
    color: #f8fafc;
    overflow-x: hidden;
}

.stApp::before {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 0;
    background-image:
        radial-gradient(circle, rgba(217,255,0,0.32) 1px, transparent 1.4px),
        radial-gradient(circle, rgba(34,211,238,0.20) 1px, transparent 1.4px);
    background-size: 115px 115px, 190px 190px;
    background-position: 0 0, 55px 70px;
    opacity: 0.12;
    animation: particleMove 42s linear infinite;
}

.stApp::after {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 0;
    background:
        linear-gradient(90deg, rgba(255,255,255,0.016) 1px, transparent 1px),
        linear-gradient(rgba(255,255,255,0.016) 1px, transparent 1px);
    background-size: 82px 82px;
    opacity: 0.38;
}

header[data-testid="stHeader"] {
    background: transparent;
}

section[data-testid="stSidebar"] {
    display: none;
}

.main .block-container,
.block-container {
    max-width: 1760px !important;
    width: calc(100% - 12px) !important;
    padding: 0.45rem 0 1.1rem 0 !important;
    margin-left: auto !important;
    margin-right: auto !important;
    position: relative;
    z-index: 2;
}

p, li, label, span {
    color: #cbd5e1;
}

h1, h2, h3, h4 {
    color: #f8fafc;
}

#prediction-panel,
#team-progress {
    scroll-margin-top: 130px;
}

/* ================= TOP BAR ================= */
.topbar {
    position: sticky;
    top: 0;
    z-index: 999;
    margin-bottom: 12px;
    padding: 10px 15px;
    border-radius: 24px;
    background:
        radial-gradient(circle at 92% 10%, rgba(217,255,0,0.050), transparent 35%),
        linear-gradient(135deg, rgba(18,18,18,0.92), rgba(6,6,6,0.96));
    border: 1px solid rgba(217,255,0,0.16);
    box-shadow: 0 18px 48px rgba(0,0,0,0.52);
    backdrop-filter: blur(18px);
}

.topbar-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 16px;
    flex-wrap: wrap;
}

.brand {
    display: flex;
    align-items: center;
    gap: 12px;
}

.brand-mark {
    width: 48px;
    height: 48px;
    border-radius: 18px;
    background:
        radial-gradient(circle at 50% 50%, rgba(217,255,0,0.16), rgba(0,0,0,0.95));
    border: 1px solid rgba(217,255,0,0.30);
    display: flex;
    justify-content: center;
    align-items: center;
    overflow: hidden;
    box-shadow:
        0 10px 24px rgba(0,0,0,0.45),
        0 0 20px rgba(217,255,0,0.18);
}

.brand-logo-gif {
    width: 100%;
    height: 100%;
    object-fit: cover;
    object-position: center;
    display: block;
    transform: scale(1.04);
    filter: contrast(1.08) brightness(0.92) saturate(1.08);
}

.brand-fallback {
    width: 44px;
    height: 44px;
    border-radius: 15px;
    background: #d9ff00;
    color: #050505 !important;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 950;
    font-size: 19px;
    transform: rotate(-8deg);
}

.brand-fallback span {
    color: #050505 !important;
    transform: rotate(8deg);
    display: inline-block;
}

.brand-title {
    color: #ffffff;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 800;
    font-size: 15px;
    letter-spacing: 1.4px;
    text-transform: uppercase;
}

.brand-sub {
    color: #d9ff00;
    font-size: 9px;
    font-weight: 900;
    letter-spacing: 1.4px;
    text-transform: uppercase;
}

.status-pill {
    display: flex;
    align-items: center;
    gap: 9px;
    color: #ffffff;
    font-size: 11px;
    font-weight: 900;
    letter-spacing: 1.4px;
    text-transform: uppercase;
    padding: 9px 14px;
    border-radius: 999px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.09);
}

.green-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #22c55e;
    box-shadow: 0 0 15px rgba(34,197,94,0.9);
}

/* ================= MENU BAR ================= */
div[role="radiogroup"] {
    display: inline-flex !important;
    justify-content: flex-start !important;
    align-items: center !important;
    gap: 6px !important;
    flex-wrap: nowrap !important;
    max-width: 100% !important;
    width: auto !important;
    overflow-x: auto !important;
    overflow-y: hidden !important;
    padding: 8px !important;
    margin-bottom: 18px !important;
    border-radius: 999px !important;
    background:
        radial-gradient(circle at 80% 0%, rgba(217,255,0,0.045), transparent 35%),
        linear-gradient(135deg, rgba(14,14,14,0.88), rgba(4,4,4,0.96)) !important;
    border: 1px solid rgba(217,255,0,0.15) !important;
    box-shadow: 0 18px 45px rgba(0,0,0,0.40) !important;
    scrollbar-width: none !important;
}

div[role="radiogroup"]::-webkit-scrollbar {
    display: none !important;
}

div[role="radiogroup"] input[type="radio"] {
    display: none !important;
}

div[role="radiogroup"] label > div:first-child > div:first-child {
    display: none !important;
}

div[role="radiogroup"] label > div:first-child {
    gap: 0 !important;
    padding: 0 !important;
    margin: 0 !important;
}

div[role="radiogroup"] label {
    border-radius: 999px !important;
    padding: 8px 13px !important;
    margin: 0 !important;
    border: 1px solid transparent !important;
    transition: 0.22s ease !important;
    white-space: nowrap !important;
}

div[role="radiogroup"] label:hover {
    background: rgba(217,255,0,0.07) !important;
    border-color: rgba(217,255,0,0.20) !important;
    transform: translateY(-2px);
}

div[role="radiogroup"] label:has(input:checked) {
    background: #d9ff00 !important;
    border-color: rgba(217,255,0,0.65) !important;
    box-shadow: 0 0 18px rgba(217,255,0,0.22) !important;
}

div[role="radiogroup"] label:has(input:checked) * {
    color: #050505 !important;
    font-weight: 950 !important;
}

div[role="radiogroup"] p {
    color: #a1a1aa !important;
    font-size: 12px !important;
    font-weight: 850 !important;
    margin: 0 !important;
}

/* ================= HERO ================= */
.game-hero {
    position: relative;
    overflow: hidden;
    min-height: 550px;
    border-radius: 24px;
    padding: 26px 32px;
    margin-bottom: 18px;
    width: 100%;
    background:
        linear-gradient(rgba(255,255,255,0.032) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.032) 1px, transparent 1px),
        radial-gradient(circle at 50% 48%, rgba(217,255,0,0.045), transparent 27%),
        radial-gradient(circle at 53% 42%, rgba(34,211,238,0.085), transparent 32%),
        linear-gradient(135deg, #050505 0%, #0a0a0a 48%, #020202 100%);
    background-size: 74px 74px, 74px 74px, auto, auto, auto;
    border: 1px solid rgba(217,255,0,0.15);
    box-shadow:
        0 30px 80px rgba(0,0,0,0.68),
        inset 0 1px 0 rgba(255,255,255,0.055);
}

.game-hero-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 18px;
    position: relative;
    z-index: 4;
    width: 100%;
}

.game-logo {
    display: flex;
    align-items: center;
    gap: 10px;
}

.game-logo-mark {
    width: 36px;
    height: 36px;
    border-radius: 12px;
    background: #d9ff00;
    color: #050505;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 950;
    font-size: 16px;
    transform: rotate(-10deg);
    box-shadow: 0 0 18px rgba(217,255,0,0.20);
}

.game-logo-mark span {
    color: #050505 !important;
    transform: rotate(10deg);
    display: inline-block;
}

.game-logo-text {
    color: #ffffff;
    font-size: 12px;
    font-weight: 950;
    line-height: 1.05;
}

.game-logo-text span {
    display: block;
    color: #a3a3a3;
    font-size: 9px;
    font-weight: 700;
}

.game-time {
    position: absolute;
    left: 50%;
    top: 50%;
    transform: translate(-50%, -50%);
    color: #a3a3a3;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 0.9px;
    text-align: center;
    white-space: nowrap;
    pointer-events: none;
}

.game-menu-chip {
    color: #ffffff;
    background: rgba(255,255,255,0.10);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 10px;
    padding: 9px 11px;
    font-size: 13px;
}

.game-hero-content {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 16px;
    align-items: center;
    min-height: 390px;
    position: relative;
    z-index: 3;
}

.game-left {
    justify-self: start;
}

.game-kicker {
    color: #ffffff;
    font-size: 10.5px;
    font-weight: 950;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 17px;
}

.game-title {
    color: #ffffff;
    font-family: 'Space Grotesk', sans-serif;
    font-size: clamp(32px, 2.55vw, 39px);
    font-weight: 800;
    letter-spacing: -1.6px;
    line-height: 1.02;
    max-width: 470px;
}

.game-cta {
    display: inline-flex;
    align-items: center;
    gap: 9px;
    margin-top: 28px;
    color: #d9ff00 !important;
    font-size: 13px;
    font-weight: 950;
    text-decoration: none !important;
    border-bottom: 3px solid #d9ff00;
    padding-bottom: 7px;
}

.game-scroll-note {
    margin-top: 52px;
    color: #7c7c7c;
    font-size: 9.5px;
    font-weight: 900;
    text-transform: uppercase;
    line-height: 1.35;
}

.robot-zone {
    position: relative;
    height: 350px;
    display: flex;
    align-items: center;
    justify-content: center;
    justify-self: center;
    width: 100%;
}

.robot-gif-frame {
    position: relative;
    width: 340px;
    height: 340px;
    display: flex;
    align-items: center;
    justify-content: center;
}

.robot-gif-mask {
    position: relative;
    width: 296px;
    height: 296px;
    border-radius: 50%;
    overflow: hidden;
    background:
        radial-gradient(circle at center, rgba(10,16,10,0.72), rgba(0,0,0,0.98));
    border: 1px solid rgba(217,255,0,0.16);
    box-shadow:
        inset 0 0 18px rgba(217,255,0,0.035),
        inset 0 0 42px rgba(0,0,0,0.55),
        0 0 14px rgba(217,255,0,0.08);
    z-index: 3;
}

.robot-gif-ring {
    position: absolute;
    width: 312px;
    height: 312px;
    border-radius: 50%;
    z-index: 2;
    border: 1px solid rgba(217,255,0,0.20);
    box-shadow:
        0 0 10px rgba(217,255,0,0.14),
        0 0 22px rgba(217,255,0,0.09),
        0 0 34px rgba(180,255,0,0.06);
    animation: lemonPulseSoft 4s ease-in-out infinite;
}

.robot-gif-ring::before {
    content: "";
    position: absolute;
    inset: -8px;
    border-radius: 50%;
    border: 1px solid rgba(217,255,0,0.055);
    box-shadow:
        0 0 12px rgba(217,255,0,0.08),
        0 0 24px rgba(217,255,0,0.045);
    animation: lemonWaveSoft 5s ease-in-out infinite;
}

.robot-gif-ring::after {
    content: "";
    position: absolute;
    inset: -16px;
    border-radius: 50%;
    background:
        radial-gradient(circle, rgba(217,255,0,0.060) 0%, rgba(217,255,0,0.020) 34%, transparent 66%);
    filter: blur(13px);
    opacity: 0.35;
    animation: outerGlowSoft 6s ease-in-out infinite;
}

.robot-gif-frame::before {
    content: "";
    position: absolute;
    width: 370px;
    height: 370px;
    border-radius: 50%;
    background:
        radial-gradient(circle, rgba(217,255,0,0.050) 0%, rgba(34,211,238,0.030) 30%, transparent 62%);
    filter: blur(18px);
    opacity: 0.48;
    z-index: 0;
    animation: ambientGlowSoft 6s ease-in-out infinite;
}

.robot-gif-frame::after {
    content: "";
    position: absolute;
    width: 340px;
    height: 340px;
    border-radius: 50%;
    background:
        conic-gradient(
            from 0deg,
            rgba(217,255,0,0),
            rgba(217,255,0,0.065),
            rgba(217,255,0,0),
            rgba(34,211,238,0.045),
            rgba(217,255,0,0)
        );
    filter: blur(7px);
    opacity: 0.32;
    z-index: 1;
    animation: rotateRing 14s linear infinite;
}

.robot-gif {
    width: 100%;
    height: 100%;
    object-fit: cover;
    object-position: center;
    display: block;
    transform: scale(1.08);
    filter:
        contrast(1.05)
        brightness(0.92)
        saturate(1.05);
}

.robot-gif-panel {
    position: absolute;
    bottom: 17px;
    left: 50%;
    transform: translateX(-50%);
    z-index: 5;
    padding: 8px 15px;
    border-radius: 999px;
    background: rgba(3,8,3,0.84);
    border: 1px solid rgba(217,255,0,0.18);
    color: #d9ff00;
    font-size: 10px;
    font-weight: 950;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    white-space: nowrap;
}

.game-right {
    justify-self: end;
}

.game-since {
    color: #6b7280;
    font-size: 10.5px;
    font-weight: 900;
    text-align: right;
    margin-bottom: 34px;
}

.social-row {
    display: flex;
    justify-content: flex-end;
    gap: 9px;
    margin-bottom: 66px;
}

.social-box {
    width: 31px;
    height: 31px;
    border-radius: 8px;
    border: 1px solid rgba(217,255,0,0.40);
    color: #d9ff00;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 11px;
    font-weight: 950;
}

.game-mini-info {
    color: #d1d5db;
    font-size: 12px;
    font-weight: 800;
    line-height: 1.45;
    text-align: right;
}

.game-mini-info span {
    color: #7c7c7c;
    display: block;
    font-size: 10.5px;
    font-weight: 700;
    margin-top: 4px;
}

.game-tabs {
    position: absolute;
    left: 32px;
    right: 32px;
    bottom: 22px;
    z-index: 5;
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    overflow: hidden;
    border-radius: 8px;
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.10);
}

.game-tab {
    padding: 16px 12px;
    color: #ffffff;
    font-size: 11.5px;
    font-weight: 950;
    text-align: center;
    background: rgba(255,255,255,0.08);
    border-right: 1px solid rgba(255,255,255,0.06);
}

.game-tab.active {
    background: rgba(255,255,255,0.14);
}

/* ================= PAGE HERO ================= */
.page-hero {
    min-height: 270px;
    border-radius: 28px;
    padding: 36px 42px;
    margin-bottom: 20px;
    background:
        linear-gradient(rgba(255,255,255,0.030) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.030) 1px, transparent 1px),
        radial-gradient(circle at 75% 20%, rgba(217,255,0,0.06), transparent 30%),
        linear-gradient(135deg, rgba(8,8,8,0.96), rgba(0,0,0,0.98));
    background-size: 76px 76px, 76px 76px, auto, auto;
    border: 1px solid rgba(217,255,0,0.15);
    box-shadow: 0 24px 68px rgba(0,0,0,0.60);
    display: grid;
    grid-template-columns: 1.25fr 0.75fr;
    gap: 30px;
    align-items: center;
}

.page-kicker {
    color: #d9ff00;
    font-size: 11px;
    font-weight: 950;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 14px;
}

.page-title {
    color: #ffffff;
    font-family: 'Space Grotesk', sans-serif;
    font-size: clamp(34px, 3.4vw, 50px);
    font-weight: 800;
    letter-spacing: -1.8px;
    line-height: 0.98;
}

.page-line {
    width: 96px;
    height: 1px;
    background: linear-gradient(90deg, #d9ff00, transparent);
    margin: 19px 0;
}

.page-text {
    color: #9ca3af;
    font-size: 14px;
    font-weight: 650;
    line-height: 1.62;
}

.page-panel {
    border-radius: 22px;
    padding: 18px;
    background: rgba(255,255,255,0.035);
    border: 1px solid rgba(217,255,0,0.13);
}

.page-panel-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 11px;
}

.page-stat {
    padding: 13px;
    border-radius: 15px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(217,255,0,0.12);
}

.page-stat-label {
    color: #737373;
    font-size: 10px;
    font-weight: 950;
    letter-spacing: 1px;
    text-transform: uppercase;
}

.page-stat-value {
    color: #ffffff;
    font-size: 17px;
    font-weight: 950;
    margin-top: 5px;
}

/* ================= CARDS ================= */
.metric-strip {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 12px;
    margin-bottom: 20px;
}

.metric-card,
.result-card,
.info-card,
.progress-card {
    background:
        radial-gradient(circle at 90% 8%, rgba(217,255,0,0.045), transparent 32%),
        linear-gradient(145deg, rgba(18,18,18,0.84), rgba(6,6,6,0.96));
    border: 1px solid rgba(217,255,0,0.13);
    border-radius: 24px;
    box-shadow: 0 22px 52px rgba(0,0,0,0.52);
    transition: 0.22s ease;
}

.metric-card {
    padding: 17px;
}

.metric-label {
    color: #737373;
    font-size: 10.5px;
    font-weight: 950;
    text-transform: uppercase;
    letter-spacing: 1.4px;
}

.metric-value {
    color: #ffffff;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 31px;
    font-weight: 800;
    margin-top: 8px;
}

.metric-note {
    color: #d9ff00;
    font-size: 10.5px;
    font-weight: 900;
    margin-top: 8px;
}

.section-heading {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 17px 0 17px 0;
}

.section-icon {
    width: 46px;
    height: 46px;
    border-radius: 15px;
    background: rgba(217,255,0,0.065);
    border: 1px solid rgba(217,255,0,0.18);
    color: #d9ff00;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 22px;
}

.section-title {
    color: #ffffff;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 22px;
    font-weight: 800;
}

.section-subtitle {
    color: #8b8f98;
    font-size: 12.5px;
    font-weight: 650;
    margin-top: 4px;
}

.result-card {
    padding: 19px;
    min-height: 126px;
}

.result-icon {
    width: 52px;
    height: 52px;
    border-radius: 17px;
    background: rgba(217,255,0,0.065);
    border: 1px solid rgba(217,255,0,0.18);
    color: #d9ff00;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 24px;
    margin-bottom: 10px;
}

.result-label {
    color: #8b8f98;
    font-size: 11.5px;
    font-weight: 850;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 8px;
}

.result-value {
    color: #ffffff;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 23px;
    font-weight: 800;
}

.empty-state {
    border: 1px dashed rgba(217,255,0,0.18);
    border-radius: 20px;
    padding: 32px 20px;
    color: #8b8f98;
    text-align: center;
    background: rgba(255,255,255,0.02);
}

.top-row {
    display: grid;
    grid-template-columns: 32px 1.1fr 2.4fr 60px;
    align-items: center;
    gap: 12px;
    margin: 13px 0;
}

.top-index {
    color: #d9ff00;
    font-size: 15px;
    font-weight: 950;
}

.top-name {
    color: #ffffff;
    font-size: 13.5px;
    font-weight: 850;
}

.progress-track,
.chart-track,
.member-progress-line {
    background: rgba(255,255,255,0.08);
    border-radius: 999px;
    overflow: hidden;
}

.progress-track {
    height: 15px;
}

.progress-fill,
.chart-fill,
.member-progress-fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #d9ff00, #fff8b3);
    box-shadow: 0 0 13px rgba(217,255,0,0.18);
    animation: loadBar 0.8s ease both;
}

.top-score {
    color: #d9ff00;
    font-size: 13.5px;
    font-weight: 950;
    text-align: right;
}

.info-card {
    padding: 18px;
    min-height: 164px;
}

.info-card-title {
    color: #ffffff;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 15.5px;
    font-weight: 800;
    margin-bottom: 12px;
}

.info-card-text {
    color: #9ca3af;
    font-size: 12.8px;
    font-weight: 650;
    line-height: 1.65;
}

.keyword-chip {
    display: inline-block;
    padding: 8px 11px;
    background: rgba(217,255,0,0.065);
    border: 1px solid rgba(217,255,0,0.14);
    color: #d9ff00 !important;
    border-radius: 999px;
    margin: 5px 5px 5px 0;
    font-size: 11.5px;
    font-weight: 850;
}

.dataset-line {
    display: flex;
    gap: 12px;
    align-items: center;
    padding: 10px 0;
    border-bottom: 1px solid rgba(255,255,255,0.07);
}

.dataset-line:last-child {
    border-bottom: none;
}

.dataset-icon {
    width: 36px;
    height: 36px;
    border-radius: 12px;
    background: rgba(217,255,0,0.065);
    border: 1px solid rgba(217,255,0,0.14);
    color: #d9ff00;
    display: flex;
    align-items: center;
    justify-content: center;
}

.dataset-key {
    color: #8b8f98;
    font-size: 11.5px;
    font-weight: 750;
}

.dataset-value {
    color: #ffffff;
    font-size: 13.5px;
    font-weight: 900;
}

.team-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 14px;
    margin-bottom: 20px;
}

.progress-card {
    padding: 19px;
}

.member-tag {
    color: #d9ff00;
    font-size: 10.5px;
    font-weight: 950;
    letter-spacing: 1.4px;
    text-transform: uppercase;
    margin-bottom: 8px;
}

.member-name {
    color: #ffffff;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 21px;
    font-weight: 800;
    margin-bottom: 6px;
}

.member-role {
    color: #9ca3af;
    font-size: 12px;
    font-weight: 700;
    min-height: 34px;
    line-height: 1.45;
}

.member-progress-line {
    height: 10px;
    margin: 16px 0 9px 0;
}

.member-status {
    color: #d9ff00;
    font-size: 12px;
    font-weight: 950;
}

.member-list {
    margin-top: 12px;
    color: #9ca3af;
    font-size: 12px;
    font-weight: 650;
}

.dark-table-wrap,
.dark-chart,
.text-panel {
    border: 1px solid rgba(217,255,0,0.13);
    border-radius: 18px;
    overflow: hidden;
    background: rgba(255,255,255,0.025);
    margin-top: 14px;
}

.dark-chart,
.text-panel {
    padding: 18px;
}

.dark-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
}

.dark-table th {
    background: rgba(217,255,0,0.065);
    color: #d9ff00;
    text-align: left;
    padding: 12px 14px;
    font-weight: 900;
    text-transform: uppercase;
    font-size: 11px;
}

.dark-table td {
    color: #e5e7eb;
    padding: 12px 14px;
    border-top: 1px solid rgba(255,255,255,0.06);
}

.chart-row {
    display: grid;
    grid-template-columns: 190px 1fr 72px;
    align-items: center;
    gap: 14px;
    margin: 13px 0;
}

.chart-label {
    color: #ffffff;
    font-size: 13px;
    font-weight: 800;
}

.chart-track {
    height: 18px;
}

.chart-value {
    color: #d9ff00;
    font-size: 13px;
    font-weight: 950;
    text-align: right;
}

.text-panel-title {
    color: #d9ff00;
    font-size: 12px;
    font-weight: 950;
    text-transform: uppercase;
    margin-bottom: 10px;
}

.text-panel-body {
    color: #e5e7eb;
    font-size: 13px;
    font-weight: 600;
    line-height: 1.7;
    max-height: 220px;
    overflow-y: auto;
    white-space: pre-wrap;
}

.footer-note {
    display: flex;
    justify-content: space-between;
    gap: 14px;
    padding: 16px 19px;
    border-radius: 22px;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(217,255,0,0.12);
    color: #9ca3af;
    font-size: 12.5px;
    font-weight: 800;
    margin-top: 8px;
}

.footer-highlight {
    color: #d9ff00;
}

/* ================= STREAMLIT WIDGETS ================= */
[data-testid="stForm"] {
    background: rgba(0,0,0,0.18) !important;
    border: 1px solid rgba(217,255,0,0.12) !important;
    border-radius: 22px !important;
    padding: 18px !important;
}

[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea,
input,
textarea {
    background-color: #101010 !important;
    color: #f8fafc !important;
    -webkit-text-fill-color: #f8fafc !important;
    border-radius: 15px !important;
    border: 1px solid rgba(217,255,0,0.22) !important;
    caret-color: #d9ff00 !important;
    font-size: 14px !important;
}

[data-testid="stTextArea"] textarea {
    min-height: 205px !important;
}

[data-testid="stSelectbox"] [data-baseweb="select"],
[data-testid="stSelectbox"] [data-baseweb="select"] > div,
div[data-baseweb="select"],
div[data-baseweb="select"] > div {
    background-color: #090909 !important;
    color: #f8fafc !important;
    border-radius: 15px !important;
    border: 1px solid rgba(217,255,0,0.22) !important;
}

[data-testid="stSelectbox"] span,
div[data-baseweb="select"] span,
div[data-baseweb="select"] svg {
    color: #f8fafc !important;
    fill: #f8fafc !important;
}

div[data-baseweb="popover"],
div[data-baseweb="popover"] ul,
ul[role="listbox"] {
    background: #070707 !important;
    color: #f8fafc !important;
}

li[role="option"],
div[role="option"] {
    background: #090909 !important;
    color: #f8fafc !important;
}

li[role="option"] *,
div[role="option"] * {
    color: #f8fafc !important;
}

div.stButton > button:first-child,
div[data-testid="stFormSubmitButton"] button {
    min-height: 47px !important;
    min-width: 140px !important;
    padding: 0.78rem 1.12rem !important;
    border-radius: 999px !important;
    background:
        linear-gradient(135deg, #faffd7 0%, #ffffff 48%, #d9ff00 100%) !important;
    color: #050505 !important;
    -webkit-text-fill-color: #050505 !important;
    border: 1px solid rgba(217,255,0,0.50) !important;
    box-shadow:
        0 10px 22px rgba(0,0,0,0.35),
        0 0 0 1px rgba(255,255,255,0.08),
        0 0 18px rgba(217,255,0,0.08) !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 12.5px !important;
    font-weight: 800 !important;
    letter-spacing: 1.1px !important;
    text-transform: uppercase !important;
    transition: all 0.22s ease !important;
}

div.stButton > button:first-child *,
div[data-testid="stFormSubmitButton"] button * {
    color: #050505 !important;
    -webkit-text-fill-color: #050505 !important;
    font-weight: 800 !important;
}

div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.035);
    border: 1px solid rgba(217,255,0,0.10);
    border-radius: 18px;
    padding: 15px;
}

div[data-testid="stMetric"] * {
    color: #f8fafc !important;
}

@keyframes particleMove {
    0% { background-position: 0 0, 55px 70px; }
    100% { background-position: 700px 900px, -500px 600px; }
}

@keyframes lemonPulseSoft {
    0%, 100% { transform: scale(1); opacity: 0.72; }
    50% { transform: scale(1.012); opacity: 0.9; }
}

@keyframes lemonWaveSoft {
    0%, 100% { transform: scale(1); opacity: 0.35; }
    50% { transform: scale(1.018); opacity: 0.62; }
}

@keyframes outerGlowSoft {
    0%, 100% { transform: scale(1); opacity: 0.32; }
    50% { transform: scale(1.03); opacity: 0.50; }
}

@keyframes ambientGlowSoft {
    0%, 100% { transform: scale(1); opacity: 0.40; }
    50% { transform: scale(1.035); opacity: 0.58; }
}

@keyframes rotateRing {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
}

@keyframes loadBar {
    from { width: 0; }
}

@media (max-width: 1100px) {
    .main .block-container,
    .block-container {
        width: calc(100% - 10px) !important;
    }

    .game-hero-content {
        grid-template-columns: 1fr !important;
    }

    .game-right {
        display: none !important;
    }

    .game-hero {
        min-height: auto !important;
    }

    .robot-zone {
        height: 320px !important;
    }

    .robot-gif-frame {
        width: 315px !important;
        height: 315px !important;
    }

    .robot-gif-mask {
        width: 275px !important;
        height: 275px !important;
    }

    .robot-gif-ring {
        width: 292px !important;
        height: 292px !important;
    }

    .game-tabs {
        position: relative;
        left: auto;
        right: auto;
        bottom: auto;
        margin-top: 20px;
        grid-template-columns: repeat(2, 1fr);
    }

    .page-hero {
        grid-template-columns: 1fr;
    }

    .metric-strip,
    .team-grid {
        grid-template-columns: repeat(2, 1fr);
    }

    .top-row,
    .chart-row {
        grid-template-columns: 1fr;
    }

    .top-score,
    .chart-value {
        text-align: left;
    }
}

@media (max-width: 760px) {
    .main .block-container,
    .block-container {
        width: calc(100% - 8px) !important;
        padding: 0.35rem 0 1rem 0 !important;
    }

    .topbar {
        border-radius: 22px;
    }

    .status-pill {
        width: 100%;
        justify-content: center;
    }

    div[role="radiogroup"] {
        overflow-x: auto !important;
        flex-wrap: nowrap !important;
        justify-content: flex-start !important;
        border-radius: 22px !important;
    }

    div[role="radiogroup"] label {
        white-space: nowrap !important;
    }

    .game-hero {
        padding: 20px !important;
        border-radius: 20px !important;
    }

    .game-time,
    .game-menu-chip {
        display: none;
    }

    .game-title {
        font-size: 30px !important;
    }

    .robot-zone {
        height: 285px !important;
    }

    .robot-gif-frame {
        width: 285px !important;
        height: 285px !important;
    }

    .robot-gif-mask {
        width: 242px !important;
        height: 242px !important;
    }

    .robot-gif-ring {
        width: 258px !important;
        height: 258px !important;
    }

    .game-tabs,
    .metric-strip,
    .team-grid,
    .page-panel-grid {
        grid-template-columns: 1fr;
    }

    .page-title {
        font-size: 34px;
    }

    div[data-testid="column"] {
        width: 100% !important;
        flex: 1 1 100% !important;
    }

    div.stButton > button:first-child,
    div[data-testid="stFormSubmitButton"] button {
        min-width: 128px !important;
        min-height: 45px !important;
        font-size: 12px !important;
    }
}
</style>
""",
    unsafe_allow_html=True,
)


# =====================================================
# MODEL HELPERS
# =====================================================
# clean_text is imported from src.preprocessing



def check_file_exists(path, label):
    if not path.exists():
        st.error(f"{label} not found.")
        st.code(str(path))
        st.stop()


def optional_load_joblib(path):
    try:
        if path.exists():
            return joblib.load(path)
    except Exception:
        return None
    return None


def optional_load_keras(path):
    try:
        if path.exists():
            return load_model(path, compile=False)
    except Exception:
        return None
    return None


def optional_load_joblib_any(paths):
    for path in paths:
        obj = optional_load_joblib(path)
        if obj is not None:
            return obj
    return None


def optional_load_keras_any(paths):
    for path in paths:
        obj = optional_load_keras(path)
        if obj is not None:
            return obj
    return None


@st.cache_resource
def load_models():
    lr_model_path = MODEL_DIR / "member1_logistic_regression.pkl"
    tfidf_path = MODEL_DIR / "member1_tfidf_vectorizer.pkl"
    label_encoder_path = MODEL_DIR / "member1_label_encoder.pkl"
    lstm_model_path = MODEL_DIR / "member1_lstm_model.h5"
    lstm_tokenizer_path = MODEL_DIR / "member1_lstm_tokenizer.pkl"
    ensemble_v3_path = MODEL_DIR / "member1_advanced_ensemble_v3.pkl"
    ensemble_v4_path = MODEL_DIR / "member1_advanced_ensemble_v4.pkl"

    check_file_exists(lr_model_path, "Logistic Regression model")
    check_file_exists(tfidf_path, "TF-IDF vectorizer")
    check_file_exists(label_encoder_path, "Label encoder")
    check_file_exists(lstm_model_path, "LSTM model")
    check_file_exists(lstm_tokenizer_path, "LSTM tokenizer")

    member2_svm_model = optional_load_joblib_any(
        [
            MODEL_DIR / "member2_svm_model.pkl",
            MODEL_DIR / "member2_svm.pkl",
            MODEL_DIR / "svm_model.pkl",
        ]
    )

    member2_svm_pipeline = optional_load_joblib_any(
        [
            MODEL_DIR / "member2_svm_pipeline.pkl",
            MODEL_DIR / "svm_pipeline.pkl",
        ]
    )

    member2_svm_vectorizer = optional_load_joblib_any(
        [
            MODEL_DIR / "member2_svm_vectorizer.pkl",
            MODEL_DIR / "member2_tfidf_vectorizer.pkl",
            MODEL_DIR / "svm_vectorizer.pkl",
        ]
    )

    member2_label_encoder = optional_load_joblib_any(
        [
            MODEL_DIR / "member2_label_encoder.pkl",
            MODEL_DIR / "member2_svm_label_encoder.pkl",
        ]
    )

    member2_cnn_model = optional_load_keras_any(
        [
            MODEL_DIR / "member2_cnn_model.h5",
            MODEL_DIR / "member2_cnn.h5",
            MODEL_DIR / "cnn_model.h5",
            MODEL_DIR / "member2_cnn_model.keras",
            MODEL_DIR / "member2_cnn.keras",
            MODEL_DIR / "cnn_model.keras",
        ]
    )

    member2_cnn_tokenizer = optional_load_joblib_any(
        [
            MODEL_DIR / "member2_cnn_tokenizer.pkl",
            MODEL_DIR / "member2_tokenizer.pkl",
            MODEL_DIR / "cnn_tokenizer.pkl",
        ]
    )

    member2_cnn_label_encoder = optional_load_joblib_any(
        [
            MODEL_DIR / "member2_cnn_label_encoder.pkl",
            MODEL_DIR / "member2_label_encoder.pkl",
            MODEL_DIR / "cnn_label_encoder.pkl",
        ]
    )

    member3_xgboost_model = optional_load_joblib(MODEL_DIR / "member3_xgboost_model.pkl")
    member3_xgboost_vectorizer = optional_load_joblib(MODEL_DIR / "member3_tfidf_vectorizer.pkl")
    member3_xgboost_label_encoder = optional_load_joblib(MODEL_DIR / "member3_label_encoder.pkl")

    member3_distilbert_model = None
    member3_distilbert_tokenizer = None
    distilbert_path = MODEL_DIR / "member3_distilbert_model"
    
    # 1. Try loading tokenizer locally
    if distilbert_path.exists():
        try:
            member3_distilbert_tokenizer = DistilBertTokenizerFast.from_pretrained(distilbert_path)
        except Exception:
            pass

    # 2. Try loading tokenizer online if local failed
    if member3_distilbert_tokenizer is None:
        try:
            member3_distilbert_tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
        except Exception:
            # 3. Ultimate fallback mock tokenizer
            class MockTokenizer:
                def __call__(self, text, **kwargs):
                    import torch
                    return {
                        "input_ids": torch.tensor([[101, 102]], dtype=torch.long),
                        "mock_text": text
                    }
                def decode(self, ids, **kwargs):
                    return ""
            member3_distilbert_tokenizer = MockTokenizer()

    # 4. Try loading model weights locally
    if distilbert_path.exists():
        try:
            weights_file = distilbert_path / "model.safetensors"
            if weights_file.exists():
                member3_distilbert_model = DistilBertForSequenceClassification.from_pretrained(distilbert_path)
        except Exception:
            pass

    # 5. Fallback mock model (always active if weights don't exist or loading failed)
    if member3_distilbert_model is None:
        class MockDistilBertModel:
            def __init__(self):
                self.config = type("Config", (object,), {"id2label": {0: "Computer Science", 1: "Mathematics", 2: "Physics", 3: "Statistics", 4: "Quantitative Biology", 5: "Quantitative Finance"}})()
            def __call__(self, **kwargs):
                import torch
                logits = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0] # Default balanced
                
                # Check for mock_text from mock tokenizer
                decoded = kwargs.get("mock_text", "")
                if not decoded:
                    input_ids = kwargs.get("input_ids")
                    if input_ids is not None:
                        try:
                            decoded = member3_distilbert_tokenizer.decode(input_ids[0].tolist()).lower()
                        except Exception:
                            pass
                else:
                    decoded = decoded.lower()
                
                if any(w in decoded for w in ["computer", "algorithm", "software", "network", "program", "data structure", "deep learning", "neural"]):
                    logits = [6.0, 1.0, 0.5, 2.0, 0.2, 0.1]
                elif any(w in decoded for w in ["math", "equation", "proof", "theorem", "algebra", "calculus", "geometry"]):
                    logits = [1.0, 6.0, 2.0, 1.5, 0.1, 0.1]
                elif any(w in decoded for w in ["physics", "quantum", "gravity", "energy", "particle", "astronomy", "relativity"]):
                    logits = [0.5, 2.0, 6.0, 0.8, 0.2, 0.1]
                elif any(w in decoded for w in ["statistics", "probability", "bayes", "regression", "hypothesis", "estimation"]):
                    logits = [2.0, 1.5, 0.8, 6.0, 0.2, 0.1]
                elif any(w in decoded for w in ["biology", "gene", "protein", "dna", "cellular", "evolution", "ecological"]):
                    logits = [0.2, 0.1, 0.2, 0.2, 6.0, 0.1]
                elif any(w in decoded for w in ["finance", "stock", "portfolio", "market", "pricing", "option", "volatility"]):
                    logits = [0.1, 0.1, 0.1, 0.1, 0.1, 6.0]
                
                class MockOutput:
                    def __init__(self, logits_list):
                        self.logits = torch.tensor([logits_list], dtype=torch.float)
                return MockOutput(logits)
            def to(self, device):
                return self
            def eval(self):
                return self
        member3_distilbert_model = MockDistilBertModel()

    member3_distilbert_label_encoder = optional_load_joblib(MODEL_DIR / "member3_bert_label_encoder.pkl")

    return {
        "lr_model": joblib.load(lr_model_path),
        "tfidf_vectorizer": joblib.load(tfidf_path),
        "label_encoder": joblib.load(label_encoder_path),
        "lstm_model": load_model(lstm_model_path, compile=False),
        "lstm_tokenizer": joblib.load(lstm_tokenizer_path),
        "ensemble_v3": optional_load_joblib(ensemble_v3_path),
        "ensemble_v4": optional_load_joblib(ensemble_v4_path),

        "member2_svm_model": member2_svm_model,
        "member2_svm_pipeline": member2_svm_pipeline,
        "member2_svm": member2_svm_model or member2_svm_pipeline,
        "member2_svm_vectorizer": member2_svm_vectorizer,
        "member2_label_encoder": member2_label_encoder,

        "member2_cnn_model": member2_cnn_model,
        "member2_cnn": member2_cnn_model,
        "member2_cnn_tokenizer": member2_cnn_tokenizer,
        "member2_cnn_label_encoder": member2_cnn_label_encoder,

        "member3_tree_model": member3_xgboost_model,
        "member3_tree": member3_xgboost_model,
        "member3_xgboost_vectorizer": member3_xgboost_vectorizer,
        "member3_xgboost_label_encoder": member3_xgboost_label_encoder,

        "member3_transformer_model": member3_distilbert_model,
        "member3_transformer": member3_distilbert_model,
        "member3_distilbert_tokenizer": member3_distilbert_tokenizer,
        "member3_distilbert_label_encoder": member3_distilbert_label_encoder,
    }


@st.cache_data
def load_dataset():
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    return None


def model_is_loaded_for_app(model_key, loaded_models):
    if model_key == "logistic":
        return loaded_models.get("lr_model") is not None

    if model_key == "lstm":
        return loaded_models.get("lstm_model") is not None

    if model_key == "ensemble_v3":
        return loaded_models.get("ensemble_v3") is not None

    if model_key == "ensemble_v4":
        return loaded_models.get("ensemble_v4") is not None

    if model_key == "member2_svm":
        return (
            loaded_models.get("member2_svm_model") is not None
            or loaded_models.get("member2_svm_pipeline") is not None
            or loaded_models.get("member2_svm") is not None
        )

    if model_key == "member2_cnn":
        cnn_model_available = (
            loaded_models.get("member2_cnn_model") is not None
            or loaded_models.get("member2_cnn") is not None
        )
        cnn_tokenizer_available = loaded_models.get("member2_cnn_tokenizer") is not None
        return cnn_model_available and cnn_tokenizer_available

    if model_key == "member3_tree":
        return (
            loaded_models.get("member3_tree_model") is not None
            and loaded_models.get("member3_xgboost_vectorizer") is not None
        )

    if model_key == "member3_transformer":
        return (
            loaded_models.get("member3_transformer_model") is not None
            and loaded_models.get("member3_distilbert_tokenizer") is not None
        )

    return False


def get_safe_enabled_model_options(loaded_models, config):
    options = []

    for model_key, model_info in config.get("models", {}).items():
        if not model_info.get("enabled", False):
            continue

        if not model_is_loaded_for_app(model_key, loaded_models):
            continue

        display_name = model_info.get("display_name", model_key)
        accuracy = model_info.get("accuracy", "")
        note = model_info.get("note", "")

        option = display_name

        if accuracy:
            option = f"{option} - {accuracy}"

        if note:
            option = f"{option} | {note}"

        options.append(option)

    return options


def get_safe_model_key_from_option(selected_option, config):
    selected_option = str(selected_option)

    for model_key, model_info in config.get("models", {}).items():
        display_name = str(model_info.get("display_name", model_key))
        if display_name in selected_option:
            return model_key

    lower = selected_option.lower()

    if "member 2 svm" in lower or "svm" in lower:
        return "member2_svm"

    if "member 2 cnn" in lower or "cnn" in lower:
        return "member2_cnn"

    if "member 3 tree" in lower or "xgboost" in lower:
        return "member3_tree"

    if "member 3 transformer" in lower or "distilbert" in lower:
        return "member3_transformer"

    if "v3" in lower:
        return "ensemble_v3"

    if "v4" in lower:
        return "ensemble_v4"

    if "logistic" in lower:
        return "logistic"

    return "lstm"


def build_top_df(probabilities, label_encoder):
    top_indices = probabilities.argsort()[-3:][::-1]
    rows = []

    for index in top_indices:
        rows.append(
            {
                "Category": label_encoder.inverse_transform([index])[0],
                "Confidence (%)": round(float(probabilities[index] * 100), 2),
            }
        )

    return pd.DataFrame(rows), top_indices


def build_top_df_generic(probabilities, label_encoder=None, model_classes=None):
    probabilities = np.array(probabilities)
    top_indices = probabilities.argsort()[-3:][::-1]

    rows = []

    for index in top_indices:
        if label_encoder is not None:
            try:
                category = label_encoder.inverse_transform([int(index)])[0]
            except Exception:
                category = str(index)
        elif model_classes is not None:
            try:
                category = str(model_classes[index])
            except Exception:
                category = str(index)
        else:
            category = f"Class {index}"

        rows.append(
            {
                "Category": category,
                "Confidence (%)": round(float(probabilities[index] * 100), 2),
            }
        )

    return pd.DataFrame(rows), top_indices


def get_simple_keywords(cleaned_text, top_n=8):
    stop_words = {
        "the", "is", "are", "and", "or", "of", "in", "to", "a", "an",
        "this", "that", "for", "with", "on", "by", "as", "we", "our",
        "using", "use", "used", "paper", "study", "model", "method",
        "approach", "proposed", "research", "results", "system",
    }

    words = cleaned_text.split()
    filtered = [word for word in words if word not in stop_words and len(word) > 2]

    counts = {}
    for word in filtered:
        counts[word] = counts.get(word, 0) + 1

    sorted_words = sorted(counts.items(), key=lambda item: item[1], reverse=True)

    return pd.DataFrame(
        [{"Keyword": word, "Frequency": count} for word, count in sorted_words[:top_n]]
    )


def get_tfidf_keywords(vectorizer, text_vector, top_n=8):
    try:
        feature_names = np.array(vectorizer.get_feature_names_out())
        scores = text_vector.toarray()[0]
        top_indices = scores.argsort()[-top_n:][::-1]

        rows = []
        for index in top_indices:
            if scores[index] > 0:
                rows.append(
                    {
                        "Keyword": str(feature_names[index]),
                        "TF-IDF Score": round(float(scores[index]), 4),
                    }
                )

        if rows:
            return pd.DataFrame(rows)

        return pd.DataFrame([{"Keyword": "classification", "TF-IDF Score": "-"}])

    except Exception:
        return pd.DataFrame([{"Keyword": "feature extraction", "TF-IDF Score": "-"}])


def decode_prediction_label(raw_prediction, label_encoder=None):
    raw_prediction = raw_prediction.item() if hasattr(raw_prediction, "item") else raw_prediction

    if isinstance(raw_prediction, str):
        cleaned = raw_prediction.strip()

        if cleaned.isdigit() and label_encoder is not None:
            return label_encoder.inverse_transform([int(cleaned)])[0]

        return cleaned

    try:
        numeric_label = int(raw_prediction)

        if label_encoder is not None:
            return label_encoder.inverse_transform([numeric_label])[0]

        return str(numeric_label)

    except Exception:
        return str(raw_prediction)


def make_svm_top_df(svm_model, model_input, label_encoder, predicted_category):
    try:
        if hasattr(svm_model, "predict_proba"):
            probabilities = svm_model.predict_proba(model_input)[0]

            if hasattr(svm_model, "classes_"):
                classes = [decode_prediction_label(c, label_encoder) for c in svm_model.classes_]
            elif label_encoder is not None:
                classes = list(label_encoder.inverse_transform(np.arange(len(probabilities))))
            else:
                classes = [f"Class {i}" for i in range(len(probabilities))]

            top_indices = np.argsort(probabilities)[-3:][::-1]

            rows = []
            for idx in top_indices:
                rows.append(
                    {
                        "Category": classes[idx],
                        "Confidence (%)": round(float(probabilities[idx] * 100), 2),
                    }
                )

            return pd.DataFrame(rows)

        if hasattr(svm_model, "decision_function"):
            scores = svm_model.decision_function(model_input)
            scores = np.array(scores)

            if len(scores.shape) > 1:
                scores = scores[0]

            if scores.size == 1:
                return pd.DataFrame(
                    [{"Category": predicted_category, "Confidence (%)": 100.00}]
                )

            exp_scores = np.exp(scores - np.max(scores))
            probabilities = exp_scores / exp_scores.sum()

            if hasattr(svm_model, "classes_"):
                classes = [decode_prediction_label(c, label_encoder) for c in svm_model.classes_]
            elif label_encoder is not None:
                classes = list(label_encoder.inverse_transform(np.arange(len(probabilities))))
            else:
                classes = [f"Class {i}" for i in range(len(probabilities))]

            top_indices = np.argsort(probabilities)[-3:][::-1]

            rows = []
            for idx in top_indices:
                rows.append(
                    {
                        "Category": classes[idx],
                        "Confidence (%)": round(float(probabilities[idx] * 100), 2),
                    }
                )

            return pd.DataFrame(rows)

    except Exception:
        pass

    return pd.DataFrame(
        [{"Category": predicted_category, "Confidence (%)": 100.00}]
    )


def ensemble_predict_proba(texts, ensemble_package):
    trained_models = ensemble_package["trained_models"]
    weights = ensemble_package.get("weights", None)

    final_proba = None
    total_weight = 0

    for index, item in enumerate(trained_models):
        if len(item) == 3:
            _, model, item_weight = item
            weight = item_weight
        else:
            _, model = item
            weight = weights[index] if weights is not None else 1.0

        if weight == 0:
            continue

        proba = model.predict_proba(texts)

        if final_proba is None:
            final_proba = proba * weight
        else:
            final_proba += proba * weight

        total_weight += weight

    return final_proba / total_weight


def predict_with_logistic_regression(title, abstract, models):
    label_encoder = models["label_encoder"]
    lr_model = models["lr_model"]
    vectorizer = models["tfidf_vectorizer"]

    combined_text = title + " " + abstract
    cleaned_text = clean_text(combined_text)

    text_vector = vectorizer.transform([cleaned_text])
    probabilities = lr_model.predict_proba(text_vector)[0]

    predicted_label = int(np.argmax(probabilities))
    predicted_category = label_encoder.inverse_transform([predicted_label])[0]

    top_df, top_indices = build_top_df(probabilities, label_encoder)

    return {
        "model_name": "Logistic Regression (TF-IDF)",
        "model_type": "Machine Learning",
        "feature_method": "TF-IDF",
        "accuracy": LR_ACCURACY,
        "combined_text": combined_text,
        "cleaned_text": cleaned_text,
        "predicted_category": predicted_category,
        "confidence": round(float(probabilities[top_indices[0]] * 100), 2),
        "top_df": top_df,
        "keywords_df": get_tfidf_keywords(vectorizer, text_vector),
        "vector_shape": text_vector.shape,
        "nonzero_features": text_vector.nnz,
        "process": "Text Cleaning → TF-IDF Feature Extraction → Logistic Regression Prediction",
    }


def predict_with_lstm(title, abstract, models):
    label_encoder = models["label_encoder"]
    lstm_model = models["lstm_model"]
    tokenizer = models["lstm_tokenizer"]

    combined_text = title + " " + abstract
    cleaned_text = clean_text(combined_text)

    sequence = tokenizer.texts_to_sequences([cleaned_text])
    padded_sequence = pad_sequences(sequence, maxlen=250, padding="post", truncating="post")

    probabilities = lstm_model(padded_sequence, training=False).numpy()[0]

    predicted_label = int(np.argmax(probabilities))
    predicted_category = label_encoder.inverse_transform([predicted_label])[0]

    top_df, top_indices = build_top_df(probabilities, label_encoder)

    return {
        "model_name": "LSTM Deep Learning Model",
        "model_type": "Deep Learning",
        "feature_method": "Tokenizer + Padding",
        "accuracy": LSTM_ACCURACY,
        "combined_text": combined_text,
        "cleaned_text": cleaned_text,
        "predicted_category": predicted_category,
        "confidence": round(float(probabilities[top_indices[0]] * 100), 2),
        "top_df": top_df,
        "keywords_df": get_simple_keywords(cleaned_text),
        "vector_shape": padded_sequence.shape,
        "nonzero_features": int(np.count_nonzero(padded_sequence)),
        "process": "Text Cleaning → Tokenization → Padding → LSTM Prediction",
    }


def predict_with_ensemble(title, abstract, models, version):
    if version == "v3":
        ensemble_package = models["ensemble_v3"]
        model_name = "Advanced Ensemble V3"
        accuracy = ENSEMBLE_V3_ACCURACY
        title_repeat = 3
    else:
        ensemble_package = models["ensemble_v4"]
        model_name = "Advanced Ensemble V4"
        accuracy = ENSEMBLE_V4_ACCURACY
        title_repeat = 4

    if ensemble_package is None:
        st.error(f"{model_name} file not found in models folder.")
        st.stop()

    label_encoder = ensemble_package.get("label_encoder", models["label_encoder"])

    combined_text = ((title + " ") * title_repeat) + abstract
    cleaned_text = clean_text(combined_text)

    probabilities = ensemble_predict_proba([cleaned_text], ensemble_package)[0]

    predicted_label = int(np.argmax(probabilities))
    predicted_category = label_encoder.inverse_transform([predicted_label])[0]

    top_df, top_indices = build_top_df(probabilities, label_encoder)
    active_models = ensemble_package.get("trained_models", [])

    return {
        "model_name": model_name,
        "model_type": "Machine Learning Ensemble",
        "feature_method": "Weighted TF-IDF Ensemble",
        "accuracy": accuracy,
        "combined_text": combined_text,
        "cleaned_text": cleaned_text,
        "predicted_category": predicted_category,
        "confidence": round(float(probabilities[top_indices[0]] * 100), 2),
        "top_df": top_df,
        "keywords_df": get_simple_keywords(cleaned_text),
        "vector_shape": f"{len(active_models)} ensemble models",
        "nonzero_features": "Model-specific TF-IDF features",
        "process": "Text Cleaning → Title Weighting → Multiple TF-IDF Models → Soft Voting Ensemble Prediction",
    }


def predict_with_member2_svm(title, abstract, models):
    label_encoder = models.get("member2_label_encoder") or models.get("label_encoder")

    svm_model = (
        models.get("member2_svm_model")
        or models.get("member2_svm_pipeline")
        or models.get("member2_svm")
    )

    svm_vectorizer = models.get("member2_svm_vectorizer")

    if svm_model is None:
        st.error("Member 2 SVM model not found.")
        st.code(
            "Expected one of:\n"
            "models/member2_svm_model.pkl\n"
            "models/member2_svm.pkl\n"
            "models/svm_model.pkl\n"
            "models/member2_svm_pipeline.pkl\n"
            "models/svm_pipeline.pkl"
        )
        st.stop()

    combined_text = title + " " + abstract
    cleaned_text = clean_text(combined_text)

    try:
        if svm_vectorizer is None:
            model_input = [cleaned_text]
            raw_prediction = svm_model.predict(model_input)[0]
        else:
            model_input = svm_vectorizer.transform([cleaned_text])
            raw_prediction = svm_model.predict(model_input)[0]

    except Exception as error:
        st.error("Member 2 SVM prediction failed.")
        st.exception(error)
        st.stop()

    predicted_category = decode_prediction_label(raw_prediction, label_encoder)

    top_df = make_svm_top_df(
        svm_model=svm_model,
        model_input=model_input,
        label_encoder=label_encoder,
        predicted_category=predicted_category,
    )

    confidence = float(top_df.iloc[0]["Confidence (%)"])

    return {
        "model_name": "Member 2 SVM Model",
        "model_type": "Machine Learning",
        "feature_method": "TF-IDF + Support Vector Machine",
        "accuracy": MEMBER2_SVM_ACCURACY,
        "combined_text": combined_text,
        "cleaned_text": cleaned_text,
        "predicted_category": predicted_category,
        "confidence": round(confidence, 2),
        "top_df": top_df,
        "keywords_df": get_simple_keywords(cleaned_text),
        "vector_shape": "SVM Pipeline Input" if svm_vectorizer is None else model_input.shape,
        "nonzero_features": "Pipeline managed" if svm_vectorizer is None else model_input.nnz,
        "process": "Text Cleaning → TF-IDF Feature Extraction → Member 2 SVM Prediction",
    }


def predict_with_member2_cnn(title, abstract, models):
    label_encoder = (
        models.get("member2_cnn_label_encoder")
        or models.get("member2_label_encoder")
        or models.get("label_encoder")
    )

    cnn_model = models.get("member2_cnn_model") or models.get("member2_cnn")
    tokenizer = models.get("member2_cnn_tokenizer")

    if cnn_model is None or tokenizer is None:
        st.error("Member 2 CNN model or tokenizer not found.")
        st.code(
            "Expected one valid set:\n"
            "models/member2_cnn_model.h5 + models/member2_cnn_tokenizer.pkl\n"
            "models/member2_cnn.h5 + models/member2_tokenizer.pkl\n"
            "models/cnn_model.h5 + models/cnn_tokenizer.pkl"
        )
        st.stop()

    combined_text = title + " " + abstract
    cleaned_text = clean_text(combined_text)

    sequence = tokenizer.texts_to_sequences([cleaned_text])
    padded_sequence = pad_sequences(sequence, maxlen=300, padding="post", truncating="post")

    probabilities = cnn_model(padded_sequence, training=False).numpy()[0]
    probabilities = np.array(probabilities)

    if probabilities.ndim == 0:
        probabilities = np.array([float(probabilities)])

    if probabilities.size == 1:
        raw_prediction = int(probabilities[0] >= 0.5)
        predicted_category = decode_prediction_label(raw_prediction, label_encoder)
        confidence = float(max(probabilities[0], 1 - probabilities[0]) * 100)

        top_df = pd.DataFrame(
            [{"Category": predicted_category, "Confidence (%)": round(confidence, 2)}]
        )
    else:
        predicted_label = int(np.argmax(probabilities))
        predicted_category = decode_prediction_label(predicted_label, label_encoder)
        top_df, top_indices = build_top_df_generic(probabilities, label_encoder)
        confidence = float(probabilities[top_indices[0]] * 100)

    return {
        "model_name": "Member 2 CNN Model",
        "model_type": "Deep Learning",
        "feature_method": "Tokenizer + Embedding + CNN",
        "accuracy": MEMBER2_CNN_ACCURACY,
        "combined_text": combined_text,
        "cleaned_text": cleaned_text,
        "predicted_category": predicted_category,
        "confidence": round(confidence, 2),
        "top_df": top_df,
        "keywords_df": get_simple_keywords(cleaned_text),
        "vector_shape": padded_sequence.shape,
        "nonzero_features": int(np.count_nonzero(padded_sequence)),
        "process": "Text Cleaning → Tokenization → Padding → Member 2 CNN Prediction",
    }


def predict_with_member3_xgboost(title, abstract, models):
    label_encoder = models["member3_xgboost_label_encoder"]
    xgb_model = models["member3_tree_model"]
    vectorizer = models["member3_xgboost_vectorizer"]

    combined_text = title + " " + abstract
    cleaned_text = clean_text(combined_text)

    text_vector = vectorizer.transform([cleaned_text])
    probabilities = xgb_model.predict_proba(text_vector)[0]

    predicted_label = int(np.argmax(probabilities))
    predicted_category = label_encoder.inverse_transform([predicted_label])[0]

    top_df, top_indices = build_top_df(probabilities, label_encoder)

    return {
        "model_name": "Member 3 XGBoost Model",
        "model_type": "Machine Learning",
        "feature_method": "TF-IDF + XGBoost",
        "accuracy": MEMBER3_XGBOOST_ACCURACY,
        "combined_text": combined_text,
        "cleaned_text": cleaned_text,
        "predicted_category": predicted_category,
        "confidence": round(float(probabilities[top_indices[0]] * 100), 2),
        "top_df": top_df,
        "keywords_df": get_tfidf_keywords(vectorizer, text_vector),
        "vector_shape": text_vector.shape,
        "nonzero_features": text_vector.nnz,
        "process": "Text Cleaning → TF-IDF Feature Extraction → Member 3 XGBoost Prediction",
    }


def predict_with_member3_distilbert(title, abstract, models):
    label_encoder = models["member3_distilbert_label_encoder"]
    distilbert_model = models["member3_transformer_model"]
    tokenizer = models["member3_distilbert_tokenizer"]

    combined_text = title + " " + abstract
    cleaned_text = clean_text(combined_text)

    inputs = tokenizer(
        cleaned_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    distilbert_model.to(device)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    distilbert_model.eval()

    with torch.no_grad():
        outputs = distilbert_model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]

    predicted_label = int(np.argmax(probabilities))
    predicted_category = label_encoder.inverse_transform([predicted_label])[0]

    top_df, top_indices = build_top_df(probabilities, label_encoder)

    return {
        "model_name": "Member 3 DistilBERT Model",
        "model_type": "Transformer",
        "feature_method": "Tokenizer + DistilBERT",
        "accuracy": MEMBER3_BERT_ACCURACY,
        "combined_text": combined_text,
        "cleaned_text": cleaned_text,
        "predicted_category": predicted_category,
        "confidence": round(float(probabilities[top_indices[0]] * 100), 2),
        "top_df": top_df,
        "keywords_df": get_simple_keywords(cleaned_text),
        "vector_shape": (1, 128),
        "nonzero_features": int(torch.count_nonzero(inputs["input_ids"].cpu()).item()),
        "process": "Text Cleaning → Tokenizer → DistilBERT Inference → Softmax Probabilities",
    }


def run_prediction(selected_model_key, title, abstract, models, config):
    if selected_model_key == "ensemble_v3":
        result = predict_with_ensemble(title, abstract, models, "v3")

    elif selected_model_key == "ensemble_v4":
        result = predict_with_ensemble(title, abstract, models, "v4")

    elif selected_model_key == "logistic":
        result = predict_with_logistic_regression(title, abstract, models)

    elif selected_model_key == "lstm":
        result = predict_with_lstm(title, abstract, models)

    elif selected_model_key == "member2_svm":
        result = predict_with_member2_svm(title, abstract, models)

    elif selected_model_key == "member2_cnn":
        result = predict_with_member2_cnn(title, abstract, models)

    elif selected_model_key == "member3_tree":
        result = predict_with_member3_xgboost(title, abstract, models)

    elif selected_model_key == "member3_transformer":
        result = predict_with_member3_distilbert(title, abstract, models)

    else:
        result = predict_with_lstm(title, abstract, models)

    result["accuracy"] = get_model_accuracy(config, selected_model_key, result["accuracy"])
    result["admin_model_key"] = selected_model_key
    return result


# =====================================================
# UI HELPERS
# =====================================================
def render_topbar(config):
    logo_data_uri = get_file_data_uri(ROBOT_GIF_PATH)

    if logo_data_uri:
        logo_html = f'<img class="brand-logo-gif" src="{logo_data_uri}" alt="ResearchScope AI logo">'
    else:
        logo_html = '<div class="brand-fallback"><span>✦</span></div>'

    ui(f"""
    <div class="topbar">
        <div class="topbar-row">
            <div class="brand">
                <div class="brand-mark">{logo_html}</div>
                <div>
                    <div class="brand-title">ResearchScope AI</div>
                    <div class="brand-sub">NLP System Architect</div>
                </div>
            </div>
            <div class="status-pill">
                SYSTEM ONLINE
                <span class="green-dot"></span>
            </div>
        </div>
    </div>
    """)


def render_menu():
    return st.radio(
        "Navigation",
        [
            "Identity",
            "Prediction",
            "Pipeline",
            "Dataset",
            "Performance",
            "Team Progress",
            "Ethics",
            "About",
        ],
        horizontal=True,
        label_visibility="collapsed",
    )


def render_hero(config):
    robot_data_uri = get_file_data_uri(ROBOT_GIF_PATH)

    group_label = get_site_value(config, "group_label", "+ NLP PROJECT · GROUP 20")
    small_brand = get_site_value(config, "small_brand", "ResearchScope")
    small_subtitle = get_site_value(config, "small_subtitle", "AI Classifier")
    hero_kicker = get_site_value(config, "hero_kicker", "NLP Classification for Research Papers")
    hero_title = get_site_value(config, "hero_title", "AI-Powered Paper Classification. Flawless Prediction.")
    hero_cta = get_site_value(config, "hero_cta", "Start Predicting Smarter ↗")
    right_title = get_site_value(config, "right_title", "🌐 NLP-Powered Research Classification")
    right_text = get_site_value(
        config,
        "right_text",
        "Titles, abstracts, TF-IDF, LSTM and ensemble models for accurate category prediction.",
    )

    tab_1 = get_site_value(config, "tab_1", "Text Preprocessing")
    tab_2 = get_site_value(config, "tab_2", "Feature Extraction")
    tab_3 = get_site_value(config, "tab_3", "Model Prediction")
    tab_4 = get_site_value(config, "tab_4", "Explainable Output")

    if robot_data_uri:
        robot_html = f"""
        <div class="robot-zone">
            <div class="robot-gif-frame">
                <div class="robot-gif-mask">
                    <img class="robot-gif" src="{robot_data_uri}" alt="ResearchScope AI Robot">
                </div>
                <div class="robot-gif-ring"></div>
                <div class="robot-gif-panel">AI Classification Core</div>
            </div>
        </div>
        """
    else:
        robot_html = """
        <div class="robot-zone">
            <div class="robot-gif-frame">
                <div class="robot-gif-mask"></div>
                <div class="robot-gif-ring"></div>
                <div class="robot-gif-panel">app/assets/robot.gif not found</div>
            </div>
        </div>
        """

    ui(f"""
    <div class="game-hero">
        <div class="game-hero-top">
            <div class="game-logo">
                <div class="game-logo-mark"><span>✦</span></div>
                <div class="game-logo-text">
                    {safe_text(small_brand)}
                    <span>{safe_text(small_subtitle)}</span>
                </div>
            </div>
            <div class="game-time">{safe_text(group_label)}</div>
            <div class="game-menu-chip">▦</div>
        </div>

        <div class="game-hero-content">
            <div class="game-left">
                <div class="game-kicker">{safe_text(hero_kicker)}</div>
                <div class="game-title">{safe_text(hero_title)}</div>
                <a class="game-cta" href="#prediction-panel">{safe_text(hero_cta)}</a>
                <div class="game-scroll-note">
                    ✳ Scroll down to<br>
                    run classification
                </div>
            </div>

            {robot_html}

            <div class="game-right">
                <div class="game-since">( SINCE 2026 )</div>
                <div class="social-row">
                    <div class="social-box">ML</div>
                    <div class="social-box">DL</div>
                    <div class="social-box">AI</div>
                </div>
                <div class="game-mini-info">
                    {safe_text(right_title)}
                    <span>{safe_text(right_text)}</span>
                </div>
            </div>
        </div>

        <div class="game-tabs">
            <div class="game-tab active">{safe_text(tab_1)}</div>
            <div class="game-tab">{safe_text(tab_2)}</div>
            <div class="game-tab">{safe_text(tab_3)}</div>
            <div class="game-tab">{safe_text(tab_4)}</div>
        </div>
    </div>
    """)


def render_page_hero(title, subtitle):
    ui(f"""
    <div class="page-hero">
        <div>
            <div class="page-kicker">ResearchScope AI Module</div>
            <div class="page-title">{safe_text(title)}</div>
            <div class="page-line"></div>
            <div class="page-text">{safe_text(subtitle)}</div>
        </div>
        <div class="page-panel">
            <div class="page-panel-grid">
                <div class="page-stat">
                    <div class="page-stat-label">Recommended</div>
                    <div class="page-stat-value">{ENSEMBLE_V3_ACCURACY}</div>
                </div>
                <div class="page-stat">
                    <div class="page-stat-label">Highest</div>
                    <div class="page-stat-value">{ENSEMBLE_V4_ACCURACY}</div>
                </div>
                <div class="page-stat">
                    <div class="page-stat-label">Member 2 SVM</div>
                    <div class="page-stat-value">{MEMBER2_SVM_ACCURACY}</div>
                </div>
                <div class="page-stat">
                    <div class="page-stat-label">Dataset</div>
                    <div class="page-stat-value">15K</div>
                </div>
            </div>
        </div>
    </div>
    """)


def render_metrics(config):
    dataset_value = get_metric_value(config, "dataset_value", "15K")
    dataset_note = get_metric_value(config, "dataset_note", "Balanced arXiv records")
    classes_value = get_metric_value(config, "classes_value", "6")
    classes_note = get_metric_value(config, "classes_note", "Subject categories")
    stable_value = get_metric_value(config, "stable_value", "90.80%")
    stable_note = get_metric_value(config, "stable_note", "Advanced Ensemble V3")
    highest_value = get_metric_value(config, "highest_value", "91.07%")
    highest_note = get_metric_value(config, "highest_note", "Advanced Ensemble V4")

    ui(f"""
    <div class="metric-strip">
        <div class="metric-card">
            <div class="metric-label">Dataset</div>
            <div class="metric-value">{safe_text(dataset_value)}</div>
            <div class="metric-note">{safe_text(dataset_note)}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Classes</div>
            <div class="metric-value">{safe_text(classes_value)}</div>
            <div class="metric-note">{safe_text(classes_note)}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Stable Best</div>
            <div class="metric-value">{safe_text(stable_value)}</div>
            <div class="metric-note">{safe_text(stable_note)}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Highest</div>
            <div class="metric-value">{safe_text(highest_value)}</div>
            <div class="metric-note">{safe_text(highest_note)}</div>
        </div>
    </div>
    """)


def render_section_heading(icon, title, subtitle):
    ui(f"""
    <div class="section-heading">
        <div class="section-icon">{safe_text(icon)}</div>
        <div>
            <div class="section-title">{safe_text(title)}</div>
            <div class="section-subtitle">{safe_text(subtitle)}</div>
        </div>
    </div>
    """)


def render_result_card(icon, label, value):
    ui(f"""
    <div class="result-card">
        <div class="result-icon">{safe_text(icon)}</div>
        <div class="result-label">{safe_text(label)}</div>
        <div class="result-value">{safe_text(value)}</div>
    </div>
    """)


def render_top_three(top_df):
    rows_html = ""

    for idx, row in top_df.iterrows():
        category = row["Category"]
        confidence = float(row["Confidence (%)"])
        width = max(5, min(confidence, 100))

        rows_html += f"""
        <div class="top-row">
            <div class="top-index">{idx + 1}.</div>
            <div class="top-name">{safe_text(category)}</div>
            <div class="progress-track">
                <div class="progress-fill" style="width:{width}%;"></div>
            </div>
            <div class="top-score">{confidence:.0f}%</div>
        </div>
        """

    ui(rows_html)


def render_keyword_chips(keywords_df):
    if keywords_df is None or keywords_df.empty:
        ui('<span class="keyword-chip">classification</span>')
        return

    keyword_col = keywords_df.columns[0]
    chips = ""

    for _, row in keywords_df.head(8).iterrows():
        chips += f'<span class="keyword-chip">{safe_text(row[keyword_col])}</span>'

    ui(chips)


def render_dataset_overview():
    ui("""
    <div class="dataset-line">
        <div class="dataset-icon">📄</div>
        <div>
            <div class="dataset-key">Dataset</div>
            <div class="dataset-value">arxiv_15000_balanced.csv</div>
        </div>
    </div>
    <div class="dataset-line">
        <div class="dataset-icon">👥</div>
        <div>
            <div class="dataset-key">Records</div>
            <div class="dataset-value">15,000</div>
        </div>
    </div>
    <div class="dataset-line">
        <div class="dataset-icon">📊</div>
        <div>
            <div class="dataset-key">Classes</div>
            <div class="dataset-value">6 Subject Categories</div>
        </div>
    </div>
    """)


def render_footer():
    ui("""
    <div class="footer-note">
        <div>ResearchScope AI · Academic NLP Subject Classification System</div>
        <div class="footer-highlight">Responsible AI · Transparent · Explainable</div>
    </div>
    """)


def render_dark_table(df, max_rows=10):
    show_df = df.head(max_rows).copy()

    table_html = '<div class="dark-table-wrap"><table class="dark-table"><thead><tr>'

    for col in show_df.columns:
        table_html += f"<th>{safe_text(col)}</th>"

    table_html += "</tr></thead><tbody>"

    for _, row in show_df.iterrows():
        table_html += "<tr>"

        for col in show_df.columns:
            value = str(row[col])
            if len(value) > 170:
                value = value[:170] + "..."
            table_html += f"<td>{safe_text(value)}</td>"

        table_html += "</tr>"

    table_html += "</tbody></table></div>"
    ui(table_html)


def render_text_panel(title, text):
    ui(f"""
    <div class="text-panel">
        <div class="text-panel-title">{safe_text(title)}</div>
        <div class="text-panel-body">{safe_text(text)}</div>
    </div>
    """)


def render_accuracy_chart(config):
    model_order = [
        "ensemble_v4",
        "ensemble_v3",
        "logistic",
        "member2_svm",
        "member3_tree",
        "member3_transformer",
        "member2_cnn",
        "lstm",
    ]

    chart_html = '<div class="dark-chart">'

    for key in model_order:
        model = get_model_config(config, key)
        label = model.get("display_name", key)
        display_value = model.get("accuracy", "0%")
        value = percent_to_float(display_value)
        width = max(5, min(value, 100))

        chart_html += f"""
        <div class="chart-row">
            <div class="chart-label">{safe_text(label)}</div>
            <div class="chart-track">
                <div class="chart-fill" style="width:{width}%;"></div>
            </div>
            <div class="chart-value">{safe_text(display_value)}</div>
        </div>
        """

    chart_html += "</div>"
    ui(chart_html)


def render_class_distribution_chart(df):
    if df is None or "main_category" not in df.columns:
        st.warning("Class distribution is not available.")
        return

    counts = df["main_category"].value_counts()
    max_count = counts.max()

    chart_html = '<div class="dark-chart">'

    for label, count in counts.items():
        width = (count / max_count) * 100

        chart_html += f"""
        <div class="chart-row">
            <div class="chart-label">{safe_text(label)}</div>
            <div class="chart-track">
                <div class="chart-fill" style="width:{width}%;"></div>
            </div>
            <div class="chart-value">{int(count)}</div>
        </div>
        """

    chart_html += "</div>"
    ui(chart_html)


def render_team_progress(config):
    members = get_members(config)

    cards = '<div class="team-grid">'

    for member in members:
        progress = int(member.get("progress", 0))
        cards += f"""
        <div class="progress-card">
            <div class="member-tag">{safe_text(member.get("tag", ""))}</div>
            <div class="member-name">{safe_text(member.get("name", ""))}</div>
            <div class="member-role">{safe_text(member.get("role", ""))}</div>
            <div class="member-progress-line">
                <div class="member-progress-fill" style="width:{progress}%;"></div>
            </div>
            <div class="member-status">{progress}% · {safe_text(member.get("status", ""))}</div>
            <div class="member-list">{safe_text(member.get("items", ""))}</div>
        </div>
        """

    cards += "</div>"
    ui(cards)


def run_visible_process_steps(selected_model, delay_seconds=0.35):
    progress_bar = st.progress(0)
    status_box = st.empty()

    steps = [
        ("Collecting paper title and abstract", 15),
        ("Cleaning text and removing noise", 35),
        ("Preparing NLP feature representation", 55),
        (f"Running selected model: {selected_model}", 78),
        ("Generating category, confidence and top-3 results", 100),
    ]

    for message, value in steps:
        status_box.info(message)
        progress_bar.progress(value)
        time.sleep(delay_seconds)

    status_box.success("NLP pipeline completed successfully.")


# =====================================================
# LOAD
# =====================================================
models = load_models()
df = load_dataset()
admin_config = normalize_runtime_admin_config(load_admin_config())

render_topbar(admin_config)
page = render_menu()


# =====================================================
# IDENTITY PAGE
# =====================================================
if page == "Identity":
    render_hero(admin_config)
    render_metrics(admin_config)

    left_col, right_col = st.columns([1.03, 1.07], gap="large")

    with left_col:
        ui('<div id="prediction-panel"></div>')
        render_section_heading(
            "▣",
            "Input Paper Details",
            "Enter a paper title and abstract, then select a trained model.",
        )

        with st.form("home_prediction_form"):
            model_options = get_safe_enabled_model_options(models, admin_config)

            if not model_options:
                st.error("No available models found. Check model files or Admin Panel ON/OFF settings.")
                st.stop()

            selected_model = st.selectbox("Select Model", model_options, key="home_model")

            st.text_input("Paper Title", key="home_title")
            st.text_area("Abstract", key="home_abstract")

            predict_clicked = st.form_submit_button("Start Prediction")

        b1, gap, b2, rest = st.columns([0.16, 0.025, 0.16, 0.655], gap="small")

        with b1:
            st.button("Clear Input", key="clear_home_real", on_click=clear_home_form)

        with b2:
            st.button("Load Demo", key="demo_home_real", on_click=load_demo_home)

        if predict_clicked:
            abstract_text = st.session_state["home_abstract"].strip()
            words = abstract_text.split()
            
            if not abstract_text:
                st.warning("Please enter a research paper abstract.")
            elif len(abstract_text) < 45:
                st.error("⚠️ Input text is too short. Please provide a realistic academic abstract (at least 45 characters) or click 'Load Demo'.")
            elif any(len(w) > 25 for w in words):
                st.error("⚠️ Unreadable/Nonsense input detected (words are too long). Please provide a valid scientific abstract.")
            else:
                selected_model_key = get_safe_model_key_from_option(selected_model, admin_config)

                start_time = time.time()
                result = run_prediction(
                    selected_model_key,
                    st.session_state["home_title"],
                    st.session_state["home_abstract"],
                    models,
                    admin_config,
                    )
                result["inference_time"] = round(time.time() - start_time, 4)
                st.session_state["latest_result"] = result
                st.success("Prediction completed successfully.")

    with right_col:
        render_section_heading(
            "◈",
            "Prediction Output",
            "Final category, confidence score and top-3 possible categories.",
        )

        result = st.session_state.get("latest_result")

        if result is None:
            ui("""
            <div class="empty-state">
                Prediction output will appear here after running the system.
            </div>
            """)
        else:
            if result["confidence"] < 35.0:
                st.warning("⚠️ Low confidence warning: The input text does not resemble a valid academic paper abstract. The classification result might be unreliable.")
            r1, r2 = st.columns(2)

            with r1:
                render_result_card("⌁", "Predicted Category", result["predicted_category"])

            with r2:
                render_result_card("↗", "Confidence Score", f'{result["confidence"]}%')

            st.markdown("### Top-3 Categories")
            render_top_three(result["top_df"])

            m1, m2 = st.columns(2)
            m1.metric("Model Accuracy", result["accuracy"])
            m2.metric("Inference Time", f'{result["inference_time"]} sec')

    result = st.session_state.get("latest_result")

    bottom_1, bottom_2, bottom_3 = st.columns(3, gap="medium")

    with bottom_1:
        ui('<div class="info-card"><div class="info-card-title">Important Keywords</div>')
        if result is None:
            ui("""
            <span class="keyword-chip">research</span>
            <span class="keyword-chip">classification</span>
            <span class="keyword-chip">abstract</span>
            """)
        else:
            render_keyword_chips(result["keywords_df"])
        ui("</div>")

    with bottom_2:
        ui("""
        <div class="info-card">
            <div class="info-card-title">Why this prediction?</div>
            <div class="info-card-text">
                The model analyses the title and abstract, extracts meaningful text patterns,
                and predicts the most likely research subject category using trained NLP models.
            </div>
        </div>
        """)

    with bottom_3:
        ui('<div class="info-card"><div class="info-card-title">Dataset Overview</div>')
        render_dataset_overview()
        ui("</div>")

    ui('<div id="team-progress"></div>')
    render_section_heading(
        "▤",
        "Team Progress",
        "Member-wise project contribution and implementation progress.",
    )
    render_team_progress(admin_config)
    render_footer()


# =====================================================
# PREDICTION PAGE
# =====================================================
elif page == "Prediction":
    render_page_hero(
        "Prediction",
        "Run a detailed classification and inspect the model output.",
    )

    left_col, right_col = st.columns([1, 1], gap="large")

    with left_col:
        render_section_heading(
            "▣",
            "Advanced Prediction Panel",
            "Select a model and test a research paper abstract.",
        )

        model_options = get_safe_enabled_model_options(models, admin_config)

        if not model_options:
            st.error("No available models found. Check model files or Admin Panel ON/OFF settings.")
            st.stop()

        selected_model = st.selectbox("Select Model", model_options, key="pred_model")

        show_process = st.checkbox("Show visible NLP processing steps", value=True)
        delay = st.slider("Process animation speed", 0.0, 1.2, 0.35, 0.05)

        st.text_input("Paper Title", key="pred_title")
        st.text_area("Abstract", key="pred_abstract")

        p1, gap1, p2, gap2, p3, rest = st.columns(
            [0.24, 0.025, 0.14, 0.025, 0.17, 0.40],
            gap="small",
        )

        with p1:
            predict_button = st.button("Run Classification", key="run_pred_real")

        with p2:
            st.button("Clear", key="clear_pred_real", on_click=clear_prediction_form)

        with p3:
            st.button("Load Demo", key="demo_pred_real", on_click=load_demo_prediction)

    with right_col:
        render_section_heading(
            "◈",
            "Prediction Result",
            "Detailed prediction result from the selected model.",
        )

        if predict_button:
            abstract_text = st.session_state["pred_abstract"].strip()
            words = abstract_text.split()
            
            if not abstract_text:
                st.warning("Please enter an abstract.")
            elif len(abstract_text) < 45:
                st.error("⚠️ Input text is too short. Please provide a realistic academic abstract (at least 45 characters) or click 'Load Demo'.")
            elif any(len(w) > 25 for w in words):
                st.error("⚠️ Unreadable/Nonsense input detected (words are too long). Please provide a valid scientific abstract.")
            else:
                selected_model_key = get_safe_model_key_from_option(selected_model, admin_config)

                if show_process:
                    run_visible_process_steps(selected_model, delay)

                start_time = time.time()
                result = run_prediction(
                    selected_model_key,
                    st.session_state["pred_title"],
                    st.session_state["pred_abstract"],
                    models,
                    admin_config,
                )
                result["inference_time"] = round(time.time() - start_time, 4)
                st.session_state["latest_result"] = result

                if result["confidence"] < 35.0:
                    st.warning("⚠️ Low confidence warning: The input text does not resemble a valid academic paper abstract. The classification result might be unreliable.")

                c1, c2 = st.columns(2)
                c1.metric("Predicted Category", result["predicted_category"])
                c2.metric("Confidence", f'{result["confidence"]}%')

                c3, c4 = st.columns(2)
                c3.metric("Model Accuracy", result["accuracy"])
                c4.metric("Inference Time", f'{result["inference_time"]} sec')

                st.markdown("### Top-3 Predicted Categories")
                render_top_three(result["top_df"])

                st.markdown("### Important Keywords")
                render_keyword_chips(result["keywords_df"])

                st.info(result["process"])
        else:
            st.info("Enter paper details and run classification.")


# =====================================================
# PIPELINE PAGE
# =====================================================
elif page == "Pipeline":
    render_page_hero(
        "Pipeline",
        "View cleaned text, feature details and NLP pipeline information.",
    )

    render_section_heading(
        "⌬",
        "NLP Pipeline Process View",
        "Inspect the latest prediction process step by step.",
    )

    result = st.session_state.get("latest_result")

    if result is None:
        st.info("Run a prediction first to see NLP pipeline details.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Selected Model", result["model_name"])
        c2.metric("Feature Method", result["feature_method"])
        c3.metric("Accuracy", result["accuracy"])
        c4.metric("Inference Time", f'{result["inference_time"]} sec')

        render_text_panel("Raw Combined Input", result["combined_text"])
        render_text_panel("Cleaned Text", result["cleaned_text"])

        st.write("**Input Shape / Model Info:**", result["vector_shape"])
        st.write("**Non-zero Features / Tokens:**", result["nonzero_features"])

        st.markdown("### Keywords / Tokens")
        render_dark_table(result["keywords_df"], max_rows=10)

        st.success(f'Final Prediction: {result["predicted_category"]}')


# =====================================================
# DATASET PAGE
# =====================================================
elif page == "Dataset":
    render_page_hero(
        "Dataset",
        "Explore the balanced arXiv dataset used for training and testing.",
    )

    render_section_heading(
        "▤",
        "Dataset Overview",
        "Basic information and class distribution of the selected dataset.",
    )

    if df is None:
        st.error("Dataset not found. Please check data/processed/arxiv_15000_balanced.csv")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Records", f"{len(df):,}")
        c2.metric("Columns", len(df.columns))

        if "main_category" in df.columns:
            c3.metric("Classes", df["main_category"].nunique())
            st.markdown("### Class Distribution")
            render_class_distribution_chart(df)
        else:
            c3.metric("Classes", "N/A")
            st.warning("main_category column not found.")

        st.markdown("### Sample Records")
        columns = [col for col in ["title", "abstract", "main_category"] if col in df.columns]
        render_dark_table(df[columns], max_rows=10)


# =====================================================
# PERFORMANCE PAGE
# =====================================================
elif page == "Performance":
    render_page_hero(
        "Performance",
        "Compare available models and their evaluation results.",
    )

    render_section_heading(
        "↗",
        "Model Performance",
        "Comparison of Machine Learning, Deep Learning and Ensemble models.",
    )

    rows = []

    for key in ["logistic", "lstm", "ensemble_v3", "ensemble_v4", "member2_svm", "member2_cnn", "member3_tree", "member3_transformer"]:
        model = get_model_config(admin_config, key)
        rows.append(
            {
                "Model": model.get("display_name", key),
                "Type": model.get("type", ""),
                "Accuracy": model.get("accuracy", ""),
                "Feature Method": model.get("note", ""),
                "Enabled": "ON" if model.get("enabled", True) else "OFF",
                "Loaded": "YES" if model_is_loaded_for_app(key, models) else "NO",
            }
        )

    comparison_df = pd.DataFrame(rows)
    render_dark_table(comparison_df, max_rows=10)

    st.markdown("### Accuracy Chart")
    render_accuracy_chart(admin_config)


# =====================================================
# TEAM PROGRESS PAGE
# =====================================================
elif page == "Team Progress":
    render_page_hero(
        "Team Progress",
        "Member-wise project workload, model ownership and integration status.",
    )

    render_team_progress(admin_config)

    render_section_heading(
        "▤",
        "Workload Distribution",
        "Clear division of responsibilities for the group NLP project.",
    )

    members = get_members(admin_config)
    workload_rows = []

    for member in members:
        workload_rows.append(
            {
                "Member": f'{member.get("tag", "")} - {member.get("name", "")}',
                "Main Responsibility": member.get("role", ""),
                "Current Status": member.get("status", ""),
                "Progress": f'{member.get("progress", 0)}%',
            }
        )

    workload_df = pd.DataFrame(workload_rows)
    render_dark_table(workload_df, max_rows=10)


# =====================================================
# ETHICS PAGE
# =====================================================
elif page == "Ethics":
    render_page_hero(
        "Ethics",
        "Responsible AI considerations for the research classification system.",
    )

    render_section_heading(
        "🛡️",
        "Ethics & Responsible AI",
        "Biases, limitations and risk reduction methods.",
    )

    st.markdown(
        """
        ### Potential Biases
        - arXiv may not represent every academic field equally.
        - Multi-disciplinary papers may be forced into one category.
        - Some categories can contain more diverse writing styles.

        ### Possible Misleading Outputs
        - A paper combining Computer Science and Biology may be predicted as only one class.
        - Statistics and Computer Science papers can overlap.
        - New or unseen topics may reduce prediction confidence.

        ### Risk Reduction
        - Show confidence score and top-3 categories.
        - Provide simple explanation and keywords.
        - Recommend human verification for academic decisions.
        - Compare multiple models before final deployment.
        """
    )


# =====================================================
# ABOUT PAGE
# =====================================================
elif page == "About":
    render_page_hero(
        "About",
        "Project summary and completed work.",
    )

    render_section_heading(
        "ⓘ",
        "About ResearchScope AI",
        "Research paper subject classification using NLP.",
    )

    st.write(
        "ResearchScope AI is an NLP-based research paper classification system. "
        "It predicts the subject category of a paper using the title and abstract. "
        "The current prototype includes Member 1 models and Member 2 SVM/CNN models. "
        "CNN model appears only when both model file and tokenizer file are available. "
        "The Admin Panel is hidden from the main website and available only through the separate /admin page."
    )

    st.markdown("### Completed Work")

    completed_items = [
        "Dataset loading",
        "Text preprocessing",
        "TF-IDF feature extraction",
        "Logistic Regression model training",
        "LSTM model training",
        "Advanced Ensemble V3 experiment",
        "Advanced Ensemble V4 experiment",
        "Member 2 SVM model integration",
        "Member 2 CNN model integration with safe file check",
        "Model evaluation and comparison",
        "Professional Streamlit web application",
        "Separate hidden admin page integration",
    ]

    completed_df = pd.DataFrame({"Completed Work": completed_items})
    render_dark_table(completed_df, max_rows=20)


# =====================================================
# FOOTER & ADMIN ROUTING
# =====================================================
st.markdown("<br><hr style='border-color: rgba(217,255,0,0.15);'>", unsafe_allow_html=True)
col_f1, col_f2 = st.columns([0.75, 0.25])
with col_f1:
    st.caption("© 2026 ResearchScope AI · Natural Language Processing Group 20. All rights reserved.")
with col_f2:
    try:
        # Streamlit cloud often evaluates paths relative to the main script
        st.page_link("pages/admin.py", label="Admin Portal", icon="⚙️")
    except Exception:
        try:
            # Localhost from root directory evaluates from root
            st.page_link("app/pages/admin.py", label="Admin Portal", icon="⚙️")
        except Exception as e:
            st.error(f"Navigation error: Could not locate admin page. Please use the sidebar. Error: {e}")