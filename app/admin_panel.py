import copy
import json
from pathlib import Path

import streamlit as st


CONFIG_PATH = Path(__file__).resolve().parent / "admin_config.json"


DEFAULT_CONFIG = {
    "users": {
        "member1": {
            "password": "member1",
            "role": "member1",
            "name": "Oshadha",
            "access": "full",
        },
        "member2": {
            "password": "member2",
            "role": "member2",
            "name": "Thiranji",
            "access": "models_only",
        },
        "member3": {
            "password": "member3",
            "role": "member3",
            "name": "Ravindu",
            "access": "models_only",
        },
    },
    "site": {
        "group_label": "+ NLP PROJECT · GROUP 20",
        "small_brand": "ResearchScope",
        "small_subtitle": "AI Classifier",
        "hero_kicker": "NLP Classification for Research Papers",
        "hero_title": "AI-Powered Paper Classification. Flawless Prediction.",
        "hero_cta": "Start Predicting Smarter ↗",
        "right_title": "🌐 NLP-Powered Research Classification",
        "right_text": "Titles, abstracts, TF-IDF, LSTM and ensemble models for accurate category prediction.",
        "tab_1": "Text Preprocessing",
        "tab_2": "Feature Extraction",
        "tab_3": "Model Prediction",
        "tab_4": "Explainable Output",
    },
    "metrics": {
        "dataset_value": "15K",
        "dataset_note": "Balanced arXiv records",
        "classes_value": "6",
        "classes_note": "Subject categories",
        "stable_value": "90.80%",
        "stable_note": "Advanced Ensemble V3",
        "highest_value": "91.07%",
        "highest_note": "Advanced Ensemble V4",
    },
    "models": {
        "logistic": {
            "owner": "member1",
            "enabled": True,
            "display_name": "Logistic Regression",
            "type": "Machine Learning",
            "accuracy": "89.33%",
            "note": "TF-IDF baseline model",
        },
        "lstm": {
            "owner": "member1",
            "enabled": True,
            "display_name": "LSTM Deep Learning Model",
            "type": "Deep Learning",
            "accuracy": "85.17%",
            "note": "Tokenizer + Padding + LSTM",
        },
        "ensemble_v3": {
            "owner": "member1",
            "enabled": True,
            "display_name": "Advanced Ensemble V3",
            "type": "ML Ensemble",
            "accuracy": "90.80%",
            "note": "Recommended stable ensemble",
        },
        "ensemble_v4": {
            "owner": "member1",
            "enabled": True,
            "display_name": "Advanced Ensemble V4",
            "type": "ML Ensemble",
            "accuracy": "91.07%",
            "note": "Highest experiment",
        },
        "member2_svm": {
            "owner": "member2",
            "enabled": True,
            "display_name": "Member 2 SVM Model",
            "type": "Machine Learning",
            "accuracy": "88.67%",
            "note": "TF-IDF + Support Vector Machine classifier",
        },
        "member2_cnn": {
            "owner": "member2",
            "enabled": True,
            "display_name": "Member 2 CNN Model",
            "type": "Deep Learning",
            "accuracy": "86.40%",
            "note": "Tokenizer + Embedding + CNN text classifier",
        },
        "member3_tree": {
            "owner": "member3",
            "enabled": True,
            "display_name": "Member 3 XGBoost Model",
            "type": "Machine Learning",
            "accuracy": "86.90%",
            "note": "TF-IDF + XGBoost Classifier",
        },
        "member3_transformer": {
            "owner": "member3",
            "enabled": True,
            "display_name": "Member 3 DistilBERT Model",
            "type": "Transformer",
            "accuracy": "86.67%",
            "note": "DistilBERT Sequence Classifier",
        },
    },
    "members": [
        {
            "tag": "Member 1",
            "role_key": "member1",
            "name": "Oshadha",
            "role": "Preprocessing, Logistic Regression, LSTM, Advanced Ensemble and Streamlit Web App",
            "progress": 100,
            "status": "Completed",
            "items": "TF-IDF · Logistic Regression · LSTM · Ensemble V3/V4 · Web UI · Admin Panel",
        },
        {
            "tag": "Member 2",
            "role_key": "member2",
            "name": "Thiranji",
            "role": "SVM and CNN model development, model evaluation and branch integration",
            "progress": 100,
            "status": "Completed",
            "items": "SVM · CNN · Accuracy Evaluation · Model Export · Git Branch Integration",
        },
        {
            "tag": "Member 3",
            "role_key": "member3",
            "name": "Ravindu",
            "role": "Tree-based ML / Transformer model branch and final comparison support",
            "progress": 100,
            "status": "Completed",
            "items": "Random Forest/XGBoost · BERT/DistilBERT · Evaluation",
        },
    ],
}


def deep_merge(default_data, saved_data):
    result = copy.deepcopy(default_data)

    if not isinstance(saved_data, dict):
        return result

    for key, value in saved_data.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value

    return result


def normalize_config(config):
    default_models = DEFAULT_CONFIG["models"]

    for model_key in ["member2_svm", "member2_cnn", "member3_tree", "member3_transformer"]:
        model = config.setdefault("models", {}).setdefault(
            model_key,
            copy.deepcopy(default_models[model_key]),
        )

        old_accuracy = str(model.get("accuracy", "")).strip().lower()

        if old_accuracy in ["", "pending", "none", "not set", "n/a"]:
            model["accuracy"] = default_models[model_key]["accuracy"]

        if not model.get("note") or "future" in str(model.get("note")).lower():
            model["note"] = default_models[model_key]["note"]

        if not model.get("type"):
            model["type"] = default_models[model_key]["type"]

        if not model.get("display_name"):
            model["display_name"] = default_models[model_key]["display_name"]

        model["enabled"] = bool(model.get("enabled", True))
        model["owner"] = default_models[model_key]["owner"]

    members = config.setdefault("members", copy.deepcopy(DEFAULT_CONFIG["members"]))

    for index, member in enumerate(members):
        role_key = member.get("role_key", "")

        if role_key == "member2" or member.get("tag") == "Member 2":
            default_member2 = DEFAULT_CONFIG["members"][1]

            if int(member.get("progress", 0)) < 100:
                members[index]["progress"] = default_member2["progress"]

            if str(member.get("status", "")).lower().startswith("pending"):
                members[index]["status"] = default_member2["status"]

            if "future" in str(member.get("items", "")).lower() or "git branch" in str(member.get("items", "")).lower():
                members[index]["items"] = default_member2["items"]

            members[index]["role"] = default_member2["role"]

        elif role_key == "member3" or member.get("tag") == "Member 3":
            default_member3 = DEFAULT_CONFIG["members"][2]

            if int(member.get("progress", 0)) < 100:
                members[index]["progress"] = default_member3["progress"]

            if str(member.get("status", "")).lower().startswith("pending"):
                members[index]["status"] = default_member3["status"]

            members[index]["role"] = default_member3["role"]
            members[index]["items"] = default_member3["items"]

    return config


def load_admin_config():
    if not CONFIG_PATH.exists():
        config = copy.deepcopy(DEFAULT_CONFIG)
        save_admin_config(config)
        return config

    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as file:
            saved_config = json.load(file)

        config = deep_merge(DEFAULT_CONFIG, saved_config)
        config = normalize_config(config)
        save_admin_config(config)
        return config

    except Exception:
        config = copy.deepcopy(DEFAULT_CONFIG)
        save_admin_config(config)
        return config


def save_admin_config(config):
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(CONFIG_PATH, "w", encoding="utf-8") as file:
        json.dump(config, file, indent=4, ensure_ascii=False)


def get_site_value(config, key, fallback=""):
    return config.get("site", {}).get(key, fallback)


def get_metric_value(config, key, fallback=""):
    return config.get("metrics", {}).get(key, fallback)


def get_members(config):
    return config.get("members", DEFAULT_CONFIG["members"])


def get_model_config(config, model_key):
    return config.get("models", {}).get(
        model_key,
        DEFAULT_CONFIG["models"].get(model_key, {}),
    )


def get_model_accuracy(config, model_key, fallback=""):
    return get_model_config(config, model_key).get("accuracy", fallback)


def _model_is_loaded_in_app(model_key, loaded_models):
    if model_key == "logistic":
        return loaded_models.get("lr_model") is not None

    if model_key == "lstm":
        return loaded_models.get("lstm_model") is not None

    if model_key == "ensemble_v3":
        return loaded_models.get("ensemble_v3") is not None

    if model_key == "ensemble_v4":
        return loaded_models.get("ensemble_v4") is not None

    if model_key == "member2_svm":
        svm_model_available = (
            loaded_models.get("member2_svm_model") is not None
            or loaded_models.get("member2_svm_pipeline") is not None
            or loaded_models.get("member2_svm") is not None
        )

        svm_vectorizer_available = loaded_models.get("member2_svm_vectorizer") is not None

        if loaded_models.get("member2_svm_pipeline") is not None:
            return True

        return svm_model_available and svm_vectorizer_available

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
            or loaded_models.get("member3_tree") is not None
        )

    if model_key == "member3_transformer":
        return (
            loaded_models.get("member3_transformer_model") is not None
            or loaded_models.get("member3_transformer") is not None
        )

    return False


def get_enabled_model_options(loaded_models, config):
    options = []

    for model_key, model_info in config.get("models", {}).items():
        if not model_info.get("enabled", False):
            continue

        if not _model_is_loaded_in_app(model_key, loaded_models):
            continue

        display_name = model_info.get("display_name", model_key)
        accuracy = model_info.get("accuracy", "")
        note = model_info.get("note", "")

        if accuracy:
            option = f"{display_name} - {accuracy}"
        else:
            option = display_name

        if note:
            option = f"{option} | {note}"

        options.append(option)

    return options


def get_model_key_from_option(selected_option, config):
    selected_option = str(selected_option)

    for model_key, model_info in config.get("models", {}).items():
        display_name = model_info.get("display_name", model_key)

        if display_name in selected_option:
            return model_key

    lower = selected_option.lower()

    if "svm" in lower:
        return "member2_svm"

    if "cnn" in lower:
        return "member2_cnn"

    if "v3" in lower:
        return "ensemble_v3"

    if "v4" in lower:
        return "ensemble_v4"

    if "logistic" in lower:
        return "logistic"

    return "lstm"


def _get_logged_user():
    return st.session_state.get("admin_user")


def _set_logged_user(username):
    st.session_state["admin_user"] = username


def _clear_logged_user():
    st.session_state.pop("admin_user", None)


def _user_has_full_access(config, username):
    user = config.get("users", {}).get(username, {})
    return user.get("access") == "full"


def _get_user_name(config, username):
    return config.get("users", {}).get(username, {}).get("name", username)


def _get_user_role(config, username):
    return config.get("users", {}).get(username, {}).get("role", username)


def _render_login(config):
    st.markdown("## 🔐 Admin Login")
    st.caption("Use member1 / member2 / member3 login. Member 1 has full admin access.")

    with st.form("admin_login_form"):
        username = st.text_input("Username", placeholder="member1 / member2 / member3")
        password = st.text_input("Password", type="password")
        login_clicked = st.form_submit_button("Login")

    if login_clicked:
        username = username.strip().lower()
        user = config.get("users", {}).get(username)

        if user and password == user.get("password"):
            _set_logged_user(username)
            st.success(f"Logged in as {user.get('name', username)}")
            st.rerun()
        else:
            st.error("Invalid username or password.")


def _render_admin_header(config, username):
    user_name = _get_user_name(config, username)
    role = _get_user_role(config, username)

    col1, col2 = st.columns([0.78, 0.22])

    with col1:
        st.markdown("## ⚙️ Admin Panel")
        st.caption(f"Logged in as **{user_name}** · Role: **{role}**")

    with col2:
        if st.button("Logout"):
            _clear_logged_user()
            st.rerun()


def _render_model_toggle_only(config, username):
    role = _get_user_role(config, username)

    model_items = {
        key: value
        for key, value in config.get("models", {}).items()
        if value.get("owner") == role
    }

    st.markdown("### 🤖 My Model Controls")
    st.info("මෙතන ඔයාට තමන්ගේ models ON / OFF විතරයි change කරන්න පුළුවන්.")

    if not model_items:
        st.warning("No models assigned to this member yet.")
        return config

    new_config = copy.deepcopy(config)

    for model_key, model_info in model_items.items():
        with st.container(border=True):
            col1, col2 = st.columns([0.72, 0.28])

            with col1:
                st.markdown(f"**{model_info.get('display_name', model_key)}**")
                st.caption(
                    f"Type: {model_info.get('type', '')} · "
                    f"Accuracy: {model_info.get('accuracy', '')} · "
                    f"{model_info.get('note', '')}"
                )

            with col2:
                new_config["models"][model_key]["enabled"] = st.toggle(
                    "ON / OFF",
                    value=bool(model_info.get("enabled", False)),
                    key=f"member_toggle_{model_key}",
                )

    if st.button("Save My Model Settings", type="primary"):
        save_admin_config(new_config)
        st.success("Model settings saved.")
        st.rerun()

    return new_config


def _render_member1_models_tab(config):
    st.markdown("### 🤖 All Model Controls")

    new_config = copy.deepcopy(config)

    for model_key, model_info in new_config.get("models", {}).items():
        with st.expander(
            f"{model_info.get('display_name', model_key)} · {model_info.get('owner', '')}",
            expanded=False,
        ):
            c1, c2 = st.columns([0.25, 0.75])

            with c1:
                new_config["models"][model_key]["enabled"] = st.toggle(
                    "Model ON / OFF",
                    value=bool(model_info.get("enabled", False)),
                    key=f"full_model_enabled_{model_key}",
                )

            with c2:
                new_config["models"][model_key]["display_name"] = st.text_input(
                    "Display Name",
                    value=model_info.get("display_name", ""),
                    key=f"full_model_name_{model_key}",
                )

                new_config["models"][model_key]["accuracy"] = st.text_input(
                    "Accuracy",
                    value=model_info.get("accuracy", ""),
                    key=f"full_model_accuracy_{model_key}",
                )

                new_config["models"][model_key]["type"] = st.text_input(
                    "Model Type",
                    value=model_info.get("type", ""),
                    key=f"full_model_type_{model_key}",
                )

                new_config["models"][model_key]["note"] = st.text_area(
                    "Note",
                    value=model_info.get("note", ""),
                    key=f"full_model_note_{model_key}",
                    height=80,
                )

    if st.button("Save Model Settings", type="primary"):
        save_admin_config(new_config)
        st.success("Model settings saved.")
        st.rerun()

    return new_config


def _render_site_tab(config):
    st.markdown("### 🏠 Site Content Editor")

    new_config = copy.deepcopy(config)
    site = new_config.setdefault("site", {})

    site["group_label"] = st.text_input("Center Group Label", value=site.get("group_label", ""))
    site["small_brand"] = st.text_input("Small Brand", value=site.get("small_brand", ""))
    site["small_subtitle"] = st.text_input("Small Subtitle", value=site.get("small_subtitle", ""))

    site["hero_kicker"] = st.text_input("Hero Kicker", value=site.get("hero_kicker", ""))
    site["hero_title"] = st.text_area("Hero Title", value=site.get("hero_title", ""), height=90)
    site["hero_cta"] = st.text_input("Hero Button Text", value=site.get("hero_cta", ""))

    site["right_title"] = st.text_input("Right Side Title", value=site.get("right_title", ""))
    site["right_text"] = st.text_area("Right Side Text", value=site.get("right_text", ""), height=90)

    st.markdown("#### Hero Tabs")
    site["tab_1"] = st.text_input("Tab 1", value=site.get("tab_1", ""))
    site["tab_2"] = st.text_input("Tab 2", value=site.get("tab_2", ""))
    site["tab_3"] = st.text_input("Tab 3", value=site.get("tab_3", ""))
    site["tab_4"] = st.text_input("Tab 4", value=site.get("tab_4", ""))

    if st.button("Save Site Content", type="primary"):
        save_admin_config(new_config)
        st.success("Site content saved.")
        st.rerun()

    return new_config


def _render_metrics_tab(config):
    st.markdown("### 📊 Dashboard Metrics Editor")

    new_config = copy.deepcopy(config)
    metrics = new_config.setdefault("metrics", {})

    c1, c2 = st.columns(2)

    with c1:
        metrics["dataset_value"] = st.text_input("Dataset Value", value=metrics.get("dataset_value", ""))
        metrics["dataset_note"] = st.text_input("Dataset Note", value=metrics.get("dataset_note", ""))

        metrics["classes_value"] = st.text_input("Classes Value", value=metrics.get("classes_value", ""))
        metrics["classes_note"] = st.text_input("Classes Note", value=metrics.get("classes_note", ""))

    with c2:
        metrics["stable_value"] = st.text_input("Stable Best Value", value=metrics.get("stable_value", ""))
        metrics["stable_note"] = st.text_input("Stable Best Note", value=metrics.get("stable_note", ""))

        metrics["highest_value"] = st.text_input("Highest Value", value=metrics.get("highest_value", ""))
        metrics["highest_note"] = st.text_input("Highest Note", value=metrics.get("highest_note", ""))

    if st.button("Save Metrics", type="primary"):
        save_admin_config(new_config)
        st.success("Metrics saved.")
        st.rerun()

    return new_config


def _render_members_tab(config):
    st.markdown("### 👥 Member Progress Editor")

    new_config = copy.deepcopy(config)
    members = new_config.setdefault("members", [])

    for index, member in enumerate(members):
        with st.expander(f'{member.get("tag", "Member")} · {member.get("name", "")}', expanded=False):
            c1, c2 = st.columns(2)

            with c1:
                member["tag"] = st.text_input(
                    "Member Tag",
                    value=member.get("tag", ""),
                    key=f"member_tag_{index}",
                )

                member["name"] = st.text_input(
                    "Member Name",
                    value=member.get("name", ""),
                    key=f"member_name_{index}",
                )

                member["status"] = st.text_input(
                    "Status",
                    value=member.get("status", ""),
                    key=f"member_status_{index}",
                )

                member["progress"] = st.slider(
                    "Progress",
                    0,
                    100,
                    int(member.get("progress", 0)),
                    key=f"member_progress_{index}",
                )

            with c2:
                member["role"] = st.text_area(
                    "Main Responsibility",
                    value=member.get("role", ""),
                    key=f"member_role_{index}",
                    height=100,
                )

                member["items"] = st.text_area(
                    "Completed / Pending Items",
                    value=member.get("items", ""),
                    key=f"member_items_{index}",
                    height=100,
                )

    if st.button("Save Member Progress", type="primary"):
        save_admin_config(new_config)
        st.success("Member progress saved.")
        st.rerun()

    return new_config


def _render_config_tab(config):
    st.markdown("### 🗂️ Config Preview")
    st.caption("Current saved admin_config.json data.")
    st.json(config)

    if st.button("Reset All Admin Config"):
        save_admin_config(DEFAULT_CONFIG)
        st.warning("Config reset completed.")
        st.rerun()


def _render_full_admin(config):
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "Site Content",
            "Dashboard Metrics",
            "Models",
            "Member Progress",
            "Config",
        ]
    )

    with tab1:
        config = _render_site_tab(config)

    with tab2:
        config = _render_metrics_tab(config)

    with tab3:
        config = _render_member1_models_tab(config)

    with tab4:
        config = _render_members_tab(config)

    with tab5:
        _render_config_tab(config)

    return config


def render_admin_panel(config=None):
    if config is None:
        config = load_admin_config()

    username = _get_logged_user()

    if not username:
        _render_login(config)
        return config

    _render_admin_header(config, username)

    if _user_has_full_access(config, username):
        return _render_full_admin(config)

    return _render_model_toggle_only(config, username)