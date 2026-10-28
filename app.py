import io
import os
import base64
from html import escape
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageOps

# 1. PAGE CONFIG

st.set_page_config(
    page_title="Crop Disease Detection AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# 2. PATHS + SETTINGS

BASE_DIR = Path(__file__).parent
ASSETS_DIR = BASE_DIR / "assets"
MODEL_PATH = BASE_DIR / "model" / "crop_disease_detection.keras"
IMG_SIZE = (224, 224)

CLASS_NAMES = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy", "Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy", "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot", "Peach___healthy",
    "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy", "Potato___Early_blight",
    "Potato___Late_blight", "Potato___healthy", "Raspberry___healthy", "Soybean___healthy",
    "Squash___Powdery_mildew", "Strawberry___Leaf_scorch", "Strawberry___healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot", "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus", "Tomato___healthy",
]


# 3. HELPERS

def to_data_uri(img: Image.Image, max_size: int) -> str:
    # Convert an image to base64 text so it can be used inside HTML / CSS
    img = img.convert("RGB")
    img.thumbnail((max_size, max_size))
    buffer = io.BytesIO()
    img.save(buffer, "JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


@st.cache_data(show_spinner=False)
def load_asset(file_name: str, max_size: int) -> str:
    return to_data_uri(Image.open(ASSETS_DIR / file_name), max_size)


def format_class_name(name: str) -> str:
    # "Apple___Apple_scab" -> "Apple — Apple Scab"
    return name.replace("___", " — ").replace("_", " ").replace(",", "").title()


def crop_name(name: str) -> str:
    # "Cherry_(including_sour)___healthy" -> "Cherry"
    return name.split("___")[0].replace("_", " ").split(" (")[0].replace(",", "").title()


def preprocess_image(img: Image.Image) -> np.ndarray:
    # Resize to 224x224 and add a batch dimension (model rescales pixels itself)
    img = ImageOps.fit(img.convert("RGB"), IMG_SIZE, method=Image.Resampling.LANCZOS)
    return np.expand_dims(np.array(img, dtype=np.float32), axis=0)


def predict_disease(img: Image.Image):
    # Returns the predicted class and its confidence
    preds = model.predict(preprocess_image(img), verbose=0)[0]
    idx = int(np.argmax(preds))
    return CLASS_NAMES[idx], float(preds[idx] * 100)


def show_html(html: str, target=st):
    # Streamlit treats indented lines as code, so join all lines without indentation
    target.markdown(" ".join(line.strip() for line in html.splitlines() if line.strip()), unsafe_allow_html=True)


# 4. CSS

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --green: #1b6b2f;
    --text: #0b1f14;
    --muted: #5b6b62;
    --danger: #dc2626;
}

/* Page background */
.stApp {
    background: url("__BG__") center / cover fixed no-repeat !important;
    color: var(--text);
}

/* Hide default Streamlit parts */
header[data-testid="stHeader"], #MainMenu, footer,
[data-testid="stSidebar"], [data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"] { display: none !important; }

/* Page spacing (compact so everything fits in one screen, no scroll) */
.block-container, [data-testid="stMainBlockContainer"] {
    max-width: 880px !important;
    padding: 78px 1rem 0.5rem 1rem !important;
}
[data-testid="stVerticalBlock"] { gap: 0.6rem !important; }

/* Navbar */
.navbar {
    position: fixed;
    top: 0; left: 0; right: 0;
    height: 64px;
    background: rgba(255, 255, 255, 0.97);
    box-shadow: 0 2px 14px rgba(15, 77, 32, 0.08);
    z-index: 1000;
    display: flex;
    align-items: center;
    padding: 0 9%;
}
.logo {
    display: flex;
    align-items: center;
    gap: 10px;
    font-family: 'Inter', 'Segoe UI', sans-serif !important;
    font-size: 1.6rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    color: var(--text);
}
.logo span { color: #5aa21b; }

/* Hero */
.hero { text-align: center; }
.hero-divider {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 14px;
}
.hero-divider i {
    width: 90px;
    height: 2px;
    background: #4f9a3b;
    border-radius: 2px;
}
.hero-title {
    font-family: 'Inter', 'Segoe UI', sans-serif !important;
    font-size: clamp(2rem, 6.4vh, 3.4rem);
    font-weight: 800;
    letter-spacing: -0.03em;
    line-height: 1.1;
    margin: 8px 0 6px 0;
    color: #0b1f14 !important;
}
.hero-title span { color: #1b6b2f !important; }
.hero-sub {
    font-family: 'Inter', 'Segoe UI', sans-serif !important;
    font-size: clamp(0.95rem, 2.4vh, 1.15rem);
    color: var(--muted);
    margin: 0 0 6px 0;
}

/* Upload card: white card + dashed green border inside it */
.st-key-upload_card {
    position: relative;
    background: rgba(255, 255, 255, 0.95);
    border-radius: 28px;
    padding: clamp(34px, 7vh, 58px) 36px;
    box-shadow: 0 18px 50px -12px rgba(15, 77, 32, 0.22);
}
.st-key-upload_card::before {
    content: "";
    position: absolute;
    inset: 14px;
    border: 1.5px dashed rgba(27, 107, 47, 0.55);
    border-radius: 20px;
    pointer-events: none;
}

/* Thin line between left and right half */
.st-key-upload_card [data-testid="stColumn"]:nth-child(2),
.st-key-upload_card [data-testid="column"]:nth-child(2) {
    border-left: 1px solid #dfe8e1;
    padding-left: 30px;
}

.upload-text { text-align: center; font-family: 'Inter', 'Segoe UI', sans-serif !important; }
.upload-text h3 {
    margin: 6px 0 2px 0;
    padding: 0;
    font-size: 1.35rem;
    font-weight: 700;
    color: var(--text) !important;
}
.upload-text p { margin: 0; padding-bottom: 4px; color: var(--muted); font-size: 0.92rem; }
.upload-hint {
    text-align: center;
    color: #7b8a81;
    font-size: 0.85rem;
    font-family: 'Inter', 'Segoe UI', sans-serif !important;
}

/* File uploader: the whole box becomes a green "Choose Image" button */
[data-testid="stFileUploader"] { width: 100%; margin-top: 18px; }
[data-testid="stFileUploaderDropzone"] {
    position: relative;
    width: min(100%, 340px);
    min-height: 52px;
    margin: 0 auto;
    padding: 0 !important;
    border: none !important;
    border-radius: 12px !important;
    background: linear-gradient(135deg, #1f8a3d, #0f5a25) !important;
    box-shadow: 0 8px 20px rgba(15, 90, 37, 0.3);
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
}
[data-testid="stFileUploaderDropzone"]::after {
    content: "Upload Image";
    color: white;
    font-size: 1.05rem;
    font-weight: 600;
    pointer-events: none;
    width: 100%;
    text-align: center;
    display: flex;
    align-items: center;
    justify-content: center;
}
[data-testid="stFileUploaderDropzoneInstructions"] { display: none !important; }
[data-testid="stFileUploaderDropzone"] button {
    position: absolute !important;
    inset: 0;
    width: 100%;
    height: 100%;
    opacity: 0;
    cursor: pointer;
}

/* Uploaded file name box + red cross button (cross = start again) */
.st-key-chip_row { width: min(100%, 340px); margin: 18px auto 0 auto; }
.file-chip {
    display: flex;
    align-items: center;
    gap: 8px;
    height: 44px;
    padding: 0 14px;
    background: #f1f7f2;
    border: 1px solid #dfe8e1;
    border-radius: 12px;
    font-family: 'Inter', 'Segoe UI', sans-serif !important;
    font-size: 0.9rem;
    color: var(--text);
    overflow: hidden;
}
.file-chip span { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.st-key-remove_btn button {
    width: 44px;
    height: 44px;
    min-height: 0;
    padding: 0;
    border-radius: 50%;
    background: white;
    color: var(--danger);
    border: 1.5px solid #fca5a5;
    font-size: 1.15rem;
    font-weight: 700;
    transition: all 0.2s ease;
}
.st-key-remove_btn button:hover { background: var(--danger); color: white; border-color: var(--danger); }

/* Leaf image on the right side */
.leaf-img {
    width: clamp(220px, 40vh, 330px);
    height: clamp(220px, 40vh, 330px);
    margin: 0 auto;
    border-radius: 22px;
    background-size: cover;
    background-position: center;
    box-shadow: 0 10px 28px rgba(0, 0, 0, 0.18);
}

/* Analyze button */
.st-key-analyze-btn { width: min(100%, 340px); margin: 0 auto; }
.st-key-analyze-btn button {
    width: 100%;
    min-height: 52px;
    border-radius: 12px;
    font-weight: 600;
    font-size: 1.05rem;
    background: white;
    color: var(--green);
    border: 1.5px solid var(--green);
    transition: all 0.2s ease;
}
.st-key-analyze-btn button:hover { background: var(--green); color: white; border-color: var(--green); }

/* Stat cards + result card */
.stat, .result {
    background: rgba(255, 255, 255, 0.93);
    border-radius: 20px;
    box-shadow: 0 10px 30px -10px rgba(15, 77, 32, 0.18);
    font-family: 'Inter', 'Segoe UI', sans-serif !important;
}
.stat {
    padding: 14px 22px;
    display: flex;
    align-items: center;
    gap: 20px;
}
.stat-icon {
    width: 56px;
    height: 56px;
    border-radius: 50%;
    background: #e6f2e8;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}
.stat-value { font-size: 2rem; font-weight: 800; line-height: 1; color: var(--text); }
.stat-label { color: var(--muted); font-size: 0.95rem; margin-top: 3px; }

.result {
    padding: 16px 32px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 20px;
}
.badge {
    display: inline-block;
    padding: 5px 14px;
    border-radius: 999px;
    font-size: 0.74rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
.badge-ok { background: rgba(27, 107, 47, 0.1); color: var(--green); border: 1px solid rgba(27, 107, 47, 0.3); }
.badge-bad { background: rgba(220, 38, 38, 0.08); color: var(--danger); border: 1px solid rgba(220, 38, 38, 0.3); }
.diagnosis { font-size: 1.3rem; font-weight: 700; margin-top: 6px; color: var(--text); }
.result-right { text-align: right; flex-shrink: 0; }
.confidence { font-size: 2.4rem; font-weight: 800; color: var(--green); line-height: 1; }
.confidence-label { color: var(--muted); font-size: 0.85rem; margin-top: 2px; }
</style>
"""

st.markdown(CSS.replace("__BG__", load_asset("background.jpg", 1920)), unsafe_allow_html=True)


# 5. LOAD MODEL

@st.cache_resource(show_spinner="Loading model...")
def load_model():
    import tensorflow as tf
    return tf.keras.models.load_model(MODEL_PATH)


if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()

model = load_model()
total_crops = len({crop_name(c) for c in CLASS_NAMES})


# 6. NAVBAR + HERO

LOGO_SVG = """
<svg width="36" height="36" viewBox="0 0 32 32" fill="none">
  <path d="M16 30V16" stroke="#2e7d32" stroke-width="2.4" stroke-linecap="round"/>
  <path d="M16 17C16 9 11 5 3 5c0 8 5 12 13 12z" fill="#2e7d32"/>
  <path d="M16 14c0-7 4-11 13-12 0 8-5 12-13 12z" fill="#6cb52d"/>
</svg>
"""

show_html(
    f"""
    <div class="navbar">
        <div class="logo">{LOGO_SVG}<div>Crop Disease Detection <span>AI</span></div></div>
    </div>

    <div class="hero">
        <div class="hero-divider">
            <i></i>
            <svg width="34" height="40" viewBox="0 0 32 32" fill="#2e7d32">
                <path d="M26 4C14 4 6 11 6 21c0 3 1 5 2 7 2-9 8-14 15-17-6 4-11 9-13 17 11 0 18-8 16-24z"/>
            </svg>
            <i></i>
        </div>
        <div class="hero-title">Crop Disease <span>Detection</span></div>
        <p class="hero-sub">Upload a leaf image to identify possible plant diseases in seconds.</p>
    </div>
    """
)


# 7. UPLOAD CARD

# Keep the uploaded image in session, so we can show our own cross (remove) button
if "img_bytes" not in st.session_state:
    st.session_state.img_bytes = None
    st.session_state.img_name = ""
    st.session_state.uploader_id = 0

with st.container(key="upload_card"):
    left, right = st.columns([1, 1], gap="large", vertical_alignment="center")

    with left:
        show_html(
            """
            <div class="upload-text">
                <svg width="52" height="52" viewBox="0 0 24 24" fill="none" stroke="#1b6b2f" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M16 16l-4-4-4 4"/>
                    <path d="M12 12v9"/>
                    <path d="M20.39 18.39A5 5 0 0 0 18 9h-1.26A8 8 0 1 0 3 16.3"/>
                </svg>
                <h3>Upload a plant leaf image</h3>
                <p>Drag and drop an image here, or click to browse</p>
            </div>
            """
        )

        analyze_clicked = False
        if st.session_state.img_bytes is None:
            # Start state: show the "Choose Image" button
            uploaded_file = st.file_uploader(
                "Upload Plant Image",
                type=["jpg", "jpeg", "png"],
                label_visibility="collapsed",
                key=f"uploader_{st.session_state.uploader_id}",
            )
            if uploaded_file is not None:
                st.session_state.img_bytes = uploaded_file.getvalue()
                st.session_state.img_name = uploaded_file.name
                st.rerun()
            show_html('<div class="upload-hint">JPG or PNG &nbsp;•&nbsp; Clear leaf photos work best</div>')
        else:
            # Uploaded state: file name, cross button and Analyze button
            with st.container(key="chip_row"):
                chip_col, remove_col = st.columns([5, 1], gap="small", vertical_alignment="center")
                show_html(
                    f'<div class="file-chip">🖼️ <span>{escape(st.session_state.img_name)}</span></div>',
                    target=chip_col,
                )
                # Cross button: clears the image and goes back to the start state
                if remove_col.button("✕", key="remove_btn"):
                    st.session_state.img_bytes = None
                    st.session_state.img_name = ""
                    st.session_state.uploader_id += 1
                    st.rerun()
            analyze_clicked = st.button("Analyze Leaf", key="analyze-btn")

    with right:
        # Show the uploaded image, or the sample leaf if nothing is uploaded
        if st.session_state.img_bytes is not None:
            image = ImageOps.exif_transpose(Image.open(io.BytesIO(st.session_state.img_bytes))).convert("RGB")
            leaf_src = to_data_uri(image, 700)
        else:
            leaf_src = load_asset("leaf.jpg", 700)
        show_html(f'<div class="leaf-img" style="background-image:url(\'{leaf_src}\')"></div>')


# 8. RESULT (replaces the stat cards after Analyze) / STAT CARDS

_, middle, _ = st.columns([0.5, 4, 0.5], gap="medium")

with middle:
    if analyze_clicked:
        with st.spinner("Analyzing..."):
            disease, confidence = predict_disease(image)

        is_healthy = "healthy" in disease.lower()
        badge = (
            '<span class="badge badge-ok">Healthy</span>'
            if is_healthy
            else '<span class="badge badge-bad">Disease Detected</span>'
        )

        show_html(
            f"""
            <div class="result">
                <div>
                    {badge}
                    <div class="diagnosis">{format_class_name(disease)}</div>
                </div>
                <div class="result-right">
                    <div class="confidence">{confidence:.1f}%</div>
                    <div class="confidence-label">Model confidence</div>
                </div>
            </div>
            """
        )
    else:
        col1, col2 = st.columns(2, gap="medium")

        show_html(
            f"""
            <div class="stat">
                <div class="stat-icon">
                    <svg width="30" height="30" viewBox="0 0 24 24" fill="none" stroke="#1b6b2f" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M12 21v-9"/><path d="M12 12C12 7 9 4 4 4c0 5 3 8 8 8z"/><path d="M12 14c0-4 3-7 8-7 0 5-3 7-8 7z"/>
                    </svg>
                </div>
                <div><div class="stat-value">{total_crops}</div><div class="stat-label">Crop Types</div></div>
            </div>
            """,
            target=col1,
        )

        show_html(
            f"""
            <div class="stat">
                <div class="stat-icon">
                    <svg width="30" height="30" viewBox="0 0 24 24" fill="none" stroke="#1b6b2f" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
                        <circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1.5"/>
                        <path d="M12 3v2M12 19v2M3 12h2M19 12h2M5.6 5.6l1.4 1.4M17 17l1.4 1.4M5.6 18.4L7 17M17 7l1.4-1.4"/>
                    </svg>
                </div>
                <div><div class="stat-value">{len(CLASS_NAMES)}</div><div class="stat-label">Disease Classes</div></div>
            </div>
            """,
            target=col2,
        )
