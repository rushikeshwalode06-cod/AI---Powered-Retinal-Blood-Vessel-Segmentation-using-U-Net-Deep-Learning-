import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from pathlib import Path


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="RetinaVision AI",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS - ATTRACTIVE UI
# ============================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Main background */

.stApp {
    background:
        radial-gradient(circle at 10% 10%, rgba(88, 166, 255, 0.18), transparent 25%),
        radial-gradient(circle at 90% 20%, rgba(168, 85, 247, 0.18), transparent 25%),
        radial-gradient(circle at 50% 100%, rgba(20, 184, 166, 0.12), transparent 30%),
        #07111f;
    color: white;
}


/* Header */

.hero {
    padding: 35px 35px 30px 35px;
    border-radius: 25px;
    background:
        linear-gradient(
            135deg,
            rgba(15, 23, 42, 0.96),
            rgba(30, 41, 59, 0.88)
        );
    border: 1px solid rgba(148, 163, 184, 0.20);
    box-shadow: 0 20px 60px rgba(0,0,0,0.35);
    margin-bottom: 25px;
}

.hero-title {
    font-size: 48px;
    font-weight: 800;
    margin-bottom: 8px;
    background: linear-gradient(
        90deg,
        #60a5fa,
        #a78bfa,
        #2dd4bf
    );
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle {
    font-size: 18px;
    color: #cbd5e1;
    line-height: 1.7;
}


/* Cards */

.card {
    padding: 25px;
    border-radius: 20px;
    background: rgba(15, 23, 42, 0.78);
    border: 1px solid rgba(148, 163, 184, 0.16);
    box-shadow: 0 15px 45px rgba(0,0,0,0.22);
    margin-bottom: 20px;
}

.card-title {
    font-size: 22px;
    font-weight: 700;
    margin-bottom: 12px;
}

.card-text {
    color: #cbd5e1;
    line-height: 1.7;
}


/* Metric cards */

.metric-box {
    padding: 20px;
    text-align: center;
    border-radius: 18px;
    background: linear-gradient(
        135deg,
        rgba(30,41,59,0.90),
        rgba(15,23,42,0.90)
    );
    border: 1px solid rgba(96,165,250,0.20);
}

.metric-number {
    font-size: 30px;
    font-weight: 800;
    color: #60a5fa;
}

.metric-label {
    color: #94a3b8;
    font-size: 14px;
}


/* Upload box */

[data-testid="stFileUploader"] {
    background: rgba(15, 23, 42, 0.75);
    border: 2px dashed rgba(96,165,250,0.40);
    border-radius: 20px;
    padding: 15px;
}


/* Buttons */

.stButton > button {
    width: 100%;
    border: none;
    border-radius: 14px;
    padding: 14px 20px;
    font-size: 17px;
    font-weight: 700;
    color: white;
    background: linear-gradient(
        90deg,
        #2563eb,
        #7c3aed,
        #0d9488
    );
    box-shadow: 0 10px 30px rgba(37,99,235,0.25);
    transition: all 0.25s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 15px 40px rgba(124,58,237,0.35);
}


/* Sidebar */

[data-testid="stSidebar"] {
    background:
        linear-gradient(
            180deg,
            #0b1324,
            #111827
        );
    border-right: 1px solid rgba(148,163,184,0.15);
}


/* Success box */

.success-box {
    padding: 16px;
    border-radius: 15px;
    background: rgba(16,185,129,0.10);
    border: 1px solid rgba(16,185,129,0.30);
    color: #6ee7b7;
}


/* Warning */

.warning-box {
    padding: 16px;
    border-radius: 15px;
    background: rgba(245,158,11,0.10);
    border: 1px solid rgba(245,158,11,0.30);
    color: #fcd34d;
}


/* Footer */

.footer {
    text-align: center;
    color: #64748b;
    padding: 30px 10px;
    font-size: 13px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# MODEL PATH
# ============================================================

APP_FOLDER = Path(__file__).resolve().parent

MODEL_PATH = APP_FOLDER / "retinal_blood_vessel_unet.h5"

IMG_SIZE = 256


# ============================================================
# COMPATIBILITY FIX
# ============================================================

class CompatibleConv2DTranspose(
    tf.keras.layers.Conv2DTranspose
):

    def __init__(self, *args, **kwargs):

        # Older model contains groups=1.
        # Current Keras version may not accept it.

        kwargs.pop("groups", None)

        super().__init__(
            *args,
            **kwargs
        )


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource(show_spinner=False)
def load_unet_model():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"""
Model file not found.

Expected location:

{MODEL_PATH}

Please make sure that:
retinal_blood_vessel_unet.h5

is in the same folder as app.py.
"""
        )

    model = tf.keras.models.load_model(
        str(MODEL_PATH),
        custom_objects={
            "Conv2DTranspose":
                CompatibleConv2DTranspose,
            "CompatibleConv2DTranspose":
                CompatibleConv2DTranspose
        },
        compile=False
    )

    return model


# ============================================================
# LOAD MODEL SAFELY
# ============================================================

try:

    model = load_unet_model()

    MODEL_STATUS = True

except Exception as e:

    MODEL_STATUS = False
    MODEL_ERROR = str(e)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("""
    <div style="text-align:center; padding:10px;">
        <div style="font-size:55px;">👁️</div>
        <h2 style="margin:0;">RetinaVision AI</h2>
        <p style="color:#94a3b8;">
            U-Net Blood Vessel Segmentation
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### 🧠 AI Model")

    st.markdown("""
    <div class="card">
        <b>Architecture</b><br>
        U-Net<br><br>

        
    </div>
    """, unsafe_allow_html=True)

    if MODEL_STATUS:

        st.markdown("""
        <div class="success-box">
        🟢 Model Ready
        </div>
        """, unsafe_allow_html=True)

    else:

        st.markdown("""
        <div class="warning-box">
        🔴 Model Loading Failed
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown(
        "### ⚙️ Processing"
    )

    st.write("Image Resize: 256 × 256")
    st.write("Normalization: 0–1")
    st.write("Threshold: 0.5")

    st.markdown("---")

    st.caption(
        "AI-Powered Retinal Blood Vessel Segmentation"
    )


# ============================================================
# HERO SECTION
# ============================================================

st.markdown("""
<div class="hero">

<div class="hero-title">
👁️ RetinaVision AI
</div>

<div class="hero-subtitle">
AI-Powered Retinal Blood Vessel Segmentation
using U-Net Deep Learning
<br><br>

Upload a retinal fundus image and let the
U-Net model identify and segment retinal
blood vessels automatically.
</div>

</div>
""", unsafe_allow_html=True)


# ============================================================
# MODEL ERROR
# ============================================================

if not MODEL_STATUS:

    st.error("❌ Could not load the U-Net model.")

    st.code(MODEL_ERROR)

    st.info(
        "Make sure retinal_blood_vessel_unet.h5 "
        "is present in the same folder as app.py."
    )

    st.stop()


# ============================================================
# MODEL READY
# ============================================================

st.markdown("""
<div class="success-box">
🟢 U-Net model loaded successfully and is ready for prediction.
</div>
""", unsafe_allow_html=True)

st.write("")


# ============================================================
# INFORMATION CARDS
# ============================================================

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.markdown("""
    <div class="metric-box">
        <div class="metric-number">U-Net</div>
        <div class="metric-label">Deep Learning Model</div>
    </div>
    """, unsafe_allow_html=True)

with col2:

    st.markdown("""
    <div class="metric-box">
        <div class="metric-number">256²</div>
        <div class="metric-label">Input Resolution</div>
    </div>
    """, unsafe_allow_html=True)

with col3:

    st.markdown("""
    <div class="metric-box">
        <div class="metric-number">1</div>
        <div class="metric-label">Segmentation Channel</div>
    </div>
    """, unsafe_allow_html=True)

with col4:

    st.markdown("""
    <div class="metric-box">
        <div class="metric-number">AI</div>
        <div class="metric-label">Automated Analysis</div>
    </div>
    """, unsafe_allow_html=True)


st.write("")
st.write("")


# ============================================================
# UPLOAD SECTION
# ============================================================

st.markdown("""
<div class="card">

<div class="card-title">
📤 Upload Retinal Fundus Image
</div>

<div class="card-text">
Choose a retinal fundus image in JPG, JPEG or PNG format.
The image will be resized and processed by the U-Net model.
</div>

</div>
""", unsafe_allow_html=True)


uploaded_file = st.file_uploader(
    "Choose retinal image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ],
    label_visibility="collapsed"
)


# ============================================================
# PREPROCESSING
# ============================================================

def preprocess_image(image):

    image = np.array(image)

    # Grayscale → RGB

    if image.ndim == 2:

        image = cv2.cvtColor(
            image,
            cv2.COLOR_GRAY2RGB
        )

    # RGBA → RGB

    elif image.shape[-1] == 4:

        image = cv2.cvtColor(
            image,
            cv2.COLOR_RGBA2RGB
        )

    # Resize

    image = cv2.resize(
        image,
        (IMG_SIZE, IMG_SIZE),
        interpolation=cv2.INTER_AREA
    )

    # Normalize

    image = image.astype(
        np.float32
    ) / 255.0

    # Add batch dimension

    image = np.expand_dims(
        image,
        axis=0
    )

    return image


# ============================================================
# PREDICTION
# ============================================================

def predict_vessels(image):

    input_image = preprocess_image(
        image
    )

    prediction = model.predict(
        input_image,
        verbose=0
    )

    prediction = np.squeeze(
        prediction
    )

    # Probability mask

    binary_mask = (
        prediction >= 0.5
    ).astype(np.uint8)

    return prediction, binary_mask


# ============================================================
# OVERLAY
# ============================================================

def create_overlay(
    original,
    mask
):

    original = np.array(
        original.convert("RGB")
    )

    original = cv2.resize(
        original,
        (IMG_SIZE, IMG_SIZE)
    )

    # Create RGB mask

    vessel_layer = np.zeros_like(
        original
    )

    # Bright red vessel visualization

    vessel_layer[:, :, 0] = (
        mask * 255
    )

    # Blend original + vessel mask

    overlay = cv2.addWeighted(
        original,
        0.70,
        vessel_layer,
        0.80,
        0
    )

    return overlay


# ============================================================
# RESULT SECTION
# ============================================================

if uploaded_file:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.write("")

    st.markdown("""
    <div class="card-title">
    🔍 Image Preview
    </div>
    """, unsafe_allow_html=True)

    preview_col1, preview_col2 = st.columns(
        [1, 1]
    )

    with preview_col1:

        st.image(
            image,
            caption="Original Retinal Image",
            use_container_width=True
        )

    with preview_col2:

        st.markdown("""
        <div class="card">

        <div class="card-title">
        📋 Image Information
        </div>

        <div class="card-text">

        <b>File:</b> Uploaded Image<br><br>

        <b>Original Size:</b>
        """
        + f"{image.size[0]} × {image.size[1]}"
        + """
        <br><br>

        <b>Model Input:</b>
        256 × 256 × 3
        <br><br>

        <b>Segmentation:</b>
        Binary Blood Vessel Mask

        </div>

        </div>
        """, unsafe_allow_html=True)


    st.write("")
    st.write("")


    # ========================================================
    # ANALYZE BUTTON
    # ========================================================

    if st.button(
        "🚀 ANALYZE RETINAL BLOOD VESSELS"
    ):

        with st.spinner(
            "🧠 U-Net is analyzing the retinal image..."
        ):

            try:

                prediction, mask = predict_vessels(
                    image
                )

                overlay = create_overlay(
                    image,
                    mask
                )

                # =================================================
                # CALCULATE STATS
                # =================================================

                vessel_pixels = np.sum(
                    mask > 0
                )

                total_pixels = mask.size

                vessel_percentage = (
                    vessel_pixels /
                    total_pixels
                ) * 100

                avg_probability = (
                    np.mean(prediction)
                )

                # =================================================
                # RESULTS
                # =================================================

                st.success(
                    "✅ Retinal blood vessel segmentation completed!"
                )

                st.write("")

                st.markdown("""
                <div class="card-title">
                🎯 Segmentation Results
                </div>
                """, unsafe_allow_html=True)

                result_col1, result_col2, result_col3 = st.columns(
                    3
                )

                with result_col1:

                    st.markdown(
                        f"""
                        <div class="metric-box">
                            <div class="metric-number">
                                {vessel_percentage:.2f}%
                            </div>
                            <div class="metric-label">
                                Vessel Area
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                with result_col2:

                    st.markdown(
                        f"""
                        <div class="metric-box">
                            <div class="metric-number">
                                {avg_probability:.2f}
                            </div>
                            <div class="metric-label">
                                Mean Prediction
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                with result_col3:

                    st.markdown(
                        f"""
                        <div class="metric-box">
                            <div class="metric-number">
                                {vessel_pixels:,}
                            </div>
                            <div class="metric-label">
                                Vessel Pixels
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )


                st.write("")
                st.write("")


                # =================================================
                # IMAGE RESULTS
                # =================================================

                result1, result2, result3 = st.columns(
                    3
                )

                with result1:

                    st.markdown(
                        "### 🖼️ Original"
                    )

                    st.image(
                        image,
                        use_container_width=True
                    )

                with result2:

                    st.markdown(
                        "### 🩸 Vessel Mask"
                    )

                    st.image(
                        mask * 255,
                        use_container_width=True,
                        clamp=True
                    )

                with result3:

                    st.markdown(
                        "### ✨ Overlay"
                    )

                    st.image(
                        overlay,
                        use_container_width=True
                    )


                st.write("")
                st.write("")


                # =================================================
                # DOWNLOAD MASK
                # =================================================

                mask_image = (
                    mask * 255
                ).astype(
                    np.uint8
                )

                success, encoded_mask = cv2.imencode(
                    ".png",
                    mask_image
                )

                if success:

                    st.download_button(
                        label="⬇️ Download Segmentation Mask",
                        data=encoded_mask.tobytes(),
                        file_name="retinal_vessel_segmentation.png",
                        mime="image/png"
                    )


                st.write("")

                # =================================================
                # INTERPRETATION
                # =================================================

                st.markdown("""
                <div class="card">

                <div class="card-title">
                📊 Analysis Summary
                </div>

                <div class="card-text">

                The U-Net model has generated a binary
                segmentation mask highlighting the predicted
                retinal blood vessel regions.

                <br><br>

                <b>White regions:</b>
                Predicted blood vessels

                <br>

                <b>Black regions:</b>
                Background / non-vessel regions

                <br><br>

                The overlay combines the original retinal image
                with the predicted vessel segmentation for
                easier visual interpretation.

                </div>

                </div>
                """, unsafe_allow_html=True)


                st.warning(
                    "⚠️ This application is a deep-learning "
                    "demonstration and should not be used as "
                    "a medical diagnosis."
                )


            except Exception as e:

                st.error(
                    "❌ Error while processing the image."
                )

                st.exception(e)


# ============================================================
# NO IMAGE UPLOADED
# ============================================================

else:

    st.write("")

    st.markdown("""
    <div class="card">

    <div class="card-title">
    👆 Get Started
    </div>

    <div class="card-text">

    1. Upload a retinal fundus image above.<br>
    2. Click <b>Analyze Retinal Blood Vessels</b>.<br>
    3. The U-Net model will segment the blood vessels.<br>
    4. View the original image, vessel mask and overlay.<br>
    5. Download the generated segmentation mask.

    </div>

    </div>
    """, unsafe_allow_html=True)


# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="footer">

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

<br>

👁️ <b>RetinaVision AI</b>

<br><br>

AI-Powered Retinal Blood Vessel Segmentation
<br>
U-Net Deep Learning Project

<br><br>

Built with Python • TensorFlow • Keras • Streamlit • OpenCV

<br><br>

© 2026 RetinaVision AI

</div>
""", unsafe_allow_html=True)