"""Khybersole AI Foot Size Finder — Streamlit Web Application.

Provides a customer-facing photogrammetry and AI sizing interface for
authentic handcrafted Peshawari Chappals.
"""

import json
from pathlib import Path
import io
import cv2
import numpy as np
from PIL import Image
import streamlit as st

from src.model_loader import ModelLoader, ModelError
from src.foot_segmentation import FootSegmenter
from src.foot_landmarks import FootLandmarkExtractor
from src.calibration import (
    ReferenceCardCalibrator,
    STANDARD_CARD_WIDTH_MM,
    STANDARD_CARD_HEIGHT_MM,
    A4_WIDTH_MM,
    A4_HEIGHT_MM
)
from src.measurement import measure_foot
from src.size_recommender import SizeRecommender
from src.image_validation import ImageValidator

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Khybersole AI Foot Size Finder",
    page_icon="👞",
    layout="wide",
    initial_sidebar_state="expanded"
)

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Playfair+Display:ital,wght@0,600;0,700;1,600&display=swap');

    :root {
        --khyber-dark: #120D0A;
        --khyber-espresso: #1E1510;
        --khyber-leather: #8B4513;
        --khyber-gold: #D4AF37;
        --khyber-brass: #C59B27;
        --khyber-amber: #E67E22;
        --khyber-cream: #FDFBF7;
        --khyber-surface: #261D17;
        --khyber-border: #3D2E24;
    }

    .main .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
        max-width: 1250px;
    }

    h1, h2, h3 {
        font-family: 'Playfair Display', Georgia, serif !important;
        letter-spacing: -0.01em;
    }

    body, p, span, div, label {
        font-family: 'Plus Jakarta Sans', -apple-system, sans-serif !important;
    }

    .khyber-header {
        background: linear-gradient(135deg, #1A120C 0%, #2B1B10 60%, #150E09 100%);
        border: 1px solid rgba(197, 155, 39, 0.35);
        border-radius: 14px;
        padding: 1.8rem 2.2rem;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.45);
        display: flex;
        align-items: center;
        justify-content: space-between;
    }

    .khyber-title {
        color: #FDFBF7;
        font-size: 2.1rem;
        font-weight: 700;
        margin: 0;
    }

    .khyber-subtitle {
        color: #C59B27;
        font-size: 0.95rem;
        font-weight: 500;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-top: 0.3rem;
    }

    .metric-card {
        background: #1E1712;
        border: 1px solid #3D2E24;
        border-radius: 12px;
        padding: 1.25rem 1.4rem;
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }

    .metric-card:hover {
        border-color: #C59B27;
        transform: translateY(-2px);
    }

    .metric-val {
        font-size: 2.2rem;
        font-weight: 800;
        color: #FDFBF7;
        font-family: 'Playfair Display', serif !important;
    }

    .metric-lbl {
        font-size: 0.8rem;
        font-weight: 600;
        color: #C59B27;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-top: 0.2rem;
    }

    .size-recommend-box {
        background: linear-gradient(135deg, #2A1D13 0%, #1A120B 100%);
        border: 2px solid #C59B27;
        border-radius: 14px;
        padding: 1.6rem 2rem;
        text-align: center;
        margin: 1.5rem 0;
        box-shadow: 0 8px 25px rgba(197, 155, 39, 0.15);
    }

    .size-badge {
        font-size: 3.4rem;
        font-weight: 800;
        color: #D4AF37;
        line-height: 1.1;
        font-family: 'Playfair Display', serif !important;
    }

    .stButton>button {
        background: linear-gradient(135deg, #C59B27 0%, #9E7B1E 100%);
        color: #120D0A !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.65rem 1.8rem !important;
        transition: all 0.25s ease !important;
    }

    .stButton>button:hover {
        background: linear-gradient(135deg, #D4AF37 0%, #B28B23 100%);
        box-shadow: 0 4px 15px rgba(197, 155, 39, 0.35);
        transform: translateY(-1px);
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------
# Cached Application Resources
# ---------------------------------------------------------
@st.cache_resource(show_spinner="Initializing AI Segmentation Engine...")
def get_segmenter():
    try:
        return FootSegmenter(model_path="models/best.pt")
    except ModelError as e:
        return None

@st.cache_resource
def get_recommender():
    try:
        return SizeRecommender("config/size_chart.json")
    except Exception:
        return None

segmenter = get_segmenter()
recommender = get_recommender()
landmark_extractor = FootLandmarkExtractor()
image_validator = ImageValidator()

# ---------------------------------------------------------
# Header Branding
# ---------------------------------------------------------
st.markdown("""
<div class="khyber-header">
    <div>
        <div class="khyber-title">👞 KHYBERSOLE</div>
        <div class="khyber-subtitle">AI Foot Size Finder & Heritage Chappal Sizing</div>
    </div>
    <div style="text-align: right; color: #A08C7D; font-size: 0.85rem;">
        Final Year Project &bull; Handcrafted Quality<br>
        <span style="color: #C59B27; font-weight: 600;">Calibrated Photogrammetry</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Check model availability
if segmenter is None:
    st.error(
        "⚠️ **AI Model Checkpoint Missing or Incompatible.**\n\n"
        "Could not load `best.pt`. Please ensure your model weights file is placed in `models/best.pt` "
        "or your Downloads folder. Run `python diagnose_model.py` to inspect the error."
    )
    st.stop()

# ---------------------------------------------------------
# Sidebar Controls & Guidelines
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### 📐 Measurement Setup")

    ref_choice = st.selectbox(
        "Reference Calibration Object",
        options=[
            "Standard Bank / Credit Card (85.60 x 53.98 mm)",
            "A4 Paper Sheet (297 x 210 mm)",
            "Custom Dimensions (mm)"
        ],
        index=0
    )

    if "Bank" in ref_choice:
        ref_w_mm = STANDARD_CARD_WIDTH_MM
        ref_h_mm = STANDARD_CARD_HEIGHT_MM
        ref_name = "Bank Card (ID-1)"
    elif "A4" in ref_choice:
        ref_w_mm = A4_WIDTH_MM
        ref_h_mm = A4_HEIGHT_MM
        ref_name = "A4 Sheet"
    else:
        col_w, col_h = st.columns(2)
        with col_w:
            ref_w_mm = st.number_input("Width (mm)", value=85.6, min_value=10.0, max_value=500.0)
        with col_h:
            ref_h_mm = st.number_input("Height (mm)", value=53.98, min_value=10.0, max_value=500.0)
        ref_name = "Custom Reference"

    st.markdown("---")
    st.markdown("### 📋 Photography Instructions")
    st.info(
        "1. **Flat Surface**: Stand or rest your bare foot on a hard, flat floor.\n"
        "2. **Place Reference Card**: Place your bank card on the floor right beside your foot.\n"
        "3. **Direct Top-Down Angle**: Hold camera directly above your foot (perpendicular to floor).\n"
        "4. **Good Lighting**: Ensure no harsh shadows obscure the card edges or toe tips."
    )

    st.markdown("---")
    st.markdown("### ⚙️ Calibration Fine-Tuning")
    enable_manual_override = st.checkbox("Manual Card Corner Override", value=False, help="Enable if automatic card detection fails or card contrast is low.")
    toe_hint = st.selectbox("Toe Orientation Hint", ["Auto", "Up (Toe at top of image)", "Down (Toe at bottom)", "Left", "Right"], index=0)
    hint_map = {"Auto": None, "Up (Toe at top of image)": "up", "Down (Toe at bottom)": "down", "Left": "left", "Right": "right"}
    selected_hint = hint_map[toe_hint]

    st.markdown("---")
    st.markdown("### 📱 Mobile Access")
    mobile_url = "http://192.168.1.17:8501"
    st.markdown(f"**URL:** [{mobile_url}]({mobile_url})")
    try:
        import qrcode
        qr_img = qrcode.make(mobile_url)
        st.image(qr_img.get_image(), caption="Scan to open on phone (same Wi-Fi)", use_container_width=True)
    except Exception:
        pass

    st.markdown("---")
    st.markdown("<div style='font-size:0.75rem; color:#888;'>Khybersole AI v1.0 &bull; Local CPU Inference &bull; No Cloud Upload</div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Image Input Tabs
# ---------------------------------------------------------
tab_upload, tab_camera, tab_samples = st.tabs(["📁 Upload Photograph", "📸 Live Camera", "🖼️ Benchmark Test Samples"])

uploaded_file = None

with tab_upload:
    uploaded_file = st.file_uploader(
        "Upload a photograph of your foot beside a reference card (JPEG, PNG, WEBP)",
        type=["jpg", "jpeg", "png", "webp"],
        key="uploader"
    )

with tab_camera:
    camera_file = st.camera_input("Capture foot photograph directly")
    if camera_file is not None:
        uploaded_file = camera_file

with tab_samples:
    test_candidates = list(Path("test_images").glob("*.jpg"))
    if test_candidates:
        names = [p.name for p in test_candidates]
        if "demo_foot_with_card.jpg" in names:
            names.remove("demo_foot_with_card.jpg")
            names.insert(0, "demo_foot_with_card.jpg")
        sample_names = names[:10]
        chosen_sample = st.selectbox("Select a test dataset image:", sample_names)
        if st.button("Load Selected Test Image"):
            sample_path = Path("test_images") / chosen_sample
            uploaded_file = str(sample_path)
    else:
        st.write("No sample images located in `test_images/`.")

# ---------------------------------------------------------
# Image Processing Pipeline
# ---------------------------------------------------------
if uploaded_file is not None:
    # Read image
    if isinstance(uploaded_file, str):
        img_bgr = cv2.imread(uploaded_file)
    else:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img_bgr is None:
        st.error("Failed to decode image. Please upload a valid image file.")
        st.stop()

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h_orig, w_orig = img_bgr.shape[:2]

    # Pre-validation
    val_report = image_validator.validate_image(img_bgr)
    if not val_report.is_valid:
        st.warning("⚠️ **Image Quality Advisory**")
        for r in val_report.rejection_reasons:
            st.error(f"&bull; {r}")
        st.info("Please capture a sharper photograph with better lighting and try again.")
        st.stop()

    if val_report.warnings:
        for w in val_report.warnings:
            st.caption(f"ℹ️ Notice: {w}")

    # Process foot segmentation
    with st.spinner("Analyzing foot anatomy and reference card..."):
        detections = segmenter.segment(img_bgr)

        if not detections:
            st.error(
                "❌ **No Foot Detected.**\n\n"
                "The AI model could not detect a foot in the photograph. "
                "Ensure your entire foot is clearly visible against the background without obstruction."
            )
            st.stop()

        primary_foot = detections[0]

        # Landmark extraction
        landmarks = landmark_extractor.extract_landmarks(
            primary_foot.mask,
            expected_toe_direction=selected_hint
        )

        if landmarks is None:
            st.error(
                "❌ **Landmark Extraction Incomplete.**\n\n"
                "Could not reliably determine the longitudinal axis of the foot. "
                "Ensure your foot is resting flat and the heel and toes are unobscured."
            )
            st.stop()

        # Session state management for manual landmark points
        img_id = f"{h_orig}_{w_orig}_{len(detections)}"
        if "last_img_id" not in st.session_state or st.session_state["last_img_id"] != img_id:
            st.session_state["last_img_id"] = img_id
            st.session_state["manual_heel_x"] = int(round(landmarks.heel_point[0]))
            st.session_state["manual_heel_y"] = int(round(landmarks.heel_point[1]))
            st.session_state["manual_toe_x"] = int(round(landmarks.toe_point[0]))
            st.session_state["manual_toe_y"] = int(round(landmarks.toe_point[1]))

        # Manual Landmark Adjustment Controls
        with st.expander("🎯 Manual Heel & Toe Fine-Tuning (Nudge Points)", expanded=False):
            st.markdown(
                "If the AI heel point or toe tip needs fine-tuning, use the sliders below "
                "to position the markers at the exact edge of your heel and the tip of your longest toe."
            )
            col_adj_h, col_adj_t = st.columns(2)
            with col_adj_h:
                st.markdown("##### 🟠 Heel Position")
                h_x = st.slider(
                    "Heel X (Left ↔ Right)",
                    min_value=0, max_value=w_orig,
                    value=int(st.session_state["manual_heel_x"]),
                    key="slider_hx"
                )
                h_y = st.slider(
                    "Heel Y (Top ↕ Bottom)",
                    min_value=0, max_value=h_orig,
                    value=int(st.session_state["manual_heel_y"]),
                    key="slider_hy"
                )
            with col_adj_t:
                st.markdown("##### 🔵 Toe Tip Position")
                t_x = st.slider(
                    "Toe X (Left ↔ Right)",
                    min_value=0, max_value=w_orig,
                    value=int(st.session_state["manual_toe_x"]),
                    key="slider_tx"
                )
                t_y = st.slider(
                    "Toe Y (Top ↕ Bottom)",
                    min_value=0, max_value=h_orig,
                    value=int(st.session_state["manual_toe_y"]),
                    key="slider_ty"
                )

            col_btn_reset, col_status = st.columns([1, 2])
            with col_btn_reset:
                if st.button("↺ Reset to AI Landmarks"):
                    st.session_state["manual_heel_x"] = int(round(landmarks.heel_point[0]))
                    st.session_state["manual_heel_y"] = int(round(landmarks.heel_point[1]))
                    st.session_state["manual_toe_x"] = int(round(landmarks.toe_point[0]))
                    st.session_state["manual_toe_y"] = int(round(landmarks.toe_point[1]))
                    st.rerun()

            # Update session state with current slider values
            st.session_state["manual_heel_x"] = h_x
            st.session_state["manual_heel_y"] = h_y
            st.session_state["manual_toe_x"] = t_x
            st.session_state["manual_toe_y"] = t_y

        # Apply adjusted landmark endpoints
        is_manually_adjusted = (
            abs(st.session_state["manual_heel_x"] - landmarks.heel_point[0]) > 1 or
            abs(st.session_state["manual_heel_y"] - landmarks.heel_point[1]) > 1 or
            abs(st.session_state["manual_toe_x"] - landmarks.toe_point[0]) > 1 or
            abs(st.session_state["manual_toe_y"] - landmarks.toe_point[1]) > 1
        )

        effective_heel = (float(st.session_state["manual_heel_x"]), float(st.session_state["manual_heel_y"]))
        effective_toe = (float(st.session_state["manual_toe_x"]), float(st.session_state["manual_toe_y"]))
        landmarks.heel_point = effective_heel
        landmarks.toe_point = effective_toe
        dx_adj = effective_toe[0] - effective_heel[0]
        dy_adj = effective_toe[1] - effective_heel[1]
        landmarks.length_pixels = float(np.sqrt(dx_adj * dx_adj + dy_adj * dy_adj))

        # Calibration
        calibrator = ReferenceCardCalibrator(card_width_mm=ref_w_mm, card_height_mm=ref_h_mm)

        manual_corners = None
        if enable_manual_override:
            st.markdown("#### 📍 Manual Reference Card Corner Fine-Tuning")
            st.caption("Adjust corner coordinates if auto-detection missed the card.")
            col_tl, col_tr = st.columns(2)
            col_br, col_bl = st.columns(2)

            with col_tl:
                tl_x = st.number_input("Top-Left X", value=int(w_orig * 0.1), min_value=0, max_value=w_orig)
                tl_y = st.number_input("Top-Left Y", value=int(h_orig * 0.1), min_value=0, max_value=h_orig)
            with col_tr:
                tr_x = st.number_input("Top-Right X", value=int(w_orig * 0.35), min_value=0, max_value=w_orig)
                tr_y = st.number_input("Top-Right Y", value=int(h_orig * 0.1), min_value=0, max_value=h_orig)
            with col_br:
                br_x = st.number_input("Bottom-Right X", value=int(w_orig * 0.35), min_value=0, max_value=w_orig)
                br_y = st.number_input("Bottom-Right Y", value=int(h_orig * 0.25), min_value=0, max_value=h_orig)
            with col_bl:
                bl_x = st.number_input("Bottom-Left X", value=int(w_orig * 0.1), min_value=0, max_value=w_orig)
                bl_y = st.number_input("Bottom-Left Y", value=int(h_orig * 0.25), min_value=0, max_value=h_orig)

            manual_corners = np.array([[tl_x, tl_y], [tr_x, tr_y], [br_x, br_y], [bl_x, bl_y]], dtype=np.float32)

        calibration_data = calibrator.calibrate(
            img_bgr,
            manual_corners=manual_corners,
            exclusion_mask=primary_foot.mask
        )

        # Fallback if card not detected
        if calibration_data is None:
            st.warning(
                "⚠️ **Reference Object Not Automatically Detected.**\n\n"
                "The system could not identify a card with 1.58:1 aspect ratio. "
                "Applying standard estimated optical calibration (~2.5 px/mm). For certified accuracy, "
                "place a standard bank card beside the foot and tick **Manual Card Corner Override**."
            )
            # Create a conservative fallback calibration
            fallback_ppm = 2.4
            from src.calibration import CalibrationData
            calibration_data = CalibrationData(
                pixels_per_mm=fallback_ppm,
                reference_object="Estimated Focal Scale (Uncalibrated)",
                reference_dims_mm=(ref_w_mm, ref_h_mm),
                card_corners=None,
                homography_matrix=None,
                is_perspective_corrected=False,
                calibration_quality="Low",
                quality_notes="Automatic card detection not found; uncalibrated optical estimation applied."
            )

        # Perform physical foot measurement
        measurement = measure_foot(
            foot_mask=primary_foot.mask,
            heel_point=landmarks.heel_point,
            toe_point=landmarks.toe_point,
            calibration_data=calibration_data,
            segmentation_confidence=primary_foot.confidence,
            pixel_width=landmarks.width_pixels,
            image_shape=(h_orig, w_orig)
        )

        # Sizing recommendation
        rec_result = recommender.recommend_size(measurement.length_mm) if recommender else None

        # Build visual overlay
        annotated = FootSegmenter.draw_overlay(img_bgr, primary_foot, color_bgr=(0, 200, 255), alpha=0.35)
        annotated = FootLandmarkExtractor.draw_landmarks(annotated, landmarks, draw_axis=True)
        if calibration_data.card_corners is not None:
            annotated = ReferenceCardCalibrator.draw_calibration(annotated, calibration_data)
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    # ---------------------------------------------------------
    # Card Detection Status Indicator
    # ---------------------------------------------------------
    if calibration_data.card_corners is not None:
        st.success(f"✅ **Reference Card Detected ({calibration_data.reference_object})**: {calibration_data.pixels_per_mm:.2f} px/mm — Perspective Homography Active")
    else:
        st.warning("⚠️ **No Reference Card Detected**: Measurement uses uncalibrated optical estimation. Place a standard bank card or A4 paper beside your foot for certified millimeter accuracy.")

    # ---------------------------------------------------------
    # Visual Output Display (Side by Side)
    # ---------------------------------------------------------
    col_orig, col_annot = st.columns(2)

    with col_orig:
        st.markdown("#### 📷 Original Image")
        st.image(img_rgb, use_container_width=True)

    with col_annot:
        st.markdown("#### 🎯 AI Photogrammetry & Segmentation")
        st.image(annotated_rgb, use_container_width=True)

    st.markdown("---")

    # ---------------------------------------------------------
    # Measurement & Sizing Results
    # ---------------------------------------------------------
    st.markdown("### 📊 Measurement Analysis & Sizing Recommendation")

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)

    with col_m1:
        adj_tag = "<br><span style='color:#D4AF37; font-size:0.75rem;'>✏️ Manually Fine-Tuned</span>" if is_manually_adjusted else ""
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">{measurement.length_cm} cm</div>
            <div class="metric-lbl">Foot Length ({measurement.length_mm} mm){adj_tag}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">{measurement.width_cm} cm</div>
            <div class="metric-lbl">Foot Width ({measurement.width_mm} mm)</div>
        </div>
        """, unsafe_allow_html=True)

    with col_m3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">{primary_foot.confidence * 100:.1f}%</div>
            <div class="metric-lbl">AI Mask Confidence</div>
        </div>
        """, unsafe_allow_html=True)

    with col_m4:
        rel_color = "#2ECC71" if measurement.reliability_status == "Reliable" else ("#F39C12" if "Caution" in measurement.reliability_status else "#E74C3C")
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val" style="color: {rel_color}; font-size: 1.6rem; margin-top: 0.35rem;">{measurement.reliability_status}</div>
            <div class="metric-lbl">Reliability Index</div>
        </div>
        """, unsafe_allow_html=True)

    # Recommendation Box
    if rec_result and rec_result.is_covered:
        st.markdown(f"""
        <div class="size-recommend-box">
            <div style="font-size: 0.9rem; color: #C59B27; text-transform: uppercase; letter-spacing: 0.1em; font-weight: 600;">
                Recommended Peshawari Chappal Size
            </div>
            <div class="size-badge">Size {rec_result.recommended_size} (UK / PK)</div>
            <div style="color: #FDFBF7; font-size: 1.15rem; margin-top: 0.4rem; font-weight: 600;">
                EU Size: {rec_result.size_eu} &bull; US Size: {rec_result.size_us}
            </div>
            <div style="color: #A08C7D; font-size: 0.95rem; margin-top: 0.6rem;">
                Fit Character: <strong>{rec_result.fit_type}</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if rec_result.boundary_warning:
            st.info(f"💡 **Fitting Advice:** {rec_result.boundary_warning}")
    else:
        st.warning(f"⚠️ **Sizing Chart Notice:** {rec_result.label if rec_result else 'Sizing chart unavailable'}")
        if rec_result and rec_result.boundary_warning:
            st.caption(rec_result.boundary_warning)

    # Reliability and Warnings
    if measurement.warnings:
        with st.expander("⚠️ Measurement Quality Flags", expanded=True):
            for w in measurement.warnings:
                st.write(f"- {w}")

    if measurement.rejection_reasons:
        with st.expander("🛑 Measurement Rejection Reasons", expanded=True):
            for r in measurement.rejection_reasons:
                st.error(f"- {r}")

    # Technical Details Expander
    with st.expander("🛠️ Photogrammetry & Landmark Details"):
        st.write(f"- **Calibration Scale:** {measurement.pixels_per_mm:.3f} px/mm")
        st.write(f"- **Calibration Method:** {measurement.calibration_method}")
        st.write(f"- **Card Aspect Ratio:** {calibration_data.reference_dims_mm[0]:.1f} x {calibration_data.reference_dims_mm[1]:.1f} mm")
        st.write(f"- **Foot Longitudinal Angle:** {landmarks.orientation_deg:.1f}°")
        st.write(f"- **Foot Aspect Ratio (L/W):** {landmarks.aspect_ratio:.2f}")
        st.write(f"- **Heel Coordinate:** ({landmarks.heel_point[0]:.1f}, {landmarks.heel_point[1]:.1f})")
        st.write(f"- **Toe Tip Coordinate:** ({landmarks.toe_point[0]:.1f}, {landmarks.toe_point[1]:.1f})")
        st.write(f"- **Landmark Quality Notes:** {landmarks.quality_notes}")

    # Downloadable Report
    report_dict = {
        "project": "Khybersole AI Foot Size Finder",
        "timestamp": "2026-10-18",
        "measured_foot_length_mm": measurement.length_mm,
        "measured_foot_length_cm": measurement.length_cm,
        "measured_foot_width_mm": measurement.width_mm,
        "recommended_chappal_size_pk_uk": rec_result.recommended_size if rec_result else None,
        "recommended_size_eu": rec_result.size_eu if rec_result else None,
        "recommended_size_us": rec_result.size_us if rec_result else None,
        "confidence": primary_foot.confidence,
        "reliability_status": measurement.reliability_status,
        "calibration_scale_ppm": measurement.pixels_per_mm,
        "chart_status": rec_result.chart_status if rec_result else "n/a"
    }

    report_json = json.dumps(report_dict, indent=2)
    st.download_button(
        label="📥 Download Measurement Report (JSON)",
        data=report_json,
        file_name="khybersole_foot_measurement.json",
        mime="application/json"
    )

else:
    # Landing instructions & prompt to upload
    st.markdown("""
    <div style="background: #18120D; border: 1px dashed #3D2E24; border-radius: 12px; padding: 2.5rem; text-align: center; margin: 2rem 0;">
        <h3 style="color: #FDFBF7; margin-bottom: 0.5rem;">Ready to Find Your Peshawari Chappal Size</h3>
        <p style="color: #A08C7D; max-width: 600px; margin: 0 auto 1.5rem auto;">
            Upload a top-down photograph of your foot resting on a flat floor beside a standard bank card or A4 paper.
            Our calibrated AI model will estimate your exact millimeter foot length and recommend the ideal chappal size.
        </p>
    </div>
    """, unsafe_allow_html=True)
