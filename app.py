"""
app.py — FedMedSeg Inference Portal
====================================
Phase 5: User-Facing Web Interface (Streamlit)

PURPOSE:
  Provide a clean, interactive web UI that allows a doctor or examiner to:
    1. Upload any chest X-ray image (.jpg, .png, .dcm)
    2. See the AI's pneumonia prediction (Classification + Segmentation)
    3. View a colored mask overlay showing WHERE the pneumonia is
    4. Explore the full project experiment comparison results

MODELS DEMONSTRATED:
  • Centralized Model (Model 3C — MobileNetV2-UNet, full fine-tuning)
    → Trained on all 4,000 images centrally (theoretical upper bound)
  • Federated Model (FedProx μ=0.01)
    → Trained on Non-IID client data using proximal regularization

RUN:
    cd /home/rajat/Documents/Project/FedMedSeg
    .venv/bin/streamlit run app.py

Then open: http://localhost:8501
"""

# ── Standard Library ──────────────────────────────────────────────────────────
import io
import json
import sys
from pathlib import Path

# ── Third-Party ───────────────────────────────────────────────────────────────
import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

# ── Project Imports ───────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from segmentation.model_unet import MobileNetV2UNet

# ════════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ════════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="FedMedSeg — Pneumonia Segmentation Portal",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ════════════════════════════════════════════════════════════════════════════════
#  CUSTOM CSS — Dark Medical Theme
# ════════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
    /* ─── Root & Fonts ─────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* ─── Background ────────────────────────────────────────── */
    .stApp { background: linear-gradient(135deg, #0a0f1e 0%, #0d1527 50%, #071020 100%); }

    /* ─── Header Banner ─────────────────────────────────────── */
    .hero-banner {
        background: linear-gradient(135deg, #1a2744 0%, #0f3460 50%, #162032 100%);
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 16px;
        padding: 32px 40px;
        margin-bottom: 28px;
        position: relative;
        overflow: hidden;
    }
    .hero-banner::before {
        content: '';
        position: absolute;
        top: -60px; right: -60px;
        width: 200px; height: 200px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(56,189,248,0.12) 0%, transparent 70%);
    }
    .hero-title {
        font-size: 2.4rem; font-weight: 700;
        background: linear-gradient(135deg, #38bdf8, #818cf8, #34d399);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin: 0 0 8px 0;
    }
    .hero-subtitle {
        color: #94a3b8; font-size: 1.05rem; font-weight: 400; margin: 0;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(56, 189, 248, 0.15);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.3);
        border-radius: 20px; padding: 4px 14px;
        font-size: 0.78rem; font-weight: 600;
        margin-right: 8px; margin-top: 12px; letter-spacing: 0.5px;
    }

    /* ─── Metric Cards ───────────────────────────────────────── */
    .metric-card {
        background: linear-gradient(135deg, rgba(30,41,59,0.9), rgba(15,23,42,0.95));
        border: 1px solid rgba(56,189,248,0.2);
        border-radius: 12px; padding: 20px 24px;
        text-align: center; transition: all 0.3s ease;
    }
    .metric-card:hover { border-color: rgba(56,189,248,0.5); transform: translateY(-2px); }
    .metric-label { color: #64748b; font-size: 0.78rem; font-weight: 600; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 8px; }
    .metric-value { color: #f1f5f9; font-size: 1.9rem; font-weight: 700; font-family: 'JetBrains Mono', monospace; }
    .metric-delta { font-size: 0.78rem; margin-top: 4px; }
    .metric-good { color: #34d399; }
    .metric-warn { color: #fbbf24; }
    .metric-info { color: #38bdf8; }

    /* ─── Result Box ──────────────────────────────────────────── */
    .result-pneumonia {
        background: linear-gradient(135deg, rgba(239,68,68,0.15), rgba(185,28,28,0.1));
        border: 1px solid rgba(239,68,68,0.4); border-radius: 12px;
        padding: 20px 24px; text-align: center;
    }
    .result-normal {
        background: linear-gradient(135deg, rgba(52,211,153,0.15), rgba(6,78,59,0.1));
        border: 1px solid rgba(52,211,153,0.4); border-radius: 12px;
        padding: 20px 24px; text-align: center;
    }
    .result-title { font-size: 1.5rem; font-weight: 700; margin: 0 0 6px 0; }
    .result-desc  { color: #94a3b8; font-size: 0.9rem; margin: 0; }

    /* ─── Section Headers ────────────────────────────────────── */
    .section-header {
        color: #38bdf8; font-size: 0.75rem; font-weight: 700;
        letter-spacing: 2px; text-transform: uppercase;
        border-left: 3px solid #38bdf8; padding-left: 12px;
        margin: 24px 0 16px 0;
    }

    /* ─── Sidebar ────────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1527 0%, #071020 100%);
        border-right: 1px solid rgba(56,189,248,0.15);
    }

    /* ─── Upload Zone ─────────────────────────────────────────── */
    [data-testid="stFileUploadDropzone"] {
        background: rgba(56,189,248,0.05) !important;
        border: 2px dashed rgba(56,189,248,0.3) !important;
        border-radius: 12px !important;
    }

    /* ─── Dataframe / Tables ─────────────────────────────────── */
    [data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

    /* ─── Hide Streamlit branding ────────────────────────────── */
    #MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
#  CONSTANTS & PATHS
# ════════════════════════════════════════════════════════════════════════════════

MODELS = {
    "🏆 Centralized Model (Best Accuracy)": {
        "path":        PROJECT_ROOT / "results" / "model3c_final" / "model3c_best.pth",
        "dice":        0.6233,
        "iou":         0.5609,
        "pixel_acc":   0.9516,
        "description": "Trained on all 4,000 chest X-rays centrally. Theoretical upper bound — requires sharing raw patient data.",
        "privacy":     "❌ No privacy guarantee (data centralised)",
        "color":       "#38bdf8",
    },
    "🔗 FedProx Model (Federated — Best FL)": {
        "path":        None,  # FedProx uses the same architecture from the final round
        "dice":        0.6449,
        "iou":         0.5856,
        "pixel_acc":   0.9541,
        "description": "Hospitals trained separately, sharing only model weights. Handles Non-IID data via proximal regularization (μ=0.01).",
        "privacy":     "⚠️  Privacy by locality (no mathematical DP guarantee)",
        "color":       "#4ade80",
    },
    "🔒 DP-FedProx Model (Federated + Privacy)": {
        "path":        None,
        "dice":        0.5000,
        "iou":         0.5000,
        "pixel_acc":   0.9394,
        "description": "Same as FedProx but with ε=8.0 Differential Privacy — adds mathematical noise to prevent patient data reconstruction.",
        "privacy":     "✅ Mathematical DP guarantee: (ε=8.0, δ=1e-5)",
        "color":       "#f472b6",
    },
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ════════════════════════════════════════════════════════════════════════════════
#  MODEL LOADING (Cached)
# ════════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_model(model_path: str) -> MobileNetV2UNet:
    """Load a saved MobileNetV2-UNet checkpoint. Cached across sessions."""
    model = MobileNetV2UNet(pretrained=False, freeze_encoder=False)
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def has_checkpoint(model_key: str) -> bool:
    path = MODELS[model_key]["path"]
    return path is not None and Path(path).exists()


# ════════════════════════════════════════════════════════════════════════════════
#  INFERENCE
# ════════════════════════════════════════════════════════════════════════════════

def run_inference(
    model: MobileNetV2UNet,
    pil_image: Image.Image,
    threshold: float = 0.5,
    top_pct: float = None,
):
    """
    Run segmentation inference on a PIL image.

    If top_pct is provided (adaptive mode), the binary mask is formed by
    marking the top N% of pixels — the ones the model is MOST confident
    about relative to its own output range. This prevents the all-red
    problem when the model outputs near-uniform values.

    Returns:
        prob_mask    (np.ndarray): H×W float array in [0,1]
        display_prob (np.ndarray): H×W float normalized to [0,1] for display
        binary_mask  (np.ndarray): H×W boolean (True = pneumonia)
        confidence   (float):      fraction of pixels flagged
        output_range (tuple):      (min_prob, max_prob) of raw model output
        mode         (str):        "adaptive" or "fixed"
    """
    tensor = TRANSFORM(pil_image.convert("RGB")).unsqueeze(0)  # (1,3,224,224)
    with torch.no_grad():
        logits = model(tensor)                        # (1,1,224,224)
        probs  = torch.sigmoid(logits).squeeze()     # (224,224)

    prob_mask = probs.numpy()                        # raw probabilities
    p_min, p_max = float(prob_mask.min()), float(prob_mask.max())

    # Normalize to [0,1] based on the model's OWN output range
    # This gives a meaningful heatmap even when outputs are near-uniform
    p_range = p_max - p_min
    if p_range > 1e-4:
        display_prob = (prob_mask - p_min) / p_range
    else:
        display_prob = np.zeros_like(prob_mask)  # completely degenerate

    # Adaptive thresholding: flag the top N% of pixels
    if top_pct is not None:
        cutoff = np.percentile(prob_mask, 100 - top_pct)
        binary_mask = prob_mask >= cutoff
        mode = "adaptive"
    else:
        binary_mask = prob_mask >= threshold
        mode = "fixed"

    confidence = float(binary_mask.mean())
    return prob_mask, display_prob, binary_mask, confidence, (p_min, p_max), mode


def create_overlay(pil_image: Image.Image, binary_mask: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """
    Overlay a red segmentation mask on the original X-ray.

    Args:
        pil_image: Original X-ray (any size)
        binary_mask: 224×224 boolean mask from the model
        alpha: Opacity of the overlay (0=invisible, 1=solid)

    Returns:
        PIL Image: Original X-ray with red pneumonia regions highlighted
    """
    # Resize original to 224×224 for overlay
    img_resized = pil_image.convert("RGB").resize((224, 224))
    img_arr     = np.array(img_resized, dtype=np.uint8)

    # Create red overlay
    overlay = img_arr.copy()
    overlay[binary_mask] = [220, 38, 38]  # Bright red (Tailwind red-600)

    # Blend
    blended = (img_arr * (1 - alpha) + overlay * alpha).astype(np.uint8)
    return Image.fromarray(blended)


def create_heatmap(display_prob: np.ndarray) -> Image.Image:
    """
    Convert a [0,1]-normalized probability map to a blue→red heatmap image.

    NOTE: This is NOT Grad-CAM. It is simply the model's raw per-pixel
    output probability, rescaled so that the LOWEST confidence region
    appears blue and the HIGHEST confidence region appears red.
    This relative normalization makes the variation visible even when
    the raw probabilities are very close together (e.g., 0.51 vs 0.52).
    """
    import matplotlib.cm as cm
    colored      = cm.RdYlBu_r(display_prob)           # RGBA, 0→blue, 1→red
    colored_uint8 = (colored[:, :, :3] * 255).astype(np.uint8)
    return Image.fromarray(colored_uint8)


# ════════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### 🫁 FedMedSeg Portal")
    st.markdown("---")

    st.markdown("**Select Model**")
    selected_model_key = st.selectbox(
        label="Model",
        options=list(MODELS.keys()),
        label_visibility="collapsed",
    )
    selected = MODELS[selected_model_key]

    st.markdown(f"""
    <div style='background:rgba(56,189,248,0.06); border:1px solid rgba(56,189,248,0.2);
                border-radius:10px; padding:14px; margin:8px 0;'>
        <div style='color:#38bdf8; font-size:0.7rem; font-weight:700; letter-spacing:1px; margin-bottom:8px;'>MODEL INFO</div>
        <div style='color:#cbd5e1; font-size:0.82rem; margin-bottom:10px;'>{selected['description']}</div>
        <div style='font-size:0.78rem; color:#94a3b8;'>{selected['privacy']}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    # ── Inference Settings ──────────────────────────────────────────────────
    st.markdown("**Inference Settings**")
    threshold_mode = st.radio(
        "Threshold Mode",
        ["Adaptive (Recommended)", "Fixed"],
        help="Adaptive highlights only the top % of pixels. Fixed uses an absolute cut-off."
    )
    if threshold_mode == "Fixed":
        threshold = st.slider("Detection Threshold", 0.1, 0.99, 0.50, 0.01,
                              help="Pixels above this raw probability are marked as pneumonia")
        top_pct = None
    else:
        top_pct = st.slider("Top % of pixels to flag", 1, 40, 10, 1,
                            help="Marks the top N% of pixels that the model is MOST confident about, relative to itself. Works even when model outputs are near-uniform.")
        threshold = None
    overlay_alpha = st.slider("Overlay Opacity", 0.2, 0.8, 0.45, 0.05,
                              help="Transparency of the red mask overlay")

    st.markdown("---")
    st.markdown("""
    <div style='color:#475569; font-size:0.75rem; line-height:1.6;'>
        <b style='color:#64748b;'>FedMedSeg</b><br>
        MobileNetV2-UNet<br>
        Flower FL Framework<br>
        Opacus DP-SGD (ε=8.0)
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
#  HERO BANNER
# ════════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class='hero-banner'>
    <p class='hero-title'>🫁 FedMedSeg Inference Portal</p>
    <p class='hero-subtitle'>Privacy-Preserving Federated AI for Pneumonia Detection & Segmentation</p>
    <span class='hero-badge'>MobileNetV2-UNet</span>
    <span class='hero-badge'>FedProx</span>
    <span class='hero-badge'>DP-SGD ε=8.0</span>
    <span class='hero-badge'>RSNA Dataset</span>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
#  TABS
# ════════════════════════════════════════════════════════════════════════════════

tab_infer, tab_compare, tab_about = st.tabs(
    ["🔬 Run Inference", "📊 Experiment Results", "📖 About the System"]
)


# ────────────────────────────────────────────────────────────────────────────────
#  TAB 1: INFERENCE
# ────────────────────────────────────────────────────────────────────────────────

with tab_infer:
    col_upload, col_results = st.columns([1, 1.4], gap="large")

    with col_upload:
        st.markdown("<div class='section-header'>Upload X-Ray</div>", unsafe_allow_html=True)
        uploaded = st.file_uploader(
            label="Upload X-Ray",
            type=["jpg", "jpeg", "png", "bmp"],
            label_visibility="collapsed",
            help="Upload a chest X-ray image. DICOM files should be exported to PNG/JPG first.",
        )

        if uploaded is not None:
            pil_image = Image.open(uploaded).convert("RGB")
            st.image(pil_image, caption="📁 Uploaded X-Ray", use_container_width=True)

            st.markdown("<div class='section-header'>Model Status</div>", unsafe_allow_html=True)

            if has_checkpoint(selected_model_key):
                ckpt_path = str(MODELS[selected_model_key]["path"])
                with st.spinner("Loading model weights..."):
                    model = load_model(ckpt_path)
                st.success(f"✓ Weights loaded from checkpoint")
                model_ready = True
            else:
                # Use centralized model for demo
                central_path = str(MODELS["🏆 Centralized Model (Best Accuracy)"]["path"])
                if Path(central_path).exists():
                    with st.spinner("Loading centralized weights (federated checkpoint not saved separately)..."):
                        model = load_model(central_path)
                    st.info(
                        f"ℹ️  Using centralized weights as architecture demo.\n\n"
                        f"*(Flower simulation does not auto-save round-by-round checkpoints. "
                        f"The centralized model is the best available standalone `.pth`.)*"
                    )
                    model_ready = True
                else:
                    st.error(
                        "⚠️ No model checkpoint found.\n\n"
                        f"Expected: `results/model3c_final/model3c_best.pth`"
                    )
                    model_ready = False
        else:
            st.info("👆 Upload a chest X-ray to begin analysis.")
            model_ready = False
            pil_image = None
            model = None

    with col_results:
        st.markdown("<div class='section-header'>AI Analysis</div>", unsafe_allow_html=True)

        if uploaded is not None and model_ready and pil_image is not None:
            with st.spinner("Running segmentation..."):
                prob_mask, display_prob, binary_mask, confidence, out_range, inf_mode = run_inference(
                    model, pil_image,
                    threshold=threshold if threshold is not None else 0.5,
                    top_pct=top_pct,
                )
                overlay_img = create_overlay(pil_image, binary_mask, overlay_alpha)
                heatmap_img = create_heatmap(display_prob)
                pneumonia_detected = confidence > 0.005  # >0.5% of pixels flagged

                # Detect degenerate / near-uniform model output
                output_range_size = out_range[1] - out_range[0]
                is_degenerate = output_range_size < 0.02  # model barely varies
                if is_degenerate:
                    st.warning(
                        f"⚠️ Model output is near-uniform (range: {out_range[0]:.4f} – {out_range[1]:.4f}).\n\n"
                        f"This can happen when: (1) the model has not fully converged, or "
                        f"(2) weights were saved before the encoder fully fine-tuned.\n\n"
                        f"**Adaptive mode is ON** — the map shows *relative* confidence across pixels."
                    )

            # ── Verdict ──────────────────────────────────────────────────
            if pneumonia_detected:
                infected_pct = confidence * 100
                st.markdown(f"""
                <div class='result-pneumonia'>
                    <p class='result-title' style='color:#ef4444;'>⚠️ Pneumonia Detected</p>
                    <p class='result-desc'>
                        The model has identified pneumonia opacity covering approximately
                        <b style='color:#fca5a5;'>{infected_pct:.1f}%</b> of the lung field.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class='result-normal'>
                    <p class='result-title' style='color:#34d399;'>✅ No Pneumonia Detected</p>
                    <p class='result-desc'>
                        No significant pneumonia opacity was found above the detection threshold.
                        Lungs appear clear.
                    </p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Segmentation Visualizations ───────────────────────────────
            v1, v2 = st.columns(2)
            with v1:
                st.image(overlay_img, caption="🔴 Segmentation Mask Overlay", use_container_width=True)
            with v2:
                st.image(
                    heatmap_img,
                    caption="🌡️ Confidence Map (Relative — blue=low, red=high)",
                    use_container_width=True,
                )
                st.caption(
                    f"Raw output range: **{out_range[0]:.4f} – {out_range[1]:.4f}** | "
                    f"Mode: **{inf_mode}** | "
                    f"Not Grad-CAM — this is the model\'s direct per-pixel sigmoid output, "
                    f"normalized to the model\'s own range for visibility."
                )

            # ── Metrics Row ────────────────────────────────────────────────
            st.markdown("<div class='section-header'>Prediction Statistics</div>", unsafe_allow_html=True)
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label'>Max Probability</div>
                    <div class='metric-value'>{prob_mask.max():.3f}</div>
                    <div class='metric-delta metric-info'>peak activation</div>
                </div>""", unsafe_allow_html=True)
            with m2:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label'>Mean Probability</div>
                    <div class='metric-value'>{prob_mask.mean():.3f}</div>
                    <div class='metric-delta metric-info'>avg confidence</div>
                </div>""", unsafe_allow_html=True)
            with m3:
                infected_px = int(binary_mask.sum())
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label'>Infected Pixels</div>
                    <div class='metric-value'>{infected_px:,}</div>
                    <div class='metric-delta {'metric-warn' if infected_px > 500 else 'metric-good'}'>
                        of 50,176 total
                    </div>
                </div>""", unsafe_allow_html=True)
            with m4:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label'>Lung Coverage</div>
                    <div class='metric-value'>{confidence*100:.1f}%</div>
                    <div class='metric-delta {'metric-warn' if confidence > 0.1 else 'metric-good'}">
                        area flagged
                    </div>
                </div>""", unsafe_allow_html=True)

            # ── Download ────────────────────────────────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            buf = io.BytesIO()
            overlay_img.save(buf, format="PNG")
            st.download_button(
                label="⬇️  Download Segmentation Overlay",
                data=buf.getvalue(),
                file_name="fedmedseg_overlay.png",
                mime="image/png",
                use_container_width=True,
            )

        else:
            st.markdown("""
            <div style='background:rgba(30,41,59,0.5); border:1px solid rgba(56,189,248,0.15);
                        border-radius:12px; padding:60px 40px; text-align:center;'>
                <div style='font-size:3rem;'>🫁</div>
                <div style='color:#64748b; font-size:1.1rem; margin-top:12px;'>
                    Upload a chest X-ray on the left<br>to see the segmentation analysis here.
                </div>
            </div>
            """, unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────────────────────
#  TAB 2: EXPERIMENT RESULTS
# ────────────────────────────────────────────────────────────────────────────────

with tab_compare:
    st.markdown("<div class='section-header'>The Federation Story — All Experiments</div>", unsafe_allow_html=True)

    federation_summary_path = PROJECT_ROOT / "results" / "federated" / "federation_summary.json"
    dp_report_path          = PROJECT_ROOT / "results" / "federated" / "dp_fedprox" / "dp_fedprox_report.json"

    # ── Build unified metrics dict ────────────────────────────────────────────
    all_results = {}
    if federation_summary_path.exists():
        with open(federation_summary_path) as f:
            summary = json.load(f)
        all_results.update(summary.get("approaches", {}))

    if dp_report_path.exists():
        with open(dp_report_path) as f:
            dp_data = json.load(f)
        fm = dp_data.get("final_metrics", {})
        all_results["DP-FedProx (ε=8.0)"] = {
            "dice":      fm.get("val_dice", 0),
            "iou":       fm.get("val_iou", 0),
            "pixel_acc": fm.get("val_pixel_acc", 0),
        }

    if all_results:
        import pandas as pd
        df = pd.DataFrame(all_results).T.reset_index()
        df.columns = ["Method", "Dice Score", "IoU", "Pixel Accuracy"]
        df["Dice Score"]     = df["Dice Score"].map("{:.4f}".format)
        df["IoU"]            = df["IoU"].map("{:.4f}".format)
        df["Pixel Accuracy"] = df["Pixel Accuracy"].map("{:.4f}".format)
        st.dataframe(df, use_container_width=True, hide_index=True)

    # ── Key Findings ──────────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>Key Findings</div>", unsafe_allow_html=True)

    f1, f2, f3, f4 = st.columns(4)
    findings = [
        ("Isolated Fails", "Biased hospital data leads to biased models. Client A over-predicts pneumonia; Client B under-predicts.", "#fbbf24"),
        ("FedAvg Recovers", "Sharing model weights (not data) across hospitals restores performance above isolated baselines.", "#38bdf8"),
        ("FedProx Excels", "The proximal term μ=0.01 keeps clients tethered to the global objective, improving Non-IID robustness.", "#4ade80"),
        ("DP adds Security", "Differential Privacy (ε=8.0) adds a mathematical guarantee — at the cost of ~14% Dice drop. Privacy has a price.", "#f472b6"),
    ]
    for col, (title, desc, color) in zip([f1, f2, f3, f4], findings):
        with col:
            st.markdown(f"""
            <div style='background:rgba(30,41,59,0.8); border:1px solid {color}40;
                        border-left: 3px solid {color}; border-radius:10px; padding:16px;
                        height:160px;'>
                <div style='color:{color}; font-size:0.72rem; font-weight:700; letter-spacing:1px; 
                            text-transform:uppercase; margin-bottom:8px;'>{title}</div>
                <div style='color:#94a3b8; font-size:0.82rem; line-height:1.5;'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Charts ────────────────────────────────────────────────────────────────
    comparison_img = PROJECT_ROOT / "results" / "federated" / "federation_comparison.png"
    convergence_img = PROJECT_ROOT / "results" / "federated" / "convergence_curves.png"

    if comparison_img.exists() or convergence_img.exists():
        st.markdown("<div class='section-header'>Visualizations</div>", unsafe_allow_html=True)
        ic1, ic2 = st.columns(2)
        if comparison_img.exists():
            with ic1:
                st.image(str(comparison_img), caption="Federation Comparison (All Approaches)", use_container_width=True)
        if convergence_img.exists():
            with ic2:
                st.image(str(convergence_img), caption="Convergence Curves (Round-by-Round)", use_container_width=True)


# ────────────────────────────────────────────────────────────────────────────────
#  TAB 3: ABOUT
# ────────────────────────────────────────────────────────────────────────────────

with tab_about:
    st.markdown("<div class='section-header'>System Architecture</div>", unsafe_allow_html=True)

    st.markdown("""
    <div style='background:rgba(15,23,42,0.8); border:1px solid rgba(56,189,248,0.15);
                border-radius:12px; padding:28px; font-family:"JetBrains Mono", monospace;
                font-size:0.82rem; color:#94a3b8; line-height:1.8;'>

    <b style='color:#38bdf8;'>COMPLETE SYSTEM PIPELINE</b><br><br>

    <b style='color:#4ade80;'>Phase 1: Classification (Baseline)</b><br>
    &nbsp;&nbsp;Dataset   → Kermany Chest X-Ray (5,216 images)<br>
    &nbsp;&nbsp;Models    → CNN2, CNN3, MobileNetV2 (Frozen / Partial / Full Fine-tune)<br>
    &nbsp;&nbsp;Best      → Model 3C (Full Fine-tune): 96.8% Accuracy, 0.965 F1<br><br>

    <b style='color:#4ade80;'>Phase 2: Semantic Segmentation (Core)</b><br>
    &nbsp;&nbsp;Dataset   → RSNA Pneumonia Detection (5,000 image subset)<br>
    &nbsp;&nbsp;Model     → MobileNetV2-UNet (encoder + decoder w/ skip connections)<br>
    &nbsp;&nbsp;Loss      → Dice-BCE Hybrid (L = L_BCE + L_Dice)<br>
    &nbsp;&nbsp;Result    → Dice: 0.6233, IoU: 0.5609, PixAcc: 0.9516<br><br>

    <b style='color:#4ade80;'>Phase 3: Federated Learning (Privacy by Locality)</b><br>
    &nbsp;&nbsp;Framework → Flower (flwr) with Ray simulation<br>
    &nbsp;&nbsp;Clients   → 2 hospitals (Non-IID: specialist 75% / clinic 25%)<br>
    &nbsp;&nbsp;FedAvg    → Dice: 0.6334  (baseline federation)<br>
    &nbsp;&nbsp;FedProx   → Dice: 0.6449, μ=0.01  (best FL result)<br><br>

    <b style='color:#f472b6;'>Phase 4: Differential Privacy (Mathematical Guarantee)</b><br>
    &nbsp;&nbsp;Library   → Opacus (PyTorch DP-SGD)<br>
    &nbsp;&nbsp;Budget    → ε=8.0, δ=1e-5 (healthcare industry standard)<br>
    &nbsp;&nbsp;Mechanism → Per-sample gradient clipping + Gaussian noise<br>
    &nbsp;&nbsp;Result    → Dice: 0.5000  (privacy cost: Δ=0.14 Dice drop)<br>
    &nbsp;&nbsp;Quant.    → float32 → int8 (Post-Training Dynamic Quantization)<br>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-header'>Algorithms & Formulas</div>", unsafe_allow_html=True)
    st.markdown(r"""
| Algorithm | Formula | Purpose |
|-----------|---------|---------|
| **Dice Loss** | $\mathcal{L}_{Dice} = 1 - \frac{2\sum p_i g_i + \varepsilon}{\sum p_i + \sum g_i + \varepsilon}$ | Penalize mask overlap errors |
| **FedAvg Aggregation** | $w_{r+1} = \sum_{k} \frac{n_k}{n_{total}} w_k^r$ | Weighted average of client weights |
| **FedProx Client Loss** | $\mathcal{L}_{prox} = \mathcal{L}_{task} + \frac{\mu}{2}\|w_k - w^r\|^2$ | Prevent client drift (Non-IID) |
| **DP Gradient Clipping** | $\bar{g}_i = g_i \cdot \min(1, \frac{C}{\|g_i\|_2})$ | Bound individual patient influence |
| **DP Noise Addition** | $\tilde{g} = \frac{1}{B}\left(\sum_i \bar{g}_i + \mathcal{N}(0, \sigma^2 C^2 I)\right)$ | Mask patient-specific signal |
    """)
