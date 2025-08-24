# streamlit_app.py
import os
import io
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt

# Optional SHAP import (explanations)
try:
    import shap  # pip install shap
    _HAS_SHAP = True
except Exception:
    _HAS_SHAP = False

# ============================================
# Theming and Page Config
# ============================================
st.set_page_config(
    page_title="Prolonged Hospital Stay Risk Calculator (SCI)",
    layout="wide",
    page_icon="🧠",
    initial_sidebar_state="expanded",
)


# ======= Fancy UI: global styles & hero banner =======
st.markdown(
    """
    <style>
      /* Base typography & max width */
      html, body, [class*="css"] { font-family: Inter, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; }
      .block-container { max-width: 1200px; padding-top: 0.5rem; }

      /* Hero banner */
      .app-hero {
        background: linear-gradient(135deg, #0ea5e9 0%, #22c55e 100%);
        border-radius: 18px; color: white; padding: 22px 26px; margin-bottom: 18px;
        box-shadow: 0 10px 30px rgba(34,197,94,0.20);
      }
      .app-hero h1 { margin: 0; font-size: 1.6rem; font-weight: 700; letter-spacing: 0.2px; }
      .app-hero p { margin: 6px 0 0; opacity: 0.95; }

      /* Cards */
      .card {
        border: 1px solid rgba(0,0,0,0.06);
        border-radius: 14px;
        padding: 16px;
        margin: 8px 0 16px;
        background: #ffffff;
        box-shadow: 0 4px 14px rgba(0,0,0,0.04);
      }
      .muted { color: #64748b; }

      /* Pills (badges) */
      .pill { display:inline-block; padding: 4px 10px; border-radius: 999px; font-size: 0.85rem; font-weight: 600; }
      .pill-green { background: #ecfdf5; color: #065f46; border: 1px solid #a7f3d0; }
      .pill-amber { background: #fffbeb; color: #92400e; border: 1px solid #fde68a; }
      .pill-red { background: #fef2f2; color: #991b1b; border: 1px solid #fecaca; }

      /* Buttons */
      .stButton>button {
        border-radius: 10px; background-color: #0ea5e9; color: white; font-weight: 600;
        transition: 0.25s; border: 0;
      }
      .stButton>button:hover { background-color: #0c8fc6; }

      /* Sidebar */
      section[data-testid="stSidebar"] {
        background: #f7fafc; border-right: 1px solid #e5e7eb;
      }

      /* Inputs labels */
      .stRadio > label, .stSelectbox > label, .stNumberInput > label { font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="app-hero">
      <h1>🧠 Prolonged Hospital Stay Risk Calculator (Penetrating Spinal Cord Injury)</h1>
      <p>Estimate risk, flag intervention thresholds, and explain drivers with Visualization.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

HUMAN_MAP = {
    'num__totalventdays': 'Total ventilator days',
    'num__iss': 'Injury Severity Score (ISS)',
    'num__ageyears': 'Age (years)',
    'num__hospitalarrivalhrs': 'Time from injury to hospital arrival (hours)',

    'cat__hospdischargedisposition_recoded_Deceased': 'Hospital discharge: Deceased',
    'cat__hospdischargedisposition_recoded_Home': 'Hospital discharge: Home',
    'cat__hospdischargedisposition_recoded_Rehab/Short_or_LongTermCare': 'Hospital discharge: Rehab/Short- or Long-Term Care',
    'cat__hospdischargedisposition_recoded_OtherInstitution': 'Hospital discharge: Other institution',

    'cat__eddischargedisposition_recoded_Deceased': 'ED discharge: Deceased',
    'cat__eddischargedisposition_recoded_Operating Room': 'ED discharge: Operating Room',
    'cat__eddischargedisposition_recoded_ICU': 'ED discharge: ICU',
    'cat__eddischargedisposition_recoded_Hosp Admission': 'ED discharge: Hospital admission',
    'cat__eddischargedisposition_recoded_Home': 'ED discharge: Home',

    'cat__pressureulcer_No': 'Pressure ulcer: No',
    'cat__pressureulcer_Yes': 'Pressure ulcer: Yes',

    'cat__primarymethodpayment_recoded_Private Insurance': 'Primary payment: Private Insurance',
    'cat__primarymethodpayment_recoded_Self-Pay': 'Primary payment: Self-Pay',
    'cat__primarymethodpayment_recoded_Medicaid/Medicare': 'Primary payment: Medicaid/Medicare',
    'cat__primarymethodpayment_recoded_Other': 'Primary payment: Other',
    'cat__primarymethodpayment_recoded_Unknown': 'Primary payment: Unknown',

    'cat__mechanism_of_injury_Firearm': 'Mechanism of injury: Firearm',
    'cat__mechanism_of_injury_Cut/Pierce': 'Mechanism of injury: Cut/Pierce',
    'cat__mechanism_of_injury_Bite/Sting': 'Mechanism of injury: Bite/Sting',

    'cat__icu_admission_No': 'ICU admission: No',
    'cat__icu_admission_Yes': 'ICU admission: Yes',

    'cat__dvt_No': 'Deep-vein thrombosis (DVT): No',
    'cat__dvt_Yes': 'Deep-vein thrombosis (DVT): Yes',

    'cat__neurologic_level_injury_Cervical': 'Neurologic level of injury: Cervical',
    'cat__neurologic_level_injury_Thoracic': 'Neurologic level of injury: Thoracic',
    'cat__neurologic_level_injury_Lumbar': 'Neurologic level of injury: Lumbar',

    'cat__bedsize_recoded_> 600': 'Hospital bed size: >600',
    'cat__bedsize_recoded_401-600': 'Hospital bed size: 401–600',
    'cat__bedsize_recoded_201-400': 'Hospital bed size: 201–400',
    'cat__bedsize_recoded_<= 200': 'Hospital bed size: ≤200',

    # Other features from your earlier mapping
    'cat__congenital_anomalies_Unknown': 'Congenital anomalies: Unknown',
    'cat__asia_impairment_scale_B/C/D': 'AIS scale: B/C/D',
    'cat__current_smoker_No': 'Current smoker: No',
    'cat__mental_personality_disorder_No': 'Mental/personality disorder: No',
    'cat__cauti_No': 'CAUTI: No',
    'cat__Race_White': 'Race: White',
}

def _map_human_readable(raw_feature_names: List[str]) -> List[str]:
    """
    Convert raw model-space feature names to human-readable ones using HUMAN_MAP.
    Falls back to the raw name if not found.
    """
    return [HUMAN_MAP.get(name, name) for name in raw_feature_names]
# ============================================
# Hand‑curated categorical choices (extend as needed)
# ============================================
PREFERRED_CATS: Dict[str, List[str]] = {
    "hospdischargedisposition_recoded": [
        "Rehab/Short_or_LongTermCare",
        "Unknown",
        "Home",
        "Deceased",
        "OtherInstitution",
    ],
    "eddischargedisposition_recoded": [
        "ICU",
        "Operating Room",
        "Deceased",
        "Hosp Admission",
        "Other",
        "Transferred",
        "Home",
    ],
    "pressureulcer": ["No", "Yes"],
    "primarymethodpayment_recoded": [
        "Medicaid/Medicare",
        "Self-Pay",
        "Private Insurance",
        "Other",
        "Unknown",
    ],
    "mechanism_of_injury": ["Firearm", "Cut/Pierce", "Bite/Sting"],
    "icu_admission": ["No", "Yes"],
    "dvt": ["No", "Yes"],
    "neurologic_level_injury": ["Thoracic", "Cervical", "Lumbar"],
    "bedsize_recoded": ["> 600", "401-600", "201-400", "<= 200"],
}

# Friendly display labels for confusing raw values
DISPLAY_LABELS: Dict[str, Dict[str, str]] = {
    "hospdischargedisposition_recoded": {
        "Rehab/Short_or_LongTermCare": "Rehabilitation / Long-Term Care",
        "Unknown": "Unknown / Not Recorded",
        "Home": "Home",
        "Deceased": "Deceased",
        "OtherInstitution": "Other Institution (SNF, Facility)",
    },
    "eddischargedisposition_recoded": {
        "ICU": "Admitted to ICU",
        "Operating Room": "Taken to Operating Room",
        "Deceased": "Deceased in ED",
        "Hosp Admission": "Admitted to Hospital (non-ICU)",
        "Other": "Other",
        "Transferred": "Transferred to Another Facility",
        "Home": "Discharged Home",
    },
    "pressureulcer": {"No": "No Pressure Ulcer", "Yes": "Has Pressure Ulcer"},
    "primarymethodpayment_recoded": {
        "Medicaid/Medicare": "Medicaid / Medicare",
        "Self-Pay": "Self-Pay / Uninsured",
        "Private Insurance": "Private Insurance",
        "Other": "Other",
        "Unknown": "Unknown / Not Recorded",
    },
    "mechanism_of_injury": {
        "Firearm": "Firearm Injury",
        "Cut/Pierce": "Cut / Pierce Injury",
        "Bite/Sting": "Bite / Sting Injury",
    },
    "icu_admission": {"No": "No ICU Admission", "Yes": "ICU Admission"},
    "dvt": {"No": "No DVT", "Yes": "Has DVT"},
    "neurologic_level_injury": {
        "Thoracic": "Thoracic Level Injury",
        "Cervical": "Cervical Level Injury",
        "Lumbar": "Lumbar Level Injury",
    },
    "bedsize_recoded": {
        "> 600": ">600 Beds",
        "401-600": "401–600 Beds",
        "201-400": "201–400 Beds",
        "<= 200": "≤200 Beds",
    },
}
FRIENDLY_FEATURE_NAMES = {
    "iss": "Injury Severity Score (ISS)",
    "ageyears": "Age (years)",
    "totalventdays": "Total Ventilator Days",
    "hospdischargedisposition_recoded": "Hospital Discharge Disposition",
    "eddischargedisposition_recoded": "ED Discharge Disposition",
    "pressureulcer": "Pressure Ulcer",
    "primarymethodpayment_recoded": "Primary Method of Payment",
    "mechanism_of_injury": "Mechanism of Injury",
    "icu_admission": "ICU Admission",
    "dvt": "Deep Vein Thrombosis (DVT)",
    "neurologic_level_injury": "Neurologic Level of Injury",
    "bedsize_recoded": "Hospital Bed Size",
}
# Numeric feature hints (for nicer UI; pipeline still decides types)
NUMERIC_HINTS = {
    "totalventdays": {"min_value": 0.0, "max_value": None, "step": 1.0, "help": "Total ventilator days (non‑negative)."},
    "iss":           {"min_value": 0.0, "max_value": 75.0, "step": 1.0, "help": "Injury Severity Score (0–75)."},
    "ageyears":      {"min_value": 0.0, "max_value": None, "step": 1.0, "help": "Age in years."},
}

# ============================================
# Utilities
# ============================================
@st.cache_resource
def load_model(pkl_path: str = "final_model_xgb_reduced.pkl"):
    model = joblib.load(pkl_path)
    return model


def get_feature_schema(model) -> Tuple[List[str], List[str], Dict[str, List[str]]]:
    """Recover numeric & categorical column names and categories from the fitted pipeline.
    Returns (num_cols, cat_cols, cat_map). cat_map is overridden by PREFERRED_CATS if provided.
    """
    preproc = model.named_steps['preproc']

    # Robustly extract column lists from ColumnTransformer without converting to dict()
    num_cols: List[str] = []
    cat_cols: List[str] = []
    for name, trans, cols in preproc.transformers:
        if name == 'num':
            try:
                num_cols = list(cols)
            except Exception:
                num_cols = []
        elif name == 'cat':
            try:
                cat_cols = list(cols)
            except Exception:
                cat_cols = []

    # Derive categories from a fitted OneHotEncoder if present
    cat_map: Dict[str, List[str]] = {}
    try:
        cat_pipe = preproc.named_transformers_['cat']
        ohe = cat_pipe.named_steps.get('ohe')
        if hasattr(ohe, 'categories_') and cat_cols:
            for col_name, cats in zip(cat_cols, ohe.categories_):
                cat_map[col_name] = [str(c) for c in cats]
    except Exception:
        pass

    # Override with curated lists
    for col, opts in PREFERRED_CATS.items():
        if not cat_cols or col in cat_cols:
            cat_map[col] = list(opts)

    return num_cols, cat_cols, cat_map


def _display_name(col: str, raw_val: Any) -> str:
    if raw_val is None:
        return ""
    if isinstance(raw_val, str) and raw_val.strip() == "(leave blank)":
        return raw_val
    return DISPLAY_LABELS.get(col, {}).get(str(raw_val), str(raw_val))


def ensure_dataframe(feature_order: List[str], record: Dict[str, Any]) -> pd.DataFrame:
    row = {col: record.get(col, np.nan) for col in feature_order}
    return pd.DataFrame([row], columns=feature_order)


def predict_with_threshold(model, X: pd.DataFrame, threshold: float = 0.5):
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)
    return proba, pred


def render_schema_table(num_cols: List[str], cat_cols: List[str]):
    st.markdown("### Expected Feature Schema")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Numeric features**")
        if num_cols:
            joined_num = "\n".join([str(x) for x in num_cols])
            st.code(joined_num)
        else:
            st.write("None found.")
    with c2:
        st.markdown("**Categorical features**")
        if cat_cols:
            joined_cat = "\n".join([str(x) for x in cat_cols])
            st.code(joined_cat)
        else:
            st.write("None found.")



def _transform_to_model_space(model, X_df: pd.DataFrame):
    """Apply preprocessor to X_df and return (X_processed, feature_names)."""
    preproc = model.named_steps['preproc']
    X_proc = preproc.transform(X_df)
    try:
        feature_names = preproc.get_feature_names_out()
    except Exception:
        feature_names = [f"feature_{i}" for i in range(X_proc.shape[1])]
    return X_proc, feature_names

def _friendly_feature_names(raw_feature_names: List[str]) -> List[str]:
    """
    Map model-space feature names (e.g., 'cat__hospdischargedisposition_recoded_Rehab/Short_or_LongTermCare')
    to human-friendly labels (e.g., 'Hospital Discharge Disposition = Rehabilitation / Long-Term Care').

    Rules:
    - Strip ColumnTransformer prefixes like 'num__' or 'cat__'
    - For OneHotEncoder outputs: '<col>_<category>' -> '<Friendly Col> = <Friendly Category>'
    - For numeric features: '<col>' -> '<Friendly Col>'
    """
    friendly = []
    for feat in raw_feature_names:
        name = str(feat)

        # Remove ColumnTransformer prefixes (e.g., 'num__', 'cat__')
        if "__" in name:
            _, name = name.split("__", 1)

        # If it's an OHE feature, it will look like '<col>_<category...>'
        parts = name.split("_", 1)
        col_key = parts[0]
        if col_key in FRIENDLY_FEATURE_NAMES and len(parts) == 2:
            cat_val_raw = parts[1]
            pretty_col = FRIENDLY_FEATURE_NAMES.get(col_key, col_key)
            # Map category to friendly display if available
            pretty_val = DISPLAY_LABELS.get(col_key, {}).get(cat_val_raw, cat_val_raw)
            friendly.append(f"{pretty_col} = {pretty_val}")
        else:
            # Likely a numeric (or non-OHE) feature
            friendly.append(FRIENDLY_FEATURE_NAMES.get(name, name))

    return friendly


@st.cache_resource
def _get_shap_explainer(_model):
    """Cache a SHAP TreeExplainer for the current model's classifier.
    Leading underscore in `_model` tells Streamlit not to hash it.
    """
    if not _HAS_SHAP:
        return None
    try:
        clf = _model.named_steps['clf']
        return shap.TreeExplainer(clf)
    except Exception:
        return None



def _plot_threshold_band(current_threshold: float, band: Tuple[float, float], patient_prob: float = None):
    fig, ax = plt.subplots(figsize=(6, 1.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    # Band background
    ax.axvspan(band[0], band[1], alpha=0.2)
    # Threshold line
    ax.axvline(current_threshold, linestyle='--')
    # Patient prob marker
    if patient_prob is not None:
        ax.axvline(patient_prob, linewidth=3)
    ax.set_yticks([])
    ax.set_xlabel('Risk probability')
    return fig


# ============================================
# App Layout
# ============================================
# Sidebar – model loader & threshold
with st.sidebar:
    st.header("⚙️ Model & Settings")
    default_path = "final_model_xgb_reduced.pkl"
    use_uploaded_model = st.toggle("Upload a different .pkl", value=False)
    if use_uploaded_model:
        up = st.file_uploader("Upload a joblib .pkl pipeline", type=["pkl", "joblib"])
        if up is not None:
            tmp_path = "uploaded_model.pkl"
            with open(tmp_path, "wb") as f:
                f.write(up.read())
            model = load_model(tmp_path)
        else:
            st.stop()
    else:
        if not os.path.exists(default_path):
            st.error(f"Can't find `{default_path}` in the working directory.")
            st.stop()
        model = load_model(default_path)

    try:
        num_cols, cat_cols, cat_map = get_feature_schema(model)
        feature_order = list(num_cols) + list(cat_cols)
    except Exception as e:
        st.error(f"Unable to inspect feature schema from the saved pipeline: {e}")
        st.stop()

    st.subheader("🎯 Decision Threshold")
    # Clinically-motivated band from decision curve analysis
    DCA_RECOMMENDED_BAND = (0.10, 0.60)
    threshold = st.slider("Positive class threshold", min_value=0.05, max_value=0.95, value=0.50, step=0.01)

    # Guidance text based on DCA band
    if DCA_RECOMMENDED_BAND[0] <= threshold <= DCA_RECOMMENDED_BAND[1]:
        st.success(
            f"Operating within suggested DCA band {DCA_RECOMMENDED_BAND[0]:.2f}–{DCA_RECOMMENDED_BAND[1]:.2f}. "
            "This range showed greatest net benefit for the calibrated XGBoost model."
        )
    else:
        st.warning(
            f"Threshold is outside suggested DCA band {DCA_RECOMMENDED_BAND[0]:.2f}–{DCA_RECOMMENDED_BAND[1]:.2f}. "
            "You may still proceed, but net benefit was lower outside this range."
        )

    # Visualize threshold vs recommended band (no patient probability here)
    st.pyplot(_plot_threshold_band(threshold, DCA_RECOMMENDED_BAND))

    st.caption("Predicted probability ≥ threshold → Positive (flag for intervention)")


# Tabs
single_tab, batch_tab, info_tab = st.tabs(["👤 Single Prediction", "📄 Batch Scoring (CSV)", "ℹ️ Model Info"])

# ============================================
# Single Prediction
# ============================================
with single_tab:
    st.subheader("📋 Patient Inputs")
    st.caption("Missing entries will be imputed by the model's preprocessing pipeline. Unknown categories are ignored.")

    with st.form("single_form"):
        grid = st.columns(3)
        record: Dict[str, Any] = {}

        # Numeric inputs first (as per ColumnTransformer order)
        for i, col in enumerate(num_cols):
            with grid[i % 3]:
                hints = NUMERIC_HINTS.get(col, {})
                label = FRIENDLY_FEATURE_NAMES.get(col, col)
                record[col] = st.number_input(
                    label,
                    value=None,
                    min_value=hints.get("min_value"),
                    max_value=hints.get("max_value"),
                    step=hints.get("step", 1.0),
                    help=hints.get("help"),
                    placeholder="Leave empty if unknown",
                )

        # Categorical inputs (binary → radio, else selectbox)
        for j, col in enumerate(cat_cols):
            with grid[(j + len(num_cols)) % 3]:
                options = cat_map.get(col)
                label = FRIENDLY_FEATURE_NAMES.get(col, col)
                if options and len(options) == 2:
                    choice = st.radio(
                        label,
                        options + ["(leave blank)"],
                        horizontal=True,
                        format_func=lambda v, _col=col: _display_name(_col, v),
                    )
                    record[col] = None if choice == "(leave blank)" else choice
                elif options and len(options) >= 3:
                    choice = st.selectbox(
                        label,
                        ["(leave blank)"] + options,
                        format_func=lambda v, _col=col: _display_name(_col, v),
                    )
                    record[col] = None if choice == "(leave blank)" else choice
                else:
                    txt = st.text_input(label, value="", placeholder="Type category or leave blank")
                    record[col] = None if txt == "" else txt

        submitted = st.form_submit_button("Predict")

    # Handle submission outside the form
    if submitted:
        X_input = ensure_dataframe(feature_order, record)
        try:
            proba, pred = predict_with_threshold(model, X_input, threshold)
            st.success("Prediction complete.")
            st.metric("Predicted probability (positive)", f"{proba[0]:.4f}")

            # Friendly, patient-facing risk classification with emoji
            if int(pred[0]) == 1:
                label_text = "🚨 At risk for prolonged stay"
            else:
                label_text = "✅ Not at risk for prolonged stay"
            st.metric("Risk classification", label_text)

            # Fancy risk card
            risk = float(proba[0])
            col_risk, col_flag = st.columns([2, 1])
            with col_risk:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("**Predicted Risk**")
                st.progress(min(max(risk, 0.0), 1.0))
                st.markdown(
                    f"<div class='small-note muted'>Probability: <b>{risk:.2%}</b></div>",
                    unsafe_allow_html=True,
                )
                st.markdown("</div>", unsafe_allow_html=True)
            with col_flag:
                st.markdown('<div class="card" style="text-align:center;">', unsafe_allow_html=True)
                if risk >= threshold:
                    st.markdown("<span class='pill pill-red'>FLAG: INTERVENE</span>", unsafe_allow_html=True)
                else:
                    st.markdown("<span class='pill pill-green'>No Flag</span>", unsafe_allow_html=True)
                st.markdown(f"<div class='small-note muted'>Threshold: {threshold:.2f}</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

            # Visualize patient probability vs threshold and DCA band
            st.pyplot(_plot_threshold_band(threshold, DCA_RECOMMENDED_BAND, patient_prob=risk))

            with st.expander("Show input row dataframe"):
                st.dataframe(X_input)

            # SHAP explanation – waterfall plot with human-readable feature names
            st.markdown("### Explain this prediction (SHAP)")

            # Local helper to prettify model-space feature names
            def _friendly_feature_names(raw_feature_names: List[str]) -> List[str]:
                """
                Map model-space names (e.g., 'cat__hospdischargedisposition_recoded_Rehab/Short_or_LongTermCare')
                to friendly labels (e.g., 'Hospital Discharge Disposition = Rehabilitation / Long-Term Care').
                """
                friendly = []
                for feat in raw_feature_names:
                    name = str(feat)
                    # Drop ColumnTransformer prefixes like 'num__' or 'cat__'
                    if "__" in name:
                        _, name = name.split("__", 1)

                    # If it's an OHE feature => '<col>_<category...>'
                    parts = name.split("_", 1)
                    col_key = parts[0]
                    if col_key in FRIENDLY_FEATURE_NAMES and len(parts) == 2:
                        cat_val_raw = parts[1]
                        pretty_col = FRIENDLY_FEATURE_NAMES.get(col_key, col_key)
                        pretty_val = DISPLAY_LABELS.get(col_key, {}).get(cat_val_raw, cat_val_raw)
                        friendly.append(f"{pretty_col} = {pretty_val}")
                    else:
                        # Likely a numeric feature (or non-OHE)
                        friendly.append(FRIENDLY_FEATURE_NAMES.get(name, name))
                return friendly

            if not _HAS_SHAP:
                st.info("Install SHAP to enable explanations: `pip install shap`.")
            else:
                try:
                    explainer = _get_shap_explainer(model)
                    X_proc, feature_names = _transform_to_model_space(model, X_input)
                    shap_values = explainer.shap_values(X_proc)

                    # Select the vector for the positive class if returned as a list
                    if isinstance(shap_values, list):
                        shap_vec = np.array(shap_values[1][0]) if len(shap_values) > 1 else np.array(shap_values[0][0])
                    else:
                        shap_vec = np.array(shap_values[0])

                    # Base value (expected value). If binary and returns two, use positive class (index 1)
                    base_val = explainer.expected_value
                    if isinstance(base_val, (list, tuple, np.ndarray)):
                        base_val = float(base_val[1] if len(np.atleast_1d(base_val)) > 1 else base_val[0])
                    else:
                        base_val = float(base_val)

                    # Use human-friendly feature names
                    # Map to human-readable feature names
                    friendly_names = _map_human_readable([str(n) for n in feature_names])

                    # Build SHAP Explanation for a waterfall plot (model-space features)
                    exp = shap.Explanation(
                        values=shap_vec,
                        base_values=base_val,
                        data=np.array(X_proc[0]).astype(float),
                        feature_names=friendly_names,
                    )

                    fig = plt.figure(figsize=(8, 5))
                    shap.plots.waterfall(exp, show=False, max_display=12)
                    st.pyplot(fig)

                    st.caption(
    "🔎 **How to read this plot:** The baseline on the left is the model’s average prediction. "
    "Each bar shows how a feature pushed the risk **up** (red, toward prolonged stay) or **down** "
    "(blue, toward shorter stay) for this patient. The numbers on each bar are SHAP values — "
    "they quantify the size of that feature’s contribution. The steps add up to the final predicted probability."
)

                except Exception as e:
                    st.warning(f"Could not compute SHAP explanation: {e}")

        except Exception as e:
            st.error(f"Failed to predict: {e}")

# ============================================
# Batch Scoring
# ============================================
with batch_tab:
    st.subheader("Upload CSV for Batch Scoring")
    st.caption("Your CSV should include the training feature columns. Extra columns are ignored; missing ones are imputed if possible.")

    colA, colB = st.columns([1, 2])
    with colA:
        if st.button("Download empty template"):
            template_df = pd.DataFrame(columns=feature_order)
            st.download_button(
                label="Download CSV Template",
                data=template_df.to_csv(index=False).encode("utf-8"),
                file_name="xgb_reduced_template.csv",
                mime="text/csv",
            )
    with colB:
        st.write("Optionally export a template from your training DataFrame (e.g., X_selected[feature_order].head().to_csv(...)).")

    up = st.file_uploader("Upload CSV", type=["csv"])

    if up is not None:
        try:
            df = pd.read_csv(up)
            st.write("Preview of uploaded data:")
            st.dataframe(df.head(10))

            extra_cols = [c for c in df.columns if c not in feature_order]
            missing_cols = [c for c in feature_order if c not in df.columns]

            if extra_cols:
                st.info(f"Ignoring {len(extra_cols)} extra column(s): {extra_cols[:10]}{' ...' if len(extra_cols)>10 else ''}")
            if missing_cols:
                st.warning(f"Missing {len(missing_cols)} expected column(s): {missing_cols[:10]}{' ...' if len(missing_cols)>10 else ''}")

            X_batch = pd.DataFrame(columns=feature_order)
            for col in feature_order:
                X_batch[col] = df[col] if col in df.columns else np.nan

            proba, pred = predict_with_threshold(model, X_batch, threshold)
            out = df.copy()
            out["pred_proba"] = proba
            out["pred_label"] = pred.astype(int)
            out["risk_classification"] = np.where(
                out["pred_label"] == 1,
                "🚨 At risk for prolonged stay",
                "✅ Not at risk for prolonged stay",
            )
            out["intervention_flag"] = out["pred_label"]
            out["operating_threshold"] = threshold

            st.success("Batch scoring complete.")
            st.dataframe(out.head(20))

            st.download_button(
                label="Download Predictions CSV",
                data=out.to_csv(index=False).encode("utf-8"),
                file_name="xgb_reduced_predictions.csv",
                mime="text/csv",
            )
        except Exception as e:
            st.error(f"Failed to process CSV: {e}")


# ============================================
# Model Info
# ============================================
with info_tab:
    st.subheader("Pipeline / Feature Info")
    render_schema_table(num_cols, cat_cols)

    st.markdown("### Clinical Operating Guidance")
    st.write(
        "**Decision Curve Analysis (DCA):** The calibrated XGBoost model provided the greatest net benefit "
        f"across threshold probabilities **{(0.10):.2f}–{(0.60):.2f}**. "
        "A *threshold probability* represents the risk level at which a clinician would choose to intervene "
        "(e.g., allocate resources or plan step-down rehab). "
        "**Using this app:** Set the threshold slider in the sidebar. Any patient with predicted probability ≥ threshold "
        "is flagged for intervention."
    )

    with st.expander("Raw objects (advanced)"):
        try:
            st.write("**Classifier:**", type(model.named_steps['clf']).__name__)
        except Exception:
            st.write("Unable to display classifier.")
        feature_order = list(num_cols) + list(cat_cols)
        st.write("**Feature order used for input assembly:**")
        st.code(json.dumps(feature_order, indent=2))

    st.caption(
        "Note: The saved pipeline handles imputation, scaling, and one-hot encoding. Unknown categories are safely ignored."
    )


st.markdown("---")
st.markdown(
    "Built for the pSCI Prolonged LOS Prediction project. "
    "Binary features render as radio buttons; 3+ category features use dropdowns."
)