# ══════════════════════════════════════════════════════════════════════════════
# CBC Thalassemia 4-Class Classifier — Streamlit UI
# Deploy: streamlit.io (Community Cloud — free)
# Model stored on HuggingFace Hub (no 25 MB GitHub limit)
# ══════════════════════════════════════════════════════════════════════════════

from __future__ import annotations
import os, pickle, warnings, io
from typing import Optional, Dict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import streamlit as st
from huggingface_hub import hf_hub_download

warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════════════
# 1. Load model bundle from Hugging Face Hub
#    ↓ Change HF_USERNAME to your actual Hugging Face username
# ══════════════════════════════════════════════════════════════════════════════
HF_USERNAME  = "Chollanot"                 # ← your HuggingFace username
HF_REPO_ID   = f"{HF_USERNAME}/cbc-thalassemia-model"
BUNDLE_CACHE = "/tmp/model_bundle.pkl"

@st.cache_resource(show_spinner="Loading model from Hugging Face Hub...")
def load_bundle():
    # Download only once per session; cached in /tmp after that
    if not os.path.exists(BUNDLE_CACHE):
        local_path = hf_hub_download(
            repo_id   = HF_REPO_ID,
            filename  = "model_bundle.pkl",
            repo_type = "dataset",
            local_dir = "/tmp",
        )
    with open(BUNDLE_CACHE, 'rb') as fh:
        return pickle.load(fh)

bundle         = load_bundle()
BEST_MODEL     = bundle['model']
fitted_scaler  = bundle['scaler']
LOG_FEATS      = bundle['log_feats']
ALL_FEATURES   = bundle['all_features']
BEST_FEATS     = bundle['best_features']
BEST_IDXS      = bundle['best_idxs']
shap_explainer = bundle.get('shap_explainer')
lime_explainer = bundle.get('lime_explainer')
CLASS_NAMES    = bundle['class_names']
N_CLASSES      = bundle['n_classes']
feat_names     = BEST_FEATS

classifier_name = bundle.get('classifier',  'Random Forest')
resampling_name = bundle.get('resampling',  'No Resampling')
feature_set     = bundle.get('feature_set', 'Top-12')
acc       = float(bundle.get('accuracy',    0.0))
f1        = float(bundle.get('f1',          0.0))
mcc       = float(bundle.get('mcc',         0.0))
auc_roc   = float(bundle.get('auc_roc',     0.0))
auc_pr    = float(bundle.get('auc_pr',      0.0))
mean_sens = float(bundle.get('sensitivity', 0.0))

SHAP_OK = shap_explainer is not None
LIME_OK = lime_explainer is not None

RAW_DEFAULTS: Dict[str, float] = {
    'Age': 30.0, 'Sex': 0.0,
    'RBC': 4.9,  'HGB': 13.2, 'HCT': 40.0,
    'MCV': 82.0, 'MCH': 27.0, 'MCHC': 33.0, 'RDW': 13.0,
    'PLT': 277.0,'WBC': 8.7,
    'NEU': 62.0, 'absNEU': 5.6,
    'LYMP': 28.0,'absLYMP': 2.3,
    'MONO': 6.4, 'absMONO': 0.54,
    'EOS': 2.7,  'absEOS': 0.23,
    'BASO': 0.4, 'absBASO': 0.02,
}

CLASS_DESC = {
    0: 'Normal / Non-clinically significant',
    1: 'Normal Hb typing ± α-thalassemia',
    2: 'HbE Trait ± α-thalassemia (Primary target)',
    3: 'Other hemoglobinopathy (Hom.HbE / HbH / β-thal / HbCS)',
}

REQUIRED_COLS = ['Sex', 'RBC', 'HGB', 'HCT', 'MCV', 'MCH', 'MCHC']

# ══════════════════════════════════════════════════════════════════════════════
# 2. Core helpers
# ══════════════════════════════════════════════════════════════════════════════
def _build_raw(data: dict) -> dict:
    raw = dict(RAW_DEFAULTS)
    for k, v in data.items():
        if k in raw and v is not None:
            try:
                raw[k] = float(v)
            except (TypeError, ValueError):
                pass
    sex_val = data.get('Sex', data.get('sex'))
    if isinstance(sex_val, str):
        raw['Sex'] = 1.0 if sex_val.strip().lower() in ('male', 'm', '1') else 0.0
    return raw


def _preprocess(raw_dict: dict) -> np.ndarray:
    row_df = pd.DataFrame([raw_dict])[ALL_FEATURES].copy()
    for f in LOG_FEATS:
        if f in row_df.columns:
            row_df[f] = np.log1p(row_df[f].clip(lower=0))
    scaled   = fitted_scaler.transform(row_df)
    return scaled[0, BEST_IDXS]


def _predict_one(raw: dict) -> dict:
    sample = _preprocess(raw)
    X_in   = sample.reshape(1, -1)
    pred   = int(BEST_MODEL.predict(X_in)[0])
    probs  = BEST_MODEL.predict_proba(X_in)[0].tolist()
    return {
        'predicted_class':      pred,
        'predicted_class_name': CLASS_NAMES.get(pred, str(pred)),
        'class_description':    CLASS_DESC.get(pred, ''),
        'probabilities': {
            CLASS_NAMES.get(i, str(i)): round(probs[i], 5)
            for i in range(N_CLASSES)
        },
        'sample': sample,
    }


def _prob_fig(probs: list, pred_cls: int) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 3))
    clrs = ['#4CAF50', '#2196F3', '#FF9800', '#E91E63'][:N_CLASSES]
    labs = [CLASS_NAMES.get(i, str(i)) for i in range(N_CLASSES)]
    bars = ax.barh(labs, [p * 100 for p in probs], color=clrs, edgecolor='white')
    for bar, p in zip(bars, [p * 100 for p in probs]):
        ax.text(p + 0.5, bar.get_y() + bar.get_height() / 2,
                f'{p:.1f}%', va='center', fontsize=9, fontweight='bold')
    ax.set_xlim(0, 120)
    ax.set_xlabel('Probability (%)', fontsize=9)
    ax.set_title(f'Predicted: {CLASS_NAMES.get(pred_cls, str(pred_cls))}',
                 fontweight='bold', fontsize=10)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    return fig


def _shap_fig(sample: np.ndarray, pred_cls: int) -> plt.Figure:
    if not SHAP_OK:
        fig, ax = plt.subplots(figsize=(5, 2))
        ax.text(0.5, 0.5, 'SHAP explainer not available',
                ha='center', va='center', fontsize=11, color='gray')
        ax.axis('off')
        return fig
    try:
        sv_all = shap_explainer.shap_values(sample.reshape(1, -1))
        sv = sv_all[pred_cls][0] if isinstance(sv_all, list) else sv_all[0]
        if hasattr(sv, 'ndim') and sv.ndim > 1:
            sv = sv.flatten()
        order  = np.argsort(np.abs(sv))[::-1][:10]
        top_f  = np.array(feat_names)[order]
        top_sv = sv[order]
        clr    = ['#E91E63' if v > 0 else '#2196F3' for v in top_sv]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(top_f[::-1], top_sv[::-1], color=clr[::-1], edgecolor='white')
        ax.axvline(0, color='black', lw=0.8)
        ax.set_xlabel('SHAP contribution', fontsize=9)
        ax.set_title(f'SHAP — {CLASS_NAMES.get(pred_cls, str(pred_cls))}',
                     fontweight='bold', fontsize=10)
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        return fig
    except Exception as e:
        fig, ax = plt.subplots(figsize=(5, 2))
        ax.text(0.5, 0.5, f'SHAP error: {e}', ha='center', va='center',
                fontsize=9, color='red')
        ax.axis('off')
        return fig


def _lime_fig(sample: np.ndarray, pred_cls: int) -> plt.Figure:
    if not LIME_OK:
        fig, ax = plt.subplots(figsize=(5, 2))
        ax.text(0.5, 0.5, 'LIME explainer not available',
                ha='center', va='center', fontsize=11, color='gray')
        ax.axis('off')
        return fig
    try:
        exp = lime_explainer.explain_instance(
            sample, BEST_MODEL.predict_proba, num_features=8, labels=[pred_cls])
        fc  = exp.as_list(label=pred_cls)
        fl  = [c[0][:35] for c in fc]
        fv  = [c[1] for c in fc]
        clr = ['#E91E63' if v > 0 else '#2196F3' for v in fv]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(fl[::-1], fv[::-1], color=clr[::-1], edgecolor='white')
        ax.axvline(0, color='black', lw=0.8)
        ax.set_xlabel('LIME contribution', fontsize=9)
        ax.set_title(f'LIME — {CLASS_NAMES.get(pred_cls, str(pred_cls))}',
                     fontweight='bold', fontsize=10)
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        return fig
    except Exception as e:
        fig, ax = plt.subplots(figsize=(5, 2))
        ax.text(0.5, 0.5, f'LIME error: {e}', ha='center', va='center',
                fontsize=9, color='red')
        ax.axis('off')
        return fig


# ══════════════════════════════════════════════════════════════════════════════
# 3. Streamlit page config
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title='CBC Thalassemia Classifier',
    page_icon='🩸',
    layout='wide',
)

st.markdown("""
<div style="background:linear-gradient(135deg,#1a237e,#c62828);
            border-radius:12px;padding:18px 24px;margin-bottom:16px">
  <h1 style="color:white;margin:0;font-size:1.6em">🩸 CBC Thalassemia 4-Class Classifier</h1>
  <p style="color:rgba(255,255,255,0.85);margin:4px 0 0">
    Explainable AI · Raw CBC → Prediction + SHAP + LIME
  </p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
| Class | Diagnosis |
|:---:|:---|
| 0 | Normal / Non-clinically significant |
| 1 | Normal Hb typing ± α-thalassemia |
| 2 | **HbE Trait** ± α-thalassemia ← *Primary clinical target* |
| 3 | Other hemoglobinopathies (Hom.HbE · HbH · β-thal · HbCS) |
""")

# ══════════════════════════════════════════════════════════════════════════════
# 4. Tabs
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(['📋 Single Patient', '📂 Batch Prediction', 'ℹ️ Model Info'])

# ── TAB 1 — Single patient ────────────────────────────────────────────────────
with tab1:
    col_in, col_out = st.columns(2)

    with col_in:
        st.markdown('### CBC Parameters')
        sex  = st.selectbox('Sex', ['Female', 'Male'])
        age  = st.number_input('Age (yrs)', min_value=0.0, max_value=120.0, value=30.0)
        st.markdown('**Red Cell Indices (Required)**')
        c1, c2 = st.columns(2)
        rbc  = c1.number_input('RBC (×10⁶/µL)', min_value=0.5,  max_value=10.0,  value=4.9)
        hgb  = c2.number_input('HGB (g/dL)',     min_value=1.0,  max_value=25.0,  value=13.2)
        hct  = c1.number_input('HCT (%)',         min_value=5.0,  max_value=75.0,  value=40.0)
        mcv  = c2.number_input('MCV (fL)',        min_value=40.0, max_value=150.0, value=82.0)
        mch  = c1.number_input('MCH (pg)',        min_value=10.0, max_value=50.0,  value=27.0)
        mchc = c2.number_input('MCHC (g/dL)',     min_value=20.0, max_value=45.0,  value=33.0)

        with st.expander('⚙️ WBC Differential (optional)'):
            d1, d2, d3 = st.columns(3)
            rdw   = d1.number_input('RDW (%)',       value=13.0)
            plt_v = d2.number_input('PLT (×10³/µL)', value=277.0)
            wbc   = d3.number_input('WBC (×10³/µL)', value=8.7)
            e1, e2 = st.columns(2)
            neu   = e1.number_input('NEU (%)',   value=62.0)
            aneu  = e2.number_input('Abs NEU',   value=5.6)
            lymp  = e1.number_input('LYMP (%)',  value=28.0)
            alymp = e2.number_input('Abs LYMP',  value=2.3)
            mono  = e1.number_input('MONO (%)',  value=6.4)
            amono = e2.number_input('Abs MONO',  value=0.54)
            eos   = e1.number_input('EOS (%)',   value=2.7)
            aeos  = e2.number_input('Abs EOS',   value=0.23)
            baso  = e1.number_input('BASO (%)',  value=0.4)
            abaso = e2.number_input('Abs BASO',  value=0.02)

        predict_btn = st.button('🔍 Predict + Explain', type='primary', use_container_width=True)

    with col_out:
        st.markdown('### Result')
        if predict_btn:
            raw = {
                'Sex': 1.0 if sex == 'Male' else 0.0,
                'Age': age, 'RBC': rbc, 'HGB': hgb, 'HCT': hct,
                'MCV': mcv, 'MCH': mch, 'MCHC': mchc, 'RDW': rdw,
                'PLT': plt_v, 'WBC': wbc, 'NEU': neu, 'absNEU': aneu,
                'LYMP': lymp, 'absLYMP': alymp, 'MONO': mono, 'absMONO': amono,
                'EOS': eos, 'absEOS': aeos, 'BASO': baso, 'absBASO': abaso,
            }
            with st.spinner('Running prediction...'):
                res    = _predict_one(raw)
                pred   = res['predicted_class']
                probs  = [res['probabilities'].get(CLASS_NAMES.get(i, str(i)), 0.0)
                          for i in range(N_CLASSES)]
                sample = res['sample']

            color_map = {0: '🟢', 1: '🔵', 2: '🟠', 3: '🔴'}
            st.success(f"{color_map.get(pred,'🔬')} **{res['predicted_class_name']}**")
            st.caption(res['class_description'])

            st.markdown('**Class Probabilities**')
            for i in range(N_CLASSES):
                st.progress(probs[i],
                    text=f"[{i}] {CLASS_NAMES.get(i, str(i))}: {probs[i]*100:.1f}%")

            st.pyplot(_prob_fig(probs, pred))

            st.markdown(f"""
---
**Model:** {classifier_name} [{resampling_name}]
**Feature set:** {feature_set} ({len(BEST_FEATS)} features)
**MCC:** {mcc:.3f} · **AUC-PR:** {auc_pr:.3f} · **Acc:** {acc:.3f}

> ⚠️ For clinical guidance only — confirm with specialist.
""")

    if predict_btn:
        st.markdown('---')
        sh_col, li_col = st.columns(2)
        with sh_col:
            st.markdown('**SHAP — Feature Contributions**')
            st.pyplot(_shap_fig(sample, pred))
        with li_col:
            st.markdown('**LIME — Local Explanation**')
            st.pyplot(_lime_fig(sample, pred))

# ── TAB 2 — Batch prediction ──────────────────────────────────────────────────
with tab2:
    st.markdown("""
Upload a **.xlsx** or **.csv** file.
Required columns: `Sex, RBC, HGB, HCT, MCV, MCH, MCHC`
Optional: `Age, RDW, PLT, WBC, NEU, absNEU, LYMP, absLYMP, MONO, absMONO, EOS, absEOS, BASO, absBASO`
""")

    # Download template
    cols_tmpl = ['PatientID','Sex','Age','RBC','HGB','HCT','MCV','MCH',
                 'MCHC','RDW','PLT','WBC','NEU','absNEU','LYMP','absLYMP',
                 'MONO','absMONO','EOS','absEOS','BASO','absBASO']
    data_tmpl = [
        ['P001','Female',28,4.20,11.8,36,70,22,33,14.5,250,7.2,60,4.3,25,1.8,7,0.5,3,0.2,0.3,0.02],
        ['P002','Male',  35,5.10,15.0,46,88,30,34,13.0,280,8.5,65,5.5,28,2.4,6,0.5,2,0.2,0.5,0.02],
        ['P003','Female',22,5.50,10.5,33,62,18,30,16.0,220,7.0,58,4.0,30,2.1,6,0.4,4,0.3,0.4,0.01],
    ]
    tmpl_buf = io.BytesIO()
    pd.DataFrame(data_tmpl, columns=cols_tmpl).to_excel(tmpl_buf, index=False)
    tmpl_buf.seek(0)
    st.download_button('⬇️ Download Template (.xlsx)', tmpl_buf,
                       file_name='CBC_Template.xlsx',
                       mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

    uploaded = st.file_uploader('📤 Upload CBC file', type=['xlsx', 'xls', 'csv'])

    if uploaded and st.button('🔍 Run Batch Prediction', type='primary'):
        try:
            df_in = (pd.read_csv(uploaded) if uploaded.name.lower().endswith('.csv')
                     else pd.read_excel(uploaded))
            df_in.columns = df_in.columns.str.strip()
        except Exception as e:
            st.error(f'Could not read file: {e}')
            st.stop()

        missing = [c for c in REQUIRED_COLS if c not in df_in.columns]
        if missing:
            st.error(f'Missing required columns: {missing}')
            st.stop()

        out_rows = []
        progress = st.progress(0, text='Processing...')
        for idx, (_, pt) in enumerate(df_in.iterrows()):
            raw = _build_raw(pt.to_dict())
            res = _predict_one(raw)
            rec = {c: pt.get(c, '') for c in df_in.columns}
            rec['Predicted_Class'] = res['predicted_class_name']
            rec['Confidence_%']    = round(max(res['probabilities'].values()) * 100, 1)
            for i in range(N_CLASSES):
                rec[f'Prob_Cls{i}_%'] = round(
                    res['probabilities'].get(CLASS_NAMES.get(i, str(i)), 0.0) * 100, 2)
            out_rows.append(rec)
            progress.progress((idx + 1) / len(df_in),
                              text=f'Processing {idx+1}/{len(df_in)}...')

        df_out = pd.DataFrame(out_rows)
        progress.empty()

        st.success(f'✅ {len(df_out)} patients processed')
        dist = df_out['Predicted_Class'].value_counts()
        st.markdown('**Class Distribution**')
        st.bar_chart(dist)
        st.dataframe(df_out.round(3), use_container_width=True)

        out_buf = io.BytesIO()
        with pd.ExcelWriter(out_buf, engine='openpyxl') as writer:
            df_out.to_excel(writer, sheet_name='Predictions', index=False)
            df_out['Predicted_Class'].value_counts().reset_index().rename(
                columns={'Predicted_Class': 'Class', 'count': 'Count'}
            ).to_excel(writer, sheet_name='Summary', index=False)
        out_buf.seek(0)

        st.download_button('⬇️ Download Full Results (.xlsx)', out_buf,
                           file_name='Batch_Predictions.xlsx',
                           mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

# ── TAB 3 — Model info ─────────────────────────────────────────────────────────
with tab3:
    st.markdown(f"""
## Model Configuration
| Parameter | Value |
|:---|:---|
| **Classifier** | {classifier_name} |
| **Resampling** | {resampling_name} |
| **Feature set** | {feature_set} ({len(BEST_FEATS)} features) |
| **Features** | {', '.join(BEST_FEATS)} |

## Test-Set Performance
| Metric | Score |
|:---|:---:|
| Accuracy | {acc:.4f} |
| F1 (macro) | {f1:.4f} |
| **★ MCC** | **{mcc:.4f}** |
| AUC-ROC | {auc_roc:.4f} |
| **★ AUC-PR** | **{auc_pr:.4f}** |
| Sensitivity (macro) | {mean_sens:.4f} |

## Dataset
| | |
|:---|:---|
| Total | 5,734 patients |
| Class 0 — Normal | 3,209 (55.9%) |
| Class 1 — Normal Hb ± α-Thal | 709 (12.4%) |
| Class 2 — HbE Trait ± α-Thal | 1,357 (23.7%) |
| Class 3 — Other hemoglobinopathy | 459 (8.0%) |

---
⚠️ *For clinical guidance only — confirm all results with a qualified haematologist.*
""")
