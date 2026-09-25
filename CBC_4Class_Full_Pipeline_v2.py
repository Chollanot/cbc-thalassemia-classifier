# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  CBC Thalassemia 4-Class Classifier — Full Pipeline v2 (Corrected)         ║
# ║  PONE-D-26-22228                                                            ║
# ║  Fixes: CatBoost predict(n,1) bug · geometric G-mean · train-only impute   ║
# ║         MDI+SHAP+RFE-LR feature selection · all outputs saved to Drive     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# ════════════════════════════════════════════════════════════════════════════════
# SECTION 0 — Install packages
# ════════════════════════════════════════════════════════════════════════════════
import subprocess, sys
pkgs = ["xgboost","catboost","shap","imbalanced-learn","openpyxl","scikit-posthocs","matplotlib","seaborn"]
for p in pkgs:
    subprocess.run([sys.executable,"-m","pip","install",p,"--quiet"], check=False)

# ════════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Imports
# ════════════════════════════════════════════════════════════════════════════════
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')            # non-interactive backend for Colab saving
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings, pickle, os, time
from copy import deepcopy
from itertools import product

warnings.filterwarnings('ignore')
np.random.seed(42)
SEED = 42

from sklearn.model_selection   import train_test_split, StratifiedKFold
from sklearn.preprocessing     import RobustScaler, label_binarize
from sklearn.impute             import SimpleImputer
from sklearn.feature_selection import RFE
from sklearn.linear_model      import LogisticRegression as LR_sel
from sklearn.tree              import DecisionTreeClassifier
from sklearn.ensemble          import RandomForestClassifier
from sklearn.naive_bayes       import GaussianNB
from sklearn.neural_network    import MLPClassifier
from sklearn.svm               import SVC
from sklearn.linear_model      import LogisticRegression
from sklearn.metrics           import (
    accuracy_score, f1_score, recall_score,
    matthews_corrcoef, confusion_matrix,
    roc_auc_score, average_precision_score,
    roc_curve, auc, precision_recall_curve
)
from sklearn.base import clone

from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.combine       import SMOTEENN

from xgboost  import XGBClassifier
from catboost import CatBoostClassifier

import shap
from google.colab import files

# ════════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Constants & output directory
# ════════════════════════════════════════════════════════════════════════════════
CLASSES     = np.array([0, 1, 2, 3])
N_CLASSES   = 4
CLASS_NAMES = {
    0: 'Normal',
    1: 'Normal ± α-Thal',
    2: 'HbE Trait ± α-Thal',
    3: 'Other Thalassemia',
}
TARGET_COL  = 'Label'
CBC_FEATURES = [
    'RBC','HGB','HCT','MCV','MCH','MCHC','RDW','PLT',
    'WBC','NEU','abs NEU','LYMP','abs LYMP','MONO',
    'abs MONO','EOS','abs EOS','BASO','abs BASO',
]
PALETTE = ['#4CAF50','#2196F3','#FF9800','#E91E63']

# Output folder (Google Drive or /content)
OUT_DIR = '/content/CBC_4Class_Outputs'
os.makedirs(OUT_DIR, exist_ok=True)
print(f'Outputs will be saved to: {OUT_DIR}')

def savefig(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {path}')

# ════════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Upload & load data
# ════════════════════════════════════════════════════════════════════════════════
print('\n' + '='*65)
print('STEP 1: Upload CBC Excel file')
print('='*65)
uploaded = files.upload()
fname    = list(uploaded.keys())[0]

df_raw = pd.read_excel(fname)
df_raw.columns = df_raw.columns.str.strip()
print(f'\nLoaded: {df_raw.shape[0]} rows × {df_raw.shape[1]} columns')
print(f'\nColumn names: {df_raw.columns.tolist()}')
print(f'\nLabel distribution:\n{df_raw[TARGET_COL].value_counts().sort_index()}')

# Keep only available CBC features
CBC_FEATURES = [c for c in CBC_FEATURES if c in df_raw.columns]
print(f'\nCBC features found ({len(CBC_FEATURES)}): {CBC_FEATURES}')

# ════════════════════════════════════════════════════════════════════════════════
# SECTION 4 — EDA
# ════════════════════════════════════════════════════════════════════════════════
print('\n' + '='*65)
print('STEP 2: Exploratory Data Analysis (EDA)')
print('='*65)

df = df_raw[CBC_FEATURES + [TARGET_COL]].copy()
df[TARGET_COL] = df[TARGET_COL].astype(int)

# Missing values
mv = df[CBC_FEATURES].isnull().sum()
print('\nMissing values per feature:')
print(mv[mv > 0].to_string() if mv.any() else '  None')
print(f'  Total missing: {df[CBC_FEATURES].isnull().sum().sum()} '
      f'/ {len(df)*len(CBC_FEATURES)} ({df[CBC_FEATURES].isnull().mean().mean()*100:.2f}%)')

# Clip negatives
neg_counts = (df[CBC_FEATURES] < 0).sum()
if neg_counts.any():
    print(f'\nNegative values found — clipping to 0:')
    print(neg_counts[neg_counts > 0])
    df[CBC_FEATURES] = df[CBC_FEATURES].clip(lower=0)

# Descriptive stats per class
print('\nDescriptive statistics (mean ± std) for key features:')
key_feats = ['RBC','HGB','MCV','MCH','RDW']
for feat in key_feats:
    print(f'  {feat}:', end='')
    for c in CLASSES:
        grp = df[df[TARGET_COL]==c][feat].dropna()
        print(f'  C{c}={grp.mean():.2f}±{grp.std():.2f}', end='')
    print()

# Class balance plot
fig, ax = plt.subplots(figsize=(7, 4))
counts = df[TARGET_COL].value_counts().sort_index()
bars = ax.bar([CLASS_NAMES[c] for c in CLASSES],
              [counts.get(c, 0) for c in CLASSES],
              color=PALETTE, edgecolor='white', linewidth=1.2)
for bar, val in zip(bars, [counts.get(c,0) for c in CLASSES]):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+20,
            str(val), ha='center', fontsize=10, fontweight='bold')
ax.set_title('Class Distribution', fontsize=13, fontweight='bold')
ax.set_ylabel('Count')
ax.spines[['top','right']].set_visible(False)
plt.tight_layout()
savefig(fig, '01_class_distribution.png')

# Distribution plots for key features
fig, axes = plt.subplots(3, 4, figsize=(16, 11))
axes = axes.flatten()
for i, feat in enumerate(CBC_FEATURES[:12]):
    ax = axes[i]
    for lbl in CLASSES:
        grp = df[df[TARGET_COL]==lbl][feat].dropna()
        ax.hist(grp, bins=35, alpha=0.55, color=PALETTE[lbl],
                label=CLASS_NAMES[lbl], density=True, edgecolor='none')
    ax.set_title(feat, fontsize=9, fontweight='bold')
    ax.tick_params(labelsize=7)
handles = [mpatches.Patch(color=PALETTE[c], label=CLASS_NAMES[c]) for c in CLASSES]
fig.legend(handles=handles, loc='lower right', fontsize=9)
fig.suptitle('Feature Distributions by Class', fontsize=12, fontweight='bold')
plt.tight_layout()
savefig(fig, '02_feature_distributions.png')

# Skewness
skew = df[CBC_FEATURES].skew().sort_values(ascending=False).round(3)
print('\nSkewness:')
for feat, sk in skew.items():
    flag = '⚠ HIGH' if abs(sk) > 1 else ('moderate' if abs(sk) > 0.5 else 'ok')
    print(f'  {feat:<12} {sk:>8.3f}   {flag}')

# ════════════════════════════════════════════════════════════════════════════════
# SECTION 4b — Data Description: Normality · Mean±SD · Statistical Comparison
# ════════════════════════════════════════════════════════════════════════════════
print('\n' + '='*75)
print('STEP 2b: Data Description — Normality Test + Mean±SD + Statistical Comparison')
print('='*75)

from scipy import stats as sp_stats
try:
    import scikit_posthocs as sp
    POSTHOCS_OK = True
except ImportError:
    POSTHOCS_OK = False
    print('  scikit-posthocs not available — pairwise Dunn test skipped')

# ── 1. Normality test (Shapiro-Wilk per feature per class) ────────────────────
# Note: Shapiro-Wilk is most sensitive for n ≤ 5000.
# For n > 5000 per class we use D'Agostino-Pearson (normaltest).
SHAPIRO_MAX = 5000

print('\n--- Normality Test (Shapiro-Wilk; D\'Agostino-Pearson if n>5000) ---')
norm_rows = []
for feat in CBC_FEATURES:
    row = {'Feature': feat}
    all_normal = True
    for c in CLASSES:
        vals = df[df[TARGET_COL] == c][feat].dropna().values
        n_c  = len(vals)
        if n_c < 3:
            stat, pval, normal = np.nan, np.nan, False
            test_name = 'n/a'
        elif n_c <= SHAPIRO_MAX:
            stat, pval = sp_stats.shapiro(vals)
            test_name  = 'Shapiro-Wilk'
            normal     = pval > 0.05
        else:
            stat, pval = sp_stats.normaltest(vals)   # D'Agostino-Pearson
            test_name  = 'D\'Agostino'
            normal     = pval > 0.05
        row[f'C{c}_n']       = n_c
        row[f'C{c}_stat']    = round(float(stat), 4) if not np.isnan(stat) else np.nan
        row[f'C{c}_p_norm']  = round(float(pval), 4) if not np.isnan(pval) else np.nan
        row[f'C{c}_normal']  = 'Yes' if normal else 'No'
        if not normal:
            all_normal = False
    row['Test_used'] = test_name
    row['All_normal'] = 'Yes' if all_normal else 'No'
    norm_rows.append(row)

df_norm = pd.DataFrame(norm_rows)
print(f'\n{"Feature":<14}', end='')
for c in CLASSES:
    print(f'  C{c}(n={int(df[df[TARGET_COL]==c].shape[0])}) p_norm', end='')
print('  All_normal')
print('-' * 80)
for _, r in df_norm.iterrows():
    print(f'{r["Feature"]:<14}', end='')
    for c in CLASSES:
        p = r[f'C{c}_p_norm']
        flag = '*' if (isinstance(p, float) and p < 0.05) else ' '
        print(f'  {p:>10.4f}{flag}', end='')
    print(f'  {r["All_normal"]}')
print('* p<0.05 → non-normal')

df_norm.to_csv(os.path.join(OUT_DIR, 'data_normality_test.csv'), index=False)
print(f'\nSaved: data_normality_test.csv')

# ── 2. Mean ± SD per class + overall ──────────────────────────────────────────
print('\n--- Mean ± SD per Class ---')
hdr_ms = f'{"Feature":<14}'
for c in CLASSES:
    hdr_ms += f'  Class {c} ({CLASS_NAMES[c][:10]})'
hdr_ms += '  Overall'
print(hdr_ms)
print('-' * 105)

desc_rows = []
for feat in CBC_FEATURES:
    row = {'Feature': feat}
    line = f'{feat:<14}'
    all_vals = df[feat].dropna().values
    row['Overall_mean'] = round(float(np.mean(all_vals)), 3)
    row['Overall_sd']   = round(float(np.std(all_vals)), 3)
    row['Overall_mean_sd'] = f'{row["Overall_mean"]:.3f} ± {row["Overall_sd"]:.3f}'
    for c in CLASSES:
        vals = df[df[TARGET_COL] == c][feat].dropna().values
        m, s = float(np.mean(vals)), float(np.std(vals))
        row[f'Class{c}_mean']    = round(m, 3)
        row[f'Class{c}_sd']      = round(s, 3)
        row[f'Class{c}_mean_sd'] = f'{m:.3f} ± {s:.3f}'
        line += f'  {m:.3f} ± {s:.3f}'
    line += f'  {row["Overall_mean"]:.3f} ± {row["Overall_sd"]:.3f}'
    print(line)
    desc_rows.append(row)

df_desc = pd.DataFrame(desc_rows)
df_desc.to_csv(os.path.join(OUT_DIR, 'data_mean_sd_per_class.csv'), index=False)
print(f'\nSaved: data_mean_sd_per_class.csv')

# ── 3. Statistical comparison (ANOVA or Kruskal-Wallis) ───────────────────────
print('\n--- Statistical Comparison (ANOVA if all-normal, else Kruskal-Wallis) ---')
print(f'{"Feature":<14}  {"Test":<15}  {"Statistic":>10}  {"p-value":>10}  {"Sig":>5}  {"Interpretation"}')
print('-' * 80)

stat_rows = []
kw_features = []   # store per-feature group data for Dunn post-hoc
for _, nr in df_norm.iterrows():
    feat = nr['Feature']
    groups = [df[df[TARGET_COL] == c][feat].dropna().values for c in CLASSES]
    all_normal = nr['All_normal'] == 'Yes'

    if all_normal:
        stat, pval = sp_stats.f_oneway(*groups)
        test_name  = 'One-way ANOVA'
    else:
        stat, pval = sp_stats.kruskal(*groups)
        test_name  = 'Kruskal-Wallis'

    if pval < 0.001:
        sig = '***'
    elif pval < 0.01:
        sig = '**'
    elif pval < 0.05:
        sig = '*'
    else:
        sig = 'ns'

    interp = 'Significant difference among classes' if pval < 0.05 else 'No significant difference'
    print(f'{feat:<14}  {test_name:<15}  {stat:>10.3f}  {pval:>10.4f}  {sig:>5}  {interp}')
    stat_rows.append({'Feature': feat, 'Test': test_name,
                      'Statistic': round(float(stat), 4),
                      'p_value': round(float(pval), 6),
                      'Significance': sig,
                      'Significant': 'Yes' if pval < 0.05 else 'No'})
    kw_features.append({'feature': feat, 'groups': groups,
                        'pval': pval, 'test': test_name})

df_stat = pd.DataFrame(stat_rows)
df_stat.to_csv(os.path.join(OUT_DIR, 'data_statistical_comparison.csv'), index=False)
print(f'\nSaved: data_statistical_comparison.csv')
n_sig = (df_stat['Significant'] == 'Yes').sum()
print(f'\n{n_sig}/{len(CBC_FEATURES)} features show significant differences among classes (p<0.05)')

# ── 4. Post-hoc pairwise tests (Dunn with Bonferroni) ─────────────────────────
print('\n--- Post-hoc Pairwise Tests (Dunn + Bonferroni, significant features only) ---')
posthoc_rows = []
pairs = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
pair_labels = [f'C{a} vs C{b}' for a,b in pairs]

if POSTHOCS_OK:
    for item in kw_features:
        feat  = item['feature']
        pval  = item['pval']
        if pval >= 0.05:
            continue   # only post-hoc significant features
        groups = item['groups']
        # Build long-format data for scikit-posthocs
        data_long = np.concatenate(groups)
        labels    = np.concatenate([[c]*len(g) for c,g in zip(CLASSES, groups)])
        try:
            ph = sp.posthoc_dunn(
                pd.DataFrame({'val': data_long, 'group': labels}),
                val_col='val', group_col='group', p_adjust='bonferroni'
            )
            row = {'Feature': feat}
            for a, b in pairs:
                p_ab = ph.loc[a, b]
                row[f'C{a}_vs_C{b}'] = round(float(p_ab), 4)
            posthoc_rows.append(row)
        except Exception as e:
            print(f'  {feat}: Dunn test error — {e}')

    if posthoc_rows:
        df_ph = pd.DataFrame(posthoc_rows)
        df_ph.to_csv(os.path.join(OUT_DIR, 'data_posthoc_dunn.csv'), index=False)

        print(f'\n{"Feature":<14}', end='')
        for pl in pair_labels:
            print(f'  {pl:>10}', end='')
        print()
        print('-' * (14 + 12*len(pairs)))
        for _, r in df_ph.iterrows():
            print(f'{r["Feature"]:<14}', end='')
            for a, b in pairs:
                p_ab = r[f'C{a}_vs_C{b}']
                sig_m = '***' if p_ab<0.001 else ('**' if p_ab<0.01 else
                        ('*' if p_ab<0.05 else 'ns'))
                print(f'  {sig_m:>10}', end='')
            print()
        print('(Bonferroni-adjusted; * p<0.05  ** p<0.01  *** p<0.001  ns=not significant)')
        print(f'Saved: data_posthoc_dunn.csv')
    else:
        print('  No significant features to post-hoc test.')
else:
    print('  scikit-posthocs not installed — skipping Dunn post-hoc.')

# ── 5. Combined summary table (Mean±SD + p-value) ─────────────────────────────
df_summary = df_desc[['Feature'] + [f'Class{c}_mean_sd' for c in CLASSES] +
                      ['Overall_mean_sd']].copy()
df_summary = df_summary.merge(
    df_stat[['Feature','Test','p_value','Significance']], on='Feature')
# Rename columns to manuscript-ready labels
rename = {f'Class{c}_mean_sd': f'{CLASS_NAMES[c][:16]} (Mean±SD)' for c in CLASSES}
rename['Overall_mean_sd'] = 'Overall (Mean±SD)'
rename['p_value']         = 'p-value'
rename['Significance']    = 'Sig.'
df_summary.rename(columns=rename, inplace=True)
df_summary.to_csv(os.path.join(OUT_DIR, 'data_description_summary.csv'), index=False)
print(f'\nSaved: data_description_summary.csv  ← manuscript-ready table')

# ── 6. Publication-quality figure: Mean±SD box + significance ─────────────────
sig_feats = df_stat[df_stat['Significant'] == 'Yes']['Feature'].tolist()
n_plot    = min(len(sig_feats), 12)
plot_feats = sig_feats[:n_plot]

if plot_feats:
    n_cols = 4
    n_rows = int(np.ceil(n_plot / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 3.5))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_plot == 1 else axes.flatten()

    for i, feat in enumerate(plot_feats):
        ax   = axes[i]
        data = [df[df[TARGET_COL] == c][feat].dropna().values for c in CLASSES]
        bp   = ax.boxplot(data, patch_artist=True, notch=False,
                          medianprops=dict(color='black', linewidth=1.5),
                          whiskerprops=dict(linewidth=1),
                          flierprops=dict(marker='o', markersize=2, alpha=0.3))
        for patch, clr in zip(bp['boxes'], PALETTE):
            patch.set_facecolor(clr)
            patch.set_alpha(0.75)

        # Overlay mean ± SD as error bars
        means = [np.mean(g) for g in data]
        sds   = [np.std(g) for g in data]
        ax.errorbar(range(1, N_CLASSES+1), means, yerr=sds,
                    fmt='D', color='black', markersize=4,
                    elinewidth=1.2, capsize=3, zorder=5)

        ax.set_xticks(range(1, N_CLASSES+1))
        ax.set_xticklabels([f'C{c}' for c in CLASSES], fontsize=8)
        ax.set_title(feat, fontsize=10, fontweight='bold')

        # p-value annotation
        row_s = df_stat[df_stat['Feature'] == feat].iloc[0]
        p_txt = f'{row_s["Test"][:2]}: p={row_s["p_value"]:.2e} {row_s["Significance"]}'
        ax.set_xlabel(p_txt, fontsize=7, color='#555')
        ax.spines[['top','right']].set_visible(False)

    # Hide unused axes
    for j in range(n_plot, len(axes)):
        axes[j].set_visible(False)

    # Legend
    handles = [mpatches.Patch(color=PALETTE[c], alpha=0.75,
                label=f'C{c}: {CLASS_NAMES[c]}') for c in CLASSES]
    fig.legend(handles=handles, loc='lower right', fontsize=8, ncol=2)
    fig.suptitle(
        f'Feature Distribution by Class (Mean±SD + Box, n={len(df)})\n'
        f'Top-{n_plot} significant features | ♦ = Mean',
        fontsize=12, fontweight='bold')
    plt.tight_layout()
    savefig(fig, '02b_significant_features_boxplot.png')

# ── 7. Heatmap of Mean values across classes (standardised) ───────────────────
mean_mat = pd.DataFrame(
    {f'C{c}': [df[df[TARGET_COL]==c][f].mean() for f in CBC_FEATURES]
     for c in CLASSES},
    index=CBC_FEATURES)
# Z-score across classes (row-wise) for visualisation
mean_z = mean_mat.subtract(mean_mat.mean(axis=1), axis=0).divide(
    mean_mat.std(axis=1).replace(0, 1), axis=0)

fig, ax = plt.subplots(figsize=(7, 9))
sns.heatmap(mean_z, annot=mean_mat.round(2), fmt='.2f',
            cmap='RdBu_r', center=0, linewidths=0.4, ax=ax,
            annot_kws={'size': 8},
            xticklabels=[f'C{c}\n{CLASS_NAMES[c][:12]}' for c in CLASSES])
ax.set_title('Feature Means by Class\n(colour = z-score across classes; value = raw mean)',
             fontsize=11, fontweight='bold')
plt.tight_layout()
savefig(fig, '02c_mean_heatmap.png')

# ════════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Stratified 80/20 split
# ════════════════════════════════════════════════════════════════════════════════
print('\n' + '='*65)
print('STEP 3: Stratified 80/20 Train/Test Split')
print('='*65)

X_all = df[CBC_FEATURES].values
y_all = df[TARGET_COL].values

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.20, random_state=SEED, stratify=y_all
)
print(f'Training : {len(y_train)} samples  {dict(zip(*np.unique(y_train, return_counts=True)))}')
print(f'Test     : {len(y_test)}  samples  {dict(zip(*np.unique(y_test,  return_counts=True)))}')

# ════════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Imputation (train-only fit)
# ════════════════════════════════════════════════════════════════════════════════
print('\n' + '='*65)
print('STEP 4: Imputation (median, fitted on training set only)')
print('='*65)

imputer = SimpleImputer(strategy='median')
imputer.fit(X_train_raw)
X_train_imp = imputer.transform(X_train_raw)
X_test_imp  = imputer.transform(X_test_raw)
print(f'Medians fitted on {len(X_train_raw)} training samples.')
print(f'Train NaN after impute : {np.isnan(X_train_imp).sum()}')
print(f'Test  NaN after impute : {np.isnan(X_test_imp).sum()}')

# ════════════════════════════════════════════════════════════════════════════════
# SECTION 7 — Feature selection (MDI + SHAP + RFE-LR consensus)
# ════════════════════════════════════════════════════════════════════════════════
print('\n' + '='*65)
print('STEP 5: Feature Selection — MDI + SHAP + RFE-LR')
print('='*65)
TOP_N = 12

# 7a. Scale for RFE-LR (fit on train only)
sc_rfe = RobustScaler()
X_train_sc = sc_rfe.fit_transform(X_train_imp)
X_test_sc  = sc_rfe.transform(X_test_imp)

# 7b. MDI (Random Forest importance)
print('  Computing MDI (Random Forest)...')
rf_sel = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)
rf_sel.fit(X_train_imp, y_train)
mdi_rank = np.argsort(-rf_sel.feature_importances_).argsort() + 1

# 7c. SHAP (TreeExplainer on same RF)
print('  Computing SHAP values...')
shap_explainer_fs = shap.TreeExplainer(rf_sel)
shap_vals_fs = shap_explainer_fs.shap_values(X_train_imp)
# Handle both SHAP formats:
# Old (<0.44): list of (n_samples, n_features) per class
# New (>=0.44): ndarray (n_samples, n_features, n_classes)
if isinstance(shap_vals_fs, list):
    shap_mean = np.mean([np.abs(s).mean(axis=0) for s in shap_vals_fs], axis=0)
else:
    shap_mean = np.abs(shap_vals_fs).mean(axis=0).mean(axis=-1)  # → (n_features,)
shap_rank = np.argsort(-shap_mean).argsort() + 1

# 7d. RFE with Logistic Regression
print('  Computing RFE-LR...')
lr_rfe = LR_sel(max_iter=1000, random_state=SEED, C=0.1)
rfe    = RFE(lr_rfe, n_features_to_select=TOP_N, step=1)
rfe.fit(X_train_sc, y_train)
rfe_rank = rfe.ranking_

# Consensus (mean rank)
consensus = (mdi_rank + shap_rank + rfe_rank) / 3.0
top_idx   = np.argsort(consensus)[:TOP_N]
TOP_FEATS = [CBC_FEATURES[i] for i in np.sort(top_idx)]

# Print ranking table
print(f'\n{"Feature":<14} {"MDI":>5} {"SHAP":>5} {"RFE":>5} {"Mean":>7}  {"Selected"}')
print('-' * 50)
for i in np.argsort(consensus):
    marker = f'◄ Top-{TOP_N}' if i in top_idx else ''
    print(f'{CBC_FEATURES[i]:<14} {int(mdi_rank[i]):>5} {int(shap_rank[i]):>5} '
          f'{int(rfe_rank[i]):>5} {consensus[i]:>7.2f}  {marker}')
print(f'\nTop-{TOP_N} features: {TOP_FEATS}')

# Save ranking table
df_ranks = pd.DataFrame({
    'Feature':   CBC_FEATURES,
    'MDI_rank':  mdi_rank,
    'SHAP_rank': shap_rank,
    'RFE_rank':  rfe_rank,
    'Consensus': consensus,
    'Selected':  ['Yes' if i in top_idx else 'No' for i in range(len(CBC_FEATURES))],
}).sort_values('Consensus')
df_ranks.to_csv(os.path.join(OUT_DIR, 'feature_ranking.csv'), index=False)

# Feature ranking heatmap
fig, ax = plt.subplots(figsize=(9, 7))
heat_data = df_ranks.set_index('Feature')[['MDI_rank','SHAP_rank','RFE_rank']].astype(float)
sns.heatmap(heat_data, annot=True, fmt='.0f', cmap='RdYlGn_r',
            linewidths=0.4, ax=ax, annot_kws={'size': 9})
for i, feat in enumerate(df_ranks['Feature']):
    if feat in TOP_FEATS:
        ax.add_patch(plt.Rectangle((0, i), 3, 1,
                     fill=True, color='gold', alpha=0.2, lw=0))
ax.set_title(f'Feature Selection Ranking (3 Methods)\nGold = Top-{TOP_N} selected',
             fontsize=12, fontweight='bold')
plt.tight_layout()
savefig(fig, '03_feature_ranking.png')

# Feature index arrays
FEAT_SETS = {
    'All Features': np.arange(len(CBC_FEATURES)),
    f'Top-{TOP_N}':  np.sort(top_idx),
}

# ════════════════════════════════════════════════════════════════════════════════
# SECTION 8 — Metric helpers
# ════════════════════════════════════════════════════════════════════════════════
def geo_gmean(y_true, y_pred):
    """Geometric mean of per-class sensitivities (correct formula)."""
    log_sens = []
    for c in CLASSES:
        n_c = np.sum(y_true == c)
        hit = np.sum((y_true == c) & (y_pred == c))
        s   = max(hit / n_c, 1e-10) if n_c > 0 else 1e-10
        log_sens.append(np.log(s))
    return float(np.exp(np.mean(log_sens)))

def macro_spec(y_true, y_pred):
    """Mean per-class specificity = TN/(TN+FP)."""
    specs = []
    for c in CLASSES:
        TN = int(np.sum((y_true != c) & (y_pred != c)))
        FP = int(np.sum((y_true != c) & (y_pred == c)))
        specs.append(TN / (TN + FP) if (TN + FP) > 0 else 0.0)
    return float(np.mean(specs))

def compute_all_metrics(y_true, y_pred, y_prob):
    """Return dict of all scalar metrics. y_pred MUST be 1-D."""
    y_pred = np.ravel(y_pred)   # guard against (n,1) from CatBoost
    y_bin  = label_binarize(y_true, classes=CLASSES)
    return {
        'Accuracy':    accuracy_score(y_true, y_pred),
        'F1 (macro)':  f1_score(y_true, y_pred, average='macro', zero_division=0),
        'MCC':         matthews_corrcoef(y_true, y_pred),
        'G-mean':      geo_gmean(y_true, y_pred),
        'Sens (macro)':recall_score(y_true, y_pred, average='macro', zero_division=0),
        'Spec (macro)':macro_spec(y_true, y_pred),
        'AUC-ROC':     roc_auc_score(y_bin, y_prob, multi_class='ovr', average='macro'),
        'AUC-PR':      average_precision_score(y_bin, y_prob, average='macro'),
    }

# ════════════════════════════════════════════════════════════════════════════════
# SECTION 9 — Classifier & resampler definitions
# ════════════════════════════════════════════════════════════════════════════════
def make_classifiers():
    return {
        'RF':         RandomForestClassifier(
                          n_estimators=300, max_depth=None, min_samples_leaf=1,
                          max_features='sqrt', random_state=SEED, n_jobs=-1),
        'XGB':        XGBClassifier(
                          n_estimators=300, max_depth=6, learning_rate=0.1,
                          subsample=0.8, colsample_bytree=0.8,
                          eval_metric='mlogloss', random_state=SEED, n_jobs=-1, verbosity=0),
        'CatBoost':   CatBoostClassifier(
                          iterations=300, depth=6, learning_rate=0.1,
                          random_seed=SEED, verbose=0),
        'SVM':        SVC(kernel='rbf', C=1.0, gamma='scale',
                          probability=True, random_state=SEED),
        'LR':         LogisticRegression(
                          C=1.0, max_iter=1000, solver='lbfgs',
                          multi_class='multinomial', random_state=SEED, n_jobs=-1),
        'MLP':        MLPClassifier(
                          hidden_layer_sizes=(128, 64), activation='relu',
                          learning_rate_init=0.003, alpha=0.001,
                          max_iter=500, early_stopping=True, random_state=SEED),
        'NaiveBayes': GaussianNB(var_smoothing=1.77e-9),
        'DT':         DecisionTreeClassifier(
                          max_depth=5, criterion='gini', min_samples_leaf=7,
                          random_state=SEED),
    }

RESAMPLERS = {
    'None':      None,
    'ROS':       RandomOverSampler(random_state=SEED),
    'SMOTE':     SMOTE(random_state=SEED, k_neighbors=5),
    'ADASYN':    ADASYN(random_state=SEED),
    'SMOTE-ENN': SMOTEENN(random_state=SEED),
}

# ════════════════════════════════════════════════════════════════════════════════
# SECTION 10 — 5-fold CV grid search (80 configs)
# ════════════════════════════════════════════════════════════════════════════════
print('\n' + '='*70)
print(f'STEP 6: Grid Search — {len(make_classifiers())} classifiers × '
      f'{len(RESAMPLERS)} resampling × {len(FEAT_SETS)} feature sets = '
      f'{len(make_classifiers())*len(RESAMPLERS)*len(FEAT_SETS)} configurations')
print('  5-fold stratified cross-validation | Scored by MCC')
print('='*70)

SKF = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
METRIC_KEYS = ['Accuracy','F1 (macro)','MCC','G-mean','Sens (macro)','Spec (macro)','AUC-ROC','AUC-PR']

all_results = []
config_num  = 0
total       = len(make_classifiers()) * len(RESAMPLERS) * len(FEAT_SETS)

for clf_name, feat_name in product(make_classifiers().keys(), FEAT_SETS.keys()):
    fidx = FEAT_SETS[feat_name]

    for res_name in RESAMPLERS:
        config_num += 1
        tag = (f'[{config_num:02d}/{total}] {clf_name:<12} + '
               f'{res_name:<10} + {feat_name}')
        t0  = time.time()
        fold_m = {k: [] for k in METRIC_KEYS}

        try:
            for tr_idx, val_idx in SKF.split(X_train_imp, y_train):
                # ── Within-fold: select features ──────────────────────────────
                Xtr = X_train_imp[tr_idx][:, fidx]
                Xvl = X_train_imp[val_idx][:, fidx]
                ytr = y_train[tr_idx]
                yvl = y_train[val_idx]

                # ── Scale ─────────────────────────────────────────────────────
                sc  = RobustScaler()
                Xtr = sc.fit_transform(Xtr)
                Xvl = sc.transform(Xvl)

                # ── Resample (training fold only) ──────────────────────────────
                resampler = RESAMPLERS[res_name]
                if resampler is not None:
                    rs = clone(resampler)
                    try:
                        Xtr, ytr = rs.fit_resample(Xtr, ytr)
                    except Exception:
                        pass  # e.g. ADASYN fails if neighbours > samples

                # ── Fit + predict ──────────────────────────────────────────────
                clf  = make_classifiers()[clf_name]
                clf.fit(Xtr, ytr)
                yp   = np.ravel(clf.predict(Xvl))          # ← ravel: CatBoost fix
                yprb = clf.predict_proba(Xvl)

                m = compute_all_metrics(yvl, yp, yprb)
                for k in METRIC_KEYS:
                    fold_m[k].append(m[k])

            row = {
                'Resampling':  res_name,
                'Classifier':  clf_name,
                'Feature Set': feat_name,
                **{k: round(np.mean(v), 6) for k, v in fold_m.items()},
                **{k+'_sd': round(np.std(v), 6) for k, v in fold_m.items()},
                'Time(s)': round(time.time()-t0, 1),
                'Status':   'OK',
            }
        except Exception as e:
            row = {
                'Resampling': res_name, 'Classifier': clf_name,
                'Feature Set': feat_name,
                **{k: 0 for k in METRIC_KEYS},
                **{k+'_sd': 0 for k in METRIC_KEYS},
                'Time(s)': round(time.time()-t0, 1),
                'Status': f'ERROR: {e}',
            }

        all_results.append(row)
        s = row['Status']
        if s == 'OK':
            print(f'{tag}  Acc={row["Accuracy"]*100:.1f}%  MCC={row["MCC"]:.3f}  '
                  f'Gm={row["G-mean"]:.3f}  AUC-ROC={row["AUC-ROC"]:.3f}  '
                  f'({row["Time(s)"]}s)')
        else:
            print(f'{tag}  {s}')

# ════════════════════════════════════════════════════════════════════════════════
# SECTION 11 — Results table
# ════════════════════════════════════════════════════════════════════════════════
DISP_COLS = ['Resampling','Classifier','Feature Set',
             'Accuracy','F1 (macro)','MCC','G-mean',
             'AUC-ROC','AUC-PR','Sens (macro)','Spec (macro)']

df_results = pd.DataFrame(all_results)
df_ok      = df_results[df_results['Status']=='OK'].copy()
df_sorted  = df_ok.sort_values('MCC', ascending=False).reset_index(drop=True)

print('\n' + '='*110)
print('RESULTS — All configurations ranked by MCC (5-fold CV mean)')
print('='*110)
print(df_sorted[DISP_COLS].to_string(index=False, float_format='{:.3f}'.format))

# Save full results CSV
csv_path = os.path.join(OUT_DIR, 'grid_search_results.csv')
df_results.to_csv(csv_path, index=False)
print(f'\nSaved: {csv_path}')

# Top-10 print
print('\n' + '='*110)
print('TOP-10 CONFIGURATIONS — Manuscript Table 2')
print('='*110)
print(df_sorted[DISP_COLS].head(10).to_string(index=False, float_format='{:.3f}'.format))

# ── Heatmap of MCC by Classifier × Resampling ─────────────────────────────────
for feat_label in FEAT_SETS.keys():
    sub   = df_sorted[df_sorted['Feature Set']==feat_label].copy()
    pivot = sub.pivot_table(index='Resampling', columns='Classifier',
                            values='MCC', aggfunc='mean')
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn',
                vmin=0.45, vmax=0.70, linewidths=0.5, ax=ax,
                annot_kws={'size': 9, 'fontweight': 'bold'})
    ax.set_title(f'MCC Heatmap — {feat_label}', fontsize=12, fontweight='bold')
    plt.tight_layout()
    savefig(fig, f'04_mcc_heatmap_{feat_label.replace(" ","_").replace("-","")}.png')

# ── Bar chart: top-5 configs × key metrics ────────────────────────────────────
top5        = df_sorted.head(5)
bar_metrics = ['MCC','G-mean','AUC-PR','Sens (macro)','Spec (macro)']
x           = np.arange(len(top5))
width       = 0.15
clrs_bar    = ['#E91E63','#FF9800','#2196F3','#4CAF50','#9C27B0']

fig, ax = plt.subplots(figsize=(14, 6))
for i, (met, clr) in enumerate(zip(bar_metrics, clrs_bar)):
    vals = [float(r[met]) for _, r in top5.iterrows()]
    bars = ax.bar(x+i*width, vals, width, label=met, color=clr,
                  alpha=0.85, edgecolor='white')
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.004,
                f'{v:.3f}', ha='center', va='bottom', fontsize=7, rotation=90)
ax.set_xticks(x + width*2)
ax.set_xticklabels(
    [f'{r["Classifier"]}\n+{r["Resampling"]}\n+{r["Feature Set"]}'
     for _, r in top5.iterrows()], fontsize=8)
ax.set_ylim(0, 1.05)
ax.set_ylabel('Score')
ax.set_title('Top-5 Configurations — Key Metrics (5-fold CV)', fontweight='bold')
ax.legend(fontsize=9)
ax.spines[['top','right']].set_visible(False)
plt.tight_layout()
savefig(fig, '05_top5_bar.png')

# ════════════════════════════════════════════════════════════════════════════════
# SECTION 12 — Best config: retrain on full training set & evaluate on test
# ════════════════════════════════════════════════════════════════════════════════
print('\n' + '='*70)
best_row  = df_sorted.iloc[0]
BEST_CLF  = best_row['Classifier']
BEST_RES  = best_row['Resampling']
BEST_FEAT = best_row['Feature Set']
BEST_FIDX = FEAT_SETS[BEST_FEAT]

print(f'STEP 7: Best Configuration → {BEST_CLF} + {BEST_RES} + {BEST_FEAT}')
print(f'  5-fold CV MCC = {best_row["MCC"]:.4f}')
print('='*70)

X_tr_best = X_train_imp[:, BEST_FIDX]
X_te_best = X_test_imp[:, BEST_FIDX]
BEST_FEAT_NAMES = [CBC_FEATURES[i] for i in BEST_FIDX]

# Scale
sc_final = RobustScaler()
X_tr_sc  = sc_final.fit_transform(X_tr_best)
X_te_sc  = sc_final.transform(X_te_best)

# Resample
X_tr_rs, y_tr_rs = X_tr_sc, y_train
if RESAMPLERS[BEST_RES] is not None:
    try:
        X_tr_rs, y_tr_rs = clone(RESAMPLERS[BEST_RES]).fit_resample(X_tr_sc, y_train)
        print(f'  After resampling: {len(y_tr_rs)} samples  '
              f'{dict(zip(*np.unique(y_tr_rs, return_counts=True)))}')
    except Exception as e:
        print(f'  Resampler warning: {e}')

# Fit final model
BEST_MODEL = make_classifiers()[BEST_CLF]
BEST_MODEL.fit(X_tr_rs, y_tr_rs)
y_pred = np.ravel(BEST_MODEL.predict(X_te_sc))         # ← ravel: CatBoost fix
y_prob = BEST_MODEL.predict_proba(X_te_sc)

# ── Overall metrics ────────────────────────────────────────────────────────────
m_test = compute_all_metrics(y_test, y_pred, y_prob)
print(f'\nHeld-out test set (n={len(y_test)}):')
for k, v in m_test.items():
    print(f'  {k:<14}: {v:.4f}{"  (= {:.1f}%)".format(v*100) if "Acc" in k else ""}')

# ── Per-class metrics table ────────────────────────────────────────────────────
print('\nPer-class metrics (one-vs-rest):')
y_test_bin = label_binarize(y_test, classes=CLASSES)

hdr = (f'{"Class":<28} {"N":>5} {"Sens":>6} {"Spec":>6} {"PPV":>6} '
       f'{"NPV":>6} {"F1":>6} {"AUC-ROC":>8} {"AUC-PR":>7}')
print(hdr)
print('-' * 90)

per_class_rows = []
ss = {}
for c in CLASSES:
    n_c = int(np.sum(y_test == c))
    TP  = int(np.sum((y_test == c) & (y_pred == c)))
    FN  = int(np.sum((y_test == c) & (y_pred != c)))
    FP  = int(np.sum((y_test != c) & (y_pred == c)))
    TN  = int(np.sum((y_test != c) & (y_pred != c)))
    sens  = TP/(TP+FN) if (TP+FN) > 0 else 0
    spec_ = TN/(TN+FP) if (TN+FP) > 0 else 0
    ppv   = TP/(TP+FP) if (TP+FP) > 0 else 0
    npv   = TN/(TN+FN) if (TN+FN) > 0 else 0
    f1_c  = 2*ppv*sens/(ppv+sens) if (ppv+sens) > 0 else 0
    auc_c = roc_auc_score(y_test_bin[:,c], y_prob[:,c])
    pr_c  = average_precision_score(y_test_bin[:,c], y_prob[:,c])
    ss[c] = dict(TP=TP, FN=FN, FP=FP, TN=TN,
                 Sens=sens, Spec=spec_, PPV=ppv, NPV=npv)
    lbl = f'{c}: {CLASS_NAMES[c]}'
    print(f'{lbl:<28} {n_c:>5} {sens:>6.3f} {spec_:>6.3f} {ppv:>6.3f} '
          f'{npv:>6.3f} {f1_c:>6.3f} {auc_c:>8.3f} {pr_c:>7.3f}')
    per_class_rows.append({'Class': lbl, 'N': n_c,
                           'Sens': round(sens,4), 'Spec': round(spec_,4),
                           'PPV':  round(ppv,4),  'NPV':  round(npv,4),
                           'F1':   round(f1_c,4),
                           'AUC-ROC': round(auc_c,4),
                           'AUC-PR':  round(pr_c,4)})

# Save per-class table
df_pc = pd.DataFrame(per_class_rows)
df_pc.to_csv(os.path.join(OUT_DIR, 'per_class_metrics.csv'), index=False)
print(f'\nSaved: per_class_metrics.csv')

# ── Overall summary row ────────────────────────────────────────────────────────
print(f'\n{"Overall (macro)":<28} {len(y_test):>5} '
      f'{m_test["Sens (macro)"]:>6.3f} {m_test["Spec (macro)"]:>6.3f} '
      f'{"---":>6} {"---":>6} '
      f'{m_test["F1 (macro)"]:>6.3f} {m_test["AUC-ROC"]:>8.3f} '
      f'{m_test["AUC-PR"]:>7.3f}')
print(f'\n  Accuracy : {m_test["Accuracy"]*100:.2f}%')
print(f'  MCC      : {m_test["MCC"]:.4f}')
print(f'  G-mean   : {m_test["G-mean"]:.4f}')

# ════════════════════════════════════════════════════════════════════════════════
# SECTION 13 — Confusion matrix
# ════════════════════════════════════════════════════════════════════════════════
cm_cnt = confusion_matrix(y_test, y_pred)
cm_pct = cm_cnt.astype(float) / cm_cnt.sum(axis=1, keepdims=True) * 100
cls_labels = [CLASS_NAMES[c] for c in CLASSES]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, data, fmt, title in zip(
    axes, [cm_cnt, cm_pct], ['d', '.1f'],
    ['Confusion Matrix (Counts)', 'Confusion Matrix (Row %)']
):
    sns.heatmap(data, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=cls_labels, yticklabels=cls_labels,
                linewidths=0.6, ax=ax, annot_kws={'size': 11})
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('True', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
plt.suptitle(f'Best: {BEST_CLF} + {BEST_RES} + {BEST_FEAT}',
             fontsize=11, y=1.02)
plt.tight_layout()
savefig(fig, '06_confusion_matrix.png')

# ════════════════════════════════════════════════════════════════════════════════
# SECTION 14 — Per-class 2×2 confusion matrices with metrics
# ════════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(13, 10))
axes = axes.flatten()
for c, ax in zip(CLASSES, axes):
    d  = ss[c]
    cm_c = np.array([[d['TN'], d['FP']], [d['FN'], d['TP']]])
    ann  = np.array([[f'TN\n{d["TN"]}', f'FP\n{d["FP"]}'],
                     [f'FN\n{d["FN"]}', f'TP\n{d["TP"]}']])
    sns.heatmap(cm_c, annot=ann, fmt='', cmap='RdYlGn', center=0,
                vmin=0, ax=ax, linewidths=0.8, annot_kws={'size': 12})
    ax.set_xticklabels([f'Pred NOT {c}', f'Pred {c}'], fontsize=9)
    ax.set_yticklabels([f'True NOT {c}', f'True {c}'], fontsize=9, rotation=0)
    ax.set_title(f'Class {c}: {CLASS_NAMES[c]}', fontsize=11, fontweight='bold')
    info = (f'Sens={d["Sens"]:.3f}  Spec={d["Spec"]:.3f}\n'
            f'PPV={d["PPV"]:.3f}   NPV={d["NPV"]:.3f}')
    ax.text(1, -0.22, info, transform=ax.transAxes, ha='right',
            fontsize=9, color='#333', style='italic')
plt.suptitle(f'Per-Class 2×2 Matrices — {BEST_CLF} + {BEST_RES}',
             fontsize=12, fontweight='bold')
plt.tight_layout()
savefig(fig, '07_per_class_cm.png')

# ════════════════════════════════════════════════════════════════════════════════
# SECTION 15 — ROC curves
# ════════════════════════════════════════════════════════════════════════════════
clrs_roc = ['#4CAF50','#2196F3','#FF9800','#E91E63']
fig, axes = plt.subplots(1, N_CLASSES, figsize=(5*N_CLASSES, 5))
for c, ax, clr in zip(CLASSES, axes, clrs_roc):
    fpr, tpr, _ = roc_curve(y_test_bin[:, c], y_prob[:, c])
    auc_v       = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=clr, lw=2.5, label=f'AUC = {auc_v:.3f}')
    ax.fill_between(fpr, tpr, alpha=0.10, color=clr)
    ax.plot([0,1],[0,1],'k--', lw=0.8)
    ax.set_xlabel('1 − Specificity (FPR)', fontsize=10)
    ax.set_ylabel('Sensitivity (TPR)', fontsize=10)
    ax.set_title(f'ROC — Class {c}\n{CLASS_NAMES[c]}',
                 fontsize=10, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_xlim([-0.02,1.02]); ax.set_ylim([-0.02,1.02])
plt.suptitle(f'ROC Curves — {BEST_CLF} + {BEST_RES} + {BEST_FEAT}',
             fontsize=12, fontweight='bold')
plt.tight_layout()
savefig(fig, '08_roc_curves.png')

# ════════════════════════════════════════════════════════════════════════════════
# SECTION 16 — Precision-Recall curves
# ════════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, N_CLASSES, figsize=(5*N_CLASSES, 5))
for c, ax, clr in zip(CLASSES, axes, clrs_roc):
    prec, rec, _ = precision_recall_curve(y_test_bin[:, c], y_prob[:, c])
    ap_v         = average_precision_score(y_test_bin[:, c], y_prob[:, c])
    ax.plot(rec, prec, color=clr, lw=2.5, label=f'AP = {ap_v:.3f}')
    ax.fill_between(rec, prec, alpha=0.10, color=clr)
    ax.set_xlabel('Recall (Sensitivity)', fontsize=10)
    ax.set_ylabel('Precision (PPV)', fontsize=10)
    ax.set_title(f'PR Curve — Class {c}\n{CLASS_NAMES[c]}',
                 fontsize=10, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_xlim([-0.02,1.02]); ax.set_ylim([-0.02,1.02])
plt.suptitle('Precision-Recall Curves', fontsize=12, fontweight='bold')
plt.tight_layout()
savefig(fig, '09_pr_curves.png')

# ════════════════════════════════════════════════════════════════════════════════
# SECTION 17 — XAI: SHAP
# ════════════════════════════════════════════════════════════════════════════════
print('\n' + '='*70)
print('STEP 8: Explainable AI — SHAP')
print('='*70)

tree_types = (RandomForestClassifier, DecisionTreeClassifier,
              XGBClassifier, CatBoostClassifier)

if isinstance(BEST_MODEL, tree_types):
    shap_exp    = shap.TreeExplainer(BEST_MODEL)
    shap_values = shap_exp.shap_values(X_te_sc)
else:
    print('Using KernelExplainer (slower)...')
    bg = shap.sample(X_tr_sc, 100, random_state=SEED)
    shap_exp    = shap.KernelExplainer(BEST_MODEL.predict_proba, bg)
    shap_values = shap_exp.shap_values(X_te_sc[:200])

# Handle both SHAP output formats
if isinstance(shap_values, list):
    # Old format: list of (n, p) per class
    shap_abs_mean = np.mean([np.abs(s).mean(axis=0) for s in shap_values], axis=0)
    shap_plot_vals = shap_values
    shap_plot_arr  = None
else:
    # New format: (n, p, k)
    shap_abs_mean  = np.abs(shap_values).mean(axis=(0, 2))
    shap_plot_vals = None
    shap_plot_arr  = shap_values

# SHAP feature importance bar
feat_imp_df = pd.DataFrame({
    'Feature':    BEST_FEAT_NAMES,
    'Mean |SHAP|':shap_abs_mean,
}).sort_values('Mean |SHAP|', ascending=False)
feat_imp_df.to_csv(os.path.join(OUT_DIR, 'shap_feature_importance.csv'), index=False)

fig, ax = plt.subplots(figsize=(8, 5))
colors = [PALETTE[i%4] for i in range(len(BEST_FEAT_NAMES))]
bars = ax.barh(feat_imp_df['Feature'][::-1], feat_imp_df['Mean |SHAP|'][::-1],
               color=colors[::-1], edgecolor='white', alpha=0.85)
ax.set_xlabel('Mean |SHAP value| (averaged across classes)', fontsize=10)
ax.set_title(f'SHAP Feature Importance — {BEST_CLF} + {BEST_FEAT}',
             fontsize=11, fontweight='bold')
ax.spines[['top','right']].set_visible(False)
plt.tight_layout()
savefig(fig, '10_shap_importance.png')

# Per-class SHAP summary plots
for c_idx, c in enumerate(CLASSES):
    try:
        fig = plt.figure(figsize=(8, 5))
        if shap_plot_vals is not None:
            shap.summary_plot(shap_plot_vals[c_idx], X_te_sc,
                              feature_names=BEST_FEAT_NAMES,
                              show=False, plot_type='dot')
        else:
            shap.summary_plot(shap_plot_arr[:,:,c_idx], X_te_sc,
                              feature_names=BEST_FEAT_NAMES,
                              show=False, plot_type='dot')
        plt.title(f'SHAP — Class {c}: {CLASS_NAMES[c]}', fontsize=11, fontweight='bold')
        plt.tight_layout()
        savefig(fig, f'11_shap_class{c}_{CLASS_NAMES[c].replace(" ","_").replace("/","")}.png')
    except Exception as e:
        print(f'  SHAP summary plot class {c} skipped: {e}')

# ════════════════════════════════════════════════════════════════════════════════
# SECTION 18 — Save model bundle
# ════════════════════════════════════════════════════════════════════════════════
print('\n' + '='*70)
print('STEP 9: Save model bundle')
print('='*70)

bundle = {
    'model':            BEST_MODEL,
    'scaler':           sc_final,
    'imputer':          imputer,
    'all_features':     CBC_FEATURES,
    'best_features':    BEST_FEAT_NAMES,
    'best_feat_idxs':   BEST_FIDX.tolist(),
    'log_feats':        [],          # no log1p in this pipeline (RobustScaler used)
    'shap_explainer':   shap_exp,
    'class_names':      CLASS_NAMES,
    'n_classes':        N_CLASSES,
    'classifier':       BEST_CLF,
    'resampling':       BEST_RES,
    'feature_set':      BEST_FEAT,
    'cv_metrics':       {k: best_row[k] for k in METRIC_KEYS},
    'test_metrics':     m_test,
    'per_class':        per_class_rows,
}
bundle_path = os.path.join(OUT_DIR, 'model_bundle.pkl')
with open(bundle_path, 'wb') as f:
    pickle.dump(bundle, f)
print(f'Saved: {bundle_path}')

# ════════════════════════════════════════════════════════════════════════════════
# SECTION 19 — External validation (n = 625)
# ════════════════════════════════════════════════════════════════════════════════
print('\n' + '='*70)
print('STEP 10: External Validation (n = 625)')
print('  CRITICAL: same imputer + scaler fitted on training set only — NO refitting')
print('='*70)

print('\nUpload the external validation file (e.g. "External Dataset-625.xlsx"):')
ext_uploaded = files.upload()
ext_fname    = list(ext_uploaded.keys())[0]

df_ext = pd.read_excel(ext_fname)
df_ext.columns = df_ext.columns.str.strip()
print(f'\nLoaded external: {df_ext.shape[0]} rows × {df_ext.shape[1]} columns')

# Keep only CBC features present in the model's training features
ext_cbc = [c for c in CBC_FEATURES if c in df_ext.columns]
missing  = [c for c in CBC_FEATURES if c not in df_ext.columns]
if missing:
    print(f'  WARNING: these training features are missing in external file: {missing}')
    print(f'  Missing features will be imputed with training-set medians.')

X_ext_raw = np.zeros((len(df_ext), len(CBC_FEATURES)))
for i, feat in enumerate(CBC_FEATURES):
    if feat in df_ext.columns:
        X_ext_raw[:, i] = pd.to_numeric(df_ext[feat], errors='coerce').values
    else:
        X_ext_raw[:, i] = np.nan  # will be filled by imputer

# Clip negatives
X_ext_raw = np.clip(X_ext_raw, 0, None)

y_ext = df_ext['Label'].astype(int).values
print(f'External label distribution: {dict(zip(*np.unique(y_ext, return_counts=True)))}')

# ── Apply TRAINING imputer → TRAINING scaler → select features ────────────────
X_ext_imp = imputer.transform(X_ext_raw)        # training medians, no refit
X_ext_f   = X_ext_imp[:, BEST_FIDX]
X_ext_sc  = sc_final.transform(X_ext_f)         # training scaler, no refit

print(f'\nPreprocessing (train-fitted):')
print(f'  Imputer  : training-set medians applied (no refit)')
print(f'  Scaler   : RobustScaler trained on {len(y_train)} samples applied')
print(f'  Features : {BEST_FEAT} ({len(BEST_FIDX)} features)')
print(f'  NaN after impute: {np.isnan(X_ext_sc).sum()}')

# ── Predict ────────────────────────────────────────────────────────────────────
y_ext_pred = np.ravel(BEST_MODEL.predict(X_ext_sc))     # ravel: CatBoost fix
y_ext_prob = BEST_MODEL.predict_proba(X_ext_sc)

# ── Overall metrics ────────────────────────────────────────────────────────────
m_ext = compute_all_metrics(y_ext, y_ext_pred, y_ext_prob)
print(f'\nExternal validation results (n={len(y_ext)}):')
for k, v in m_ext.items():
    print(f'  {k:<14}: {v:.4f}{"  (= {:.1f}%)".format(v*100) if "Acc" in k else ""}')

# ── Per-class metrics ──────────────────────────────────────────────────────────
print('\nPer-class metrics (external, one-vs-rest):')
y_ext_bin = label_binarize(y_ext, classes=CLASSES)

hdr_e = (f'{"Class":<28} {"N":>5} {"Sens":>6} {"Spec":>6} {"PPV":>6} '
         f'{"NPV":>6} {"F1":>6} {"AUC-ROC":>8} {"AUC-PR":>7}')
print(hdr_e)
print('-' * 90)

ext_pc_rows = []
ss_ext = {}
for c in CLASSES:
    n_c  = int(np.sum(y_ext == c))
    TP   = int(np.sum((y_ext == c) & (y_ext_pred == c)))
    FN   = int(np.sum((y_ext == c) & (y_ext_pred != c)))
    FP   = int(np.sum((y_ext != c) & (y_ext_pred == c)))
    TN   = int(np.sum((y_ext != c) & (y_ext_pred != c)))
    sens  = TP/(TP+FN) if (TP+FN) > 0 else 0
    spec_ = TN/(TN+FP) if (TN+FP) > 0 else 0
    ppv   = TP/(TP+FP) if (TP+FP) > 0 else 0
    npv   = TN/(TN+FN) if (TN+FN) > 0 else 0
    f1_c  = 2*ppv*sens/(ppv+sens) if (ppv+sens) > 0 else 0
    auc_c = roc_auc_score(y_ext_bin[:, c], y_ext_prob[:, c])
    pr_c  = average_precision_score(y_ext_bin[:, c], y_ext_prob[:, c])
    ss_ext[c] = dict(TP=TP, FN=FN, FP=FP, TN=TN,
                     Sens=sens, Spec=spec_, PPV=ppv, NPV=npv)
    lbl = f'{c}: {CLASS_NAMES[c]}'
    print(f'{lbl:<28} {n_c:>5} {sens:>6.3f} {spec_:>6.3f} {ppv:>6.3f} '
          f'{npv:>6.3f} {f1_c:>6.3f} {auc_c:>8.3f} {pr_c:>7.3f}')
    ext_pc_rows.append({'Class': lbl, 'N': n_c,
                        'Sens': round(sens,4), 'Spec': round(spec_,4),
                        'PPV':  round(ppv,4),  'NPV':  round(npv,4),
                        'F1':   round(f1_c,4),
                        'AUC-ROC': round(auc_c,4),
                        'AUC-PR':  round(pr_c,4)})

print(f'\n{"Overall (macro)":<28} {len(y_ext):>5} '
      f'{m_ext["Sens (macro)"]:>6.3f} {m_ext["Spec (macro)"]:>6.3f} '
      f'{"---":>6} {"---":>6} {m_ext["F1 (macro)"]:>6.3f} '
      f'{m_ext["AUC-ROC"]:>8.3f} {m_ext["AUC-PR"]:>7.3f}')
print(f'  Accuracy = {m_ext["Accuracy"]*100:.2f}%  |  MCC = {m_ext["MCC"]:.4f}  '
      f'|  G-mean = {m_ext["G-mean"]:.4f}')

# Save external per-class CSV
df_ext_pc = pd.DataFrame(ext_pc_rows)
df_ext_pc.to_csv(os.path.join(OUT_DIR, 'external_per_class_metrics.csv'), index=False)

# ── Confusion matrix (external) ────────────────────────────────────────────────
cm_ext_cnt = confusion_matrix(y_ext, y_ext_pred)
cm_ext_pct = cm_ext_cnt.astype(float) / cm_ext_cnt.sum(axis=1, keepdims=True) * 100
cls_labels  = [CLASS_NAMES[c] for c in CLASSES]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, data, fmt, title in zip(
    axes, [cm_ext_cnt, cm_ext_pct], ['d', '.1f'],
    ['External CM (Counts)', 'External CM (Row %)']
):
    sns.heatmap(data, annot=True, fmt=fmt, cmap='Oranges',
                xticklabels=cls_labels, yticklabels=cls_labels,
                linewidths=0.6, ax=ax, annot_kws={'size': 11})
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('True', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
plt.suptitle(f'External Validation (n=625) — {BEST_CLF} + {BEST_RES} + {BEST_FEAT}',
             fontsize=11, y=1.02)
plt.tight_layout()
savefig(fig, '12_external_confusion_matrix.png')

# ── Per-class 2×2 (external) ──────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 10))
axes = axes.flatten()
for c, ax in zip(CLASSES, axes):
    d  = ss_ext[c]
    cm_c = np.array([[d['TN'], d['FP']], [d['FN'], d['TP']]])
    ann  = np.array([[f'TN\n{d["TN"]}', f'FP\n{d["FP"]}'],
                     [f'FN\n{d["FN"]}', f'TP\n{d["TP"]}']])
    sns.heatmap(cm_c, annot=ann, fmt='', cmap='RdYlGn', center=0,
                vmin=0, ax=ax, linewidths=0.8, annot_kws={'size': 12})
    ax.set_xticklabels([f'Pred NOT {c}', f'Pred {c}'], fontsize=9)
    ax.set_yticklabels([f'True NOT {c}', f'True {c}'], fontsize=9, rotation=0)
    ax.set_title(f'External — Class {c}: {CLASS_NAMES[c]}',
                 fontsize=11, fontweight='bold')
    info = (f'Sens={d["Sens"]:.3f}  Spec={d["Spec"]:.3f}\n'
            f'PPV={d["PPV"]:.3f}   NPV={d["NPV"]:.3f}')
    ax.text(1, -0.22, info, transform=ax.transAxes, ha='right',
            fontsize=9, color='#333', style='italic')
plt.suptitle(f'External Per-Class 2×2 — {BEST_CLF} + {BEST_RES}',
             fontsize=12, fontweight='bold')
plt.tight_layout()
savefig(fig, '13_external_per_class_cm.png')

# ── ROC curves (external) ──────────────────────────────────────────────────────
fig, axes = plt.subplots(1, N_CLASSES, figsize=(5*N_CLASSES, 5))
for c, ax, clr in zip(CLASSES, axes, clrs_roc):
    fpr, tpr, _ = roc_curve(y_ext_bin[:, c], y_ext_prob[:, c])
    auc_v       = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=clr, lw=2.5, label=f'AUC = {auc_v:.3f}')
    ax.fill_between(fpr, tpr, alpha=0.10, color=clr)
    ax.plot([0,1],[0,1],'k--', lw=0.8)
    ax.set_xlabel('1 − Specificity (FPR)', fontsize=10)
    ax.set_ylabel('Sensitivity (TPR)', fontsize=10)
    ax.set_title(f'ROC — Class {c}\n{CLASS_NAMES[c]}',
                 fontsize=10, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_xlim([-0.02,1.02]); ax.set_ylim([-0.02,1.02])
plt.suptitle(f'External ROC Curves (n=625) — {BEST_CLF}',
             fontsize=12, fontweight='bold')
plt.tight_layout()
savefig(fig, '14_external_roc_curves.png')

# ── PR curves (external) ───────────────────────────────────────────────────────
fig, axes = plt.subplots(1, N_CLASSES, figsize=(5*N_CLASSES, 5))
for c, ax, clr in zip(CLASSES, axes, clrs_roc):
    prec, rec, _ = precision_recall_curve(y_ext_bin[:, c], y_ext_prob[:, c])
    ap_v         = average_precision_score(y_ext_bin[:, c], y_ext_prob[:, c])
    ax.plot(rec, prec, color=clr, lw=2.5, label=f'AP = {ap_v:.3f}')
    ax.fill_between(rec, prec, alpha=0.10, color=clr)
    ax.set_xlabel('Recall (Sensitivity)', fontsize=10)
    ax.set_ylabel('Precision (PPV)', fontsize=10)
    ax.set_title(f'PR — Class {c}\n{CLASS_NAMES[c]}',
                 fontsize=10, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_xlim([-0.02,1.02]); ax.set_ylim([-0.02,1.02])
plt.suptitle('External PR Curves (n=625)', fontsize=12, fontweight='bold')
plt.tight_layout()
savefig(fig, '15_external_pr_curves.png')

# ── Side-by-side comparison: internal test vs external ────────────────────────
print('\n' + '='*70)
print('COMPARISON: Internal Test (n=1147) vs External Validation (n=625)')
print('='*70)
print(f'{"Metric":<16} {"Internal Test":>15} {"External Val":>13}  {"Δ":>7}')
print('-'*55)
for k in METRIC_KEYS:
    v_int = m_test[k]
    v_ext = m_ext[k]
    diff  = v_ext - v_int
    print(f'{k:<16} {v_int:>15.4f} {v_ext:>13.4f}  {diff:>+7.4f}')

# Save comparison CSV
df_compare = pd.DataFrame([
    {'Set': 'Internal Test (n=1147)', **{k: round(m_test[k],4) for k in METRIC_KEYS}},
    {'Set': 'External Val  (n=625)',  **{k: round(m_ext[k],4)  for k in METRIC_KEYS}},
])
df_compare.to_csv(os.path.join(OUT_DIR, 'internal_vs_external_comparison.csv'), index=False)
print(f'\nSaved: internal_vs_external_comparison.csv')

# Update model bundle with external results
bundle['ext_metrics']   = m_ext
bundle['ext_per_class'] = ext_pc_rows
with open(bundle_path, 'wb') as f:
    pickle.dump(bundle, f)
print('Updated model_bundle.pkl with external results.')

# ════════════════════════════════════════════════════════════════════════════════
# SECTION 20 — Summary verification printout
# ════════════════════════════════════════════════════════════════════════════════
print('\n' + '='*70)
print('FINAL VERIFICATION SUMMARY')
print('='*70)
print(f'Dataset          : {len(df)} samples, {len(CBC_FEATURES)} CBC features')
print(f'Train/Test split : {len(y_train)} / {len(y_test)} (stratified 80/20)')
print(f'Imputer          : MedianImputer fitted on train only')
print(f'Feature selection: MDI + SHAP + RFE-LR consensus → Top-{TOP_N}: {TOP_FEATS}')
print(f'Grid search      : {total} configs (5-fold CV)')
print(f'Best config      : {BEST_CLF} + {BEST_RES} + {BEST_FEAT}')
print(f'\n{"Metric":<16} {"Internal Test (n="+str(len(y_test))+")":<22} {"External Val (n="+str(len(y_ext))+")":<20}')
print('-'*60)
for k in METRIC_KEYS:
    v_i = m_test[k]; v_e = m_ext[k]
    print(f'  {k:<14}: {v_i:.4f}{"  ("+"{:.1f}%".format(v_i*100)+")" if "Acc" in k else "":12}   '
          f'{v_e:.4f}{"  ("+"{:.1f}%".format(v_e*100)+")" if "Acc" in k else ""}')

print(f'\nOutputs saved to: {OUT_DIR}/')
for fname_out in sorted(os.listdir(OUT_DIR)):
    sz = os.path.getsize(os.path.join(OUT_DIR, fname_out))
    print(f'  {fname_out:<55} {sz:>8,} bytes')

# ── Pack everything into a single zip and download once ───────────────────────
# (Downloading files one-by-one in a loop triggers browser popup blockers
#  and silently skips CSVs / PKL files — zip avoids this entirely)
import zipfile, datetime
zip_name = f'CBC_4Class_Outputs_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.zip'
zip_path = f'/content/{zip_name}'

with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for fname_out in sorted(os.listdir(OUT_DIR)):
        full = os.path.join(OUT_DIR, fname_out)
        zf.write(full, arcname=fname_out)
        print(f'  Zipped: {fname_out}')

zip_size = os.path.getsize(zip_path)
print(f'\nZip archive: {zip_path}  ({zip_size/1024:.0f} KB)')
print('Downloading zip...')
files.download(zip_path)
print('\nDone. Unzip to access all CSVs, PNGs, and model bundle.')
