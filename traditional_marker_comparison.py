# -*- coding: utf-8 -*-
"""
SLE 传统标志物 vs CPD vs 联合模型——两两比较（含年龄、性别）
比较：缓解 vs 复发、初诊 vs 缓解、初诊 vs 复发
每组构建 3 个模型：
  - Traditional + Age + Sex (LR)
  - CPD + Age + Sex (LightGBM)
  - Combined (Traditional + CPD + Age + Sex) (LightGBM)
额外：
  - 仅年龄+性别基线模型
  - 年龄/性别分层 AUC
"""

import os
import re
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, roc_curve, average_precision_score,
    confusion_matrix, cohen_kappa_score, brier_score_loss
)
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from scipy.stats import norm
import lightgbm as lgb
import statsmodels.api as sm

warnings.filterwarnings('ignore')
mpl.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
mpl.rcParams['axes.unicode_minus'] = False


# ============================================================
# 1. 配置区
# ============================================================
FILE_PATH = os.environ.get("SLE_CPD_TRADITIONAL_DATA", "data/SLE初诊缓解复发含传统标志物.xlsx")
SHEET_NAME = "Sheet1"
OUTPUT_DIR = "revision_conventional_comparison_with_demo"
os.makedirs(OUTPUT_DIR, exist_ok=True)

AGE_COL_CANDIDATES = ["年龄", "age", "Age"]
SEX_COL_CANDIDATES = ["性别", "sex", "Sex", "gender", "Gender"]

# 传统标志物
MARKER_COLS_PREFERRED = ["CRP", "ESR", "C3", "C4", "C1q", "抗dsDNA抗体", "抗sm抗体"]
LOG_TRANSFORM_MARKERS = ["CRP", "ESR", "抗dsDNA抗体"]

N_SPLITS = 5
N_BOOTSTRAPS = 2000
RANDOM_STATE = 42
DCA_THRESHOLDS = np.arange(0.01, 1.00, 0.01)

# 年龄分层
AGE_BINS = [0, 40, 55, 200]
AGE_LABELS = ['<40', '40-55', '>55']


# ============================================================
# 2. 工具函数
# ============================================================
def clean_marker(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if s == '' or s.lower() in ['nan', 'none']:
        return np.nan
    s = s.replace('＜', '<').replace('＞', '>')
    s = s.replace('≤', '<=').replace('≥', '>=')
    s = re.sub(r'^[<>]=?', '', s).strip()
    try:
        return float(s)
    except Exception:
        return np.nan


def clean_feature_name(name, seen_set=None):
    cleaned = re.sub(r'[^a-zA-Z0-9_\u4e00-\u9fa5]', '_', str(name))
    cleaned = re.sub(r'_+', '_', cleaned).strip('_')
    if not cleaned:
        cleaned = 'feature'
    if cleaned[0].isdigit():
        cleaned = 'f_' + cleaned
    if seen_set is not None:
        original = cleaned
        count = 1
        while cleaned in seen_set:
            cleaned = f"{original}_{count}"
            count += 1
        seen_set.add(cleaned)
    return cleaned


def delong_test(y_true, y_score1, y_score2):
    """DeLong 检验：比较两个模型 AUC"""
    y_true = np.asarray(y_true).astype(int)
    y_score1 = np.asarray(y_score1, dtype=float)
    y_score2 = np.asarray(y_score2, dtype=float)

    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    m = len(pos_idx)
    n = len(neg_idx)
    if m == 0 or n == 0:
        return np.nan

    preds = np.vstack([y_score1, y_score2])

    def compute_midrank(x):
        J = np.argsort(x)
        Z = x[J]
        N = len(x)
        T = np.zeros(N, dtype=float)
        i = 0
        while i < N:
            j = i
            while j < N and Z[j] == Z[i]:
                j += 1
            T[i:j] = 0.5 * (i + j - 1) + 1
            i = j
        T2 = np.empty(N, dtype=float)
        T2[J] = T
        return T2

    k = len(preds)
    aucs = np.zeros(k)
    v10 = np.zeros((k, m))
    v01 = np.zeros((k, n))
    for r in range(k):
        p = preds[r]
        pos_p = p[pos_idx]
        neg_p = p[neg_idx]
        tz = compute_midrank(p)
        tx = compute_midrank(pos_p)
        ty = compute_midrank(neg_p)
        aucs[r] = (tz[pos_idx].sum() - m * (m + 1) / 2.0) / (m * n)
        v10[r] = (tz[pos_idx] - tx) / n
        v01[r] = 1.0 - (tz[neg_idx] - ty) / m

    sx = np.cov(v10)
    sy = np.cov(v01)
    if k == 1:
        sx = np.atleast_2d(sx)
        sy = np.atleast_2d(sy)
    s = sx / m + sy / n

    l = np.array([[1.0, -1.0]])
    z_stat_arr = l @ aucs            # shape (1,)
    var_arr = l @ s @ l.T            # shape (1,1)

    # 关键修复：用 .item() 提取标量，避免新版 numpy 报错
    z_stat = float(np.asarray(z_stat_arr).ravel()[0])
    var = float(np.asarray(var_arr).ravel()[0])

    se = np.sqrt(var) if var > 0 else 0.0
    if se == 0:
        return 1.0
    z = z_stat / se
    return 2 * (1 - norm.cdf(abs(z)))


def bootstrap_ci_metrics(y_true, y_prob, n_bootstraps=2000,
                          threshold=0.5, random_state=42):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    rng = np.random.RandomState(random_state)

    def compute_metrics(yt, yp, thr):
        yp_bin = (yp >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(yt, yp_bin, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        ppv = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan
        acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else np.nan
        f1 = 2 * ppv * sens / (ppv + sens) if (ppv + sens) > 0 else np.nan
        try:
            auc = roc_auc_score(yt, yp)
        except ValueError:
            auc = np.nan
        try:
            prauc = average_precision_score(yt, yp)
        except ValueError:
            prauc = np.nan
        try:
            brier = brier_score_loss(yt, yp)
        except ValueError:
            brier = np.nan
        try:
            kappa = cohen_kappa_score(yt, yp_bin)
        except ValueError:
            kappa = np.nan
        return {'AUC': auc, 'PRAUC': prauc, 'Accuracy': acc,
                'Sensitivity': sens, 'Specificity': spec,
                'PPV': ppv, 'NPV': npv, 'F1': f1,
                'Brier': brier, 'Kappa': kappa}

    point = compute_metrics(y_true, y_prob, threshold)
    boot_metrics = {k: [] for k in point.keys()}
    n = len(y_true)
    for _ in range(n_bootstraps):
        idx = rng.randint(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        m = compute_metrics(y_true[idx], y_prob[idx], threshold)
        for k in boot_metrics:
            boot_metrics[k].append(m[k])

    result = {}
    for k, vals in boot_metrics.items():
        vals = np.array([v for v in vals if not np.isnan(v)])
        if len(vals) == 0:
            result[k] = (point[k], np.nan, np.nan)
        else:
            result[k] = (point[k],
                         np.percentile(vals, 2.5),
                         np.percentile(vals, 97.5))
    return result


def calibration_analysis(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    brier = brier_score_loss(y_true, y_prob)
    prob_true, prob_pred = calibration_curve(y_true, y_prob,
                                              n_bins=10, strategy='quantile')
    eps = 1e-6
    p = np.clip(y_prob, eps, 1 - eps)
    logit_p = np.log(p / (1 - p))
    try:
        X_design = sm.add_constant(logit_p)
        model = sm.Logit(y_true, X_design).fit(disp=0)
        intercept = float(model.params[0])
        slope = float(model.params[1])
    except Exception:
        intercept, slope = np.nan, np.nan
    return {'Brier': brier,
            'Calibration_Intercept': intercept,
            'Calibration_Slope': slope,
            'prob_true': prob_true,
            'prob_pred': prob_pred}


def dca_analysis(y_true, y_prob, thresholds):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n = len(y_true)
    nb_model, nb_all, nb_none = [], [], []
    prev = y_true.mean()
    for pt in thresholds:
        yp_bin = (y_prob >= pt).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, yp_bin, labels=[0, 1]).ravel()
        nb_model.append(tp / n - (fp / n) * (pt / (1 - pt)))
        nb_all.append(prev - (1 - prev) * (pt / (1 - pt)))
        nb_none.append(0.0)
    return {'thresholds': thresholds,
            'nb_model': np.array(nb_model),
            'nb_all': np.array(nb_all),
            'nb_none': np.array(nb_none)}


def get_oof_predictions(model, X, y, n_splits=5, random_state=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                           random_state=random_state)
    oof = np.zeros(len(y))
    for tri, vai in skf.split(X, y):
        m = clone(model)
        m.fit(X.iloc[tri], y.iloc[tri])
        oof[vai] = m.predict_proba(X.iloc[vai])[:, 1]
    return oof


def fmt_ci(res, key):
    p, lo, hi = res[key]
    if np.isnan(lo):
        return f"{p:.3f}"
    return f"{p:.3f} ({lo:.3f}-{hi:.3f})"


# ============================================================
# 3. 加载数据
# ============================================================
print("=" * 72)
print("SLE 传统标志物 vs CPD vs 联合模型——两两比较（含年龄、性别）")
print("=" * 72)

df = pd.read_excel(FILE_PATH, sheet_name=SHEET_NAME)
df.columns = [str(c).strip() for c in df.columns]
print(f"原始数据维度：{df.shape}")

# 标签列
label_col = None
for c in df.columns:
    if "阴阳性标记" in str(c):
        label_col = c
        break
if label_col is None:
    raise ValueError("未找到阴阳性标记列")

# 年龄列
age_col = None
for c in AGE_COL_CANDIDATES:
    if c in df.columns:
        age_col = c
        break
if age_col is None:
    raise ValueError("未找到年龄列")

# 性别列
sex_col = None
for c in SEX_COL_CANDIDATES:
    if c in df.columns:
        sex_col = c
        break
if sex_col is None:
    raise ValueError("未找到性别列")

# 分组列
group_col = None
for c in df.columns:
    if "分组" in str(c) or "组别" in str(c):
        group_col = c
        break

if group_col is not None:
    def classify_group(s):
        s = str(s).strip()
        if "初诊" in s or "新发" in s:
            return "初诊"
        if "缓解" in s:
            return "缓解"
        if "复发" in s:
            return "复发"
        return "其他"
    df["_grp"] = df[group_col].apply(classify_group)
else:
    def lbl_to_grp(x):
        try:
            v = int(x)
        except Exception:
            return "其他"
        return {1: "初诊", 2: "缓解", 3: "复发"}.get(v, "其他")
    df["_grp"] = df[label_col].apply(lbl_to_grp)

print(f"\n分组分布：")
print(df["_grp"].value_counts().to_string())

# ---------- 年龄、性别清洗 ----------
df[age_col] = pd.to_numeric(df[age_col], errors='coerce')
df[age_col] = df[age_col].fillna(df[age_col].median())

def parse_sex(x):
    s = str(x).strip().lower()
    if s in ['男', '男性', 'male', 'm', '1']:
        return 1
    if s in ['女', '女性', 'female', 'f', '0']:
        return 0
    return np.nan

df[sex_col] = df[sex_col].apply(parse_sex)
df[sex_col] = df[sex_col].fillna(df[sex_col].mode()[0] if not df[sex_col].mode().empty else 0)
df[sex_col] = df[sex_col].astype(int)

print(f"性别分布：\n{df[sex_col].value_counts().to_string()}")
print(f"年龄：中位数={df[age_col].median():.1f}, "
      f"范围=[{df[age_col].min():.0f}, {df[age_col].max():.0f}]")

# ---------- 清洗传统标志物 ----------
valid_markers = []
for c in MARKER_COLS_PREFERRED:
    if c in df.columns:
        df[c] = df[c].apply(clean_marker)
        if c in ["C3", "C4", "C1q"]:
            df.loc[df[c] == 0, c] = np.nan
        valid_markers.append(c)

valid_markers = [c for c in valid_markers if df[c].notna().mean() >= 0.4]
print(f"\n纳入的传统标志物：{valid_markers}")

for c in LOG_TRANSFORM_MARKERS:
    if c in valid_markers:
        df[c] = np.log1p(df[c].clip(lower=0))

# ---------- CPD 特征 ----------
exclude_cols = [label_col, group_col, "_grp", age_col, sex_col] + valid_markers
exclude_cols = [c for c in exclude_cols if c in df.columns]
cpd_raw_cols = [c for c in df.columns if c not in exclude_cols]

seen = set()
cpd_name_map = {}
for c in cpd_raw_cols:
    cpd_name_map[c] = clean_feature_name(c, seen)
df = df.rename(columns=cpd_name_map)
cpd_cols = list(cpd_name_map.values())

for c in cpd_cols:
    df[c] = pd.to_numeric(df[c], errors='coerce')

allna = [c for c in cpd_cols if df[c].isnull().all()]
if allna:
    print(f"丢弃全空 CPD 列：{len(allna)}")
    cpd_cols = [c for c in cpd_cols if c not in allna]

high_miss = [c for c in cpd_cols if df[c].isnull().mean() > 0.30]
if high_miss:
    print(f"丢弃高缺失 CPD 列：{len(high_miss)}")
    cpd_cols = [c for c in cpd_cols if c not in high_miss]

for c in cpd_cols:
    if df[c].isnull().any():
        df[c] = df[c].fillna(df[c].median())
    df[c] = df[c].fillna(0)

print(f"最终 CPD 特征数：{len(cpd_cols)}")


# ============================================================
# 4. 两两比较主循环（含年龄、性别）
# ============================================================
pairs = [
    ("缓解", "复发", "Remission_vs_Relapse"),
    ("初诊", "缓解", "NewOnset_vs_Remission"),
    ("初诊", "复发", "NewOnset_vs_Relapse"),
]

all_summary = []

for grp_a, grp_b, tag in pairs:
    sub = df[(df["_grp"] == grp_a) | (df["_grp"] == grp_b)].copy()
    n_a = int((sub["_grp"] == grp_a).sum())
    n_b = int((sub["_grp"] == grp_b).sum())

    if len(sub) < 30 or n_a < 10 or n_b < 10:
        print(f"\n[跳过] {tag}：样本量不足（{grp_a}={n_a}, {grp_b}={n_b}）")
        continue

    print(f"\n{'=' * 72}")
    print(f"比较：{grp_a} (n={n_a}) vs {grp_b} (n={n_b})   [{tag}]")
    print("=" * 72)

    out_dir = os.path.join(OUTPUT_DIR, tag)
    os.makedirs(out_dir, exist_ok=True)

    # 标签
    y = (sub["_grp"] == grp_b).astype(int).reset_index(drop=True)

    # 人口学
    X_demo = sub[[age_col, sex_col]].reset_index(drop=True).copy()
    X_demo.columns = ['age', 'sex']

    # 传统标志物
    X_marker = sub[valid_markers].reset_index(drop=True).copy()
    for c in X_marker.columns:
        if X_marker[c].isnull().any():
            X_marker[c] = X_marker[c].fillna(X_marker[c].median())
        if X_marker[c].isnull().any():
            X_marker[c] = X_marker[c].fillna(0)

    # CPD
    X_cpd = sub[cpd_cols].reset_index(drop=True).copy()

    # 组装三套输入
    X_marker_demo = pd.concat([X_marker, X_demo], axis=1)
    X_cpd_demo = pd.concat([X_cpd, X_demo], axis=1)
    X_comb_demo = pd.concat([X_marker, X_cpd, X_demo], axis=1)

    print(f"传统+人口学 特征数：{X_marker_demo.shape[1]}")
    print(f"CPD+人口学 特征数：{X_cpd_demo.shape[1]}")
    print(f"联合（+人口学） 特征数：{X_comb_demo.shape[1]}")

    # ------- 模型 -------
    marker_model = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(
            C=1.0, penalty='l2', class_weight='balanced',
            max_iter=3000, random_state=RANDOM_STATE))
    ])

    lgb_params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 15,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'min_child_samples': 10,
        'n_estimators': 300,
        'class_weight': 'balanced',
        'verbose': -1,
        'random_state': RANDOM_STATE,
    }
    cpd_model = lgb.LGBMClassifier(**lgb_params)
    comb_model = lgb.LGBMClassifier(**lgb_params)

    # 仅年龄+性别基线
    demo_model = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(
            C=1.0, penalty='l2', class_weight='balanced',
            max_iter=3000, random_state=RANDOM_STATE))
    ])

    # ------- 折外预测 -------
    print("\n计算折外预测...")
    oof_marker = get_oof_predictions(marker_model, X_marker_demo, y,
                                     N_SPLITS, RANDOM_STATE)
    oof_cpd = get_oof_predictions(cpd_model, X_cpd_demo, y,
                                  N_SPLITS, RANDOM_STATE)
    oof_comb = get_oof_predictions(comb_model, X_comb_demo, y,
                                   N_SPLITS, RANDOM_STATE)
    oof_demo = get_oof_predictions(demo_model, X_demo, y,
                                   N_SPLITS, RANDOM_STATE)

    # ------- 指标 -------
    threshold = 0.5
    metrics_marker = bootstrap_ci_metrics(y.values, oof_marker,
                                           N_BOOTSTRAPS, threshold)
    metrics_cpd = bootstrap_ci_metrics(y.values, oof_cpd,
                                        N_BOOTSTRAPS, threshold)
    metrics_comb = bootstrap_ci_metrics(y.values, oof_comb,
                                         N_BOOTSTRAPS, threshold)
    metrics_demo = bootstrap_ci_metrics(y.values, oof_demo,
                                         N_BOOTSTRAPS, threshold)

    metrics_list = ['AUC', 'PRAUC', 'Accuracy', 'Sensitivity',
                    'Specificity', 'PPV', 'NPV', 'F1', 'Brier', 'Kappa']
    rows = []
    for m in metrics_list:
        rows.append({
            'Metric': m,
            'Demo_only': fmt_ci(metrics_demo, m),
            'Traditional_demo': fmt_ci(metrics_marker, m),
            'CPD_demo': fmt_ci(metrics_cpd, m),
            'Combined_demo': fmt_ci(metrics_comb, m),
            'Demo_only_point': metrics_demo[m][0],
            'Traditional_demo_point': metrics_marker[m][0],
            'CPD_demo_point': metrics_cpd[m][0],
            'Combined_demo_point': metrics_comb[m][0],
        })
    metric_df = pd.DataFrame(rows)
    metric_df.to_csv(os.path.join(out_dir, 'metrics_table.csv'),
                     index=False, encoding='utf-8-sig')
    print("\n性能指标（Bootstrap 95% CI）：")
    print(metric_df[['Metric', 'Demo_only', 'Traditional_demo',
                     'CPD_demo', 'Combined_demo']].to_string(index=False))

    # ------- DeLong -------
    p_trad_cpd = delong_test(y.values, oof_marker, oof_cpd)
    p_trad_comb = delong_test(y.values, oof_marker, oof_comb)
    p_cpd_comb = delong_test(y.values, oof_cpd, oof_comb)
    p_demo_trad = delong_test(y.values, oof_demo, oof_marker)
    p_demo_cpd = delong_test(y.values, oof_demo, oof_cpd)

    delong_df = pd.DataFrame([
        {'Comparison': 'Demo-only vs Traditional+demo',
         'p_value': p_demo_trad,
         'delta_AUC': metrics_marker['AUC'][0] - metrics_demo['AUC'][0]},
        {'Comparison': 'Demo-only vs CPD+demo',
         'p_value': p_demo_cpd,
         'delta_AUC': metrics_cpd['AUC'][0] - metrics_demo['AUC'][0]},
        {'Comparison': 'Traditional+demo vs CPD+demo',
         'p_value': p_trad_cpd,
         'delta_AUC': metrics_cpd['AUC'][0] - metrics_marker['AUC'][0]},
        {'Comparison': 'Traditional+demo vs Combined+demo',
         'p_value': p_trad_comb,
         'delta_AUC': metrics_comb['AUC'][0] - metrics_marker['AUC'][0]},
        {'Comparison': 'CPD+demo vs Combined+demo',
         'p_value': p_cpd_comb,
         'delta_AUC': metrics_comb['AUC'][0] - metrics_cpd['AUC'][0]},
    ])
    delong_df.to_csv(os.path.join(out_dir, 'delong_pvalues.csv'),
                     index=False, encoding='utf-8-sig')
    print("\nDeLong 检验：")
    print(delong_df.to_string(index=False))

    # ------- 保存预测 -------
    pred_df = pd.DataFrame({
        'y_true': y.values,
        'age': X_demo['age'].values,
        'sex': X_demo['sex'].values,
        'prob_demo': oof_demo,
        'prob_traditional_demo': oof_marker,
        'prob_cpd_demo': oof_cpd,
        'prob_combined_demo': oof_comb,
    })
    pred_df.to_csv(os.path.join(out_dir, 'predictions.csv'),
                   index=False, encoding='utf-8-sig')

    # ------- ROC -------
    plt.figure(figsize=(7, 6))
    for prob, label, color in [
        (oof_demo,   f"Demo only (AUC={metrics_demo['AUC'][0]:.3f})",        '#7f7f7f'),
        (oof_marker, f"Traditional+demo (AUC={metrics_marker['AUC'][0]:.3f})", '#1f77b4'),
        (oof_cpd,    f"CPD+demo (AUC={metrics_cpd['AUC'][0]:.3f})",           '#2ca02c'),
        (oof_comb,   f"Combined+demo (AUC={metrics_comb['AUC'][0]:.3f})",     '#d62728'),
    ]:
        fpr, tpr, _ = roc_curve(y.values, prob)
        plt.plot(fpr, tpr, label=label, color=color, linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('1 - Specificity', fontsize=12)
    plt.ylabel('Sensitivity', fontsize=12)
    plt.title(f'ROC: {grp_a} vs {grp_b}', fontsize=13)
    plt.legend(loc='lower right', fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'roc_curve.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    # ------- DCA -------
    dca_marker = dca_analysis(y.values, oof_marker, DCA_THRESHOLDS)
    dca_cpd = dca_analysis(y.values, oof_cpd, DCA_THRESHOLDS)
    dca_comb = dca_analysis(y.values, oof_comb, DCA_THRESHOLDS)
    dca_demo = dca_analysis(y.values, oof_demo, DCA_THRESHOLDS)

    plt.figure(figsize=(7, 5))
    plt.plot(DCA_THRESHOLDS, dca_demo['nb_model'],
             label='Demo only', color='#7f7f7f', linewidth=1.5)
    plt.plot(DCA_THRESHOLDS, dca_marker['nb_model'],
             label='Traditional+demo', color='#1f77b4', linewidth=2)
    plt.plot(DCA_THRESHOLDS, dca_cpd['nb_model'],
             label='CPD+demo', color='#2ca02c', linewidth=2)
    plt.plot(DCA_THRESHOLDS, dca_comb['nb_model'],
             label='Combined+demo', color='#d62728', linewidth=2)
    plt.plot(DCA_THRESHOLDS, dca_marker['nb_all'],
             label='Treat all', linestyle='--', color='gray')
    plt.plot(DCA_THRESHOLDS, dca_marker['nb_none'],
             label='Treat none', linestyle=':', color='black')
    plt.xlabel('Threshold probability', fontsize=12)
    plt.ylabel('Net benefit', fontsize=12)
    plt.title(f'DCA: {grp_a} vs {grp_b}', fontsize=13)
    plt.legend(loc='upper right', fontsize=9)
    plt.grid(alpha=0.3)
    y_min = min(dca_marker['nb_all'].min(), -0.1)
    y_max = max(dca_marker['nb_all'].max(), dca_cpd['nb_model'].max()) * 1.1
    plt.ylim(y_min, y_max)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'dca_curve.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    # ------- 校准曲线 -------
    cal_marker = calibration_analysis(y.values, oof_marker)
    cal_cpd = calibration_analysis(y.values, oof_cpd)
    cal_comb = calibration_analysis(y.values, oof_comb)
    cal_demo = calibration_analysis(y.values, oof_demo)

    plt.figure(figsize=(6, 6))
    plt.plot(cal_demo['prob_pred'], cal_demo['prob_true'],
             marker='x', color='#7f7f7f',
             label=f"Demo only (Brier={cal_demo['Brier']:.3f})")
    plt.plot(cal_marker['prob_pred'], cal_marker['prob_true'],
             marker='o', color='#1f77b4',
             label=f"Traditional+demo (Brier={cal_marker['Brier']:.3f})")
    plt.plot(cal_cpd['prob_pred'], cal_cpd['prob_true'],
             marker='s', color='#2ca02c',
             label=f"CPD+demo (Brier={cal_cpd['Brier']:.3f})")
    plt.plot(cal_comb['prob_pred'], cal_comb['prob_true'],
             marker='^', color='#d62728',
             label=f"Combined+demo (Brier={cal_comb['Brier']:.3f})")
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect')
    plt.xlabel('Mean predicted probability', fontsize=12)
    plt.ylabel('Observed fraction', fontsize=12)
    plt.title(f'Calibration: {grp_a} vs {grp_b}', fontsize=13)
    plt.legend(loc='upper left', fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'calibration_curve.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    # ------- 年龄/性别分层 AUC（用联合模型） -------
    stratum_rows = []
    pred_for_strata = pred_df.copy()
    pred_for_strata['age_bin'] = pd.cut(pred_for_strata['age'],
                                         bins=AGE_BINS, labels=AGE_LABELS,
                                         include_lowest=True)

    for (ab, sx), grp in pred_for_strata.groupby(['age_bin', 'sex']):
        if len(grp) < 8 or grp['y_true'].nunique() < 2:
            continue
        try:
            auc_c = roc_auc_score(grp['y_true'], grp['prob_combined_demo'])
            auc_m = roc_auc_score(grp['y_true'], grp['prob_traditional_demo'])
            auc_p = roc_auc_score(grp['y_true'], grp['prob_cpd_demo'])
            auc_d = roc_auc_score(grp['y_true'], grp['prob_demo'])
        except ValueError:
            continue
        stratum_rows.append({
            'age_bin': str(ab),
            'sex': 'Male' if sx == 1 else 'Female',
            'n': len(grp),
            'n_positive': int(grp['y_true'].sum()),
            'AUC_demo': auc_d,
            'AUC_traditional_demo': auc_m,
            'AUC_cpd_demo': auc_p,
            'AUC_combined_demo': auc_c,
        })

    if stratum_rows:
        strata_df = pd.DataFrame(stratum_rows)
        strata_df.to_csv(os.path.join(out_dir, 'age_sex_stratified_auc.csv'),
                         index=False, encoding='utf-8-sig')
        print("\n年龄/性别分层 AUC：")
        print(strata_df.to_string(index=False))
    else:
        print("\n分层样本不足，跳过年龄/性别分层分析")

    # ------- 特征重要性（联合模型） -------
    try:
        comb_final = lgb.LGBMClassifier(**lgb_params)
        comb_final.fit(X_comb_demo, y)
        imp = pd.DataFrame({
            'Feature': X_comb_demo.columns,
            'Importance': comb_final.feature_importances_
        }).sort_values('Importance', ascending=False)
        imp.to_csv(os.path.join(out_dir, 'feature_importance_combined.csv'),
                   index=False, encoding='utf-8-sig')
        print("\n联合模型 Top-15 特征：")
        print(imp.head(15).to_string(index=False))
    except Exception as e:
        print(f"特征重要性计算失败：{e}")

    # ------- JSON -------
    result_json = {
        'comparison': f'{grp_a} vs {grp_b}',
        'n_grp_a': n_a,
        'n_grp_b': n_b,
        'markers': valid_markers,
        'n_cpd_features': len(cpd_cols),
        'demo_only': {k: {'point': float(v[0])} for k, v in metrics_demo.items()},
        'traditional_demo': {k: {'point': float(v[0])} for k, v in metrics_marker.items()},
        'cpd_demo': {k: {'point': float(v[0])} for k, v in metrics_cpd.items()},
        'combined_demo': {k: {'point': float(v[0])} for k, v in metrics_comb.items()},
        'delong_p': {
            'demo_vs_traditional_demo': float(p_demo_trad),
            'demo_vs_cpd_demo': float(p_demo_cpd),
            'traditional_demo_vs_cpd_demo': float(p_trad_cpd),
            'traditional_demo_vs_combined_demo': float(p_trad_comb),
            'cpd_demo_vs_combined_demo': float(p_cpd_comb),
        },
        'delta_auc': {
            'cpd_demo_minus_traditional_demo':
                float(metrics_cpd['AUC'][0] - metrics_marker['AUC'][0]),
            'combined_demo_minus_traditional_demo':
                float(metrics_comb['AUC'][0] - metrics_marker['AUC'][0]),
            'combined_demo_minus_cpd_demo':
                float(metrics_comb['AUC'][0] - metrics_cpd['AUC'][0]),
        }
    }
    with open(os.path.join(out_dir, 'results.json'), 'w', encoding='utf-8') as f:
        json.dump(result_json, f, indent=2, ensure_ascii=False)

    # ------- 汇总 -------
    all_summary.append({
        'Comparison': f'{grp_a} vs {grp_b}',
        'N_positive': n_b,
        'N_negative': n_a,
        'Demo_only_AUC': metrics_demo['AUC'][0],
        'Traditional_demo_AUC': metrics_marker['AUC'][0],
        'CPD_demo_AUC': metrics_cpd['AUC'][0],
        'Combined_demo_AUC': metrics_comb['AUC'][0],
        'Delta_CPD_demo_minus_Traditional_demo':
            metrics_cpd['AUC'][0] - metrics_marker['AUC'][0],
        'Delta_Combined_demo_minus_Traditional_demo':
            metrics_comb['AUC'][0] - metrics_marker['AUC'][0],
        'Delta_Combined_demo_minus_CPD_demo':
            metrics_comb['AUC'][0] - metrics_cpd['AUC'][0],
        'p_Demo_vs_Traditional_demo': p_demo_trad,
        'p_Demo_vs_CPD_demo': p_demo_cpd,
        'p_Traditional_demo_vs_CPD_demo': p_trad_cpd,
        'p_Traditional_demo_vs_Combined_demo': p_trad_comb,
        'p_CPD_demo_vs_Combined_demo': p_cpd_comb,
    })


# ============================================================
# 5. 汇总
# ============================================================
print("\n" + "=" * 72)
print("全部比较完成")
print("=" * 72)

if all_summary:
    summary_df = pd.DataFrame(all_summary)
    summary_df.to_csv(
        os.path.join(OUTPUT_DIR, 'summary_all_comparisons_with_demo.csv'),
        index=False, encoding='utf-8-sig'
    )
    print("\n三组比较汇总（AUC）：")
    cols = ['Comparison', 'N_positive', 'N_negative',
            'Demo_only_AUC', 'Traditional_demo_AUC',
            'CPD_demo_AUC', 'Combined_demo_AUC',
            'p_Traditional_demo_vs_Combined_demo',
            'p_CPD_demo_vs_Combined_demo']
    print(summary_df[cols].round(4).to_string(index=False))

print(f"\n所有结果已保存至：{os.path.abspath(OUTPUT_DIR)}")