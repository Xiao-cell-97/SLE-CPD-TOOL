# -*- coding: utf-8 -*-
# =============================================================================
# SLE CPD ML Pipeline — Model1 / Model2 完整版（响应审稿意见）
#
# MODEL_ID = 1 → data1.xlsx + 外部验证HC-SLE.xlsx  (SLE vs HC)
# MODEL_ID = 2 → data2.xlsx + 外部验证SLE-非SLE.xlsx (SLE vs 非SLE风湿病)
#
# 核心方法学（与 Model3/4 一致）：
#  1. 五模型 AUC 比较 + 合并 ROC
#  2. 嵌套 CV 只在训练集内 → 训练集 OOF 锁定阈值与三分位
#  3. 测试集一次最终评估 + 外部验证（使用训练集锁定阈值/切点）
#  4. 校准指标 Brier / Intercept / Slope + Bootstrap CI
#  5. DCA + Bootstrap CI
#  6. 风险分层（数据驱动，非临床验证）
#  7. 缺失机制 + 插补敏感性
#  8. 年龄/性别匹配（探索性）
#  9. EPV
# 10. SHAP 全局图 + waterfall
# 11. DeLong 检验：训练集 OOF + 测试集（双报告）
# =============================================================================
import os
import re
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib

from scipy.stats import norm, t as t_dist, chi2_contingency
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, ParameterGrid, GridSearchCV
)
from sklearn.metrics import (
    roc_auc_score, roc_curve, average_precision_score,
    confusion_matrix, brier_score_loss
)
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin
import lightgbm as lgb
import xgboost as xgb
import statsmodels.api as sm
from statsmodels.stats.proportion import proportion_confint

warnings.filterwarnings('ignore')

# =============================================================================
# 0. 配置区
# =============================================================================
MODEL_ID = 1   # 改成 2 即运行 Model 2

if MODEL_ID == 1:
    FILE_PATH = os.environ.get("SLE_CPD_MODEL1_DATA", "data/data1.xlsx")
    EXT_FILE_PATH = os.environ.get("SLE_CPD_MODEL1_EXTERNAL", "data/external_HC_SLE.xlsx")
    OUTPUT_DIR = "revision_outputs_Model1_final"
    MODEL_TAG = "Model 1 (SLE初诊+复发+缓解 vs 健康对照 HC)"
    MODEL_SHORT = "Model1"
    EXT_POS_LABEL = "SLE"
    EXT_NEG_LABEL = "HC"
elif MODEL_ID == 2:
    FILE_PATH = os.environ.get("SLE_CPD_MODEL2_DATA", "data/data2.xlsx")
    EXT_FILE_PATH = os.environ.get("SLE_CPD_MODEL2_EXTERNAL", "data/external_SLE_nonSLE.xlsx")
    OUTPUT_DIR = "revision_outputs_Model2_final"
    MODEL_TAG = "Model 2 (SLE初诊+复发+缓解 vs 非SLE风湿病)"
    MODEL_SHORT = "Model2"
    EXT_POS_LABEL = "SLE"
    EXT_NEG_LABEL = "非SLE组"
else:
    raise ValueError("MODEL_ID 只能是 1 或 2")

SHEET_NAME = "全部含性别年龄"
EXT_SHEET_NAME = "全部含性别年龄"
TARGET_COL = "阴阳性标记(初诊+复发+缓解标记为1)"
EXT_GROUP_COL = "组别"
EXT_SAMPLE_COL = "样本号"
os.makedirs(OUTPUT_DIR, exist_ok=True)

AGE_COL = "年龄"
SEX_COL = "性别"

LEAK_COLS = [
    TARGET_COL, "诊断", "科室", "组别", "样本号", "亚组", "group", "Group",
    "分组", "疾病状态", "status", "Status", "SLE状态", "疾病分组",
    AGE_COL, SEX_COL
]

MIN_FEATURES = 10
RFE_STEP = 5
RFE_CV = 5
TUNE_CV = 5
RANDOM_STATE = 42
TEST_SIZE = 0.30

RUN_LASSO_REFINE = True
LASSO_MIN_FEATURES = 5
LASSO_C_GRID = np.logspace(-3, 1, 20)

N_BOOTSTRAPS = 2000
N_BOOTSTRAPS_CI = 1000
DCA_THRESHOLDS = np.arange(0.01, 1.00, 0.01)

RUN_NESTED_CV = True
NESTED_OUTER_CV = 5
NESTED_INNER_CV = 5
NESTED_PARAM_GRID = {
    'num_leaves': [15, 31],
    'learning_rate': [0.05, 0.1],
    'feature_fraction': [0.8, 1.0],
    'bagging_fraction': [0.8, 1.0],
}
NESTED_FIXED_PARAMS = {
    'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
    'n_estimators': 500, 'verbose': -1, 'is_unbalance': True,
    'random_state': RANDOM_STATE,
}

HIGH_MISS_THRESHOLD = 30.0
LOW_MISS_THRESHOLD = 10.0

print("=" * 70)
print(f"SLE CPD ML Pipeline — {MODEL_TAG}")
print("=" * 70)


# =============================================================================
# 1. 工具函数
# =============================================================================
def clean_numeric_series(s):
    return pd.to_numeric(
        pd.Series(s).replace(
            ['----', 'ERROR', 'DISCRETE', '', ' ', 'NA', 'NaN'], np.nan),
        errors='coerce'
    )


def clean_feature_name(name, seen_set):
    cleaned = re.sub(r'[^a-zA-Z0-9_\u4e00-\u9fa5]', '_', str(name))
    cleaned = re.sub(r'_+', '_', cleaned).strip('_')
    if not cleaned:
        cleaned = 'feature'
    if cleaned[0].isdigit():
        cleaned = 'f_' + cleaned
    original = cleaned
    count = 1
    while cleaned in seen_set:
        cleaned = f"{original}_{count}"
        count += 1
    seen_set.add(cleaned)
    return cleaned


def report_missingness(df, save_path):
    miss = df.isnull().sum()
    miss_pct = miss / len(df) * 100
    miss_df = pd.DataFrame({
        'Variable': df.columns,
        'Missing_Count': miss.values,
        'Missing_Percent': miss_pct.values
    }).sort_values('Missing_Percent', ascending=False)
    miss_df.to_csv(save_path, index=False)
    return miss_df


class DataImputer(BaseEstimator, TransformerMixin):
    def __init__(self, high_miss=HIGH_MISS_THRESHOLD, low_miss=LOW_MISS_THRESHOLD):
        self.high_miss = high_miss
        self.low_miss = low_miss

    def fit(self, X, y=None):
        X = pd.DataFrame(X).copy()
        miss_pct = X.isnull().mean() * 100
        self.drop_cols_ = miss_pct[miss_pct > self.high_miss].index.tolist()
        X = X.drop(columns=self.drop_cols_, errors='ignore')
        miss_pct2 = X.isnull().mean() * 100
        self.mid_cols_ = [c for c in X.columns
                          if self.low_miss < miss_pct2.get(c, 0) <= self.high_miss]
        self.low_cols_ = [c for c in X.columns
                          if 0 < miss_pct2.get(c, 0) <= self.low_miss]
        self.fill_values_ = {}
        for c in self.mid_cols_:
            self.fill_values_[c] = X[c].median()
        for c in self.low_cols_:
            try:
                sk = X[c].skew()
            except Exception:
                sk = 0
            self.fill_values_[c] = X[c].median() if abs(sk) > 1 else X[c].mean()
        self.all_cols_ = list(X.columns)
        self.feature_names_in_ = np.array(self.all_cols_)
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        X = X.drop(columns=self.drop_cols_, errors='ignore')
        for c in self.all_cols_:
            if c not in X.columns:
                X[c] = np.nan
        X = X[self.all_cols_]
        for c, v in self.fill_values_.items():
            if c in X.columns:
                X[c] = X[c].fillna(v)
        return X.fillna(0)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.all_cols_)


def delong_roc_test(y_true, y_pred1, y_pred2):
    """DeLong 检验。小样本下功效不足，p 值需谨慎解释。"""
    y_true = np.asarray(y_true); y_pred1 = np.asarray(y_pred1); y_pred2 = np.asarray(y_pred2)
    pos_idx = np.where(y_true == 1)[0]; neg_idx = np.where(y_true == 0)[0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return np.nan

    def get_u(y_true, y_pred):
        n_pos = np.sum(y_true); n_neg = len(y_true) - n_pos; u = 0
        for i in range(len(y_true)):
            if y_true[i] == 1:
                for j in range(len(y_true)):
                    if y_true[j] == 0:
                        if y_pred[i] > y_pred[j]: u += 1
                        elif y_pred[i] == y_pred[j]: u += 0.5
        return u / (n_pos * n_neg) if (n_pos * n_neg) > 0 else np.nan

    u1 = get_u(y_true, y_pred1); u2 = get_u(y_true, y_pred2)
    v10_1 = np.zeros(len(pos_idx)); v01_1 = np.zeros(len(neg_idx))
    for i, p in enumerate(pos_idx):
        v10_1[i] = np.mean((y_pred1[p] > y_pred1[neg_idx]).astype(float) +
                           0.5 * (y_pred1[p] == y_pred1[neg_idx]).astype(float))
    for j, n in enumerate(neg_idx):
        v01_1[j] = np.mean((y_pred1[pos_idx] > y_pred1[n]).astype(float) +
                           0.5 * (y_pred1[pos_idx] == y_pred1[n]).astype(float))
    v10_2 = np.zeros(len(pos_idx)); v01_2 = np.zeros(len(neg_idx))
    for i, p in enumerate(pos_idx):
        v10_2[i] = np.mean((y_pred2[p] > y_pred2[neg_idx]).astype(float) +
                           0.5 * (y_pred2[p] == y_pred2[neg_idx]).astype(float))
    for j, n in enumerate(neg_idx):
        v01_2[j] = np.mean((y_pred2[pos_idx] > y_pred2[n]).astype(float) +
                           0.5 * (y_pred2[pos_idx] == y_pred2[n]).astype(float))
    try:
        S10 = np.atleast_2d(np.cov(np.array([v10_1, v10_2])))
        S01 = np.atleast_2d(np.cov(np.array([v01_1, v01_2])))
        var = S10[0, 0] / len(pos_idx) + S01[0, 0] / len(neg_idx)
        cov = S10[0, 1] / len(pos_idx) + S01[0, 1] / len(neg_idx)
        var2 = S10[1, 1] / len(pos_idx) + S01[1, 1] / len(neg_idx)
    except Exception:
        return np.nan
    diff = u1 - u2
    se = np.sqrt(var + var2 - 2 * cov)
    if se == 0 or np.isnan(se):
        return 1.0
    return 2 * (1 - norm.cdf(abs(diff / se)))


def bootstrap_ci_metrics(y_true, y_prob, n_bootstraps=N_BOOTSTRAPS,
                         threshold=0.5, random_state=RANDOM_STATE):
    y_true = np.asarray(y_true); y_prob = np.asarray(y_prob)
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
        try: auc = roc_auc_score(yt, yp)
        except ValueError: auc = np.nan
        try: prauc = average_precision_score(yt, yp)
        except ValueError: prauc = np.nan
        try: brier = brier_score_loss(yt, yp)
        except ValueError: brier = np.nan
        return {'AUC': auc, 'PRAUC': prauc, 'Accuracy': acc,
                'Sensitivity': sens, 'Specificity': spec,
                'PPV': ppv, 'NPV': npv, 'F1': f1, 'Brier': brier}

    point = compute_metrics(y_true, y_prob, threshold)
    boot_metrics = {k: [] for k in point.keys()}
    n = len(y_true)
    for _ in range(n_bootstraps):
        idx = rng.randint(0, n, n)
        if len(np.unique(y_true[idx])) < 2: continue
        m = compute_metrics(y_true[idx], y_prob[idx], threshold)
        for k in boot_metrics: boot_metrics[k].append(m[k])
    result = {}
    for k, vals in boot_metrics.items():
        vals = np.array([v for v in vals if not np.isnan(v)])
        if len(vals) == 0: result[k] = (point[k], np.nan, np.nan)
        else: result[k] = (point[k], np.percentile(vals, 2.5), np.percentile(vals, 97.5))
    return result


def calibration_analysis(y_true, y_prob, save_dir, save_prefix="calibration"):
    y_true = np.asarray(y_true); y_prob = np.asarray(y_prob)
    brier = brier_score_loss(y_true, y_prob)
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy='quantile')
    eps = 1e-6; p = np.clip(y_prob, eps, 1 - eps); logit_p = np.log(p / (1 - p))
    try:
        X_design = sm.add_constant(logit_p)
        model = sm.Logit(y_true, X_design).fit(disp=0)
        intercept = model.params[0]; slope = model.params[1]
    except Exception:
        intercept, slope = np.nan, np.nan
    plt.figure(figsize=(6, 6))
    plt.plot(prob_pred, prob_true, marker='o', label=f'Model (Brier={brier:.3f})')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfect Calibration')
    plt.xlabel('Mean Predicted Probability'); plt.ylabel('Fraction of Positives')
    plt.title(f'Calibration (Intercept={intercept:.3f}, Slope={slope:.3f})')
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{save_prefix}_curve.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    return {'Brier': float(brier),
            'Calibration_Intercept': float(intercept),
            'Calibration_Slope': float(slope)}


def decision_curve_analysis(y_true, y_prob, thresholds=DCA_THRESHOLDS):
    y_true = np.asarray(y_true); y_prob = np.asarray(y_prob)
    n = len(y_true); nb_model, nb_all, nb_none = [], [], []
    prev = y_true.mean()
    for pt in thresholds:
        yp_bin = (y_prob >= pt).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, yp_bin, labels=[0, 1]).ravel()
        nb_model.append(tp / n - (fp / n) * (pt / (1 - pt)))
        nb_all.append(prev - (1 - prev) * (pt / (1 - pt)))
        nb_none.append(0.0)
    return {'thresholds': thresholds, 'nb_model': np.array(nb_model),
            'nb_all': np.array(nb_all), 'nb_none': np.array(nb_none)}


def bootstrap_calibration_dca(y_true, y_prob, n_bootstraps=N_BOOTSTRAPS_CI,
                              random_state=RANDOM_STATE):
    y_true = np.asarray(y_true); y_prob = np.asarray(y_prob)
    rng = np.random.RandomState(random_state)
    brier_list, intercept_list, slope_list = [], [], []
    dca_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
    dca_net_list = {t: [] for t in dca_thresholds}
    for _ in range(n_bootstraps):
        idx = rng.randint(0, len(y_true), len(y_true))
        yt = y_true[idx]; yp = y_prob[idx]
        if len(np.unique(yt)) < 2: continue
        try: brier_list.append(brier_score_loss(yt, yp))
        except Exception: pass
        try:
            eps = 1e-6; p = np.clip(yp, eps, 1 - eps); logit_p = np.log(p / (1 - p))
            X_design = sm.add_constant(logit_p)
            model = sm.Logit(yt, X_design).fit(disp=0)
            intercept_list.append(model.params[0]); slope_list.append(model.params[1])
        except Exception: pass
        for t in dca_thresholds:
            yp_bin = (yp >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(yt, yp_bin, labels=[0, 1]).ravel()
            n = len(yt); nb = tp / n - (fp / n) * (t / (1 - t))
            dca_net_list[t].append(nb)

    def ci(vals):
        vals = np.array([v for v in vals if not np.isnan(v)])
        if len(vals) == 0: return (np.nan, np.nan)
        return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))
    return {'Brier_CI': ci(brier_list),
            'Calibration_Intercept_CI': ci(intercept_list),
            'Calibration_Slope_CI': ci(slope_list),
            'DCA_CI': {str(t): ci(dca_net_list[t]) for t in dca_thresholds}}


def bootstrap_threshold_and_tertiles(oof_y_true, oof_y_prob,
                                     n_bootstraps=N_BOOTSTRAPS_CI,
                                     random_state=RANDOM_STATE):
    rng = np.random.RandomState(random_state)
    thresholds, q1s, q2s = [], [], []
    for _ in range(n_bootstraps):
        idx = rng.randint(0, len(oof_y_true), len(oof_y_true))
        yt = oof_y_true[idx]; yp = oof_y_prob[idx]
        if len(np.unique(yt)) < 2: continue
        try:
            fpr, tpr, thr = roc_curve(yt, yp)
            thresholds.append(float(thr[int(np.argmax(tpr - fpr))]))
        except Exception: pass
        q1s.append(float(np.percentile(yp, 33.33)))
        q2s.append(float(np.percentile(yp, 66.67)))

    def ci(vals):
        vals = np.array([v for v in vals if not np.isnan(v)])
        if len(vals) == 0: return (np.nan, np.nan)
        return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))
    return {'threshold_CI': ci(thresholds), 'q1_CI': ci(q1s), 'q2_CI': ci(q2s)}


# =============================================================================
# 缺失机制 / 插补敏感性 / 年龄性别匹配
# =============================================================================
def missingness_mechanism_analysis(X_raw, y, save_dir):
    y_arr = np.asarray(y)
    rows = []
    for col in X_raw.columns:
        miss = X_raw[col].isnull()
        n_miss = int(miss.sum())
        if n_miss == 0 or n_miss == len(X_raw):
            continue
        miss_arr = miss.values
        pos_rate_miss = float(y_arr[miss_arr].mean())
        pos_rate_comp = float(y_arr[~miss_arr].mean())
        try:
            table = pd.crosstab(miss_arr, y_arr)
            if table.shape == (2, 2):
                chi2, p_val, _, _ = chi2_contingency(table)
            else:
                p_val = np.nan
        except Exception:
            p_val = np.nan
        rows.append({
            'Feature': col, 'N_Missing': n_miss,
            'Missing_Percent': n_miss / len(X_raw) * 100,
            'Outcome_PosRate_Missing': pos_rate_miss,
            'Outcome_PosRate_Complete': pos_rate_comp,
            'Chi2_p_Missing_vs_Outcome': p_val,
        })
    res_df = pd.DataFrame(rows).sort_values('Missing_Percent', ascending=False)
    res_df.to_csv(os.path.join(save_dir, 'missingness_mechanism.csv'),
                  index=False, encoding='utf-8-sig')
    n_sig = int((res_df['Chi2_p_Missing_vs_Outcome'] < 0.05).sum()) if len(res_df) > 0 else 0
    summary = {
        'n_features_with_missing': int(len(res_df)),
        'n_features_missing_vs_outcome_p_lt_005': n_sig,
        'interpretation': (
            'MCAR is plausible if few features show missingness-outcome association. '
            'MAR/MNAR is likely if many features show association; '
            'imputation sensitivity analysis is recommended in that case.'),
    }
    with open(os.path.join(save_dir, 'missingness_mechanism_summary.json'),
              'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"缺失机制分析：{len(res_df)} 个特征存在缺失；"
          f"其中 {n_sig} 个特征缺失与结局显著相关 (p<0.05)")
    return res_df, summary


def imputation_sensitivity_analysis(X_train_raw, y_train, X_test_raw, y_test,
                                     selected_features, best_params, save_dir):
    results = []

    def _eval_with_imputer(imputer, label):
        try:
            imp = imputer.fit(X_train_raw, y_train)
            Xtr = imp.transform(X_train_raw)
            Xte = imp.transform(X_test_raw)
            if not isinstance(Xtr, pd.DataFrame):
                cols = list(getattr(imp, 'all_cols_', X_train_raw.columns))
                Xtr = pd.DataFrame(Xtr, columns=cols)
                Xte = pd.DataFrame(Xte, columns=cols)
            feats = [f for f in selected_features if f in Xtr.columns]
            if len(feats) == 0:
                return
            m = lgb.LGBMClassifier(**best_params)
            m.fit(Xtr[feats], y_train)
            prob = m.predict_proba(Xte[feats])[:, 1]
            auc = float(roc_auc_score(y_test, prob))
            try:
                brier = float(brier_score_loss(y_test, prob))
            except Exception:
                brier = np.nan
            results.append({
                'Strategy': label,
                'High_Miss_Threshold': getattr(imputer, 'high_miss', np.nan),
                'Low_Miss_Threshold': getattr(imputer, 'low_miss', np.nan),
                'N_Features_Used': len(feats),
                'Test_AUC': auc, 'Test_Brier': brier,
            })
            print(f"  插补策略 [{label}] AUC={auc:.4f}, Brier={brier:.4f}, "
                  f"特征数={len(feats)}")
        except Exception as e:
            print(f"  插补敏感性 [{label}] 失败: {e}")

    _eval_with_imputer(DataImputer(high_miss=30.0, low_miss=10.0), 'A_Current_H30_L10')
    _eval_with_imputer(DataImputer(high_miss=20.0, low_miss=10.0), 'B_H20_L10')
    _eval_with_imputer(DataImputer(high_miss=40.0, low_miss=10.0), 'C_H40_L10')
    _eval_with_imputer(DataImputer(high_miss=30.0, low_miss=5.0),  'D_H30_L5')
    _eval_with_imputer(DataImputer(high_miss=30.0, low_miss=0.0),  'E_H30_L0_MedianOnly')

    res_df = pd.DataFrame(results)
    res_df.to_csv(os.path.join(save_dir, 'imputation_sensitivity.csv'),
                  index=False, encoding='utf-8-sig')
    return res_df


def age_sex_matched_sensitivity_analysis(X_raw, y, age_all, sex_all,
                                          selected_features, best_params,
                                          save_dir):
    """
    【探索性分析】使用已选特征与已调参数，未在匹配子集内重做嵌套 FS/tuning。
    """
    if age_all is None or sex_all is None:
        print("  年龄或性别缺失，跳过年龄/性别匹配敏感性分析")
        return None

    age = pd.to_numeric(age_all, errors='coerce')
    sex = sex_all.copy()
    df_match = pd.DataFrame({
        'y': y.values, 'age': age.values, 'sex': sex.values,
    }, index=y.index).dropna()

    if len(df_match) < 60:
        print("  可用于匹配的样本过少，跳过")
        return None

    try:
        df_match['age_bin'] = pd.qcut(df_match['age'], q=3,
                                       labels=['young', 'mid', 'old'],
                                       duplicates='drop')
    except Exception:
        df_match['age_bin'] = pd.cut(df_match['age'], bins=3,
                                      labels=['young', 'mid', 'old'])
    df_match['stratum'] = (df_match['age_bin'].astype(str) + '_'
                            + df_match['sex'].astype(str))

    rng = np.random.RandomState(RANDOM_STATE)
    matched_idx = []
    for s, g in df_match.groupby('stratum'):
        pos = g[g['y'] == 1]; neg = g[g['y'] == 0]
        k = min(len(pos), len(neg))
        if k < 3:
            continue
        matched_idx.extend(pos.sample(k, random_state=rng).index.tolist())
        matched_idx.extend(neg.sample(k, random_state=rng).index.tolist())

    if len(matched_idx) < 40:
        print(f"  匹配后样本过少 (n={len(matched_idx)})，跳过")
        return None

    X_m = X_raw.loc[matched_idx]
    y_m = y.loc[matched_idx]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    aucs = []
    for tri, vai in skf.split(X_m, y_m):
        imp = DataImputer().fit(X_m.iloc[tri], y_m.iloc[tri])
        Xtr = imp.transform(X_m.iloc[tri])
        Xva = imp.transform(X_m.iloc[vai])
        feats = [f for f in selected_features if f in Xtr.columns]
        if len(feats) == 0:
            continue
        m = lgb.LGBMClassifier(**best_params)
        m.fit(Xtr[feats], y_m.iloc[tri])
        aucs.append(roc_auc_score(y_m.iloc[vai],
                                   m.predict_proba(Xva[feats])[:, 1]))

    if len(aucs) == 0:
        return None

    res = {
        'Analysis_Type': 'Exploratory sensitivity analysis (no nested FS/tuning)',
        'N_matched_total': int(len(matched_idx)),
        'N_pos': int((y_m == 1).sum()),
        'N_neg': int((y_m == 0).sum()),
        'Matched_CV_AUC_mean': float(np.mean(aucs)),
        'Matched_CV_AUC_std': float(np.std(aucs)),
        'Matched_CV_AUC_folds': [float(a) for a in aucs],
        'Caveat': ('Uses pre-selected features and pre-tuned params; '
                   'no nested feature selection / tuning inside matched subset.')
    }
    with open(os.path.join(save_dir, 'age_sex_matched_sensitivity.json'),
              'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"  年龄/性别匹配后 5折CV AUC = {res['Matched_CV_AUC_mean']:.4f} "
          f"(n={len(matched_idx)}, pos={res['N_pos']}, neg={res['N_neg']}) [探索性]")
    return res


# =============================================================================
# 2. 五模型比较
# =============================================================================
def compare_five_models(X_train, y_train, save_dir,
                        n_splits=5, random_state=RANDOM_STATE):
    print("\n[0] 五模型 AUC 比较（5折CV + 合并ROC）")
    neg_pos = (len(y_train) - sum(y_train)) / max(sum(y_train), 1)
    models = {
        'LightGBM': lgb.LGBMClassifier(
            objective='binary', is_unbalance=True,
            random_state=random_state, verbose=-1),
        'XGBoost': xgb.XGBClassifier(
            scale_pos_weight=neg_pos, random_state=random_state,
            use_label_encoder=False, eval_metric='logloss', verbose=0),
        'Random Forest': RandomForestClassifier(
            class_weight='balanced', random_state=random_state, n_jobs=-1),
        'Logistic Regression': LogisticRegression(
            class_weight='balanced', random_state=random_state, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(
            class_weight='balanced', random_state=random_state),
    }
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    rows = []
    roc_data = {}
    for name, model in models.items():
        fold_aucs = []
        oof_prob = np.zeros(len(y_train))
        for tri, vai in skf.split(X_train, y_train):
            model.fit(X_train.iloc[tri], y_train.iloc[tri])
            prob = model.predict_proba(X_train.iloc[vai])[:, 1]
            oof_prob[vai] = prob
            fold_aucs.append(roc_auc_score(y_train.iloc[vai], prob))
        fold_aucs = np.array(fold_aucs)
        mean_auc = fold_aucs.mean()
        se = fold_aucs.std(ddof=1) / np.sqrt(len(fold_aucs))
        try:
            ci_low, ci_high = t_dist.interval(
                0.95, len(fold_aucs) - 1, loc=mean_auc, scale=se)
        except Exception:
            ci_low, ci_high = mean_auc, mean_auc
        fpr, tpr, _ = roc_curve(y_train, oof_prob)
        oof_auc = roc_auc_score(y_train, oof_prob)
        roc_data[name] = {'fpr': fpr, 'tpr': tpr, 'auc': oof_auc}
        rows.append({'Model': name, 'AUC_mean': mean_auc, 'AUC_std': fold_aucs.std(),
                     'CI_low': ci_low, 'CI_high': ci_high})
    cmp_df = pd.DataFrame(rows).sort_values('AUC_mean', ascending=False)
    cmp_df.to_csv(os.path.join(save_dir, 'model_comparison_auc.csv'),
                  index=False, encoding='utf-8-sig')
    print(cmp_df.to_string(index=False))

    plt.figure(figsize=(9, 5.5))
    x_pos = np.arange(len(cmp_df))
    yerr_low = cmp_df['AUC_mean'] - cmp_df['CI_low']
    yerr_high = cmp_df['CI_high'] - cmp_df['AUC_mean']
    plt.errorbar(x_pos, cmp_df['AUC_mean'], yerr=[yerr_low, yerr_high],
                 fmt='o', capsize=10, capthick=2, color='tab:blue',
                 markersize=12, linewidth=2)
    for i, row in cmp_df.reset_index(drop=True).iterrows():
        plt.text(i, row['CI_high'] + 0.015,
                 f"{row['AUC_mean']:.3f}\n({row['CI_low']:.3f}-{row['CI_high']:.3f})",
                 ha='center', va='bottom', fontsize=9)
    plt.xticks(x_pos, cmp_df['Model'], rotation=15, ha='right')
    plt.ylabel('AUC (5-fold CV, 95% CI)', fontsize=12)
    plt.title(f'Five-Model AUC Comparison ({MODEL_SHORT})', fontsize=14)
    plt.ylim(max(0.4, cmp_df['CI_low'].min() - 0.05),
             min(1.02, cmp_df['CI_high'].max() + 0.08))
    plt.grid(alpha=0.3, axis='y'); plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'model_comparison_auc.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    style_map = {
        'LightGBM':            {'color': '#E41A1C', 'linestyle': '-'},
        'XGBoost':             {'color': '#FF7F00', 'linestyle': '-'},
        'Random Forest':       {'color': '#4DAF4A', 'linestyle': '-'},
        'Logistic Regression': {'color': '#377EB8', 'linestyle': '-'},
        'Decision Tree':       {'color': '#984EA3', 'linestyle': '--'},
    }
    plt.figure(figsize=(8, 8))
    for name in ['LightGBM', 'XGBoost', 'Random Forest',
                 'Logistic Regression', 'Decision Tree']:
        d = roc_data[name]; style = style_map[name]
        plt.plot(d['fpr'], d['tpr'], color=style['color'],
                 linestyle=style['linestyle'], linewidth=2.0,
                 label=f"{name} (AUC = {d['auc']:.3f})")
    plt.plot([0, 1], [0, 1], color='gray', linestyle=':', linewidth=1.5,
             label='Chance')
    plt.xlim([-0.02, 1.02]); plt.ylim([-0.02, 1.02])
    plt.xlabel('1-Specificity (False Positive Rate)', fontsize=14)
    plt.ylabel('Sensitivity (True Positive Rate)', fontsize=14)
    plt.title(f'ROC Curves of Five Machine Learning Models ({MODEL_SHORT})',
              fontsize=15, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11, frameon=True)
    plt.grid(alpha=0.3); plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'model_comparison_roc.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print("五模型 AUC 误差棒图 + 合并 ROC 图已保存")
    return cmp_df


# =============================================================================
# 3. 风险分层（数据驱动）
# =============================================================================
def risk_stratification_analysis(y_true, y_prob, q1, q2, save_dir, save_prefix):
    y_true = np.asarray(y_true); y_prob = np.asarray(y_prob)
    strata = pd.cut(y_prob, bins=[-np.inf, q1, q2, np.inf],
                    labels=['Low', 'Middle', 'High'])
    rows = []
    for s in ['Low', 'Middle', 'High']:
        mask = np.array(strata == s)
        n = int(mask.sum()); events = int(y_true[mask].sum())
        event_rate = float(y_true[mask].mean()) if n > 0 else np.nan
        if n > 0 and 0 < events < n:
            ci_low, ci_high = proportion_confint(events, n, method='wilson')
        else:
            ci_low, ci_high = np.nan, np.nan
        rows.append({'Stratum': s, 'N': n, 'Events': events,
                     'Event_Rate': event_rate,
                     'CI_Low': float(ci_low) if not np.isnan(ci_low) else np.nan,
                     'CI_High': float(ci_high) if not np.isnan(ci_high) else np.nan})
    res_df = pd.DataFrame(rows)
    res_df.to_csv(os.path.join(save_dir, f'{save_prefix}_risk_strata.csv'),
                  index=False, encoding='utf-8-sig')
    plt.figure(figsize=(7, 5))
    x = np.arange(len(res_df)); colors = ['#4CAF50', '#FFC107', '#F44336']
    err_low = (res_df['Event_Rate'] - res_df['CI_Low']).fillna(0).values
    err_high = (res_df['CI_High'] - res_df['Event_Rate']).fillna(0).values
    plt.bar(x, res_df['Event_Rate'], yerr=[err_low, err_high],
            capsize=8, color=colors, edgecolor='black', linewidth=1.2)
    for i, row in res_df.iterrows():
        y_text = row['Event_Rate'] + (row['CI_High'] - row['Event_Rate']
                                       if not np.isnan(row['CI_High']) else 0) + 0.03
        plt.text(i, y_text, f"{row['Event_Rate']:.2f}\n(n={row['N']})",
                 ha='center', va='bottom', fontsize=10)
    plt.xticks(x, res_df['Stratum']); plt.ylabel('Event Rate')
    plt.title(f'Data-driven Risk Stratification — {save_prefix} '
              f'(cut-points from training OOF; not clinically validated)')
    plt.ylim(0, 1.15); plt.grid(alpha=0.3, axis='y'); plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{save_prefix}_risk_strata.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    return res_df


# =============================================================================
# 4. 风险组校准（Train 参考 / Test / External）
# =============================================================================
def plot_risk_group_calibration(y_train_true, y_train_prob,
                                 y_test_true, y_test_prob,
                                 y_ext_true, y_ext_prob,
                                 q1, q2, save_path, model_name="Model"):
    def _group_calib(y_true, y_prob):
        y_true = np.asarray(y_true); y_prob = np.asarray(y_prob)
        strata = pd.cut(y_prob, bins=[-np.inf, q1, q2, np.inf],
                        labels=['Low', 'Middle', 'High'])
        means, obs = [], []
        for s in ['Low', 'Middle', 'High']:
            mask = np.array(strata == s)
            if mask.sum() == 0:
                means.append(np.nan); obs.append(np.nan)
            else:
                means.append(float(y_prob[mask].mean()))
                obs.append(float(y_true[mask].mean()))
        return np.array(means), np.array(obs)

    plt.figure(figsize=(8, 7))
    m_tr, o_tr = _group_calib(y_train_true, y_train_prob)
    plt.plot(m_tr, o_tr, marker='o', color='tab:blue', linewidth=2,
             markersize=10, label='Train (optimistic, reference only)')
    m_te, o_te = _group_calib(y_test_true, y_test_prob)
    plt.plot(m_te, o_te, marker='o', color='tab:orange', linewidth=2,
             markersize=10, label='Test (primary)')
    m_ex, o_ex = _group_calib(y_ext_true, y_ext_prob)
    plt.plot(m_ex, o_ex, marker='o', color='tab:green', linewidth=2,
             markersize=10, label='External')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect')
    plt.xlabel('Mean predicted probability', fontsize=14)
    plt.ylabel('Observed positive rate', fontsize=14)
    plt.title(f'Risk Group Calibration ({model_name})', fontsize=15)
    plt.legend(fontsize=12, loc='best'); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print("风险组校准曲线已保存（Test/External 为主，Train 仅参考）")


# =============================================================================
# 5. Test vs External 合并校准 + DCA
# =============================================================================
def plot_combined_calibration_test_external(y_test_true, y_test_prob,
                                             y_ext_true, y_ext_prob, save_path):
    plt.figure(figsize=(8, 8))
    pt_i, pp_i = calibration_curve(y_test_true, y_test_prob,
                                    n_bins=10, strategy='quantile')
    auc_i = roc_auc_score(y_test_true, y_test_prob)
    plt.plot(pp_i, pt_i, marker='o', color='tab:blue', linewidth=2,
             markersize=8, label=f'Test (AUC={auc_i:.3f})')
    pt_e, pp_e = calibration_curve(y_ext_true, y_ext_prob,
                                    n_bins=10, strategy='quantile')
    auc_e = roc_auc_score(y_ext_true, y_ext_prob)
    plt.plot(pp_e, pt_e, marker='o', color='tab:orange', linewidth=2,
             markersize=8, label=f'External (AUC={auc_e:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect')
    plt.xlabel('Mean predicted probability', fontsize=14)
    plt.ylabel('Observed fraction', fontsize=14)
    plt.title(f'Calibration: Test vs External ({MODEL_SHORT})', fontsize=15)
    plt.legend(fontsize=12); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print("Test vs External 校准曲线已保存")


def plot_combined_dca_test_external(y_test_true, y_test_prob,
                                     y_ext_true, y_ext_prob,
                                     save_path, ext_label="External"):
    dca_i = decision_curve_analysis(y_test_true, y_test_prob)
    dca_e = decision_curve_analysis(y_ext_true, y_ext_prob)
    plt.figure(figsize=(9, 6))
    plt.plot(dca_i['thresholds'], dca_i['nb_model'], color='tab:blue',
             linewidth=2, label='Test Model')
    plt.plot(dca_e['thresholds'], dca_e['nb_model'], color='tab:orange',
             linewidth=2, label=f'External Model ({ext_label})')
    plt.plot(dca_i['thresholds'], dca_i['nb_all'], color='gray',
             linestyle='--', linewidth=2, label='Treat all (Test)')
    plt.plot(dca_e['thresholds'], dca_e['nb_all'], color='orange',
             linestyle='--', linewidth=2, label='Treat all (External)')
    plt.plot(dca_i['thresholds'], dca_i['nb_none'], color='black',
             linestyle=':', linewidth=2, label='Treat none')
    plt.xlabel('Threshold probability', fontsize=14)
    plt.ylabel('Net benefit', fontsize=14)
    plt.title(f'Decision Curve Analysis: Test vs External ({MODEL_SHORT})', fontsize=15)
    y_max = max(dca_i['nb_all'].max(), dca_e['nb_all'].max()) * 1.15
    plt.ylim(-0.1, y_max)
    plt.legend(fontsize=11, loc='upper right'); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print("Test vs External DCA 已保存")


# =============================================================================
# 6. RFE / LASSO / 网格搜索 / 嵌套 CV
# =============================================================================
def rfe_select_features(X, y, min_features=MIN_FEATURES, step=RFE_STEP,
                        cv=RFE_CV, random_state=RANDOM_STATE):
    features = list(X.columns); best_auc = -np.inf; best_features = features.copy()
    prev_auc = None
    while len(features) > min_features:
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
        auc_scores, importances = [], []
        for tr_idx, va_idx in skf.split(X[features], y):
            X_tr = X[features].iloc[tr_idx]; X_va = X[features].iloc[va_idx]
            y_tr = y.iloc[tr_idx]; y_va = y.iloc[va_idx]
            model = lgb.LGBMClassifier(objective='binary', is_unbalance=True,
                                        n_estimators=100, learning_rate=0.05,
                                        num_leaves=31, random_state=random_state, verbose=-1)
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric='auc',
                      callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)])
            auc_scores.append(roc_auc_score(y_va, model.predict_proba(X_va)[:, 1]))
            importances.append(model.feature_importances_)
        mean_auc = np.mean(auc_scores)
        if prev_auc is not None and (prev_auc - mean_auc) > 0.01: break
        prev_auc = mean_auc
        if mean_auc > best_auc:
            best_auc = mean_auc; best_features = features.copy()
        avg_imp = np.mean(importances, axis=0)
        imp_df = pd.DataFrame({'feature': features, 'importance': avg_imp}
                              ).sort_values('importance', ascending=True)
        if len(features) - step < min_features:
            step = len(features) - min_features
            if step <= 0: break
        drop_features = imp_df.head(step)['feature'].tolist()
        features = [f for f in features if f not in drop_features]
    if len(best_features) > min_features:
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
        last_importances = []
        for tr_idx, va_idx in skf.split(X[best_features], y):
            X_tr = X[best_features].iloc[tr_idx]; X_va = X[best_features].iloc[va_idx]
            y_tr = y.iloc[tr_idx]; y_va = y.iloc[va_idx]
            model = lgb.LGBMClassifier(objective='binary', is_unbalance=True,
                                        n_estimators=100, learning_rate=0.05,
                                        num_leaves=31, random_state=random_state, verbose=-1)
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric='auc',
                      callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)])
            last_importances.append(model.feature_importances_)
        avg_imp = np.mean(last_importances, axis=0)
        imp_df = pd.DataFrame({'feature': best_features, 'importance': avg_imp}
                              ).sort_values('importance', ascending=False)
        best_features = imp_df.head(min_features)['feature'].tolist()
    return best_features


def lasso_refine_features(X, y, features, min_features=LASSO_MIN_FEATURES,
                          c_grid=LASSO_C_GRID, cv=5, random_state=RANDOM_STATE):
    if len(features) <= min_features: return features, None
    pipe = Pipeline([('scaler', StandardScaler()),
                     ('lasso', LogisticRegression(penalty='l1', solver='liblinear',
                                                   class_weight='balanced',
                                                   max_iter=5000,
                                                   random_state=random_state))])
    grid = GridSearchCV(pipe, {'lasso__C': list(c_grid)}, cv=cv,
                        scoring='roc_auc', n_jobs=-1, refit=True)
    grid.fit(X[features], y)
    best_lasso = grid.best_estimator_.named_steps['lasso']
    coef = best_lasso.coef_[0]
    selected = [f for f, c in zip(features, coef) if abs(c) > 1e-6]
    if len(selected) < min_features:
        coef_df = pd.DataFrame({'feature': features, 'coef': coef})
        coef_df['abs_coef'] = coef_df['coef'].abs()
        coef_df = coef_df.sort_values('abs_coef', ascending=False)
        selected = coef_df.head(min_features)['feature'].tolist()
    return selected, grid


def grid_search_lgbm(X, y, cv=TUNE_CV, random_state=RANDOM_STATE,
                     param_grid=None, fixed_params=None):
    if param_grid is None: param_grid = NESTED_PARAM_GRID
    if fixed_params is None: fixed_params = NESTED_FIXED_PARAMS
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    best_score = -np.inf; best_params = None
    for params in ParameterGrid(param_grid):
        current = {**fixed_params, **params}
        fold_aucs = []
        for tr_idx, va_idx in skf.split(X, y):
            X_tr = X.iloc[tr_idx]; X_va = X.iloc[va_idx]
            y_tr = y.iloc[tr_idx]; y_va = y.iloc[va_idx]
            m = lgb.LGBMClassifier(**current)
            m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric='auc',
                  callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
            fold_aucs.append(roc_auc_score(y_va, m.predict_proba(X_va)[:, 1]))
        mean_auc = np.mean(fold_aucs)
        if mean_auc > best_score:
            best_score = mean_auc; best_params = current
    return best_params, best_score


def run_nested_cv(X_raw, y, outer_cv=NESTED_OUTER_CV, inner_cv=NESTED_INNER_CV,
                  random_state=RANDOM_STATE):
    """嵌套 CV 只在训练集内调用。"""
    outer_skf = StratifiedKFold(n_splits=outer_cv, shuffle=True, random_state=random_state)
    outer_results = []
    oof_pred = np.full(len(y), np.nan); oof_index = np.zeros(len(y), dtype=bool)
    y_array = np.asarray(y)
    for fold_idx, (tr_idx, va_idx) in enumerate(outer_skf.split(X_raw, y)):
        print(f"\n--- 嵌套 CV 外层折 {fold_idx + 1}/{outer_cv} ---")
        X_tr_raw = X_raw.iloc[tr_idx]; y_tr = y.iloc[tr_idx]
        X_va_raw = X_raw.iloc[va_idx]; y_va = y.iloc[va_idx]
        imputer = DataImputer().fit(X_tr_raw, y_tr)
        X_tr_imp = imputer.transform(X_tr_raw); X_va_imp = imputer.transform(X_va_raw)
        print("  步骤 1/4: RFE ...")
        rfe_feats = rfe_select_features(X_tr_imp, y_tr, min_features=MIN_FEATURES,
                                         step=RFE_STEP, cv=inner_cv,
                                         random_state=random_state)
        print(f"  RFE 特征数: {len(rfe_feats)}")
        if RUN_LASSO_REFINE:
            print("  步骤 2/4: LASSO ...")
            sel_feats, _ = lasso_refine_features(X_tr_imp, y_tr, rfe_feats,
                                                  min_features=LASSO_MIN_FEATURES,
                                                  c_grid=LASSO_C_GRID, cv=inner_cv,
                                                  random_state=random_state)
        else:
            sel_feats = rfe_feats
        print(f"  LASSO 后特征数: {len(sel_feats)}")
        print("  步骤 3/4: 网格搜索 ...")
        best_p, best_s = grid_search_lgbm(X_tr_imp[sel_feats], y_tr, cv=inner_cv,
                                          random_state=random_state,
                                          param_grid=NESTED_PARAM_GRID,
                                          fixed_params=NESTED_FIXED_PARAMS)
        print(f"  内层 CV AUC: {best_s:.4f}")
        print("  步骤 4/4: 外层验证 ...")
        Xt, Xv, yt, yv = train_test_split(X_tr_imp[sel_feats], y_tr, test_size=0.1,
                                          random_state=random_state, stratify=y_tr)
        m = lgb.LGBMClassifier(**best_p)
        m.fit(Xt, yt, eval_set=[(Xv, yv)],
              callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        prob = m.predict_proba(X_va_imp[sel_feats])[:, 1]
        auc_va = roc_auc_score(y_va, prob)
        oof_pred[va_idx] = prob; oof_index[va_idx] = True
        outer_results.append({
            'fold': fold_idx + 1, 'n_rfe_features': len(rfe_feats),
            'n_selected_features': len(sel_feats), 'inner_cv_auc': best_s,
            'outer_valid_auc': auc_va,
            'selected_features': json.dumps(sel_feats, ensure_ascii=False),
            'best_params': json.dumps({k: v for k, v in best_p.items()
                                        if k not in ['objective', 'metric',
                                                     'boosting_type', 'verbose',
                                                     'is_unbalance', 'random_state']},
                                       ensure_ascii=False)})
        print(f"  外层折 {fold_idx + 1} AUC = {auc_va:.4f}")
    oof_y_true = y_array[oof_index]; oof_y_prob = oof_pred[oof_index]
    return {'outer_results': pd.DataFrame(outer_results),
            'overall_auc': roc_auc_score(oof_y_true, oof_y_prob),
            'overall_metrics': bootstrap_ci_metrics(oof_y_true, oof_y_prob,
                                                     n_bootstraps=N_BOOTSTRAPS,
                                                     threshold=0.5,
                                                     random_state=random_state),
            'oof_y_true': oof_y_true, 'oof_y_prob': oof_y_prob,
            'oof_index': oof_index, 'oof_pred_full': oof_pred}


# =============================================================================
# 7. 主流程：数据加载
# =============================================================================
print(f"\n[1] 数据加载与清洗 ({MODEL_SHORT})")
df = pd.read_excel(FILE_PATH, sheet_name=SHEET_NAME)
df.columns = [str(c).strip() for c in df.columns]
print(f"原始数据形状: {df.shape}")

if TARGET_COL not in df.columns:
    raise ValueError(f"目标列 {TARGET_COL} 不存在")

age_all = clean_numeric_series(df[AGE_COL]) if AGE_COL in df.columns else None
sex_all = None
if SEX_COL in df.columns:
    sex_raw = df[SEX_COL].astype(str).str.strip()
    sex_map = {'男': 1, '男性': 1, 'male': 1, 'Male': 1, 'M': 1, 'm': 1,
               '女': 0, '女性': 0, 'female': 0, 'Female': 0, 'F': 0, 'f': 0}
    sex_all = sex_raw.map(sex_map)
    if sex_all.isnull().any():
        sex_all = pd.factorize(sex_raw)[0]
    sex_all = sex_all.astype(int)

y_all = pd.to_numeric(df[TARGET_COL], errors='coerce')
if y_all.isnull().any():
    raise ValueError("目标列存在缺失或非数值")
y_all = y_all.astype(int)
print(f"阳性例数: {(y_all == 1).sum()}, 阴性例数: {(y_all == 0).sum()}")

if "诊断" in df.columns:
    print("\n诊断分布 Top 10:")
    print(df["诊断"].astype(str).value_counts().head(10).to_string())
if "科室" in df.columns:
    print("\n科室分布 Top 10:")
    print(df["科室"].astype(str).value_counts().head(10).to_string())

feature_cols_raw = [c for c in df.columns if c not in LEAK_COLS]
print(f"\n候选特征列数（已排除泄漏列）: {len(feature_cols_raw)}")

X_raw = df[feature_cols_raw].copy()
for c in X_raw.columns:
    X_raw[c] = clean_numeric_series(X_raw[c])

seen = set(); name_mapping = {}; new_cols = []
for c in X_raw.columns:
    new_c = clean_feature_name(c, seen)
    new_cols.append(new_c); name_mapping[new_c] = c
X_raw.columns = new_cols
pd.DataFrame({'Cleaned_Name': list(name_mapping.keys()),
              'Original_Name': list(name_mapping.values())}
             ).to_csv(os.path.join(OUTPUT_DIR, 'feature_name_mapping.csv'),
                      index=False, encoding='utf-8-sig')
print(f"特征名已清洗，共 {len(name_mapping)} 列")

miss_df = report_missingness(X_raw, os.path.join(OUTPUT_DIR, 'missingness_report.csv'))
all_null_cols = X_raw.columns[X_raw.isnull().all()]
if len(all_null_cols) > 0:
    X_raw = X_raw.drop(columns=all_null_cols)
    print(f"删除全空列: {len(all_null_cols)}")


# =============================================================================
# 8. 训练/测试划分
# =============================================================================
print("\n[2] 划分训练集/测试集（70/30分层抽样）")
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, y_all, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_all)
print(f"训练集: {len(X_train_raw)} 例 (阳性比例 {y_train.mean():.3f})")
print(f"测试集: {len(X_test_raw)} 例 (阳性比例 {y_test.mean():.3f})")

age_train = age_all.loc[X_train_raw.index] if age_all is not None else None
age_test = age_all.loc[X_test_raw.index] if age_all is not None else None
sex_train = sex_all.loc[X_train_raw.index] if sex_all is not None else None
sex_test = sex_all.loc[X_test_raw.index] if sex_all is not None else None

print("\n[2b] 缺失机制分析（训练集内）")
missingness_mechanism_df, missingness_mech_summary = missingness_mechanism_analysis(
    X_train_raw, y_train, OUTPUT_DIR)


# =============================================================================
# 9. 五模型比较
# =============================================================================
imputer_cmp = DataImputer().fit(X_train_raw, y_train)
X_train_cmp = imputer_cmp.transform(X_train_raw)
model_comparison_df = compare_five_models(X_train_cmp, y_train, OUTPUT_DIR)


# =============================================================================
# 10. 嵌套 CV（只在训练集内）
# =============================================================================
nested_cv_results = None
best_threshold = None
tertile_q1 = None
tertile_q2 = None

if RUN_NESTED_CV:
    print("\n[3] 嵌套交叉验证（仅在训练集内运行）")
    nested_cv_results = run_nested_cv(X_train_raw, y_train)
    nested_output_dir = os.path.join(OUTPUT_DIR, "nested_cv")
    os.makedirs(nested_output_dir, exist_ok=True)
    nested_cv_results['outer_results'].to_csv(
        os.path.join(nested_output_dir, 'nested_cv_outer_results.csv'), index=False)
    pd.DataFrame([{'Metric': k, 'Point': p, 'CI_Low': lo, 'CI_High': hi}
                  for k, (p, lo, hi) in nested_cv_results['overall_metrics'].items()]
                 ).to_csv(os.path.join(nested_output_dir,
                                       'nested_cv_overall_metrics.csv'), index=False)
    pd.DataFrame({'y_true': nested_cv_results['oof_y_true'],
                  'y_prob': nested_cv_results['oof_y_prob']}
                 ).to_csv(os.path.join(nested_output_dir,
                                       'nested_cv_oof_predictions.csv'), index=False)
    fpr_n, tpr_n, _ = roc_curve(nested_cv_results['oof_y_true'],
                                 nested_cv_results['oof_y_prob'])
    plt.figure(figsize=(6, 6))
    plt.plot(fpr_n, tpr_n,
             label=f"Training-set OOF AUC = {nested_cv_results['overall_auc']:.3f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('FPR'); plt.ylabel('TPR')
    plt.title('ROC — Nested CV OOF (Training set only)')
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(nested_output_dir, 'nested_cv_roc.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"训练集嵌套 CV OOF AUC = {nested_cv_results['overall_auc']:.4f}")

    oof_y_true = nested_cv_results['oof_y_true']
    oof_y_prob = nested_cv_results['oof_y_prob']
    fpr_o, tpr_o, thr_o = roc_curve(oof_y_true, oof_y_prob)
    best_threshold = float(thr_o[int(np.argmax(tpr_o - fpr_o))])
    tertile_q1 = float(np.percentile(oof_y_prob, 33.33))
    tertile_q2 = float(np.percentile(oof_y_prob, 66.67))
    print(f"训练集 OOF 锁定最佳阈值: {best_threshold:.4f}")
    print(f"训练集 OOF 锁定三分位切点: q1={tertile_q1:.4f}, q2={tertile_q2:.4f}")

    threshold_ci = bootstrap_threshold_and_tertiles(
        oof_y_true, oof_y_prob, n_bootstraps=N_BOOTSTRAPS_CI,
        random_state=RANDOM_STATE)
    with open(os.path.join(OUTPUT_DIR, 'locked_cutpoints.json'),
              'w', encoding='utf-8') as f:
        json.dump({
            'source': 'Training-set nested-CV OOF predictions only',
            'note': ('Threshold and tertile cut-points are locked using '
                     'training-set OOF predictions. Test set labels are NOT used.'),
            'best_threshold': best_threshold,
            'best_threshold_CI': threshold_ci['threshold_CI'],
            'tertile_q1': tertile_q1, 'tertile_q1_CI': threshold_ci['q1_CI'],
            'tertile_q2': tertile_q2, 'tertile_q2_CI': threshold_ci['q2_CI'],
            'risk_stratification_caveat': (
                'Data-driven cut-points; NOT clinically validated.')
        }, f, ensure_ascii=False, indent=2)


# =============================================================================
# 11. 训练集内最终特征选择与调参
# =============================================================================
print("\n[4] 训练集内最终 RFE + LASSO + 网格搜索")
imputer = DataImputer(high_miss=HIGH_MISS_THRESHOLD,
                     low_miss=LOW_MISS_THRESHOLD).fit(X_train_raw, y_train)
X_train_imp = imputer.transform(X_train_raw)
X_test_imp = imputer.transform(X_test_raw)

with open(os.path.join(OUTPUT_DIR, 'impute_record.json'), 'w', encoding='utf-8') as f:
    json.dump({
        'high_miss_threshold': HIGH_MISS_THRESHOLD,
        'low_miss_threshold': LOW_MISS_THRESHOLD,
        'threshold_rationale': (
            'High missing >30% dropped (structural missingness, high imputation bias). '
            'Missing <=10% imputed by mean/median based on skewness. '
            '10%-30% imputed by median (robust). '
            'See imputation_sensitivity.csv for robustness check.'),
        'high_missing_dropped': imputer.drop_cols_,
        'mid_missing_imputed': imputer.mid_cols_,
        'low_missing_imputed': imputer.low_cols_,
        'fill_values': {k: float(v) for k, v in imputer.fill_values_.items()}},
        f, ensure_ascii=False, indent=2)

rfe_features = rfe_select_features(X_train_imp, y_train)
print(f"RFE 保留特征数: {len(rfe_features)}")

selected_features = rfe_features
if RUN_LASSO_REFINE:
    selected_features, lasso_model = lasso_refine_features(
        X_train_imp, y_train, rfe_features)
    print(f"LASSO 精修后最终特征数: {len(selected_features)}")

best_params, best_cv_auc = grid_search_lgbm(X_train_imp[selected_features], y_train)
print(f"最佳 CV AUC: {best_cv_auc:.4f}")
pd.Series(selected_features, name='selected_features').to_csv(
    os.path.join(OUTPUT_DIR, 'selected_features.csv'), index=False)
with open(os.path.join(OUTPUT_DIR, 'best_params.json'), 'w', encoding='utf-8') as f:
    json.dump(best_params, f, ensure_ascii=False, indent=2, default=str)

n_events_train = int((y_train == 1).sum())
epv = n_events_train / max(len(selected_features), 1)
with open(os.path.join(OUTPUT_DIR, 'epv_report.json'), 'w', encoding='utf-8') as f:
    json.dump({
        'n_events_train': n_events_train,
        'n_selected_features': len(selected_features),
        'EPV_train': float(epv),
        'note': 'EPV = events / number of selected features. EPV>=10 preferred.'
    }, f, ensure_ascii=False, indent=2)
print(f"EPV（训练集）: {epv:.2f} (events={n_events_train}, "
      f"features={len(selected_features)})")


# =============================================================================
# 11b. 训练集 OOF 上的 DeLong 检验（嵌套 LightGBM vs 基线）
# =============================================================================
print("\n[4b] 基线模型训练集 OOF（用于 DeLong 比较）")
X_tr_sel_final = X_train_imp[selected_features]
skf_oof = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

baseline_oof_models = {
    'Logistic Regression': Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(class_weight='balanced',
                                   random_state=RANDOM_STATE, max_iter=1000))
    ]),
    'Random Forest': RandomForestClassifier(
        class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1),
    'XGBoost': xgb.XGBClassifier(
        scale_pos_weight=(len(y_train) - sum(y_train)) / max(sum(y_train), 1),
        random_state=RANDOM_STATE, use_label_encoder=False,
        eval_metric='logloss', verbose=0),
}

baseline_oof = {}
for name, m in baseline_oof_models.items():
    oof = np.zeros(len(y_train))
    for tri, vai in skf_oof.split(X_tr_sel_final, y_train):
        m.fit(X_tr_sel_final.iloc[tri], y_train.iloc[tri])
        oof[vai] = m.predict_proba(X_tr_sel_final.iloc[vai])[:, 1]
    baseline_oof[name] = oof
    print(f"  {name} 训练集 OOF AUC = {roc_auc_score(y_train, oof):.4f}")

if RUN_NESTED_CV and nested_cv_results is not None:
    lgb_oof = nested_cv_results['oof_pred_full']
    lgb_oof_true = np.asarray(y_train)
    auc_lgb = roc_auc_score(lgb_oof_true, lgb_oof)
    delong_oof_results = []
    for name, oof in baseline_oof.items():
        try:
            p_val = delong_roc_test(lgb_oof_true, lgb_oof, oof)
        except Exception:
            p_val = np.nan
        auc_cmp = roc_auc_score(lgb_oof_true, oof)
        delong_oof_results.append({
            'Comparison': f'LightGBM (Nested OOF) vs {name}',
            'AUC_A': auc_lgb, 'AUC_B': auc_cmp,
            'AUC_diff': auc_lgb - auc_cmp,
            'DeLong_p': p_val, 'N_train': len(y_train),
            'Note': ('Training-set OOF; larger n than test set. '
                     'Baseline OOF uses fixed selected features (5-fold CV).')
        })
        print(f"  LightGBM vs {name}: AUC {auc_lgb:.3f} vs {auc_cmp:.3f}, p={p_val:.4f}")
    pd.DataFrame(delong_oof_results).to_csv(
        os.path.join(OUTPUT_DIR, 'nested_cv', 'delong_test_on_training_oof.csv'),
        index=False, encoding='utf-8-sig')


# =============================================================================
# 12. 最终模型训练
# =============================================================================
print("\n[5] 训练最终模型")
X_tr_f, X_v_f, y_tr_f, y_v_f = train_test_split(
    X_train_imp[selected_features], y_train, test_size=0.1,
    random_state=RANDOM_STATE, stratify=y_train)
final_model = lgb.LGBMClassifier(**best_params)
final_model.fit(X_tr_f, y_tr_f, eval_set=[(X_v_f, y_v_f)],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
joblib.dump(final_model, os.path.join(OUTPUT_DIR, 'final_model.pkl'))
joblib.dump(imputer, os.path.join(OUTPUT_DIR, 'imputer.pkl'))


# =============================================================================
# 13. 测试集评估
# =============================================================================
print("\n[6] 测试集评估（仅一次，使用训练集锁定阈值）")
y_pred_prob = final_model.predict_proba(X_test_imp[selected_features])[:, 1]
train_prob_in = final_model.predict_proba(X_train_imp[selected_features])[:, 1]

if best_threshold is None:
    print("  未运行嵌套 CV，回退到训练集内 5 折 CV 锁定阈值")
    skf_thr = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    train_cv_prob = np.zeros(len(y_train))
    for tri, vai in skf_thr.split(X_train_imp[selected_features], y_train):
        m = lgb.LGBMClassifier(**best_params)
        m.fit(X_train_imp[selected_features].iloc[tri], y_train.iloc[tri])
        train_cv_prob[vai] = m.predict_proba(
            X_train_imp[selected_features].iloc[vai])[:, 1]
    fpr_t, tpr_t, thr_t = roc_curve(y_train, train_cv_prob)
    best_threshold = float(thr_t[int(np.argmax(tpr_t - fpr_t))])
    tertile_q1 = float(np.percentile(train_cv_prob, 33.33))
    tertile_q2 = float(np.percentile(train_cv_prob, 66.67))

print(f"使用训练集锁定阈值: {best_threshold:.4f}")
test_metrics_05 = bootstrap_ci_metrics(y_test, y_pred_prob,
                                       n_bootstraps=N_BOOTSTRAPS, threshold=0.5)
test_metrics_locked = bootstrap_ci_metrics(y_test, y_pred_prob,
                                           n_bootstraps=N_BOOTSTRAPS,
                                           threshold=best_threshold)
for name, met in [('threshold_0.5', test_metrics_05),
                  ('threshold_locked', test_metrics_locked)]:
    pd.DataFrame([{'Metric': k, 'Point': p, 'CI_Low': lo, 'CI_High': hi}
                  for k, (p, lo, hi) in met.items()]
                 ).to_csv(os.path.join(OUTPUT_DIR,
                                       f'test_metrics_{name}.csv'), index=False)
pd.DataFrame({'y_true': y_test.values, 'y_prob': y_pred_prob}
             ).to_csv(os.path.join(OUTPUT_DIR, 'test_predictions.csv'), index=False)

# 测试集校准 + CI
print("\n[6b] 测试集校准指标 + Bootstrap CI")
test_cal = calibration_analysis(y_test.values, y_pred_prob,
                                OUTPUT_DIR, save_prefix='test_calibration')
test_cal_boot = bootstrap_calibration_dca(y_test.values, y_pred_prob,
                                          n_bootstraps=N_BOOTSTRAPS_CI,
                                          random_state=RANDOM_STATE)
test_cal_report = {
    'Role': 'Primary calibration report (test set)',
    'Brier': test_cal['Brier'],
    'Brier_CI': list(test_cal_boot['Brier_CI']),
    'Calibration_Intercept': test_cal['Calibration_Intercept'],
    'Calibration_Intercept_CI': list(test_cal_boot['Calibration_Intercept_CI']),
    'Calibration_Slope': test_cal['Calibration_Slope'],
    'Calibration_Slope_CI': list(test_cal_boot['Calibration_Slope_CI']),
    'DCA_NetBenefit_CI': test_cal_boot['DCA_CI'],
}
with open(os.path.join(OUTPUT_DIR, 'test_calibration_report.json'),
          'w', encoding='utf-8') as f:
    json.dump(test_cal_report, f, ensure_ascii=False, indent=2)
print(f"测试集 Brier = {test_cal['Brier']:.4f} (95% CI {test_cal_boot['Brier_CI']})")
print(f"测试集校准截距 = {test_cal['Calibration_Intercept']:.4f} "
      f"(95% CI {test_cal_boot['Calibration_Intercept_CI']})")
print(f"测试集校准斜率 = {test_cal['Calibration_Slope']:.4f} "
      f"(95% CI {test_cal_boot['Calibration_Slope_CI']})")

# 训练集校准仅参考
train_cal = calibration_analysis(y_train.values, train_prob_in,
                                 OUTPUT_DIR, save_prefix='train_calibration_reference_only')
with open(os.path.join(OUTPUT_DIR, 'train_calibration_reference_note.json'),
          'w', encoding='utf-8') as f:
    json.dump({'Role': 'Reference only; optimistically biased.',
               'Train_Brier': train_cal['Brier'],
               'Train_Calibration_Intercept': train_cal['Calibration_Intercept'],
               'Train_Calibration_Slope': train_cal['Calibration_Slope']},
              f, ensure_ascii=False, indent=2)

# 测试集 DeLong
print("\n[6d] DeLong 检验（测试集）")
X_tr_sel = X_train_imp[selected_features]
X_te_sel = X_test_imp[selected_features]

baseline_models_test = {
    'Logistic Regression': Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(class_weight='balanced',
                                   random_state=RANDOM_STATE, max_iter=1000))
    ]),
    'Random Forest': RandomForestClassifier(
        class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1),
    'XGBoost': xgb.XGBClassifier(
        scale_pos_weight=(len(y_train) - sum(y_train)) / max(sum(y_train), 1),
        random_state=RANDOM_STATE, use_label_encoder=False,
        eval_metric='logloss', verbose=0),
}

test_probs = {'LightGBM (Final)': y_pred_prob}
for name, m in baseline_models_test.items():
    m.fit(X_tr_sel, y_train)
    test_probs[name] = m.predict_proba(X_te_sel)[:, 1]

ref_name = 'LightGBM (Final)'
ref_prob = test_probs[ref_name]
auc_ref = roc_auc_score(y_test.values, ref_prob)
delong_test_results = []
for name, prob in test_probs.items():
    if name == ref_name:
        continue
    try:
        p_val = delong_roc_test(y_test.values, ref_prob, prob)
    except Exception:
        p_val = np.nan
    auc_cmp = roc_auc_score(y_test.values, prob)
    delong_test_results.append({
        'Comparison': f'{ref_name} vs {name}',
        'AUC_A': auc_ref, 'AUC_B': auc_cmp,
        'AUC_diff': auc_ref - auc_cmp, 'DeLong_p': p_val, 'N_test': len(y_test),
        'Note': ('Small test set; DeLong test underpowered. '
                 'Non-significant p should NOT be interpreted as equivalence.')
    })
    print(f"  {ref_name} vs {name}: AUC {auc_ref:.3f} vs {auc_cmp:.3f}, p={p_val:.4f}")

pd.DataFrame(delong_test_results).to_csv(
    os.path.join(OUTPUT_DIR, 'delong_test_on_test_set.csv'),
    index=False, encoding='utf-8-sig')


# =============================================================================
# 14. 外部验证
# =============================================================================
print(f"\n[7] 外部验证 ({EXT_POS_LABEL} vs {EXT_NEG_LABEL})")
ext_df = pd.read_excel(EXT_FILE_PATH, sheet_name=EXT_SHEET_NAME)
ext_df.columns = [str(c).strip() for c in ext_df.columns]
if EXT_GROUP_COL not in ext_df.columns:
    raise ValueError(f"外部数据缺少组别列: {EXT_GROUP_COL}")
ext_df[EXT_GROUP_COL] = ext_df[EXT_GROUP_COL].astype(str).str.strip()
ext_df = ext_df[ext_df[EXT_GROUP_COL].isin([EXT_POS_LABEL, EXT_NEG_LABEL])].copy()
y_ext = (ext_df[EXT_GROUP_COL] == EXT_POS_LABEL).astype(int).values
print(f"外部验证集: n={len(y_ext)}, {EXT_POS_LABEL}={(y_ext == 1).sum()}, "
      f"{EXT_NEG_LABEL}={(y_ext == 0).sum()}")

ext_features_raw = pd.DataFrame(index=ext_df.index)
for clean_name, orig_name in name_mapping.items():
    if orig_name in ext_df.columns:
        ext_features_raw[clean_name] = ext_df[orig_name]
    else:
        ext_features_raw[clean_name] = np.nan
for col in X_raw.columns:
    if col not in ext_features_raw.columns:
        ext_features_raw[col] = np.nan
ext_features_raw = ext_features_raw[X_raw.columns]
for c in ext_features_raw.columns:
    ext_features_raw[c] = clean_numeric_series(ext_features_raw[c])

X_ext_imp = imputer.transform(ext_features_raw)
X_ext_sel = X_ext_imp[selected_features]
ext_prob = final_model.predict_proba(X_ext_sel)[:, 1]

ext_output_dir = os.path.join(OUTPUT_DIR, "external_validation")
os.makedirs(ext_output_dir, exist_ok=True)

ext_metrics_05 = bootstrap_ci_metrics(y_ext, ext_prob,
                                       n_bootstraps=N_BOOTSTRAPS, threshold=0.5)
ext_metrics_locked = bootstrap_ci_metrics(y_ext, ext_prob,
                                           n_bootstraps=N_BOOTSTRAPS,
                                           threshold=best_threshold)
pd.DataFrame([{'Metric': k, 'Point': p, 'CI_Low': lo, 'CI_High': hi}
              for k, (p, lo, hi) in ext_metrics_05.items()]
             ).to_csv(os.path.join(ext_output_dir,
                                   f'{MODEL_SHORT}_metrics_threshold_0.5.csv'), index=False)
pd.DataFrame([{'Metric': k, 'Point': p, 'CI_Low': lo, 'CI_High': hi}
              for k, (p, lo, hi) in ext_metrics_locked.items()]
             ).to_csv(os.path.join(ext_output_dir,
                                   f'{MODEL_SHORT}_metrics_threshold_locked.csv'),
                      index=False)
pd.DataFrame({
    'sample_id': ext_df[EXT_SAMPLE_COL].values if EXT_SAMPLE_COL in ext_df.columns else range(len(y_ext)),
    'group': ext_df[EXT_GROUP_COL].values, 'y_true': y_ext, 'y_prob': ext_prob}
).to_csv(os.path.join(ext_output_dir,
                      f'{MODEL_SHORT}_predictions.csv'), index=False)

# 外部校准 + CI
print("\n[7b] 外部验证校准指标 + Bootstrap CI")
ext_cal = calibration_analysis(y_ext, ext_prob,
                               ext_output_dir, save_prefix='external_calibration')
ext_cal_boot = bootstrap_calibration_dca(y_ext, ext_prob,
                                         n_bootstraps=N_BOOTSTRAPS_CI,
                                         random_state=RANDOM_STATE)
ext_cal_report = {
    'Brier': ext_cal['Brier'],
    'Brier_CI': list(ext_cal_boot['Brier_CI']),
    'Calibration_Intercept': ext_cal['Calibration_Intercept'],
    'Calibration_Intercept_CI': list(ext_cal_boot['Calibration_Intercept_CI']),
    'Calibration_Slope': ext_cal['Calibration_Slope'],
    'Calibration_Slope_CI': list(ext_cal_boot['Calibration_Slope_CI']),
    'DCA_NetBenefit_CI': ext_cal_boot['DCA_CI'],
}
with open(os.path.join(ext_output_dir, 'external_calibration_report.json'),
          'w', encoding='utf-8') as f:
    json.dump(ext_cal_report, f, ensure_ascii=False, indent=2)
print(f"外部验证 Brier = {ext_cal['Brier']:.4f} (95% CI {ext_cal_boot['Brier_CI']})")
print(f"外部验证校准截距 = {ext_cal['Calibration_Intercept']:.4f} "
      f"(95% CI {ext_cal_boot['Calibration_Intercept_CI']})")
print(f"外部验证校准斜率 = {ext_cal['Calibration_Slope']:.4f} "
      f"(95% CI {ext_cal_boot['Calibration_Slope_CI']})")


# =============================================================================
# 15. 风险分层
# =============================================================================
print("\n[8] 数据驱动风险分层（非临床验证）")
risk_strat_train = risk_stratification_analysis(
    y_test.values, y_pred_prob, tertile_q1, tertile_q2,
    OUTPUT_DIR, 'internal_test')
risk_strat_ext = risk_stratification_analysis(
    y_ext, ext_prob, tertile_q1, tertile_q2,
    ext_output_dir, 'external')
print("\n内部测试集风险分层:")
print(risk_strat_train.to_string(index=False))
print("\n外部验证集风险分层:")
print(risk_strat_ext.to_string(index=False))
print("【注意】切点为训练集 OOF 三分位，数据驱动，未临床验证。")


# =============================================================================
# 16. 风险组校准 + Test vs External 校准 + DCA
# =============================================================================
print("\n[9] 风险组校准曲线（Train/Test/External）")
plot_risk_group_calibration(
    y_train.values, train_prob_in,
    y_test.values, y_pred_prob,
    y_ext, ext_prob,
    tertile_q1, tertile_q2,
    os.path.join(OUTPUT_DIR, 'risk_group_calibration.png'),
    model_name=MODEL_TAG)

print("\n[10] Test vs External 合并校准 + DCA")
plot_combined_calibration_test_external(
    y_test.values, y_pred_prob, y_ext, ext_prob,
    os.path.join(OUTPUT_DIR, 'calibration_test_vs_external.png'))
plot_combined_dca_test_external(
    y_test.values, y_pred_prob, y_ext, ext_prob,
    os.path.join(OUTPUT_DIR, 'dca_test_vs_external.png'),
    ext_label=f'{EXT_POS_LABEL} vs {EXT_NEG_LABEL}')


# =============================================================================
# 17. 插补敏感性 + 年龄性别匹配
# =============================================================================
print("\n[11] 插补敏感性分析")
imputation_sens_df = imputation_sensitivity_analysis(
    X_train_raw, y_train, X_test_raw, y_test,
    selected_features, best_params, OUTPUT_DIR)

print("\n[12] 年龄/性别匹配敏感性分析（探索性）")
age_sex_match_result = age_sex_matched_sensitivity_analysis(
    X_raw, y_all, age_all, sex_all,
    selected_features, best_params, OUTPUT_DIR)


# =============================================================================
# 18. SHAP + waterfall
# =============================================================================
print("\n[13] SHAP 分析")
try:
    explainer = shap.TreeExplainer(final_model)
    shap_values = explainer.shap_values(X_test_imp[selected_features])
    if isinstance(shap_values, list):
        shap_arr = np.array(shap_values[1] if len(shap_values) == 2 else shap_values[0])
    else:
        shap_arr = np.array(shap_values)
        if shap_arr.ndim == 3:
            shap_arr = shap_arr[:, :, 1] if shap_arr.shape[2] == 2 else shap_arr[:, :, 0]
    original_feature_names = [name_mapping.get(f, f) for f in selected_features]

    plt.figure()
    shap.summary_plot(shap_arr, X_test_imp[selected_features],
                      feature_names=original_feature_names, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'shap_summary_beeswarm.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure()
    shap.summary_plot(shap_arr, X_test_imp[selected_features],
                      feature_names=original_feature_names,
                      plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'shap_summary_bar.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    mean_abs_shap = np.abs(shap_arr).mean(axis=0)
    pd.DataFrame({'Feature_Cleaned': selected_features,
                  'Feature_Original': original_feature_names,
                  'Mean_Abs_SHAP': mean_abs_shap}
                 ).sort_values('Mean_Abs_SHAP', ascending=False).to_csv(
        os.path.join(OUTPUT_DIR, 'shap_importance.csv'),
        index=False, encoding='utf-8-sig')

    X_test_sel = X_test_imp[selected_features]
    pos_idx = np.where(y_test.values == 1)[0]
    sample_idx = int(pos_idx[0]) if len(pos_idx) > 0 else 0
    sample_df = X_test_sel.iloc[[sample_idx]]

    try:
        exp = explainer(sample_df)
        values = np.array(exp.values)
        base_vals = np.array(exp.base_values)
        if values.ndim == 3:
            values = values[0, :, 1] if values.shape[2] == 2 else values[0, :, 0]
        elif values.ndim == 2:
            values = values[0, :]
        if base_vals.ndim == 2:
            base_val = float(base_vals[0, 1] if base_vals.shape[1] == 2 else base_vals[0, 0])
        elif base_vals.ndim == 1:
            base_val = float(base_vals[1] if base_vals.shape[0] == 2 else base_vals[0])
        else:
            base_val = float(base_vals)
    except Exception:
        sv = explainer.shap_values(sample_df)
        if isinstance(sv, list):
            values = np.array(sv[1] if len(sv) == 2 else sv[0])
        else:
            values = np.array(sv)
            if values.ndim == 3:
                values = values[0, :, 1] if values.shape[2] == 2 else values[0, :, 0]
            elif values.ndim == 2:
                values = values[0, :]
        ev = explainer.expected_value
        if isinstance(ev, (list, np.ndarray)):
            base_val = float(ev[1] if len(ev) == 2 else ev[0])
        else:
            base_val = float(ev)

    data_flat = np.array(sample_df.values).flatten()
    if len(data_flat) != len(values):
        data_flat = data_flat[:len(values)]

    plt.figure(figsize=(13, 7))
    shap.waterfall_plot(
        shap.Explanation(values=values, base_values=base_val,
                         data=data_flat, feature_names=original_feature_names),
        show=False, max_display=15)
    plt.title(f'SHAP Waterfall for Patient {sample_idx} (True=1)', fontsize=15)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'shap_waterfall_patient0.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print("SHAP 全局图 + waterfall 已保存")
except Exception as e:
    import traceback; traceback.print_exc()
    print(f"SHAP 分析失败: {e}")


# =============================================================================
# 19. 方法学说明
# =============================================================================
methodology_notes = {
    "model": MODEL_TAG,
    "nested_cv_scope": (
        "Nested CV was performed ONLY within the training set to generate OOF. "
        "Test-set labels were never used for threshold or tertile selection."),
    "threshold_locking": (
        "Optimal threshold (Youden) and tertile cut-points (33.3/66.7 percentiles) "
        "were locked using training-set OOF predictions only."),
    "test_set_usage": (
        "Test set was used exactly once for final evaluation, using the "
        "training-set locked threshold and cut-points."),
    "external_validation": (
        "External validation set used the training-set locked threshold and "
        "tertile cut-points; no threshold re-optimization was performed."),
    "risk_stratification_caveat": (
        "Risk strata are data-driven (from OOF tertiles) and have NOT been "
        "validated against independent clinical outcomes. Do not describe them "
        "as 'clinical risk stratification'."),
    "calibration_reporting": (
        "Primary calibration results are based on the test set and external set. "
        "Training-set calibration is reported for reference only."),
    "age_sex_matched_analysis": (
        "Age/sex-matched analysis is exploratory. Uses pre-selected features and "
        "pre-tuned hyperparameters; no nested FS/tuning inside matched subset."),
    "delong_test": (
        "DeLong tests were performed on both training-set OOF and the test set. "
        "Test-set DeLong is underpowered; non-significant p should NOT be "
        "interpreted as equivalence."),
}
with open(os.path.join(OUTPUT_DIR, 'methodology_notes.json'),
          'w', encoding='utf-8') as f:
    json.dump(methodology_notes, f, ensure_ascii=False, indent=2)


print("\n" + "=" * 70)
print(f"所有分析完成 ({MODEL_TAG})，结果已保存至:", OUTPUT_DIR)
if nested_cv_results is not None:
    print(f"训练集嵌套 CV OOF AUC = {nested_cv_results['overall_auc']:.4f}")
if best_threshold is not None:
    print(f"训练集锁定阈值 = {best_threshold:.4f}")
    print(f"训练集锁定三分位切点 q1 = {tertile_q1:.4f}, q2 = {tertile_q2:.4f}")
print("=" * 70)