import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, List, Dict
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.base import BaseEstimator
from skopt import BayesSearchCV
from skopt.space import Integer, Real, Categorical
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# =========================================================
# 0) 数据加载
# =========================================================
def load_data_from_excel(file_path: str, well_col: str) -> Tuple[pd.DataFrame, pd.Series, str]:
    """
    Load dataset from Excel file:
      - First row: parameter names
      - First column: well name (excluded from modeling)
      - Last column: target variable
      - Middle columns: features
    """
    df = pd.read_excel(file_path)
    assert df.columns[0] == well_col, f"首列应为井名列：{well_col}"
    target_col = df.columns[-1]
    feature_cols = df.columns[1:-1]  # 去掉井名列、目标列
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    return X, y, target_col

# =========================================================
# 1) Custom Variance-Adaptive Random Forest (with bootstrap)
# =========================================================
def _compute_best_split_numpy(X: np.ndarray, y: np.ndarray, cand_feats: np.ndarray, min_samples_split: int):
    best_feature, best_threshold, best_mse = -1, 0.0, np.inf
    n_samples = len(y)
    total_sum = np.sum(y)
    total_sum_sq = np.sum(y ** 2)

    for feature in cand_feats:
        values = X[:, feature]
        order = np.argsort(values)
        sv = values[order]
        sy = y[order]
        if sv[0] == sv[-1]:
            continue
        cumsum_y = np.cumsum(sy)
        cumsum_y_sq = np.cumsum(sy ** 2)
        for i in range(1, n_samples):
            if sv[i - 1] == sv[i]:
                continue
            left_size = i
            right_size = n_samples - i
            if left_size < min_samples_split or right_size < min_samples_split:
                continue
            left_sum = cumsum_y[i - 1]
            left_sum_sq = cumsum_y_sq[i - 1]
            right_sum = total_sum - left_sum
            right_sum_sq = total_sum_sq - left_sum_sq
            left_var = (left_sum_sq - (left_sum ** 2) / left_size) if left_size > 0 else 0.0
            right_var = (right_sum_sq - (right_sum ** 2) / right_size) if right_size > 0 else 0.0
            mse = (left_var + right_var) / n_samples
            if mse < best_mse:
                best_mse = mse
                best_feature = feature
                best_threshold = (sv[i - 1] + sv[i]) / 2.0
    return best_feature, best_threshold

class _VA_RFTree:
    def __init__(self, max_depth=7, min_samples_split=2, max_features="sqrt", k=25, c=0.5, rng=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.k = k
        self.c = c
        self.rng = np.random.RandomState(None) if rng is None else rng
        self.n_features_ = None
        self.global_var_ = None
        self.tree_ = None

    """
    A single Variance-Adaptive Random Forest tree.
    - Uses dynamic probability p(norm_var) to decide: best split vs random split
    - High variance nodes -> best split
    - Low variance nodes -> random split
    """


    def _node_var_norm(self, y):
        lv = np.var(y)
        if self.global_var_ is None or self.global_var_ == 0:
            return 0.0
        return min(lv / self.global_var_, 1.0)

    def _p_dynamic(self, norm_var):
        return 1.0 / (1.0 + np.exp(-self.k * (norm_var - self.c)))

    def _pick_features(self):
        if self.max_features == "sqrt":
            n = max(1, int(np.sqrt(self.n_features_)))
        elif isinstance(self.max_features, float):
            n = max(1, int(self.max_features * self.n_features_))
        elif isinstance(self.max_features, int):
            n = max(1, min(self.max_features, self.n_features_))
        else:
            n = self.n_features_
        return self.rng.choice(self.n_features_, size=n, replace=False)

    def _best_split(self, X, y, cand_feats):
        return _compute_best_split_numpy(X, y, cand_feats, self.min_samples_split)

    def _random_split(self, X, y, cand_feats):
        f = self.rng.choice(cand_feats)
        vals = X[:, f]
        lo, hi = np.min(vals), np.max(vals)
        if lo == hi:
            return None, None
        thr = self.rng.uniform(lo, hi)
        return f, thr

    def fit(self, X, y):
        X = np.asarray(X); y = np.asarray(y)
        self.n_features_ = X.shape[1]
        self.global_var_ = np.var(y)
        self.tree_ = self._grow(X, y, depth=0)
        return self

    def _grow(self, X, y, depth):
        if (self.max_depth is not None and depth >= self.max_depth) or \
           (len(y) < self.min_samples_split) or np.all(y == y[0]):
            return {"leaf": True, "pred": float(np.mean(y)) if len(y) else 0.0}
        nv = self._node_var_norm(y)
        p = self._p_dynamic(nv)
        cand = self._pick_features()
        if self.rng.rand() < p:
            f, thr = self._best_split(X, y, cand)
        else:
            f, thr = self._random_split(X, y, cand)
        if f is None or thr is None:
            return {"leaf": True, "pred": float(np.mean(y)) if len(y) else 0.0}
        mask_left = X[:, f] <= thr
        mask_right = ~mask_left
        if not np.any(mask_left) or not np.any(mask_right):
            return {"leaf": True, "pred": float(np.mean(y)) if len(y) else 0.0}
        left = self._grow(X[mask_left], y[mask_left], depth + 1)
        right = self._grow(X[mask_right], y[mask_right], depth + 1)
        return {"leaf": False, "f": int(f), "thr": float(thr), "L": left, "R": right}

    def _predict_one(self, x, node):
        if node["leaf"]:
            return node["pred"]
        if x[node["f"]] <= node["thr"]:
            return self._predict_one(x, node["L"])
        else:
            return self._predict_one(x, node["R"])

    def predict(self, X):
        X = np.asarray(X)
        return np.array([self._predict_one(x, self.tree_) for x in X], dtype=float)

class VARandomForest(BaseEstimator):
    """
    Variance-Adaptive Random Forest (VA-RF) ensemble, sklearn-compatible:
    - Supports bootstrap sampling
    - Each tree is trained with variance-adaptive splitting strategy
    """
    def __init__(self, n_estimators=200, max_depth=7, min_samples_split=2,
                 max_features="sqrt", k=25, c=0.5, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.k = k
        self.c = c
        self.random_state = random_state
        self.rng_ = np.random.RandomState(self.random_state)
        self.trees_: List[_VA_RFTree] = []

    def fit(self, X, y):
        X = np.asarray(X); y = np.asarray(y)
        n = len(y)
        self.trees_ = []
        for _ in range(self.n_estimators):
            idx = self.rng_.choice(n, size=n, replace=True)  # bootstrap
            tree = _VA_RFTree(max_depth=self.max_depth,
                              min_samples_split=self.min_samples_split,
                              max_features=self.max_features,
                              k=self.k, c=self.c, rng=self.rng_)
            tree.fit(X[idx], y[idx])
            self.trees_.append(tree)
        return self

    def predict(self, X):
        X = np.asarray(X)
        preds = np.stack([t.predict(X) for t in self.trees_], axis=0)
        return preds.mean(axis=0)

# =========================
# （FPRF）
# =========================
class _FP_RFTree(_VA_RFTree):
    """Tree with constant probability p_const for split choice (ablation)."""
    def __init__(self, max_depth=7, min_samples_split=2, max_features="sqrt",
                 p_const=0.5, rng=None):
        super().__init__(max_depth=max_depth, min_samples_split=min_samples_split,
                         max_features=max_features, k=25, c=0.5, rng=rng)
        self.p_const = p_const
    def _p_dynamic(self, norm_var):
        return float(self.p_const)

class FPRandomForest(BaseEstimator):
    """Ensemble of fixed-probability trees (ablation of VA-RF)."""
    def __init__(self, n_estimators=200, max_depth=7, min_samples_split=2,
                 max_features="sqrt", p_const=0.5, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.p_const = p_const
        self.random_state = random_state
        self.rng_ = np.random.RandomState(self.random_state)
        self.trees_: List[_FP_RFTree] = []
    def fit(self, X, y):
        X = np.asarray(X); y = np.asarray(y)
        n = len(y)
        self.trees_ = []
        for _ in range(self.n_estimators):
            idx = self.rng_.choice(n, size=n, replace=True)
            tree = _FP_RFTree(max_depth=self.max_depth,
                              min_samples_split=self.min_samples_split,
                              max_features=self.max_features,
                              p_const=self.p_const,
                              rng=self.rng_)
            tree.fit(X[idx], y[idx])
            self.trees_.append(tree)
        return self
    def predict(self, X):
        X = np.asarray(X)
        preds = np.stack([t.predict(X) for t in self.trees_], axis=0)
        return preds.mean(axis=0)

# —— Utility: Tree strength & correlation —— #
def _predict_each_tree_rf(rf_model, X_imp: np.ndarray) -> np.ndarray:
    """For sklearn RF: return [n_trees, n_samples] predictions."""
    return np.stack([est.predict(X_imp) for est in rf_model.estimators_], axis=0)

def _predict_each_tree_varf(varf_model, X_imp: np.ndarray) -> np.ndarray:
    """For custom VA-RF: return [n_trees, n_samples] predictions."""
    return np.stack([t.predict(X_imp) for t in varf_model.trees_], axis=0)

def _predict_each_tree_fprf(fprf_model, X_imp: np.ndarray) -> np.ndarray:
    """For custom FPRF: return [n_trees, n_samples] predictions."""
    return np.stack([t.predict(X_imp) for t in fprf_model.trees_], axis=0)

def compute_strength_corr_for_fold(best_pipe, model_name: str,
                                   X_tr_raw, y_tr, X_te_raw, y_te):
    """
    Compute per-tree strength and average correlation for one CV fold:
      - Strength s_i = 1 - MSE_i / Var_train(Y)
      - Correlation ρ̄ = avg pairwise correlation of residuals
    Return dictionary of stats.
    """
    imputer = best_pipe.named_steps["imputer"]
    model   = best_pipe.named_steps["model"]
    X_te_imp = imputer.transform(X_te_raw)

    # 逐树预测矩阵
    if model_name == "随机森林":
        preds_each = _predict_each_tree_rf(model, X_te_imp)
    elif model_name == "方差自适应随机森林":
        preds_each = _predict_each_tree_varf(model, X_te_imp)
    elif model_name == "固定概率随机森林（消融）":
        preds_each = _predict_each_tree_fprf(model, X_te_imp)
    else:
        return None

    n_trees, n_samples = preds_each.shape

    # 单树强度（式 2）
    var_train = float(np.var(y_tr.values)) if len(y_tr) > 1 else 0.0
    if var_train == 0.0:
        strengths = np.full(n_trees, np.nan)
    else:
        mse_each = np.mean((preds_each - y_te.values.reshape(1, -1)) ** 2, axis=1)
        strengths = 1.0 - (mse_each / var_train)
        strengths = np.clip(strengths, 0, None)  # 将所有负值截断为 0，保留其余值

    # 树间残差相关性（式 5）
    residuals = y_te.values.reshape(1, -1) - preds_each
    corr_mat = np.corrcoef(residuals)
    if np.isscalar(corr_mat):
        rho_bar, valid_pairs = np.nan, 0
    else:
        mask = ~np.eye(n_trees, dtype=bool)
        vals = corr_mat[mask]
        valid = ~np.isnan(vals)
        rho_bar = float(np.nanmean(vals[valid])) if valid.any() else np.nan
        valid_pairs = int(valid.sum())

    return {
        "平均强度": float(np.nanmean(strengths)),
        "平均相关性": rho_bar,
        "树数": int(n_trees),
        "有效相关对数": int(valid_pairs)
    }

# =========================================================
# 2) Bayesian Optimization + Outer 10-fold Evaluation
# =========================================================
@dataclass
class CVResult:
    """Data structure to store CV results for each model."""
    模型名: str
    RMSE均值: float
    RMSE标准差: float
    MAE均值: float
    MAE标准差: float
    RMSE逐折: List[float]
    MAE逐折: List[float]

def bayes_spaces_for(model_key: str):
    """
    Define Bayesian search spaces for each model.
    Model key: 'RF', 'VARF', 'FPRF'
    """
    if model_key == "RF":
        return {
            "model__n_estimators": Integer(50, 200),
            "model__max_depth": Integer(3, 12),
            "model__min_samples_split": Integer(2, 10),
            "model__max_features": Categorical([0.6, 0.7]),
        }

    elif model_key == "VARF":
        return {
            "model__n_estimators": Integer(50, 200),
            "model__max_depth": Integer(3, 12),
            "model__min_samples_split": Integer(2, 10),
            "model__max_features": Categorical([0.6, 0.7]),
            "model__k": Integer(1, 20),
            #"model__c": Real(0.2, 0.4)  # ← 新增
        }

    elif model_key == "FPRF":  # ← 新增：固定概率随机森林（消融）
        return {
            "model__n_estimators": Integer(50, 200),
            "model__max_depth": Integer(3, 12),
            "model__min_samples_split": Integer(2, 10),
            "model__max_features": Categorical([0.6, 0.7]),
            "model__p_const": Real(0.49, 0.51)
        }
    else:
        raise ValueError("未知模型 key")

def build_model_and_key(name: str):
    """
    Return (model_key, estimator) based on model name.
    Supported:
      - RandomForestRegressor
      - VARandomForest
      - FPRandomForest
    """
    if name == "随机森林":
        return "RF", RandomForestRegressor(bootstrap=True, random_state=42, n_jobs=-1)
    elif name == "方差自适应随机森林":
        return "VARF", VARandomForest(c=0.50, random_state=42)
    elif name == "固定概率随机森林（消融）":
        return "FPRF", FPRandomForest(random_state=42)
    else:
        raise ValueError("未知模型名")


def evaluate_with_bayes(X: pd.DataFrame, y: pd.Series,
                        n_splits=10, random_state=42,
                        n_iter=5, inner_cv=2):
    """
    Nested CV with Bayesian Optimization:
      - Outer CV: KFold (n_splits)
      - Inner CV: BayesSearchCV (hyperparameter tuning)
      - Models: RF, FPRF (ablation), VARF
    Outputs:
      - CV results summary
      - Best params per fold
      - Theory validation (strength & correlation)
      - Per-fold errors
    """

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    model_names = ["随机森林", "固定概率随机森林（消融）", "方差自适应随机森林"]

    results: List[CVResult] = []
    best_params_rows = []
    theory_rows = []
    fold_errors_rows = []  

    for model_name in model_names:
        key, base_est = build_model_and_key(model_name)
        space = bayes_spaces_for(key)

        rmse_list, mae_list = [], []

        for fold_id, (tr_idx, te_idx) in enumerate(kf.split(X), start=1):
            X_tr_raw, X_te_raw = X.iloc[tr_idx].copy(), X.iloc[te_idx].copy()
            y_tr, y_te = y.iloc[tr_idx].copy(), y.iloc[te_idx].copy()

            pipe = Pipeline(steps=[
                ("imputer", KNNImputer(n_neighbors=3, weights="distance")),
                ("model", base_est)
            ])

            opt = BayesSearchCV(
                estimator=pipe,
                search_spaces=space,
                n_iter=n_iter,
                cv=inner_cv,
                scoring="neg_root_mean_squared_error",
                n_jobs=-1,
                random_state=42,
                refit=True,
                verbose=0
            )
            opt.fit(X_tr_raw, y_tr)

            
            best_params = opt.best_params_
            clean_params = {k.replace("model__", ""): v for k, v in best_params.items() if k.startswith("model__")}
            clean_params["折号"] = fold_id
            clean_params["模型"] = model_name
            best_params_rows.append(clean_params)

            
            y_pred = opt.predict(X_te_raw)
            rmse = float(np.sqrt(mean_squared_error(y_te.values, y_pred)))
            mae  = float(mean_absolute_error(y_te.values, y_pred))
            rmse_list.append(rmse)
            mae_list.append(mae)

            
            fold_errors_rows.append({
                "模型": model_name,
                "折号": fold_id,
                "RMSE": rmse,
                "MAE": mae
            })

            
            if model_name in ["随机森林", "方差自适应随机森林"]:
                stat = compute_strength_corr_for_fold(opt.best_estimator_, model_name,
                                                      X_tr_raw, y_tr, X_te_raw, y_te)
                if stat is not None:
                    theory_rows.append({
                        "模型": model_name,
                        "折号": fold_id,
                        "平均相关性": stat["平均相关性"],
                        "平均强度": stat["平均强度"],
                        "树数": stat["树数"],
                        "有效相关对数": stat["有效相关对数"]
                    })

        results.append(
            CVResult(
                模型名=model_name,
                RMSE均值=float(np.mean(rmse_list)),
                RMSE标准差=float(np.std(rmse_list, ddof=1)),
                MAE均值=float(np.mean(mae_list)),
                MAE标准差=float(np.std(mae_list, ddof=1)),
                RMSE逐折=rmse_list,
                MAE逐折=mae_list
            )
        )

    best_params_df = pd.DataFrame(best_params_rows).sort_values(["模型", "折号"]).reset_index(drop=True)
    theory_df = pd.DataFrame(theory_rows).sort_values(["模型", "折号"]).reset_index(drop=True)
    fold_errors_df = pd.DataFrame(fold_errors_rows).sort_values(["模型", "折号"]).reset_index(drop=True)


    all_strengths = theory_df["平均强度"].dropna().values
    all_corrs = theory_df["平均相关性"].dropna().values

    if len(all_strengths) > 1:
        min_s, max_s = np.min(all_strengths), np.max(all_strengths)
        if max_s > min_s:
            theory_df["归一化平均强度"] = ( theory_df["平均强度"] - min_s ) / (max_s - min_s)
        else:
            theory_df["归一化平均强度"] = theory_df["平均强度"]  # 无差异时保持原值
    else:
        theory_df["归一化平均强度"] = theory_df["平均强度"]

    if len(all_corrs) > 1:
        min_c, max_c = np.min(all_corrs), np.max(all_corrs)
        if max_c > min_c:
            theory_df["归一化平均相关性"] = ( theory_df["平均相关性"] - min_c ) / (max_c - min_c)
        else:
            theory_df["归一化平均相关性"] = theory_df["平均相关性"]  # 无差异时保持原值
    else:
        theory_df["归一化平均相关性"] = theory_df["平均相关性"]

    return results, best_params_df, theory_df, fold_errors_df

# =========================================================
# 3) Main: Run comparison of 3 models with 10-fold CV
# =========================================================
if __name__ == "__main__":
    # ==============================
    # Paths & basic settings / 路径与基础设置
    # ==============================
    data_path = r"D:\Desktop\小论文\修改重投\数据\Github示例数据.xlsx"
    well_col  = "井名"

    # ==============================
    # Load data / 读取数据
    # ==============================
    X_all, y_all, target_col = load_data_from_excel(data_path, well_col)

    # ==============================
    # Run nested CV + Bayesian opt / 外层交叉验证 + 折内贝叶斯优化
    # ==============================
    results, best_params_df, theory_df, fold_errors_df = evaluate_with_bayes(
        X_all, y_all, n_splits=10, random_state=1, n_iter=5, inner_cv=2
    )

    # ==============================
    # (1) Theory validation summary / 理论数据验证（汇总，均值±标准差）
    # ==============================
    print("\n=== Theory Validation Summary (mean ± std) ===")
    print("=== 理论数据验证（汇总，均值±标准差） ===")

    if not theory_df.empty:
        theory_sum = theory_df.groupby("模型")[["归一化平均相关性", "归一化平均强度"]].agg(["mean", "std"]).reset_index()
        for name in ["方差自适应随机森林", "随机森林"]:
            row = theory_sum[theory_sum["模型"] == name]
            if not row.empty:
                rho_mean = float(row[("归一化平均相关性", "mean")].values[0])
                rho_std  = float(row[("归一化平均相关性", "std")].values[0])
                s_mean   = float(row[("归一化平均强度", "mean")].values[0])
                s_std    = float(row[("归一化平均强度", "std")].values[0])

                # English mapping
                eng_name = {
                    "随机森林": "Random Forest",
                    "方差自适应随机森林": "Variance-Adaptive RF"
                }[name]

                print(f"{name} ({eng_name}): ρ̄={rho_mean:.3f}±{rho_std:.3f} , s̄={s_mean:.3f}±{s_std:.3f}")

    # ==============================
    # (2) Overall 10-fold CV results / 10 折交叉验证 + 折内贝叶斯优化结果
    # ==============================
    print("\n=== 10-fold CV + Inner Bayesian Optimization (Geological features) ===")
    print("=== 10 折交叉验证 + 折内贝叶斯优化（地质参数组） ===")

    # Define bilingual names
    order_eval = [
        ("随机森林", "Random Forest"),
        ("固定概率随机森林（消融）", "Fixed-Probability RF"),
        ("方差自适应随机森林", "Variance-Adaptive RF")
    ]

    res_map = {r.模型名: r for r in results}
    for zh_name, en_name in order_eval:
        if zh_name in res_map:
            r = res_map[zh_name]
            model_name = f"{zh_name} ({en_name})"
            print(f"{model_name.ljust(28)}  RMSE: {r.RMSE均值:.4f} ± {r.RMSE标准差:.4f}   "
                  f"MAE: {r.MAE均值:.4f} ± {r.MAE标准差:.4f}")

