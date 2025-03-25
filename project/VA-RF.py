import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from sklearn.base import BaseEstimator
import warnings
from numba import jit

warnings.filterwarnings("ignore", category=FutureWarning)

# -------------------------
# 1. 数据加载函数
# -------------------------
def load_data_from_excel(file_path, well_col, target_col):
    """
    从 Excel 文件加载数据：
      - 第一行为参数名称；
      - 第一列为井名称，不参与预处理；
      - 最后一列为目标变量，其余列为特征；
    """
    df = pd.read_excel(file_path)
    df = df.dropna()  # 删除含有缺失值的行
    feature_cols = df.columns[1:-1]
    X = df[feature_cols]
    y = df[target_col]
    return X, y

# -------------------------
def mean_absolute_percentage_error(y_true, y_pred):
    """
    计算平均绝对百分比误差 (MAPE)
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # 避免除以 0 的情况，添加掩码
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

@jit(nopython=True)
def compute_best_split(X, y, candidate_features, min_samples_split):
    best_feature, best_threshold, best_mse = -1, 0.0, np.inf
    n_samples = len(y)
    total_sum = np.sum(y)
    total_sum_sq = np.sum(y ** 2)

    for feature in candidate_features:
        values = X[:, feature]
        sorted_indices = np.argsort(values)
        sorted_values = values[sorted_indices]
        sorted_y = y[sorted_indices]

        if sorted_values[0] == sorted_values[-1]:
            continue

        cumsum_y = np.cumsum(sorted_y)
        cumsum_y_sq = np.cumsum(sorted_y ** 2)

        for i in range(1, n_samples):
            if sorted_values[i - 1] == sorted_values[i]:
                continue

            left_size = i
            right_size = n_samples - i
            if left_size < min_samples_split or right_size < min_samples_split:
                continue

            left_sum = cumsum_y[i - 1]
            left_sum_sq = cumsum_y_sq[i - 1]
            right_sum = total_sum - left_sum
            right_sum_sq = total_sum_sq - left_sum_sq

            left_var = (left_sum_sq - (left_sum ** 2) / left_size) if left_size > 0 else 0
            right_var = (right_sum_sq - (right_sum ** 2) / right_size) if right_size > 0 else 0

            mse = (left_var + right_var) / n_samples

            if mse < best_mse:
                best_mse = mse
                best_feature = feature
                best_threshold = (sorted_values[i - 1] + sorted_values[i]) / 2

    return best_feature, best_threshold

# -------------------------
# 2. 模型定义
# -------------------------

class PARFTreeRegressor:
    def __init__(self, max_depth=5, min_samples_split=2, max_features="sqrt"):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.tree_ = None
        self.n_features_ = None
        self.global_var_ = None

    def fit(self, X, y):
        X, y = np.array(X), np.array(y)
        self.n_features_ = X.shape[1]
        self.global_var_ = np.var(y)
        self.tree_ = self._build_tree(X, y, depth=0)
        return self

    def predict(self, X):
        X = np.array(X)
        preds = [self._predict_one(sample, self.tree_) for sample in X]
        return np.array(preds)

    def _kernel_var(self, y):
        """
        计算目标变量 y 的局部方差，并归一化到 [0, 1]。
        """
        local_var = np.var(y)
        if self.global_var_ == 0:
            return 0.0
        norm_var = local_var / self.global_var_
        return min(norm_var, 1.0)

    def _get_candidate_features(self):
        if self.max_features == "sqrt":
            n_features = int(np.sqrt(self.n_features_))
        elif isinstance(self.max_features, float):
            n_features = int(self.max_features * self.n_features_)
        elif isinstance(self.max_features, int):
            n_features = self.max_features
        else:
            n_features = self.n_features_
        return np.random.choice(self.n_features_, size=max(1, min(n_features, self.n_features_)), replace=False)

    def _best_split(self, X, y, candidate_features):
        feature, threshold = compute_best_split(X, y, candidate_features, self.min_samples_split)
        return feature, threshold

    def _random_split(self, X, y, candidate_features):
        feature = np.random.choice(candidate_features)
        values = X[:, feature]
        min_val, max_val = np.min(values), np.max(values)
        if min_val == max_val:
            return None, None
        threshold = np.random.uniform(min_val, max_val)
        return feature, threshold

    def _build_tree(self, X, y, depth):
        if (self.max_depth is not None and depth >= self.max_depth) or (len(y) < self.min_samples_split) or np.all(
                y == y[0]):
            return {'is_leaf': True, 'prediction': np.mean(y) if len(y) > 0 else 0.0}
        var_K = self._kernel_var(y)
        norm_var_K = var_K
        p_dynamic = self.get_p_dynamic(norm_var_K)
        candidate_features = self._get_candidate_features()
        if np.random.rand() < p_dynamic:
            feature, threshold = self._best_split(X, y, candidate_features)
        else:
            feature, threshold = self._random_split(X, y, candidate_features)
        if feature is None or threshold is None:
            return {'is_leaf': True, 'prediction': np.mean(y) if len(y) > 0 else 0.0}
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask
        if not np.any(left_mask) or not np.any(right_mask):
            return {'is_leaf': True, 'prediction': np.mean(y) if len(y) > 0 else 0.0}
        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        return {'is_leaf': False, 'feature': feature, 'threshold': threshold,
                'left': left_child, 'right': right_child}

    def _predict_one(self, sample, node):
        if node['is_leaf']:
            return node['prediction']
        if sample[node['feature']] <= node['threshold']:
            return self._predict_one(sample, node['left'])
        else:
            return self._predict_one(sample, node['right'])



class PurityAdaptivePARFTreeRegressor(PARFTreeRegressor):
    def __init__(self, max_depth=11, min_samples_split=2, k=10,
                 max_features="sqrt"):
        super().__init__(max_depth, min_samples_split, max_features)
        self.k = k
        self.c = 0.5

    def get_p_dynamic(self, norm_var_K):
        p_dynamic = 1 / (1 + np.exp(-self.k * (norm_var_K - self.c)))
        return p_dynamic


class PurityAdaptivePARFEnsembleRegressor(BaseEstimator):
    def __init__(self, n_estimators=120, max_depth=14, min_samples_split=2,
                 k=10, max_features="sqrt", random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.k = k
        self.max_features = max_features
        self.random_state = random_state
        if random_state is not None:
            np.random.seed(random_state)

    def fit(self, X, y):
        X, y = np.array(X), np.array(y)
        self.trees = []
        for _ in range(self.n_estimators):
            tree = PurityAdaptivePARFTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                k=self.k,
                max_features=self.max_features
            )
            tree.fit(X, y)
            self.trees.append(tree)
        return self

    def predict(self, X):
        X = np.array(X)
        all_preds = [tree.predict(X) for tree in self.trees]
        return np.mean(all_preds, axis=0)


def print_iteration_callback(res):
    current_iter = len(res.func_vals)
    print(f"当前贝叶斯迭代次数: {current_iter}")

if __name__ == "__main__":

    train_file = r"D:\Desktop\小论文\小论文使用数据\基于随机森林的页岩油产量预测\平均日产\一起划分数据\处理后数据集\9训练数据.xlsx"
    test_file = r"D:\Desktop\小论文\小论文使用数据\基于随机森林的页岩油产量预测\平均日产\一起划分数据\处理后数据集\9测试数据.xlsx"
    well_col = "井名称"
    target_col = "平均日产"


    X_train, y_train = load_data_from_excel(train_file, well_col, target_col)
    X_test, y_test = load_data_from_excel(test_file, well_col, target_col)


    search_space = {
        'n_estimators': Integer(20, 200),
        'max_depth': Integer(3, 9),
        'min_samples_split': Integer(2, 10),
        'k': Integer(10, 50),
        'max_features': Real(0.3, 0.7, 'uniform')
    }


    opt = BayesSearchCV(
        estimator=PurityAdaptivePARFEnsembleRegressor(random_state=42),
        search_spaces=search_space,
        n_iter=50,
        cv=10,
        scoring='neg_mean_squared_error',
        random_state=42,
        verbose=0,
        return_train_score=True
    )

    print("正在对训练集进行贝叶斯优化，进行10折交叉验证...")
    opt.fit(X_train, y_train, callback=[print_iteration_callback])
    best_params = opt.best_params_
    print("最佳超参数：", best_params)

    # 使用最佳超参数，在整个训练集上训练最终模型
    best_model = opt.best_estimator_
    best_model.fit(X_train.values, y_train.values)

    # 计算训练集的预测和误差指标
    train_preds = best_model.predict(X_train.values)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
    train_mae = mean_absolute_error(y_train, train_preds)
    train_medae = np.median(np.abs(y_train - train_preds))
    train_mape = mean_absolute_percentage_error(y_train, train_preds)

    # 从交叉验证中提取验证集的误差指标
    best_index = opt.best_index_
    val_rmse_scores = []
    val_mae_scores = []
    val_medae_scores = []
    val_mape_scores = []
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    for train_idx, val_idx in kf.split(X_train):
        X_train_cv, X_val_cv = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train_cv, y_val_cv = y_train.iloc[train_idx], y_train.iloc[val_idx]
        best_model.fit(X_train_cv.values, y_train_cv.values)
        val_preds_cv = best_model.predict(X_val_cv.values)
        val_rmse_scores.append(np.sqrt(mean_squared_error(y_val_cv, val_preds_cv)))
        val_mae_scores.append(mean_absolute_error(y_val_cv, val_preds_cv))
        val_medae_scores.append(np.median(np.abs(y_val_cv - val_preds_cv)))
        val_mape_scores.append(mean_absolute_percentage_error(y_val_cv, val_preds_cv))
    val_rmse = np.mean(val_rmse_scores)
    val_mae = np.mean(val_mae_scores)
    val_medae = np.mean(val_medae_scores)
    val_mape = np.mean(val_mape_scores)

    # 计算测试集的预测和误差指标
    test_preds = best_model.predict(X_test.values)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
    test_mae = mean_absolute_error(y_test, test_preds)
    test_medae = np.median(np.abs(y_test - test_preds))
    test_mape = mean_absolute_percentage_error(y_test, test_preds)

    # 输出训练集的四个指标
    print("\n训练集误差指标（完整训练数据）：")
    print(f"训练集 RMSE: {train_rmse:.4f}")
    print(f"训练集 MAE: {train_mae:.4f}")
    print(f"训练集 MedAE: {train_medae:.4f}")
    print(f"训练集 MAPE: {train_mape:.4f}%")

    # 输出验证集的四个指标
    print("\n验证集误差指标（10折交叉验证平均值）：")
    print(f"验证集 RMSE: {val_rmse:.4f}")
    print(f"验证集 MAE: {val_mae:.4f}")
    print(f"验证集 MedAE: {val_medae:.4f}")
    print(f"验证集 MAPE: {val_mape:.4f}%")

    # 输出测试集的四个指标
    print("\n测试集误差指标：")
    print(f"测试集 RMSE: {test_rmse:.4f}")
    print(f"测试集 MAE: {test_mae:.4f}")
    print(f"测试集 MedAE: {test_medae:.4f}")
    print(f"测试集 MAPE: {test_mape:.4f}%")