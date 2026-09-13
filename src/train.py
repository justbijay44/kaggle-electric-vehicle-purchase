import os
import joblib
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold

from lightgbm import LGBMClassifier

from src.data import load_data
from src.features import build_preprocessor, compute_combo_map, apply_combo_map

MODEL_PARAMS = dict(
    random_state=42, max_bin=511, n_estimators=500, learning_rate=0.05,
    num_leaves=63, max_depth=5, min_child_samples=50, verbose=-1
)

def build_pipeline(y):
    preprocessor = build_preprocessor()
    full_preprocessor = ColumnTransformer(
        transformers=preprocessor.transformers + [('te', 'passthrough', ['Combo_te'])]
    )
    scale_pos_weight = (y == 0).sum() / (y == 1).sum()
    return Pipeline(steps=[
        ('preprocessor', full_preprocessor),
        ('model', LGBMClassifier(scale_pos_weight=scale_pos_weight, **MODEL_PARAMS))
    ])

def run_cv(X, y, n_splits=5, seed=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_preds = np.zeros(len(X))
    fold_aucs = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr_raw, X_va_raw = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]

        combo_map = compute_combo_map(X_tr_raw, y_tr)
        X_tr = apply_combo_map(X_tr_raw, combo_map)
        X_va = apply_combo_map(X_va_raw, combo_map)

        pipe = build_pipeline(y_tr)
        pipe.fit(X_tr, y_tr)

        fold_pred = pipe.predict_proba(X_va)[:, 1]
        oof_preds[val_idx] = fold_pred
        fold_auc = roc_auc_score(y_va, fold_pred)
        fold_aucs.append(fold_auc)
        print(f"Fold {fold}: AUC = {fold_auc:.5f}")

    oof_auc = roc_auc_score(y, oof_preds)
    print(f"\nMean fold AUC: {np.mean(fold_aucs):.5f}")
    print(f"Pooled OOF AUC: {oof_auc:.5f}")
    return oof_auc

def train_final_model(X, y):
    combo_map = compute_combo_map(X, y)
    X_encoded = apply_combo_map(X, combo_map)

    pipe = build_pipeline(y)
    pipe.fit(X_encoded, y)
    return pipe, combo_map

def main():
    train_df, _, _ = load_data()
    X = train_df.drop(columns=['id', 'Will_Buy_EV'])
    y = train_df['Will_Buy_EV'].map({'No': 0, 'Yes': 1})

    run_cv(X, y)

    model, combo_map = train_final_model(X, y)
    os.makedirs('models', exist_ok=True)
    joblib.dump({'pipeline': model, 'combo_map': combo_map}, 'models/model.pkl')
    print("Saved model to models/model.pkl")

if __name__ == '__main__':
    main()