"""Fábricas de modelos (cada função devolve um estimador novo, sem estado)."""
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import config as C


def lr(C_=0.5, balanced=True):
    return lambda: make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
                                 LogisticRegression(C=C_, class_weight='balanced' if balanced else None,
                                                    max_iter=3000))


def hgb(depth=3, lr_=0.05, iters=200, leaf=30, balanced=True):
    return lambda: HistGradientBoostingClassifier(
        max_depth=depth, learning_rate=lr_, max_iter=iters, min_samples_leaf=leaf,
        l2_regularization=1.0, class_weight='balanced' if balanced else None, random_state=C.SEED)


def rf(n=180, leaf=12, max_features=0.65):
    return lambda: make_pipeline(
        SimpleImputer(strategy='median'),
        RandomForestClassifier(
            n_estimators=n,
            min_samples_leaf=leaf,
            max_features=max_features,
            random_state=C.SEED,
            n_jobs=-1,
        ),
    )


def xgb(depth=3, n=300, lr_=0.03, spw=1.0):
    from xgboost import XGBClassifier
    return lambda: XGBClassifier(max_depth=depth, n_estimators=n, learning_rate=lr_, subsample=0.8,
                                 colsample_bytree=0.8, min_child_weight=5, reg_lambda=2.0,
                                 scale_pos_weight=spw, eval_metric='logloss', random_state=C.SEED,
                                 n_jobs=4, verbosity=0)
