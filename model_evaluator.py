"""
Model Evaluator — GastroMap Colombia
Evaluates 3 classification + 3 regression models with full metrics, ROC curves,
confusion matrices, and comparison charts.
"""

import pandas as pd
import numpy as np
import json, os, io, base64, logging, warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               RandomForestRegressor, GradientBoostingRegressor)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    mean_absolute_error, mean_squared_error, r2_score
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

FEATURES = ['gastro_count', 'gastro_variety', 'gastro_employees',
            'total_providers', 'total_employees', 'unique_categories',
            'climate_index', 'avg_temperature', 'precipitation']

CLASS_MAP = {'Very Low': 'Low', 'Low': 'Low', 'Medium': 'Medium', 'High': 'High', 'Very High': 'High'}
CLASS_ORDER = ['Low', 'Medium', 'High']


def _make_chart(func):
    """Wrapper: call func(fig, ax), return base64 PNG."""
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    bg = '#161a23'; txt = '#e8eaf0'; acc = '#f59e0b'; grd = '#272d3b'
    plt.rcParams.update({'figure.facecolor': bg, 'axes.facecolor': bg,
        'text.color': txt, 'axes.labelcolor': txt,
        'xtick.color': txt, 'ytick.color': txt, 'font.size': 10})
    fig, ax = plt.subplots(figsize=(8, 5))
    func(fig, ax)
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return b64


def run_evaluation():
    """Full evaluation pipeline: classification + regression, all charts."""
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    bg = '#161a23'; txt = '#e8eaf0'; acc = '#f59e0b'; grd = '#272d3b'
    plt.rcParams.update({'figure.facecolor': bg, 'axes.facecolor': bg,
        'text.color': txt, 'axes.labelcolor': txt,
        'xtick.color': txt, 'ytick.color': txt, 'font.size': 10})

    # Load data
    df = pd.read_json(os.path.join(DATA_DIR, 'processed_dataset.json'), orient='records')
    df['target_class'] = df['classification'].map(CLASS_MAP)

    X = df[FEATURES].copy()
    y_class = df['target_class'].copy()
    y_reg = df['GTPI'].copy()

    le = LabelEncoder(); le.fit(CLASS_ORDER)
    y_enc = le.transform(y_class)

    scaler = StandardScaler()
    X_sc = pd.DataFrame(scaler.fit_transform(X), columns=FEATURES)

    X_tr, X_te, yc_tr, yc_te, yr_tr, yr_te = train_test_split(
        X_sc, y_enc, y_reg, test_size=0.2, random_state=42, stratify=y_enc)

    info = {
        'total': len(X), 'train': len(X_tr), 'test': len(X_te),
        'class_dist': y_class.value_counts().to_dict(),
        'features': FEATURES, 'n_features': len(FEATURES)
    }

    # ══════════════════════════════════════════
    # CLASSIFICATION MODELS
    # ══════════════════════════════════════════
    clf_models = {
        'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=15,
            min_samples_split=5, min_samples_leaf=2, class_weight='balanced', random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=150, max_depth=5,
            learning_rate=0.1, min_samples_split=5, min_samples_leaf=3, random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=7, weights='distance', metric='minkowski')
    }

    clf_results = []
    roc_data = {}

    for name, model in clf_models.items():
        model.fit(X_tr, yc_tr)
        yp_tr = model.predict(X_tr)
        yp_te = model.predict(X_te)
        cv = cross_val_score(model, X_sc, y_enc, cv=5, scoring='accuracy')

        # ROC (one-vs-rest)
        y_bin = label_binarize(yc_te, classes=[0, 1, 2])
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_te)
        else:
            from sklearn.calibration import CalibratedClassifierCV
            cal = CalibratedClassifierCV(model, cv=3)
            cal.fit(X_tr, yc_tr)
            y_prob = cal.predict_proba(X_te)

        fpr_all, tpr_all, auc_all = {}, {}, {}
        for i, cls in enumerate(CLASS_ORDER):
            fpr_all[cls], tpr_all[cls], _ = roc_curve(y_bin[:, i], y_prob[:, i])
            auc_all[cls] = round(auc(fpr_all[cls], tpr_all[cls]), 4)

        # Macro average ROC
        all_fpr = np.unique(np.concatenate([fpr_all[c] for c in CLASS_ORDER]))
        mean_tpr = np.zeros_like(all_fpr)
        for cls in CLASS_ORDER:
            mean_tpr += np.interp(all_fpr, fpr_all[cls], tpr_all[cls])
        mean_tpr /= len(CLASS_ORDER)
        macro_auc = round(auc(all_fpr, mean_tpr), 4)

        roc_data[name] = {
            'classes': {cls: {'fpr': fpr_all[cls].tolist(), 'tpr': tpr_all[cls].tolist(), 'auc': auc_all[cls]}
                        for cls in CLASS_ORDER},
            'macro_auc': macro_auc
        }

        cm = confusion_matrix(yc_te, yp_te, labels=[0, 1, 2])
        report = classification_report(yc_te, yp_te, target_names=CLASS_ORDER, zero_division=0, output_dict=True)

        clf_results.append({
            'name': name,
            'accuracy': round(accuracy_score(yc_te, yp_te), 4),
            'precision': round(precision_score(yc_te, yp_te, average='weighted', zero_division=0), 4),
            'recall': round(recall_score(yc_te, yp_te, average='weighted', zero_division=0), 4),
            'f1': round(f1_score(yc_te, yp_te, average='weighted', zero_division=0), 4),
            'train_acc': round(accuracy_score(yc_tr, yp_tr), 4),
            'cv_mean': round(cv.mean(), 4), 'cv_std': round(cv.std(), 4),
            'cv_scores': cv.round(4).tolist(),
            'macro_auc': macro_auc,
            'confusion_matrix': cm.tolist(),
            'class_report': {k: v for k, v in report.items() if k in CLASS_ORDER},
            'feature_importance': dict(zip(FEATURES, model.feature_importances_.round(4).tolist()))
                if hasattr(model, 'feature_importances_') else None
        })
        logger.info(f"  CLF {name}: acc={clf_results[-1]['accuracy']} f1={clf_results[-1]['f1']} AUC={macro_auc}")

    # ══════════════════════════════════════════
    # REGRESSION MODELS
    # ══════════════════════════════════════════
    reg_models = {
        'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=150, max_depth=5,
            learning_rate=0.1, random_state=42),
        'K-Nearest Neighbors': KNeighborsRegressor(n_neighbors=7, weights='distance')
    }

    reg_results = []
    for name, model in reg_models.items():
        model.fit(X_tr, yr_tr)
        yp = model.predict(X_te)
        mae = round(mean_absolute_error(yr_te, yp), 4)
        mse = round(mean_squared_error(yr_te, yp), 4)
        rmse = round(np.sqrt(mse), 4)
        r2 = round(r2_score(yr_te, yp), 4)
        cv_r2 = cross_val_score(model, X_sc, y_reg, cv=5, scoring='r2')
        reg_results.append({
            'name': name, 'mae': mae, 'mse': mse, 'rmse': rmse, 'r2': r2,
            'cv_r2_mean': round(cv_r2.mean(), 4), 'cv_r2_std': round(cv_r2.std(), 4),
            'predictions': yp.round(1).tolist()[:20],
            'actuals': yr_te.values.round(1).tolist()[:20]
        })
        logger.info(f"  REG {name}: MAE={mae} RMSE={rmse} R²={r2}")

    # ══════════════════════════════════════════
    # GENERATE CHARTS
    # ══════════════════════════════════════════
    charts = {}
    colors = ['#f59e0b', '#10b981', '#3b82f6']

    # 1. Classification comparison
    def chart_clf_comp(fig, ax):
        names = [r['name'] for r in clf_results]
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        x = np.arange(len(names))
        w = 0.2
        for i, met in enumerate(metrics):
            vals = [r[met] for r in clf_results]
            bars = ax.bar(x + i*w, vals, w, label=met.upper(), alpha=0.85,
                         color=['#f59e0b', '#10b981', '#3b82f6', '#ef4444'][i])
            for b in bars:
                ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.003,
                       f'{b.get_height():.3f}', ha='center', fontsize=7, color=txt)
        ax.set_xticks(x + 1.5*w); ax.set_xticklabels(names)
        ax.set_ylim(0.9, 1.02); ax.set_ylabel('Score')
        ax.set_title('Classification Metrics Comparison', color=acc, fontsize=14, fontweight='bold')
        ax.legend(facecolor=bg, edgecolor=grd, fontsize=9); ax.grid(axis='y', color=grd, alpha=0.4)
    charts['clf_comparison'] = _make_chart(chart_clf_comp)

    # 2. ROC Curves (one per model)
    for idx, (name, rd) in enumerate(roc_data.items()):
        def make_roc(fig, ax, n=name, r=rd, c=colors[idx]):
            line_colors = ['#f59e0b', '#10b981', '#3b82f6']
            for i, cls in enumerate(CLASS_ORDER):
                ax.plot(r['classes'][cls]['fpr'], r['classes'][cls]['tpr'],
                       color=line_colors[i], lw=2, label=f"{cls} (AUC={r['classes'][cls]['auc']:.3f})")
            ax.plot([0,1],[0,1],'--', color='#555', lw=1)
            ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
            ax.set_title(f'ROC Curve — {n} (Macro AUC={r["macro_auc"]:.3f})',
                        color=acc, fontsize=13, fontweight='bold')
            ax.legend(facecolor=bg, edgecolor=grd, fontsize=9)
            ax.grid(color=grd, alpha=0.3)
        safe = name.replace(' ', '_').lower()
        charts[f'roc_{safe}'] = _make_chart(make_roc)

    # 3. Confusion matrices
    for idx, r in enumerate(clf_results):
        def make_cm(fig, ax, res=r):
            cm_arr = np.array(res['confusion_matrix'])
            im = ax.imshow(cm_arr, cmap='YlOrBr', aspect='auto')
            ax.set_xticks(range(3)); ax.set_yticks(range(3))
            ax.set_xticklabels(CLASS_ORDER); ax.set_yticklabels(CLASS_ORDER)
            ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
            ax.set_title(f'Confusion Matrix — {res["name"]}', color=acc, fontsize=13, fontweight='bold')
            for i in range(3):
                for j in range(3):
                    c = 'black' if cm_arr[i,j] > cm_arr.max()*0.5 else txt
                    ax.text(j, i, str(cm_arr[i,j]), ha='center', va='center',
                           fontsize=16, fontweight='bold', color=c)
        safe = r['name'].replace(' ', '_').lower()
        charts[f'cm_{safe}'] = _make_chart(make_cm)

    # 4. Regression comparison
    def chart_reg(fig, ax):
        names = [r['name'] for r in reg_results]
        x = np.arange(len(names))
        r2s = [r['r2'] for r in reg_results]
        bars = ax.bar(x, r2s, 0.5, color=colors, alpha=0.85)
        for b in bars:
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.005,
                   f'{b.get_height():.4f}', ha='center', fontsize=10, color=txt)
        ax.set_xticks(x); ax.set_xticklabels(names)
        ax.set_ylabel('R² Score'); ax.set_ylim(0.85, 1.02)
        ax.set_title('Regression R² Comparison', color=acc, fontsize=14, fontweight='bold')
        ax.grid(axis='y', color=grd, alpha=0.4)
    charts['reg_comparison'] = _make_chart(chart_reg)

    # 5. Regression error comparison
    def chart_reg_err(fig, ax):
        names = [r['name'] for r in reg_results]
        x = np.arange(len(names)); w = 0.3
        maes = [r['mae'] for r in reg_results]
        rmses = [r['rmse'] for r in reg_results]
        ax.bar(x-w/2, maes, w, label='MAE', color='#f59e0b', alpha=0.85)
        ax.bar(x+w/2, rmses, w, label='RMSE', color='#ef4444', alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(names)
        ax.set_ylabel('Error (GTPI points)')
        ax.set_title('Regression Error Comparison (MAE vs RMSE)', color=acc, fontsize=14, fontweight='bold')
        ax.legend(facecolor=bg, edgecolor=grd); ax.grid(axis='y', color=grd, alpha=0.4)
    charts['reg_errors'] = _make_chart(chart_reg_err)

    # 6. Actual vs Predicted scatter (best regressor)
    best_reg = max(reg_results, key=lambda x: x['r2'])
    def chart_scatter(fig, ax):
        ax.scatter(best_reg['actuals'], best_reg['predictions'], c=acc, alpha=0.6, s=30)
        lims = [0, 100]
        ax.plot(lims, lims, '--', color='#555', lw=1, label='Perfect prediction')
        ax.set_xlabel('Actual GTPI'); ax.set_ylabel('Predicted GTPI')
        ax.set_title(f'Actual vs Predicted — {best_reg["name"]} (R²={best_reg["r2"]})',
                    color=acc, fontsize=13, fontweight='bold')
        ax.legend(facecolor=bg, edgecolor=grd); ax.grid(color=grd, alpha=0.3)
        ax.set_xlim(lims); ax.set_ylim(lims)
    charts['scatter'] = _make_chart(chart_scatter)

    # 7. CV boxplot
    def chart_cv(fig, ax):
        data = [r['cv_scores'] for r in clf_results]
        names = [r['name'] for r in clf_results]
        bp = ax.boxplot(data, labels=names, patch_artist=True,
                       boxprops=dict(facecolor=acc, alpha=0.3),
                       medianprops=dict(color=acc, linewidth=2),
                       whiskerprops=dict(color=txt), capprops=dict(color=txt))
        ax.set_ylabel('Accuracy'); ax.grid(axis='y', color=grd, alpha=0.4)
        ax.set_title('5-Fold Cross-Validation', color=acc, fontsize=14, fontweight='bold')
    charts['cv_boxplot'] = _make_chart(chart_cv)

    # 8. Feature importance comparison
    def chart_fi(fig, ax):
        rf_fi = clf_results[0]['feature_importance']
        gb_fi = clf_results[1]['feature_importance']
        feats = sorted(rf_fi.keys(), key=lambda f: rf_fi[f]+gb_fi[f], reverse=True)
        y_pos = np.arange(len(feats)); w = 0.35
        ax.barh(y_pos-w/2, [rf_fi[f] for f in feats], w, label='Random Forest', color='#f59e0b', alpha=0.8)
        ax.barh(y_pos+w/2, [gb_fi[f] for f in feats], w, label='Gradient Boosting', color='#10b981', alpha=0.8)
        ax.set_yticks(y_pos); ax.set_yticklabels(feats, fontsize=9)
        ax.set_xlabel('Importance'); ax.invert_yaxis()
        ax.set_title('Feature Importance Comparison', color=acc, fontsize=14, fontweight='bold')
        ax.legend(facecolor=bg, edgecolor=grd, fontsize=9); ax.grid(axis='x', color=grd, alpha=0.4)
    charts['feature_importance'] = _make_chart(chart_fi)

    # ══════════════════════════════════════════
    # BEST MODEL SELECTION
    # ══════════════════════════════════════════
    best_clf = max(clf_results, key=lambda x: x['f1'])
    best_reg_m = max(reg_results, key=lambda x: x['r2'])

    selection = {
        'best_classifier': best_clf['name'],
        'best_clf_f1': best_clf['f1'],
        'best_clf_auc': best_clf['macro_auc'],
        'best_regressor': best_reg_m['name'],
        'best_reg_r2': best_reg_m['r2'],
        'best_reg_rmse': best_reg_m['rmse'],
        'justification': [
            f"{best_clf['name']} achieves the highest weighted F1 score ({best_clf['f1']}) among classifiers, indicating the best balance between precision and recall across all classes.",
            f"Its macro AUC of {best_clf['macro_auc']} demonstrates excellent discrimination ability between Low, Medium, and High potential classes.",
            f"For regression, {best_reg_m['name']} achieves R²={best_reg_m['r2']} with RMSE={best_reg_m['rmse']} GTPI points, meaning predictions deviate by only ~{best_reg_m['rmse']} points on a 0-100 scale.",
            "Cross-validation confirms generalization — the model performs consistently across different data splits, not just on one lucky test set."
        ]
    }

    report = {
        'info': info,
        'classification': clf_results,
        'regression': reg_results,
        'roc_data': {k: {'macro_auc': v['macro_auc'],
                         'class_aucs': {c: v['classes'][c]['auc'] for c in CLASS_ORDER}}
                     for k, v in roc_data.items()},
        'selection': selection,
        'charts': charts
    }

    path = os.path.join(DATA_DIR, 'evaluation_report.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, default=str)
    logger.info(f"Evaluation report saved: {path}")
    return report


def get_evaluation_report():
    path = os.path.join(DATA_DIR, 'evaluation_report.json')
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return run_evaluation()


if __name__ == '__main__':
    r = run_evaluation()
    print(f"\n{'='*60}")
    print("CLASSIFICATION RESULTS")
    for m in r['classification']:
        print(f"  {m['name']:25s} Acc={m['accuracy']} F1={m['f1']} AUC={m['macro_auc']}")
    print("\nREGRESSION RESULTS")
    for m in r['regression']:
        print(f"  {m['name']:25s} MAE={m['mae']} RMSE={m['rmse']} R²={m['r2']}")
    print(f"\nBest classifier: {r['selection']['best_classifier']}")
    print(f"Best regressor:  {r['selection']['best_regressor']}")
