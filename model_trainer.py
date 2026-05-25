"""
Model Trainer — GastroMap Colombia
Trains, evaluates, and compares 3 ML models for GTPI prediction.
Uses GTPI classification as target with supervised learning.
"""

import pandas as pd
import numpy as np
import json
import os
import base64
import io
import logging
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
STATIC_DIR = os.path.join(os.path.dirname(__file__), 'static', 'img')

# Features used for training
FEATURES = ['gastro_count', 'gastro_variety', 'gastro_employees',
            'total_providers', 'total_employees', 'unique_categories',
            'climate_index', 'avg_temperature', 'precipitation']

# Target: simplified 3-class classification
CLASS_MAP = {
    'Very Low': 'Low',
    'Low': 'Low',
    'Medium': 'Medium',
    'High': 'High',
    'Very High': 'High'
}
CLASS_ORDER = ['Low', 'Medium', 'High']


def load_data():
    """Load processed dataset and prepare for ML."""
    path = os.path.join(DATA_DIR, 'processed_dataset.json')
    df = pd.read_json(path, orient='records')

    # Create simplified 3-class target
    df['target'] = df['classification'].map(CLASS_MAP)

    X = df[FEATURES].copy()
    y = df['target'].copy()

    return df, X, y


def generate_charts(models_results, cm_data):
    """Generate chart images as base64 strings."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    charts = {}
    bg_color = '#161a23'
    text_color = '#e8eaf0'
    accent = '#f59e0b'
    grid_color = '#272d3b'

    plt.rcParams.update({
        'figure.facecolor': bg_color, 'axes.facecolor': bg_color,
        'text.color': text_color, 'axes.labelcolor': text_color,
        'xtick.color': text_color, 'ytick.color': text_color,
        'font.size': 10, 'font.family': 'sans-serif'
    })

    # 1. Accuracy comparison bar chart
    fig, ax = plt.subplots(figsize=(8, 4))
    names = [r['name'] for r in models_results]
    accs = [r['test_accuracy'] for r in models_results]
    f1s = [r['test_f1_weighted'] for r in models_results]
    x = np.arange(len(names))
    w = 0.35
    bars1 = ax.bar(x - w/2, accs, w, label='Accuracy', color=accent, alpha=0.9)
    bars2 = ax.bar(x + w/2, f1s, w, label='F1 (weighted)', color='#10b981', alpha=0.9)
    ax.set_ylabel('Score')
    ax.set_title('Model Comparison — Accuracy vs F1 Score', color=accent, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.05)
    ax.legend(facecolor=bg_color, edgecolor=grid_color)
    ax.grid(axis='y', color=grid_color, alpha=0.5)
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9, color=text_color)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9, color=text_color)
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    charts['comparison'] = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()

    # 2. Confusion matrices
    for model_name, cm in cm_data.items():
        fig, ax = plt.subplots(figsize=(5, 4))
        cm_array = np.array(cm)
        im = ax.imshow(cm_array, cmap='YlOrBr', aspect='auto')
        ax.set_xticks(range(len(CLASS_ORDER)))
        ax.set_yticks(range(len(CLASS_ORDER)))
        ax.set_xticklabels(CLASS_ORDER, fontsize=9)
        ax.set_yticklabels(CLASS_ORDER, fontsize=9)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(f'Confusion Matrix — {model_name}', color=accent, fontsize=12, fontweight='bold')
        for i in range(len(CLASS_ORDER)):
            for j in range(len(CLASS_ORDER)):
                color_val = 'black' if cm_array[i, j] > cm_array.max() * 0.5 else text_color
                ax.text(j, i, str(cm_array[i, j]), ha='center', va='center',
                        fontsize=14, fontweight='bold', color=color_val)
        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        safe_name = model_name.replace(' ', '_').lower()
        charts[f'cm_{safe_name}'] = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()

    # 3. Feature importance (Random Forest)
    rf_result = next((r for r in models_results if r['name'] == 'Random Forest'), None)
    if rf_result and 'feature_importance' in rf_result:
        fig, ax = plt.subplots(figsize=(8, 4))
        fi = rf_result['feature_importance']
        sorted_fi = sorted(fi.items(), key=lambda x: x[1], reverse=True)
        feat_names = [f[0] for f in sorted_fi]
        feat_vals = [f[1] for f in sorted_fi]
        colors = [accent if v > 0.1 else '#10b981' if v > 0.05 else grid_color for v in feat_vals]
        ax.barh(range(len(feat_names)), feat_vals, color=colors)
        ax.set_yticks(range(len(feat_names)))
        ax.set_yticklabels(feat_names, fontsize=9)
        ax.set_xlabel('Importance')
        ax.set_title('Feature Importance — Random Forest', color=accent, fontsize=14, fontweight='bold')
        ax.grid(axis='x', color=grid_color, alpha=0.5)
        ax.invert_yaxis()
        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        charts['feature_importance'] = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()

    # 4. Cross-validation boxplot
    fig, ax = plt.subplots(figsize=(8, 4))
    cv_data = [r['cv_scores'] for r in models_results]
    bp = ax.boxplot(cv_data, labels=names, patch_artist=True,
                    boxprops=dict(facecolor=accent, alpha=0.3),
                    medianprops=dict(color=accent, linewidth=2),
                    whiskerprops=dict(color=text_color),
                    capprops=dict(color=text_color),
                    flierprops=dict(markeredgecolor=text_color))
    ax.set_ylabel('Accuracy')
    ax.set_title('5-Fold Cross-Validation Distribution', color=accent, fontsize=14, fontweight='bold')
    ax.grid(axis='y', color=grid_color, alpha=0.5)
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    charts['cv_boxplot'] = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()

    return charts


def train_all_models():
    """Train 3 models, evaluate, compare, and generate report."""
    df, X, y = load_data()

    # Encode target
    le = LabelEncoder()
    le.fit(CLASS_ORDER)
    y_encoded = le.transform(y)

    # Scale features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=FEATURES)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    logger.info(f"Training set: {len(X_train)} | Test set: {len(X_test)}")
    logger.info(f"Class distribution (train): {dict(zip(*np.unique(y_train, return_counts=True)))}")

    # Define models
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=200, max_depth=15, min_samples_split=5,
            min_samples_leaf=2, class_weight='balanced', random_state=42, n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=150, max_depth=5, learning_rate=0.1,
            min_samples_split=5, min_samples_leaf=3, random_state=42
        ),
        'K-Nearest Neighbors': KNeighborsClassifier(
            n_neighbors=7, weights='distance', metric='minkowski', p=2
        )
    }

    results = []
    cm_data = {}
    predictions_examples = []

    for name, model in models.items():
        logger.info(f"Training {name}...")

        # Train
        model.fit(X_train, y_train)

        # Predict
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # Cross-validation
        cv_scores = cross_val_score(model, X_scaled, y_encoded, cv=5, scoring='accuracy')

        # Metrics
        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc = accuracy_score(y_test, y_pred_test)
        test_prec = precision_score(y_test, y_pred_test, average='weighted', zero_division=0)
        test_rec = recall_score(y_test, y_pred_test, average='weighted', zero_division=0)
        test_f1 = f1_score(y_test, y_pred_test, average='weighted', zero_division=0)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred_test, labels=le.transform(CLASS_ORDER))
        cm_data[name] = cm.tolist()

        # Classification report
        report = classification_report(y_test, y_pred_test,
                                       target_names=CLASS_ORDER, zero_division=0, output_dict=True)

        result = {
            'name': name,
            'train_accuracy': round(train_acc, 4),
            'test_accuracy': round(test_acc, 4),
            'test_precision': round(test_prec, 4),
            'test_recall': round(test_rec, 4),
            'test_f1_weighted': round(test_f1, 4),
            'cv_mean': round(cv_scores.mean(), 4),
            'cv_std': round(cv_scores.std(), 4),
            'cv_scores': cv_scores.round(4).tolist(),
            'confusion_matrix': cm.tolist(),
            'class_report': {k: v for k, v in report.items() if k in CLASS_ORDER},
            'hyperparameters': {}
        }

        # Model-specific info
        if name == 'Random Forest':
            result['feature_importance'] = dict(zip(FEATURES,
                model.feature_importances_.round(4).tolist()))
            result['hyperparameters'] = {
                'n_estimators': 200, 'max_depth': 15, 'min_samples_split': 5,
                'min_samples_leaf': 2, 'class_weight': 'balanced'
            }
        elif name == 'Gradient Boosting':
            result['feature_importance'] = dict(zip(FEATURES,
                model.feature_importances_.round(4).tolist()))
            result['hyperparameters'] = {
                'n_estimators': 150, 'max_depth': 5, 'learning_rate': 0.1,
                'min_samples_split': 5, 'min_samples_leaf': 3
            }
        elif name == 'K-Nearest Neighbors':
            result['hyperparameters'] = {
                'n_neighbors': 7, 'weights': 'distance', 'metric': 'minkowski', 'p': 2
            }

        results.append(result)
        logger.info(f"  {name}: train={train_acc:.4f} test={test_acc:.4f} CV={cv_scores.mean():.4f}±{cv_scores.std():.4f}")

    # Generate prediction examples
    sample_indices = np.random.RandomState(42).choice(len(X_test), min(10, len(X_test)), replace=False)
    test_df_indices = X_test.index[sample_indices]

    for idx, test_idx in zip(sample_indices, test_df_indices):
        actual_class = le.inverse_transform([y_test[idx]])[0]
        row_data = df.iloc[test_idx]
        example = {
            'municipality': str(row_data.get('muni_name', 'Unknown')),
            'department': str(row_data.get('dept_name', 'Unknown')),
            'actual_gtpi': float(row_data.get('GTPI', 0)),
            'actual_class': actual_class,
            'predictions': {}
        }
        for name, model in models.items():
            pred = model.predict(X_test.iloc[[idx]])[0]
            pred_class = le.inverse_transform([pred])[0]
            example['predictions'][name] = pred_class
        predictions_examples.append(example)

    # Generate charts
    charts = generate_charts(results, cm_data)

    # Best model
    best = max(results, key=lambda x: x['test_f1_weighted'])

    # Build report
    report = {
        'models': results,
        'best_model': best['name'],
        'best_f1': best['test_f1_weighted'],
        'predictions': predictions_examples,
        'charts': charts,
        'dataset_info': {
            'total_samples': len(X),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'test_size': 0.2,
            'features_used': FEATURES,
            'n_features': len(FEATURES),
            'class_distribution': y.value_counts().to_dict(),
            'class_mapping': CLASS_MAP,
            'scaler': 'StandardScaler',
            'random_state': 42
        }
    }

    # Save
    cache_path = os.path.join(DATA_DIR, 'model_report.json')
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, default=str)
    logger.info(f"Model report saved to {cache_path}")

    return report


def get_model_report():
    """Load model report from cache or train."""
    path = os.path.join(DATA_DIR, 'model_report.json')
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return train_all_models()


if __name__ == '__main__':
    report = train_all_models()
    print(f"\n{'='*60}")
    print(f"MODELS TRAINED SUCCESSFULLY")
    print(f"{'='*60}")
    for m in report['models']:
        print(f"\n{m['name']}:")
        print(f"  Train Acc: {m['train_accuracy']}")
        print(f"  Test Acc:  {m['test_accuracy']}")
        print(f"  F1 Score:  {m['test_f1_weighted']}")
        print(f"  CV Mean:   {m['cv_mean']} ± {m['cv_std']}")
    print(f"\nBest model: {report['best_model']} (F1: {report['best_f1']})")
    print(f"\nPrediction examples:")
    for p in report['predictions'][:5]:
        preds = ' | '.join([f"{k}: {v}" for k, v in p['predictions'].items()])
        print(f"  {p['municipality']} (GTPI {p['actual_gtpi']}, {p['actual_class']}) → {preds}")
