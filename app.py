from flask import Flask, render_template, jsonify, request
from data_processor import (
    build_dataset, get_dataset, get_summary_stats,
    get_eda_report, get_engineering_report
)
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

_df = None
_stats = None


def get_data():
    global _df, _stats
    if _df is None:
        try:
            _df = get_dataset()
        except Exception:
            _df = build_dataset()
        _stats = get_summary_stats(_df)
    return _df, _stats


# ── Page Routes (CRISP-ML Phases) ────────────────────────────

@app.route('/')
def home():
    _, stats = get_data()
    return render_template('home.html', stats=stats)


@app.route('/crisp-ml')
def crisp_ml():
    return render_template('crisp_ml.html')


@app.route('/business-understanding')
def business_understanding():
    return render_template('business_understanding.html')


@app.route('/data-understanding')
def data_understanding():
    eda = get_eda_report()
    return render_template('data_understanding.html', eda=eda)


@app.route('/data-engineering')
def data_engineering():
    eng = get_engineering_report()
    _, stats = get_data()
    return render_template('data_engineering.html', eng=eng, stats=stats)


@app.route('/dashboard')
def dashboard():
    _, stats = get_data()
    return render_template('dashboard.html', stats=stats)


@app.route('/model-engineering')
def model_engineering():
    _, stats = get_data()
    return render_template('model_engineering.html', stats=stats)


@app.route('/model-evaluation')
def model_evaluation():
    _, stats = get_data()
    return render_template('model_evaluation.html', stats=stats)


@app.route('/deployment')
def deployment():
    return render_template('deployment.html')


@app.route('/monitoring')
def monitoring():
    return render_template('monitoring.html')


# ── API Routes ───────────────────────────────────────────────

@app.route('/api/municipalities')
def api_municipalities():
    df, _ = get_data()
    region = request.args.get('region')
    department = request.args.get('department')
    min_gtpi = request.args.get('min_gtpi', type=float, default=0)
    classification = request.args.get('classification')

    filtered = df.copy()
    if region:
        filtered = filtered[filtered['region'] == region]
    if department:
        filtered = filtered[filtered['dept_name'] == department]
    if min_gtpi > 0:
        filtered = filtered[filtered['GTPI'] >= min_gtpi]
    if classification:
        filtered = filtered[filtered['classification'] == classification]

    return jsonify(filtered[[
        'ranking', 'muni_code', 'muni_name', 'dept_name', 'region',
        'latitude', 'longitude', 'GTPI', 'classification',
        'gastro_count', 'gastro_variety', 'gastro_employees',
        'total_providers', 'avg_temperature', 'precipitation',
        'top_subcategories'
    ]].to_dict('records'))


@app.route('/api/stats')
def api_stats():
    _, stats = get_data()
    return jsonify(stats)


@app.route('/api/departments')
def api_departments():
    df, _ = get_data()
    return jsonify(sorted(df['dept_name'].unique().tolist()))


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
