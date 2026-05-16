from flask import Flask, render_template, jsonify, request
from data_processor import build_dataset, get_dataset, get_summary_stats
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

_df = None
_stats = None


def get_data():
    """Load processed dataset. Uses cached JSON if available, builds from raw data otherwise."""
    global _df, _stats
    if _df is None:
        try:
            logger.info("Attempting to load from cached dataset...")
            _df = get_dataset()
        except Exception:
            logger.info("Cache not found. Building from raw data...")
            _df = build_dataset()
        _stats = get_summary_stats(_df)
    return _df, _stats


# ── Page Routes ──────────────────────────────────────────────

@app.route('/')
def index():
    """Landing page with project overview."""
    _, stats = get_data()
    return render_template('index.html', stats=stats)


@app.route('/dashboard')
def dashboard():
    """Interactive dashboard with map, rankings, and filters."""
    _, stats = get_data()
    return render_template('dashboard.html', stats=stats)


@app.route('/methodology')
def methodology():
    """Detailed methodology and ML explanation."""
    return render_template('methodology.html')


# ── API Routes ───────────────────────────────────────────────

@app.route('/api/municipalities')
def api_municipalities():
    """Return municipalities with GTPI scores. Supports query filters."""
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

    result = filtered[[
        'ranking', 'muni_code', 'muni_name', 'dept_name', 'region',
        'latitude', 'longitude', 'GTPI', 'classification',
        'gastro_count', 'gastro_variety', 'gastro_employees',
        'total_providers', 'avg_temperature', 'precipitation',
        'top_subcategories'
    ]].to_dict('records')

    return jsonify(result)


@app.route('/api/stats')
def api_stats():
    _, stats = get_data()
    return jsonify(stats)


@app.route('/api/departments')
def api_departments():
    df, _ = get_data()
    return jsonify(sorted(df['dept_name'].unique().tolist()))


@app.route('/api/municipality/<muni_code>')
def api_municipality_detail(muni_code):
    df, _ = get_data()
    muni = df[df['muni_code'] == muni_code]
    if muni.empty:
        return jsonify({'error': 'Municipality not found'}), 404
    return jsonify(muni.iloc[0].to_dict())


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)