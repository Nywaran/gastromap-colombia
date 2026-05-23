import pandas as pd
import numpy as np
import requests
import os
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


# ══════════════════════════════════════════════════════════════
# PHASE 2A — DATA UNDERSTANDING (Raw data loading + EDA)
# ══════════════════════════════════════════════════════════════

def load_divipola():
    """Load and parse the DIVIPOLA Excel file with municipality coordinates."""
    path = os.path.join(DATA_DIR, 'DIVIPOLA_Municipios.xlsx')
    df = pd.read_excel(path, header=None, skiprows=10)
    df.columns = [
        'dept_code', 'dept_name', 'muni_code', 'muni_name',
        'type', 'longitude', 'latitude', 'note'
    ]
    df = df.dropna(subset=['muni_code'])
    df['dept_code'] = df['dept_code'].astype(str).str.zfill(2)
    df['muni_code'] = df['muni_code'].astype(str).str.zfill(5)
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    df = df.dropna(subset=['latitude', 'longitude'])
    df['dept_name'] = df['dept_name'].ffill()
    logger.info(f"DIVIPOLA loaded: {len(df)} municipalities")
    return df[['dept_code', 'dept_name', 'muni_code', 'muni_name',
               'type', 'longitude', 'latitude']]


def load_rnt():
    """Load the National Tourism Registry CSV."""
    path = os.path.join(DATA_DIR, 'Registro_Nacional_de_Turismo_-_RNT_20260516.csv')
    df = pd.read_csv(path, low_memory=False)
    df['CODIGO_MUNICIPIO'] = df['CODIGO_MUNICIPIO'].astype(str).str.zfill(5)
    df['CODIGO_DEPARTAMENTO'] = df['CODIGO_DEPARTAMENTO'].astype(str).str.zfill(2)
    logger.info(f"RNT loaded: {len(df)} records")
    return df


def generate_eda_report(divipola_df, rnt_df):
    """Generate comprehensive Exploratory Data Analysis statistics."""

    # --- DIVIPOLA EDA ---
    div_eda = {
        'name': 'DIVIPOLA — Municipality Geographic Data',
        'source': 'DANE (National Administrative Department of Statistics)',
        'url': 'https://geoportal.dane.gov.co',
        'format': '.xlsx',
        'total_records': len(divipola_df),
        'total_columns': len(divipola_df.columns),
        'columns': [],
        'missing_values': {},
        'sample_rows': divipola_df.head(5).to_dict('records'),
        'numeric_stats': {},
        'unique_counts': {}
    }

    for col in divipola_df.columns:
        col_info = {
            'name': col,
            'dtype': str(divipola_df[col].dtype),
            'non_null': int(divipola_df[col].notna().sum()),
            'null_count': int(divipola_df[col].isna().sum()),
            'null_pct': round(divipola_df[col].isna().mean() * 100, 2),
            'unique': int(divipola_df[col].nunique())
        }
        div_eda['columns'].append(col_info)
        div_eda['missing_values'][col] = col_info['null_count']

    for col in ['latitude', 'longitude']:
        stats = divipola_df[col].describe()
        div_eda['numeric_stats'][col] = {
            'mean': round(float(stats['mean']), 4),
            'std': round(float(stats['std']), 4),
            'min': round(float(stats['min']), 4),
            'max': round(float(stats['max']), 4),
            'median': round(float(stats['50%']), 4)
        }

    div_eda['unique_counts'] = {
        'departments': int(divipola_df['dept_name'].nunique()),
        'municipalities': int(divipola_df['muni_name'].nunique()),
        'types': divipola_df['type'].value_counts().to_dict()
    }

    # --- RNT EDA ---
    rnt_eda = {
        'name': 'RNT — National Tourism Registry',
        'source': 'MinCIT (Ministry of Commerce, Industry and Tourism)',
        'url': 'https://www.datos.gov.co',
        'format': '.csv',
        'total_records': len(rnt_df),
        'total_columns': len(rnt_df.columns),
        'columns': [],
        'missing_values': {},
        'sample_rows': [],
        'numeric_stats': {},
        'category_distribution': {},
        'temporal_distribution': {}
    }

    for col in rnt_df.columns:
        col_info = {
            'name': col,
            'dtype': str(rnt_df[col].dtype),
            'non_null': int(rnt_df[col].notna().sum()),
            'null_count': int(rnt_df[col].isna().sum()),
            'null_pct': round(rnt_df[col].isna().mean() * 100, 2),
            'unique': int(rnt_df[col].nunique())
        }
        rnt_eda['columns'].append(col_info)
        if col_info['null_count'] > 0:
            rnt_eda['missing_values'][col] = {
                'count': col_info['null_count'],
                'percentage': col_info['null_pct']
            }

    # Numeric stats for employees
    if 'NUMERO_DE_EMPLEADOS' in rnt_df.columns:
        emp = rnt_df['NUMERO_DE_EMPLEADOS'].dropna()
        rnt_eda['numeric_stats']['NUMERO_DE_EMPLEADOS'] = {
            'mean': round(float(emp.mean()), 2),
            'std': round(float(emp.std()), 2),
            'min': int(emp.min()),
            'max': int(emp.max()),
            'median': round(float(emp.median()), 2)
        }

    # Category distribution
    rnt_eda['category_distribution'] = rnt_df['CATEGORIA'].value_counts().head(10).to_dict()

    # Gastronomy subcategories
    gastro_mask = rnt_df['CATEGORIA'].str.contains('GASTRO|BAR(?!RAN)', case=False, na=False, regex=True)
    gastro_sub = rnt_df.loc[gastro_mask, 'SUB_CATEGORIA'].value_counts().to_dict()
    rnt_eda['gastronomy_subcategories'] = gastro_sub
    rnt_eda['gastronomy_count'] = int(gastro_mask.sum())

    # Temporal
    if 'AÑO' in rnt_df.columns:
        rnt_eda['temporal_distribution'] = rnt_df['AÑO'].value_counts().sort_index().to_dict()

    # Top departments
    rnt_eda['top_departments'] = rnt_df['DEPARTAMENTO'].value_counts().head(10).to_dict()

    # Correlation (numeric columns only)
    numeric_cols = rnt_df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) >= 2:
        corr = rnt_df[numeric_cols].corr().round(3)
        rnt_eda['correlation_matrix'] = {
            'columns': numeric_cols,
            'values': corr.values.tolist()
        }

    return {'divipola': div_eda, 'rnt': rnt_eda}


# ══════════════════════════════════════════════════════════════
# PHASE 2B — DATA ENGINEERING (Cleaning, transformation, features)
# ══════════════════════════════════════════════════════════════

def filter_gastronomy(rnt_df):
    """Filter gastronomy establishments using regex on CATEGORIA column."""
    mask = rnt_df['CATEGORIA'].str.contains(
        'GASTRO|BAR(?!RAN)', case=False, na=False, regex=True
    )
    gastro = rnt_df[mask].copy()
    logger.info(f"Gastronomy establishments: {len(gastro)}")
    return gastro


def aggregate_gastronomy(gastro_df):
    """Aggregate gastronomy metrics per municipality."""
    agg = gastro_df.groupby('CODIGO_MUNICIPIO').agg(
        gastro_count=('CODIGO_RNT', 'count'),
        gastro_variety=('SUB_CATEGORIA', 'nunique'),
        gastro_employees=('NUMERO_DE_EMPLEADOS', 'sum'),
        top_subcategories=('SUB_CATEGORIA', lambda x: list(x.value_counts().head(5).index))
    ).reset_index()
    agg.rename(columns={'CODIGO_MUNICIPIO': 'muni_code'}, inplace=True)
    return agg


def aggregate_all_tourism(rnt_df):
    """Aggregate all tourism providers per municipality."""
    counts = rnt_df.groupby('CODIGO_MUNICIPIO').agg(
        total_providers=('CODIGO_RNT', 'count'),
        total_employees=('NUMERO_DE_EMPLEADOS', 'sum'),
        unique_categories=('CATEGORIA', 'nunique')
    ).reset_index()
    counts.rename(columns={'CODIGO_MUNICIPIO': 'muni_code'}, inplace=True)
    return counts


def estimate_climate(divipola_df):
    """Estimate climate from coordinates using Colombia's thermal floor system."""
    df = divipola_df.copy()

    def classify_region(row):
        lat, lon = row['latitude'], row['longitude']
        if lat > 8:
            return 'Caribbean'
        elif lon < -76.5 and lat < 7:
            return 'Pacific'
        elif lon > -72:
            return 'Orinoquia'
        elif lon > -73 and lat < 2:
            return 'Amazon'
        else:
            return 'Andean'

    df['region'] = df.apply(classify_region, axis=1)

    temp_map = {'Caribbean': 28, 'Pacific': 26, 'Andean': 18, 'Orinoquia': 27, 'Amazon': 26}
    df['avg_temperature'] = df['region'].map(temp_map)
    df['avg_temperature'] += (df['latitude'] - 4) * 0.3

    precip_map = {'Caribbean': 1200, 'Pacific': 5000, 'Andean': 1500, 'Orinoquia': 2500, 'Amazon': 3500}
    df['precipitation'] = df['region'].map(precip_map)

    df['temp_comfort'] = (1 - np.abs(df['avg_temperature'] - 23) / 15).clip(0, 1)
    df['precip_comfort'] = (1 - np.abs(df['precipitation'] - 1500) / 4000).clip(0, 1)
    df['climate_index'] = 0.6 * df['temp_comfort'] + 0.4 * df['precip_comfort']

    logger.info(f"Climate estimated for {len(df)} municipalities")
    return df


def calculate_gtpi(merged_df):
    """Calculate the Gastronomic Tourism Potential Index (GTPI)."""
    df = merged_df.copy()

    def normalize(series):
        mn, mx = series.min(), series.max()
        if mx == mn:
            return pd.Series(0.5, index=series.index)
        return (series - mn) / (mx - mn)

    df['norm_gastro_count'] = normalize(np.log1p(df['gastro_count']))
    df['norm_gastro_variety'] = normalize(df['gastro_variety'])
    df['norm_providers'] = normalize(np.log1p(df['total_providers']))
    df['norm_climate'] = normalize(df['climate_index'])
    df['norm_employment'] = normalize(np.log1p(df['gastro_employees']))

    df['GTPI'] = (
        0.35 * df['norm_gastro_count'] +
        0.20 * df['norm_gastro_variety'] +
        0.20 * df['norm_providers'] +
        0.15 * df['norm_climate'] +
        0.10 * df['norm_employment']
    )
    df['GTPI'] = (df['GTPI'] * 100).round(1)

    df['classification'] = pd.cut(
        df['GTPI'],
        bins=[0, 20, 40, 60, 80, 100],
        labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'],
        include_lowest=True
    )

    logger.info(f"GTPI calculated. Distribution:\n{df['classification'].value_counts().to_string()}")
    return df


def generate_engineering_report(divipola_df, rnt_df, merged_df):
    """Document all data engineering transformations performed."""
    gastro_mask = rnt_df['CATEGORIA'].str.contains('GASTRO|BAR(?!RAN)', case=False, na=False, regex=True)

    report = {
        'cleaning_steps': [
            {
                'step': 'DIVIPOLA Header Removal',
                'description': 'Skipped 10 header rows from Excel file (title, subtitle, blank lines) and manually assigned column names.',
                'impact': 'Enabled correct parsing of 1,122 municipality records.'
            },
            {
                'step': 'Municipality Code Normalization',
                'description': 'Zero-padded municipality codes to 5 digits (e.g., 5001 → 05001) and department codes to 2 digits using str.zfill().',
                'impact': 'Ensured consistent join keys across all datasets.'
            },
            {
                'step': 'Department Name Forward-Fill',
                'description': 'Applied ffill() to propagate department names down merged Excel cells.',
                'impact': f'All {len(divipola_df)} municipalities now have a department name.'
            },
            {
                'step': 'Null Coordinate Removal',
                'description': 'Dropped rows where latitude or longitude was null or non-numeric.',
                'impact': 'Removed invalid geographic entries, retaining only plottable municipalities.'
            },
            {
                'step': 'RNT Numeric Coercion',
                'description': 'Loaded CSV with low_memory=False to prevent mixed-type inference on 686K+ rows.',
                'impact': 'Correct data types for all columns from initial load.'
            }
        ],
        'null_handling': [
            {
                'column': 'Gastronomy features (gastro_count, gastro_variety, gastro_employees)',
                'strategy': 'Fill with 0',
                'reason': 'Municipalities without gastronomy establishments should score 0, not be excluded.'
            },
            {
                'column': 'Tourism features (total_providers, total_employees)',
                'strategy': 'Fill with 0',
                'reason': 'Same logic — absence of data means absence of providers.'
            },
            {
                'column': 'top_subcategories (list)',
                'strategy': 'Fill with empty list []',
                'reason': 'Prevents errors when iterating subcategory lists in the frontend.'
            }
        ],
        'transformations': [
            {
                'name': 'Gastronomy Filtering (Regex)',
                'formula': "CATEGORIA.str.contains('GASTRO|BAR(?!RAN)')",
                'before': f'{len(rnt_df):,} total RNT records',
                'after': f'{int(gastro_mask.sum()):,} gastronomy establishments',
                'reason': 'Isolate gastronomy-related providers. Negative lookahead (?!RAN) prevents false matches with "BARRANQUILLA".'
            },
            {
                'name': 'Log Transformation — log(1+x)',
                'formula': 'np.log1p(x)',
                'columns': ['gastro_count', 'total_providers', 'gastro_employees'],
                'reason': 'These features have extremely right-skewed distributions (Bogotá: 3,854 establishments vs median: 5). Log compresses the range so outliers do not dominate normalization.',
                'before': 'Range [0, 3854] — 99% of values below 100',
                'after': 'Range [0, 8.26] — smooth, near-normal distribution'
            },
            {
                'name': 'Min-Max Normalization',
                'formula': '(x - x_min) / (x_max - x_min)',
                'columns': ['All 5 GTPI components'],
                'reason': 'Scales all features to [0, 1] for comparable weighted combination regardless of original units.',
                'before': 'Different scales: count (0-3854), index (0-1), variety (0-25)',
                'after': 'All features in [0, 1]'
            },
            {
                'name': 'Climate Feature Engineering',
                'formula': 'comfort = 0.6 × temp_comfort + 0.4 × precip_comfort',
                'reason': 'Created a composite climate comfort index from estimated temperature and precipitation using domain knowledge of Colombia thermal floors.',
                'before': 'Raw coordinates (latitude, longitude)',
                'after': 'region, avg_temperature, precipitation, climate_index'
            }
        ],
        'feature_engineering': [
            {
                'feature': 'gastro_count',
                'type': 'Aggregation',
                'operation': 'COUNT(CODIGO_RNT) GROUP BY muni_code',
                'description': 'Number of gastronomy establishments per municipality'
            },
            {
                'feature': 'gastro_variety',
                'type': 'Aggregation',
                'operation': 'NUNIQUE(SUB_CATEGORIA) GROUP BY muni_code',
                'description': 'Diversity of gastronomy subcategories (restaurant, gastrobar, café, etc.)'
            },
            {
                'feature': 'gastro_employees',
                'type': 'Aggregation',
                'operation': 'SUM(NUMERO_DE_EMPLEADOS) GROUP BY muni_code',
                'description': 'Total employees in gastronomy sector per municipality'
            },
            {
                'feature': 'total_providers',
                'type': 'Aggregation',
                'operation': 'COUNT(CODIGO_RNT) GROUP BY muni_code (all categories)',
                'description': 'Total tourism providers — measures overall tourism infrastructure'
            },
            {
                'feature': 'climate_index',
                'type': 'Domain Engineering',
                'operation': 'Rule-based classification + comfort formula',
                'description': 'Climate comfort score (0-1) based on estimated temperature and precipitation'
            },
            {
                'feature': 'GTPI',
                'type': 'Weighted Composite',
                'operation': '0.35×GD + 0.20×GV + 0.20×TI + 0.15×CC + 0.10×GE',
                'description': 'Final Gastronomic Tourism Potential Index (0-100)'
            }
        ],
        'final_dataset_shape': {
            'rows': len(merged_df),
            'columns': len(merged_df.columns),
            'features_used': 5,
            'classification_bins': 5
        }
    }
    return report


# ══════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════

def build_dataset():
    """Full pipeline: load → clean → engineer → score → export."""
    divipola = load_divipola()
    rnt = load_rnt()

    gastro = filter_gastronomy(rnt)
    gastro_agg = aggregate_gastronomy(gastro)
    tourism_agg = aggregate_all_tourism(rnt)

    divipola_climate = estimate_climate(divipola)

    merged = divipola_climate.merge(gastro_agg, on='muni_code', how='left')
    merged = merged.merge(tourism_agg, on='muni_code', how='left')

    fill_cols = ['gastro_count', 'gastro_variety', 'gastro_employees',
                 'total_providers', 'total_employees', 'unique_categories']
    merged[fill_cols] = merged[fill_cols].fillna(0)
    merged['top_subcategories'] = merged['top_subcategories'].apply(
        lambda x: x if isinstance(x, list) else []
    )

    active = merged[merged['total_providers'] > 0].copy()
    logger.info(f"Active municipalities: {len(active)}")

    result = calculate_gtpi(active)
    result = result.sort_values('GTPI', ascending=False).reset_index(drop=True)
    result['ranking'] = range(1, len(result) + 1)

    # Generate reports
    eda_report = generate_eda_report(divipola, rnt)
    eng_report = generate_engineering_report(divipola, rnt, result)

    # Export
    export = result[[
        'ranking', 'dept_code', 'dept_name', 'muni_code', 'muni_name',
        'region', 'latitude', 'longitude', 'avg_temperature', 'precipitation',
        'climate_index', 'gastro_count', 'gastro_variety',
        'gastro_employees', 'top_subcategories', 'total_providers',
        'total_employees', 'unique_categories', 'GTPI', 'classification'
    ]].copy()
    export['classification'] = export['classification'].astype(str)

    export.to_json(os.path.join(DATA_DIR, 'processed_dataset.json'), orient='records', force_ascii=False)
    with open(os.path.join(DATA_DIR, 'eda_report.json'), 'w', encoding='utf-8') as f:
        json.dump(eda_report, f, ensure_ascii=False, default=str)
    with open(os.path.join(DATA_DIR, 'engineering_report.json'), 'w', encoding='utf-8') as f:
        json.dump(eng_report, f, ensure_ascii=False, default=str)

    logger.info("All reports saved.")
    return export


def get_dataset():
    """Load processed dataset from cache."""
    path = os.path.join(DATA_DIR, 'processed_dataset.json')
    if os.path.exists(path):
        return pd.read_json(path, orient='records')
    return build_dataset()


def get_eda_report():
    """Load EDA report from cache."""
    path = os.path.join(DATA_DIR, 'eda_report.json')
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def get_engineering_report():
    """Load engineering report from cache."""
    path = os.path.join(DATA_DIR, 'engineering_report.json')
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def get_summary_stats(df):
    """Generate summary statistics for the dashboard."""
    return {
        'total_municipalities': len(df),
        'with_gastronomy': int((df['gastro_count'] > 0).sum()),
        'total_gastro_establishments': int(df['gastro_count'].sum()),
        'total_tourism_providers': int(df['total_providers'].sum()),
        'top_10': df.head(10).to_dict('records'),
        'by_region': df.groupby('region').agg(
            municipalities=('muni_code', 'count'),
            establishments=('gastro_count', 'sum'),
            avg_gtpi=('GTPI', 'mean')
        ).round(1).reset_index().to_dict('records'),
        'by_classification': df['classification'].value_counts().to_dict(),
        'by_department': df.groupby('dept_name').agg(
            municipalities=('muni_code', 'count'),
            establishments=('gastro_count', 'sum'),
            avg_gtpi=('GTPI', 'mean'),
            best_gtpi=('GTPI', 'max')
        ).round(1).sort_values('avg_gtpi', ascending=False).reset_index().to_dict('records')
    }


if __name__ == '__main__':
    df = build_dataset()
    print(f"\nDataset: {len(df)} municipalities")
    print(df[['ranking', 'muni_name', 'dept_name', 'GTPI', 'classification']].head(10).to_string(index=False))
