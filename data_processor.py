import pandas as pd
import numpy as np
import requests
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


def load_divipola():
    """Load and clean the DIVIPOLA file with municipality coordinates."""
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
    logger.info(f"DIVIPOLA loaded: {len(df)} municipalities with coordinates")
    return df[['dept_code', 'dept_name', 'muni_code', 'muni_name',
               'type', 'longitude', 'latitude']]


def load_rnt():
    """Load the National Tourism Registry (RNT) dataset."""
    path = os.path.join(DATA_DIR, 'Registro_Nacional_de_Turismo_-_RNT_20260516.csv')
    df = pd.read_csv(path, low_memory=False)
    df['CODIGO_MUNICIPIO'] = df['CODIGO_MUNICIPIO'].astype(str).str.zfill(5)
    df['CODIGO_DEPARTAMENTO'] = df['CODIGO_DEPARTAMENTO'].astype(str).str.zfill(2)
    logger.info(f"RNT loaded: {len(df)} total records")
    return df


def filter_gastronomy(rnt_df):
    """Filter only gastronomy-related establishments from the RNT."""
    mask = rnt_df['CATEGORIA'].str.contains(
        'GASTRO|BAR(?!RAN)', case=False, na=False, regex=True
    )
    gastro = rnt_df[mask].copy()
    logger.info(f"Gastronomy establishments found: {len(gastro)}")
    return gastro


def aggregate_all_tourism(rnt_df):
    """Count all tourism service providers per municipality."""
    counts = rnt_df.groupby('CODIGO_MUNICIPIO').agg(
        total_providers=('CODIGO_RNT', 'count'),
        total_employees=('NUMERO_DE_EMPLEADOS', 'sum'),
        unique_categories=('CATEGORIA', 'nunique')
    ).reset_index()
    counts.rename(columns={'CODIGO_MUNICIPIO': 'muni_code'}, inplace=True)
    return counts


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


def estimate_climate(divipola_df):
    """
    Estimate climate variables from geographic coordinates.
    In Colombia, temperature strongly correlates with altitude and region
    (thermal floors). We use latitude/longitude as proxies.
    """
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

    # Average annual temperature by region (Celsius)
    temp_map = {
        'Caribbean': 28, 'Pacific': 26, 'Andean': 18,
        'Orinoquia': 27, 'Amazon': 26
    }
    df['avg_temperature'] = df['region'].map(temp_map)
    # Latitude adjustment (further north = warmer in Colombia)
    df['avg_temperature'] += (df['latitude'] - 4) * 0.3

    # Annual precipitation by region (mm/year)
    precip_map = {
        'Caribbean': 1200, 'Pacific': 5000, 'Andean': 1500,
        'Orinoquia': 2500, 'Amazon': 3500
    }
    df['precipitation'] = df['region'].map(precip_map)

    # Climate comfort index (0-1): optimal range 18-28C, moderate rainfall
    df['temp_comfort'] = 1 - np.abs(df['avg_temperature'] - 23) / 15
    df['temp_comfort'] = df['temp_comfort'].clip(0, 1)
    df['precip_comfort'] = 1 - np.abs(df['precipitation'] - 1500) / 4000
    df['precip_comfort'] = df['precip_comfort'].clip(0, 1)
    df['climate_index'] = 0.6 * df['temp_comfort'] + 0.4 * df['precip_comfort']

    logger.info(f"Climate estimated for {len(df)} municipalities")
    return df


def fetch_ideam_climate(limit=5000):
    """
    Attempt to fetch climate data from IDEAM via datos.gov.co API.
    Returns None if the API is unavailable.
    """
    url = (
        "https://www.datos.gov.co/resource/sbwg-7ju4.csv"
        f"?$limit={limit}"
        "&$select=CodigoEstacion,Municipio,Departamento,"
        "avg(ValorObservado) as avg_temperature"
        "&$group=CodigoEstacion,Municipio,Departamento"
        "&$where=ValorObservado IS NOT NULL"
    )
    try:
        logger.info("Querying IDEAM API...")
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200 and len(resp.text) > 100:
            df = pd.read_csv(pd.io.common.StringIO(resp.text))
            logger.info(f"IDEAM data retrieved: {len(df)} stations")
            return df
        logger.warning(f"IDEAM API returned status {resp.status_code}")
    except Exception as e:
        logger.warning(f"Could not connect to IDEAM API: {e}")
    return None


def calculate_gtpi(merged_df):
    """
    Calculate the Gastronomic Tourism Potential Index (GTPI).

    Components (weights):
    - Gastronomy density (0.35): number of gastronomy establishments
    - Gastronomy variety (0.20): diversity of subcategories
    - Tourism infrastructure (0.20): total tourism providers
    - Climate comfort (0.15): climate comfort index
    - Gastronomy employment (0.10): employees in gastronomy sector
    """
    df = merged_df.copy()

    def normalize(series):
        mn, mx = series.min(), series.max()
        if mx == mn:
            return pd.Series(0.5, index=series.index)
        return (series - mn) / (mx - mn)

    # Apply log1p to smooth skewed distributions
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

    # Scale to 0-100
    df['GTPI'] = (df['GTPI'] * 100).round(1)

    # Classification
    df['classification'] = pd.cut(
        df['GTPI'],
        bins=[0, 20, 40, 60, 80, 100],
        labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'],
        include_lowest=True
    )

    logger.info(f"GTPI calculated. Distribution:\n{df['classification'].value_counts().to_string()}")
    return df


def build_dataset():
    """Full pipeline: load, merge, and calculate scores."""
    # 1. Load sources
    divipola = load_divipola()
    rnt = load_rnt()

    # 2. Process RNT
    gastro = filter_gastronomy(rnt)
    gastro_agg = aggregate_gastronomy(gastro)
    tourism_agg = aggregate_all_tourism(rnt)

    # 3. Estimate climate
    divipola_climate = estimate_climate(divipola)

    # 4. Merge datasets (DIVIPOLA + Gastronomy + Tourism)
    merged = divipola_climate.merge(gastro_agg, on='muni_code', how='left')
    merged = merged.merge(tourism_agg, on='muni_code', how='left')

    # Fill municipalities with no gastronomy data
    fill_cols = ['gastro_count', 'gastro_variety', 'gastro_employees',
                 'total_providers', 'total_employees', 'unique_categories']
    merged[fill_cols] = merged[fill_cols].fillna(0)
    merged['top_subcategories'] = merged['top_subcategories'].apply(
        lambda x: x if isinstance(x, list) else []
    )

    # 5. Keep only municipalities with at least some tourism activity
    active = merged[merged['total_providers'] > 0].copy()
    logger.info(f"Municipalities with tourism activity: {len(active)}")

    # 6. Calculate GTPI
    result = calculate_gtpi(active)

    # 7. Sort by score
    result = result.sort_values('GTPI', ascending=False).reset_index(drop=True)
    result['ranking'] = range(1, len(result) + 1)

    # Save cache
    cache_path = os.path.join(DATA_DIR, 'processed_dataset.json')
    export = result[[
        'ranking', 'dept_code', 'dept_name', 'muni_code', 'muni_name',
        'region', 'latitude', 'longitude', 'avg_temperature', 'precipitation',
        'climate_index', 'gastro_count', 'gastro_variety',
        'gastro_employees', 'top_subcategories', 'total_providers',
        'total_employees', 'unique_categories', 'GTPI', 'classification'
    ]].copy()
    export['classification'] = export['classification'].astype(str)
    export.to_json(cache_path, orient='records', force_ascii=False)
    logger.info(f"Dataset saved to {cache_path}")

    return export


def get_dataset():
    """Return the processed dataset (uses cache if available)."""
    cache_path = os.path.join(DATA_DIR, 'processed_dataset.json')
    if os.path.exists(cache_path):
        logger.info("Loading dataset from cache...")
        return pd.read_json(cache_path, orient='records')
    return build_dataset()


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
    print(f"\n{'='*60}")
    print(f"FINAL DATASET: {len(df)} municipalities processed")
    print(f"{'='*60}")
    print(df[['ranking', 'muni_name', 'dept_name', 'GTPI', 'classification',
              'gastro_count', 'region']].head(20).to_string(index=False))
