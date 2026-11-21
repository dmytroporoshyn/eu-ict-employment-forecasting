"""Merge the Eurostat and World Bank data into a country x year panel.

Input: data/eurostat_*.csv, data/wb_*.csv
Output: data/panel_dataset.csv
"""

import os
import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

EU27_CODES = [
    'AT', 'BE', 'BG', 'HR', 'CY', 'CZ', 'DK', 'EE', 'FI', 'FR',
    'DE', 'EL', 'HU', 'IE', 'IT', 'LV', 'LT', 'LU', 'MT', 'NL',
    'PL', 'PT', 'RO', 'SK', 'SI', 'ES', 'SE'
]

# Eurostat uses EL for Greece, World Bank uses GRC/GR
GEO_RENAME = {'EL': 'GR'}  # normalize Greece code
GEO_RENAME_REVERSE = {'GR': 'EL'}

ISO3_TO_ISO2 = {
    'AUT': 'AT', 'BEL': 'BE', 'BGR': 'BG', 'HRV': 'HR', 'CYP': 'CY',
    'CZE': 'CZ', 'DNK': 'DK', 'EST': 'EE', 'FIN': 'FI', 'FRA': 'FR',
    'DEU': 'DE', 'GRC': 'GR', 'HUN': 'HU', 'IRL': 'IE', 'ITA': 'IT',
    'LVA': 'LV', 'LTU': 'LT', 'LUX': 'LU', 'MLT': 'MT', 'NLD': 'NL',
    'POL': 'PL', 'PRT': 'PT', 'ROU': 'RO', 'SVK': 'SK', 'SVN': 'SI',
    'ESP': 'ES', 'SWE': 'SE'
}

EU27_ISO2 = set(ISO3_TO_ISO2.values())  # includes 'GR' for Greece


# ============================================================
# HELPERS
# ============================================================

def melt_eurostat(df, id_cols, value_name):
    """Melt Eurostat wide format (years as columns) to long format."""
    # The first id column contains 'geo\\TIME_PERIOD' — extract geo from it
    first_col = df.columns[len(id_cols) - 1]  # last id col has geo
    year_cols = [c for c in df.columns if c not in id_cols and c != first_col]

    # Identify year columns (numeric strings)
    actual_year_cols = []
    for c in df.columns:
        if c not in id_cols:
            try:
                int(c)
                actual_year_cols.append(c)
            except (ValueError, TypeError):
                pass

    if not actual_year_cols:
        return pd.DataFrame()

    melted = df.melt(
        id_vars=id_cols,
        value_vars=actual_year_cols,
        var_name='year',
        value_name=value_name
    )
    melted['year'] = melted['year'].astype(int)
    return melted


def read_eurostat_csv(filename):
    """Read Eurostat CSV and parse the geo column."""
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        print(f"  File not found: {filename}")
        return None
    df = pd.read_csv(path)
    # Find the column with geo\TIME_PERIOD pattern
    geo_col = [c for c in df.columns if 'geo' in c.lower()]
    if geo_col:
        df = df.rename(columns={geo_col[0]: 'geo'})
    return df


# ============================================================
# 1. ICT SPECIALISTS (% of employment) — TARGET VARIABLE
# ============================================================

def process_ict_specialists_pct():
    """isoc_sks_itspt: ICT specialists as % of total employment."""
    print("Processing: ICT specialists (% of employment)...")
    df = read_eurostat_csv("eurostat_isoc_sks_itspt.csv")
    if df is None:
        return pd.DataFrame()

    # Filter: unit = PC_EMP (percentage)
    df = df[df['unit'] == 'PC_EMP'].copy()

    # Keep only EU27 countries
    df = df[df['geo'].isin(EU27_CODES)].copy()

    # Normalize Greece
    df['geo'] = df['geo'].replace(GEO_RENAME)

    # Melt
    id_cols = ['freq', 'unit', 'geo']
    result = melt_eurostat(df, id_cols, 'ict_specialists_pct')
    result = result[['geo', 'year', 'ict_specialists_pct']]

    # Convert to numeric
    result['ict_specialists_pct'] = pd.to_numeric(result['ict_specialists_pct'], errors='coerce')

    print(f"  -> {len(result)} rows, years: {result['year'].min()}-{result['year'].max()}")
    return result


# ============================================================
# 2. ICT SPECIALISTS (thousands) — ADDITIONAL TARGET
# ============================================================

def process_ict_specialists_ths():
    """isoc_sks_itspt: ICT specialists in thousands."""
    print("Processing: ICT specialists (thousands)...")
    df = read_eurostat_csv("eurostat_isoc_sks_itspt.csv")
    if df is None:
        return pd.DataFrame()

    df = df[df['unit'] == 'THS_PER'].copy()
    df = df[df['geo'].isin(EU27_CODES)].copy()
    df['geo'] = df['geo'].replace(GEO_RENAME)

    id_cols = ['freq', 'unit', 'geo']
    result = melt_eurostat(df, id_cols, 'ict_specialists_ths')
    result = result[['geo', 'year', 'ict_specialists_ths']]
    result['ict_specialists_ths'] = pd.to_numeric(result['ict_specialists_ths'], errors='coerce')

    print(f"  -> {len(result)} rows, years: {result['year'].min()}-{result['year'].max()}")
    return result


# ============================================================
# 3. HIGH-TECH EMPLOYMENT
# ============================================================

def process_hightech_employment():
    """htec_emp_nat2: Employment in high-tech sectors (% of total, both sexes)."""
    print("Processing: High-tech employment...")
    df = read_eurostat_csv("eurostat_htec_emp_nat2.csv")
    if df is None:
        return pd.DataFrame()

    # Filter: total sex (T), unit PC_EMP, nace_r2 = HTC (all high-tech sectors)
    df = df[
        (df['sex'] == 'T') &
        (df['unit'] == 'PC_EMP') &
        (df['nace_r2'] == 'HTC')
    ].copy()

    df = df[df['geo'].isin(EU27_CODES)].copy()
    df['geo'] = df['geo'].replace(GEO_RENAME)

    id_cols = ['freq', 'sex', 'nace_r2', 'unit', 'geo']
    result = melt_eurostat(df, id_cols, 'hightech_emp_pct')
    result = result[['geo', 'year', 'hightech_emp_pct']]
    result['hightech_emp_pct'] = pd.to_numeric(result['hightech_emp_pct'], errors='coerce')

    print(f"  -> {len(result)} rows, years: {result['year'].min()}-{result['year'].max()}")
    return result


# ============================================================
# 4. GDP PER CAPITA (Eurostat)
# ============================================================

def process_gdp_eurostat():
    """nama_10_pc: GDP per capita in EUR (chain linked volumes, 2010)."""
    print("Processing: GDP per capita (Eurostat)...")
    df = read_eurostat_csv("eurostat_nama_10_pc.csv")
    if df is None:
        return pd.DataFrame()

    # Filter: unit = CLV10_EUR_HAB, na_item = B1GQ (GDP)
    df = df[
        (df['unit'] == 'CLV10_EUR_HAB') &
        (df['na_item'] == 'B1GQ')
    ].copy()

    df = df[df['geo'].isin(EU27_CODES)].copy()
    df['geo'] = df['geo'].replace(GEO_RENAME)

    id_cols = ['freq', 'unit', 'na_item', 'geo']
    result = melt_eurostat(df, id_cols, 'gdp_pc_eur')
    result = result[['geo', 'year', 'gdp_pc_eur']]
    result['gdp_pc_eur'] = pd.to_numeric(result['gdp_pc_eur'], errors='coerce')

    # Keep only 2004+
    result = result[result['year'] >= 2004]

    print(f"  -> {len(result)} rows, years: {result['year'].min()}-{result['year'].max()}")
    return result


# ============================================================
# 5. R&D EXPENDITURE (Eurostat)
# ============================================================

def process_rd_eurostat():
    """rd_e_gerdtot: GERD as % of GDP."""
    print("Processing: R&D expenditure (Eurostat)...")
    df = read_eurostat_csv("eurostat_rd_e_gerdtot.csv")
    if df is None:
        return pd.DataFrame()

    # Filter: unit = PC_GDP, sectperf = TOTAL
    df = df[
        (df['unit'] == 'PC_GDP') &
        (df['sectperf'] == 'TOTAL')
    ].copy()

    df = df[df['geo'].isin(EU27_CODES)].copy()
    df['geo'] = df['geo'].replace(GEO_RENAME)

    id_cols = ['freq', 'sectperf', 'unit', 'geo']
    result = melt_eurostat(df, id_cols, 'rd_pct_gdp')
    result = result[['geo', 'year', 'rd_pct_gdp']]
    result['rd_pct_gdp'] = pd.to_numeric(result['rd_pct_gdp'], errors='coerce')
    result = result[result['year'] >= 2004]

    print(f"  -> {len(result)} rows, years: {result['year'].min()}-{result['year'].max()}")
    return result


# ============================================================
# 6. INTERNET USE BY INDIVIDUALS (Eurostat)
# ============================================================

def process_internet_use():
    """isoc_ci_ifp_iu: Individuals who used the internet in the last 12 months (%)."""
    print("Processing: Internet use by individuals (Eurostat)...")
    df = read_eurostat_csv("eurostat_isoc_ci_ifp_iu.csv")
    if df is None:
        return pd.DataFrame()

    # Filter: indic_is = I_IU3 (used internet last 3 months — most common),
    # ind_type = IND_TOTAL or CB_NAT (all individuals / nationals), unit = PC_IND
    # Try IND_TOTAL first, fall back to CB_NAT
    ind_types_priority = ['IND_TOTAL', 'CB_NAT']
    filtered = pd.DataFrame()
    for ind_type in ind_types_priority:
        filtered = df[
            (df['indic_is'] == 'I_IU3') &
            (df['ind_type'] == ind_type) &
            (df['unit'] == 'PC_IND')
        ].copy()
        if len(filtered) > 0:
            break
    df = filtered

    df = df[df['geo'].isin(EU27_CODES)].copy()
    df['geo'] = df['geo'].replace(GEO_RENAME)

    id_cols = ['freq', 'indic_is', 'unit', 'ind_type', 'geo']
    result = melt_eurostat(df, id_cols, 'internet_use_pct')
    result = result[['geo', 'year', 'internet_use_pct']]
    result['internet_use_pct'] = pd.to_numeric(result['internet_use_pct'], errors='coerce')

    print(f"  -> {len(result)} rows, years: {result['year'].min()}-{result['year'].max()}")
    return result


# ============================================================
# 7. WORLD BANK INDICATORS
# ============================================================

def process_world_bank():
    """Process all World Bank CSV files."""
    print("Processing: World Bank indicators...")

    wb_files = {
        'wb_BX_GSR_CCIS_ZS.csv': 'ict_service_exports_pct',
        'wb_TX_VAL_TECH_MF_ZS.csv': 'hightech_exports_pct',
        'wb_IT_NET_USER_ZS.csv': 'wb_internet_users_pct',
        'wb_SL_SRV_EMPL_ZS.csv': 'services_employment_pct',
        'wb_SL_IND_EMPL_ZS.csv': 'industry_employment_pct',
        'wb_SL_TLF_ACTI_ZS.csv': 'labor_force_participation_pct',
        'wb_NY_GDP_PCAP_CD.csv': 'gdp_pc_usd',
        'wb_GB_XPD_RSDV_GD_ZS.csv': 'wb_rd_pct_gdp',
        'wb_SE_TER_ENRR.csv': 'tertiary_education_pct',
        'wb_IT_NET_BBND_P2.csv': 'broadband_per100',
    }

    all_dfs = []

    for filename, col_name in wb_files.items():
        path = os.path.join(DATA_DIR, filename)
        if not os.path.exists(path):
            print(f"  File not found: {filename}")
            continue

        df = pd.read_csv(path)

        # Convert ISO-3 to ISO-2
        df['geo'] = df['economy'].map(ISO3_TO_ISO2)
        df = df.dropna(subset=['geo'])

        # Melt year columns (YR2010, YR2011, ...)
        year_cols = [c for c in df.columns if c.startswith('YR')]

        melted = df.melt(
            id_vars=['geo'],
            value_vars=year_cols,
            var_name='year',
            value_name=col_name
        )
        melted['year'] = melted['year'].str.replace('YR', '').astype(int)
        melted[col_name] = pd.to_numeric(melted[col_name], errors='coerce')

        melted = melted[['geo', 'year', col_name]]
        all_dfs.append(melted)
        print(f"  {col_name}: {melted[col_name].notna().sum()} values")

    if not all_dfs:
        return pd.DataFrame()

    # Merge all WB indicators
    result = all_dfs[0]
    for df in all_dfs[1:]:
        result = result.merge(df, on=['geo', 'year'], how='outer')

    print(f"  -> Total: {len(result)} rows")
    return result


# ============================================================
# 8. MERGE EVERYTHING
# ============================================================

def merge_all():
    """Merge all datasets into a single panel."""
    print("\n" + "=" * 60)
    print("BUILDING THE PANEL DATASET")
    print("=" * 60 + "\n")

    # Process all sources
    ict_pct = process_ict_specialists_pct()
    ict_ths = process_ict_specialists_ths()
    hightech = process_hightech_employment()
    gdp = process_gdp_eurostat()
    rd = process_rd_eurostat()
    internet = process_internet_use()
    wb = process_world_bank()

    # Start with ICT specialists (target variable)
    panel = ict_pct.merge(ict_ths, on=['geo', 'year'], how='outer')

    # Merge Eurostat features
    for df in [hightech, gdp, rd, internet]:
        if not df.empty:
            panel = panel.merge(df, on=['geo', 'year'], how='left')

    # Merge World Bank features
    if not wb.empty:
        panel = panel.merge(wb, on=['geo', 'year'], how='left')

    # Filter: keep only EU27 and years with target variable
    panel = panel[panel['geo'].isin(EU27_ISO2)].copy()
    panel = panel.sort_values(['geo', 'year']).reset_index(drop=True)

    # Add EU27 aggregate row indicator
    # (EU27_2020 was in raw data but we filtered it out — could add back if needed)

    # Country names for convenience
    country_names = {
        'AT': 'Austria', 'BE': 'Belgium', 'BG': 'Bulgaria', 'HR': 'Croatia',
        'CY': 'Cyprus', 'CZ': 'Czechia', 'DK': 'Denmark', 'EE': 'Estonia',
        'FI': 'Finland', 'FR': 'France', 'DE': 'Germany', 'GR': 'Greece',
        'HU': 'Hungary', 'IE': 'Ireland', 'IT': 'Italy', 'LV': 'Latvia',
        'LT': 'Lithuania', 'LU': 'Luxembourg', 'MT': 'Malta', 'NL': 'Netherlands',
        'PL': 'Poland', 'PT': 'Portugal', 'RO': 'Romania', 'SK': 'Slovakia',
        'SI': 'Slovenia', 'ES': 'Spain', 'SE': 'Sweden'
    }
    panel['country'] = panel['geo'].map(country_names)

    # Reorder columns
    cols_order = [
        'geo', 'country', 'year',
        # Targets
        'ict_specialists_pct', 'ict_specialists_ths',
        # Eurostat features
        'hightech_emp_pct', 'gdp_pc_eur', 'rd_pct_gdp', 'internet_use_pct',
        # World Bank features
        'ict_service_exports_pct', 'hightech_exports_pct',
        'services_employment_pct', 'industry_employment_pct',
        'labor_force_participation_pct', 'gdp_pc_usd',
        'wb_rd_pct_gdp', 'tertiary_education_pct', 'broadband_per100',
        'wb_internet_users_pct',
    ]
    cols_order = [c for c in cols_order if c in panel.columns]
    panel = panel[cols_order]

    return panel


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    panel = merge_all()

    # Save
    output_path = os.path.join(DATA_DIR, "panel_dataset.csv")
    panel.to_csv(output_path, index=False)

    # Summary
    print("\n" + "=" * 60)
    print("PANEL DATASET SUMMARY")
    print("=" * 60)
    print(f"Size: {panel.shape[0]} rows x {panel.shape[1]} columns")
    print(f"Countries: {panel['geo'].nunique()} ({', '.join(sorted(panel['geo'].unique()))})")
    print(f"Years: {panel['year'].min()}-{panel['year'].max()}")
    print(f"\nColumns:")
    for col in panel.columns:
        non_null = panel[col].notna().sum()
        pct = non_null / len(panel) * 100
        print(f"  {col:35s} {non_null:5d} values ({pct:5.1f}%)")

    print(f"\nSaved: {output_path}")

    # Preview
    print("\n--- First 10 rows ---")
    print(panel.head(10).to_string(index=False))

    # Missing data summary per country
    print("\n--- Coverage of the target variable (ict_specialists_pct) ---")
    target_coverage = panel.groupby('geo')['ict_specialists_pct'].apply(
        lambda x: f"{x.notna().sum()}/{len(x)} years"
    )
    print(target_coverage.to_string())
