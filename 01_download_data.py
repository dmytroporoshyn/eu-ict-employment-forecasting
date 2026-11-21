"""Download the Eurostat and World Bank data used in the study."""

import os
import pandas as pd
import eurostat
import wbgapi as wb

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

EU27_CODES = [
    'AT', 'BE', 'BG', 'HR', 'CY', 'CZ', 'DK', 'EE', 'FI', 'FR',
    'DE', 'GR', 'HU', 'IE', 'IT', 'LV', 'LT', 'LU', 'MT', 'NL',
    'PL', 'PT', 'RO', 'SK', 'SI', 'ES', 'SE'
]

# ISO-2 -> ISO-3 mapping for World Bank
ISO2_TO_ISO3 = {
    'AT': 'AUT', 'BE': 'BEL', 'BG': 'BGR', 'HR': 'HRV', 'CY': 'CYP',
    'CZ': 'CZE', 'DK': 'DNK', 'EE': 'EST', 'FI': 'FIN', 'FR': 'FRA',
    'DE': 'DEU', 'GR': 'GRC', 'HU': 'HUN', 'IE': 'IRL', 'IT': 'ITA',
    'LV': 'LVA', 'LT': 'LTU', 'LU': 'LUX', 'MT': 'MLT', 'NL': 'NLD',
    'PL': 'POL', 'PT': 'PRT', 'RO': 'ROU', 'SK': 'SVK', 'SI': 'SVN',
    'ES': 'ESP', 'SE': 'SWE'
}
ISO3_TO_ISO2 = {v: k for k, v in ISO2_TO_ISO3.items()}

# ============================================================
# 1. EUROSTAT
# ============================================================

def download_eurostat_dataset(code, name):
    """Download one Eurostat dataset as a DataFrame."""
    print(f"  Downloading Eurostat: {name} ({code})...")
    try:
        df = eurostat.get_data_df(code)
        print(f"    -> {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except Exception as e:
        print(f"    -> ERROR: {e}")
        return None


def download_eurostat():
    """Download all Eurostat datasets."""
    print("\n" + "=" * 60)
    print("EUROSTAT: downloading datasets")
    print("=" * 60)

    datasets = {
        # ICT specialists as % of total employment
        "isoc_sks_itspt": "ICT specialists in employment",
        # Employed ICT specialists - total
        "isoc_sks_itsps": "ICT specialists supply",
        # Employment in high-tech sectors
        "htec_emp_nat2": "Employment in high-tech sectors",
        # Individuals' level of digital skills
        "isoc_sk_dskl_i21": "Digital skills of individuals",
        # GDP per capita
        "nama_10_pc": "GDP per capita",
        # R&D expenditure (GERD)
        "rd_e_gerdtot": "R&D expenditure (GERD)",
        # ICT sector value added
        "isoc_bde15dec": "ICT sector - enterprises",
        # Individuals using the internet
        "isoc_ci_ifp_iu": "Internet use by individuals",
    }

    results = {}
    for code, name in datasets.items():
        df = download_eurostat_dataset(code, name)
        if df is not None:
            filename = f"eurostat_{code}.csv"
            df.to_csv(os.path.join(OUTPUT_DIR, filename), index=False)
            results[code] = df
            print(f"    -> Saved: {filename}")

    return results


# ============================================================
# 2. WORLD BANK
# ============================================================

def download_wb_indicator(indicator_code, indicator_name, economies, time_range):
    """Download one World Bank indicator."""
    print(f"  Downloading WB: {indicator_name} ({indicator_code})...")
    try:
        df = wb.data.DataFrame(
            indicator_code,
            economy=economies,
            time=time_range,
            labels=True,
        )
        df = df.reset_index()
        print(f"    -> {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except Exception as e:
        print(f"    -> ERROR: {e}")
        return None


def download_world_bank():
    """Download all World Bank indicators."""
    print("\n" + "=" * 60)
    print("WORLD BANK: downloading indicators")
    print("=" * 60)

    economies = list(ISO2_TO_ISO3.values())
    time_range = range(2010, 2025)

    indicators = {
        # ICT service exports (% of service exports)
        "BX.GSR.CCIS.ZS": "ICT service exports (% of service exports)",
        # High-technology exports (% of manufactured exports)
        "TX.VAL.TECH.MF.ZS": "High-tech exports (% of manufactured exports)",
        # Individuals using the Internet (% of population)
        "IT.NET.USER.ZS": "Individuals using Internet (%)",
        # Employment in services (% of total employment) (ILO)
        "SL.SRV.EMPL.ZS": "Employment in services (%)",
        # Employment in industry (% of total employment) (ILO)
        "SL.IND.EMPL.ZS": "Employment in industry (%)",
        # Labor force participation rate (% of total population ages 15+)
        "SL.TLF.ACTI.ZS": "Labor force participation rate (%)",
        # GDP per capita (current US$)
        "NY.GDP.PCAP.CD": "GDP per capita (current US$)",
        # Research and development expenditure (% of GDP)
        "GB.XPD.RSDV.GD.ZS": "R&D expenditure (% of GDP)",
        # Tertiary education enrollment (% gross)
        "SE.TER.ENRR": "Tertiary education enrollment (%)",
        # Fixed broadband subscriptions (per 100 people)
        "IT.NET.BBND.P2": "Fixed broadband subscriptions (per 100)",
    }

    results = {}
    for code, name in indicators.items():
        df = download_wb_indicator(code, name, economies, time_range)
        if df is not None:
            safe_name = code.replace(".", "_")
            filename = f"wb_{safe_name}.csv"
            df.to_csv(os.path.join(OUTPUT_DIR, filename), index=False)
            results[code] = df
            print(f"    -> Saved: {filename}")

    return results


# ============================================================
# 3. MAIN
# ============================================================

if __name__ == "__main__":
    print("Downloading data...")
    print(f"Output folder: {OUTPUT_DIR}")

    eurostat_data = download_eurostat()
    wb_data = download_world_bank()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Eurostat: {len(eurostat_data)} datasets downloaded")
    print(f"World Bank: {len(wb_data)} indicators downloaded")

    csv_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.csv')]
    print(f"\nCSV files in data/: {len(csv_files)}")
    for f in sorted(csv_files):
        size = os.path.getsize(os.path.join(OUTPUT_DIR, f))
        print(f"  {f} ({size / 1024:.1f} KB)")

    print("\nDone.")
