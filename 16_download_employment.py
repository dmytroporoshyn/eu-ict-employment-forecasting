"""Download Eurostat lfsa_egan (employment) and proj_23np (EUROPOP2023) for the
employment-weighted EU aggregate.

Output: data/eurostat_lfsa_egan.csv, data/eurostat_proj_23np.csv,
data/download_log_revision.txt
"""
import datetime as dt
import os

import eurostat
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")
EU27 = ['AT', 'BE', 'BG', 'HR', 'CY', 'CZ', 'DK', 'EE', 'FI', 'FR', 'DE', 'EL', 'HU', 'IE',
        'IT', 'LV', 'LT', 'LU', 'MT', 'NL', 'PL', 'PT', 'RO', 'SK', 'SI', 'ES', 'SE']


def main():
    emp = eurostat.get_data_df('lfsa_egan', filter_pars={
        'sex': 'T', 'citizen': 'TOTAL', 'age': ['Y15-74', 'Y20-64'], 'geo': EU27})
    emp.to_csv(os.path.join(DATA, "eurostat_lfsa_egan.csv"), index=False)
    pop = eurostat.get_data_df('proj_23np', filter_pars={
        'projection': 'BSL', 'sex': 'T', 'age': ['Y15-74', 'Y20-64'], 'geo': EU27})
    pop.to_csv(os.path.join(DATA, "eurostat_proj_23np.csv"), index=False)
    with open(os.path.join(DATA, "download_log_revision.txt"), "w", encoding="utf-8") as fh:
        fh.write(f"lfsa_egan and proj_23np downloaded on {dt.date.today().isoformat()}\n")
    print(emp.shape, pop.shape)


if __name__ == "__main__":
    main()
