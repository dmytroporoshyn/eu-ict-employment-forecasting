"""Рисунок 3 статті: частка ІКТ-спеціалістів у 2011 та 2024 роках.

2011 рік обрано як перший рік після методологічного розриву ряду Eurostat
(перехід на класифікацію професій ISCO-08), тому порівняння 2011 -> 2024
не містить артефакту розриву. Стиль повторює fig2 з 03_eda.py.
-> output/fig2_ict_2011_vs_2024.png
"""
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use('Agg')
BASE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE, "data", "panel_dataset.csv"))

comparison = df[df['year'].isin([2011, 2024])].pivot_table(
    index='country', columns='year', values='ict_specialists_pct').dropna()
comparison = comparison.sort_values(2024, ascending=True)

fig, ax = plt.subplots(figsize=(12, 8))
y_pos = np.arange(len(comparison))
ax.barh(y_pos - 0.15, comparison[2011], height=0.3, color='#4e79a7', label='2011')
ax.barh(y_pos + 0.15, comparison[2024], height=0.3, color='#e15759', label='2024')
ax.set_yticks(y_pos)
ax.set_yticklabels(comparison.index, fontsize=9)
ax.set_xlabel('ICT Specialists (% of employment)')
ax.set_title('ICT Specialists Share: 2011 vs 2024')
ax.legend()
ax.grid(True, axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(BASE, "output", "fig2_ict_2011_vs_2024.png"), bbox_inches='tight')
print("Saved: output/fig2_ict_2011_vs_2024.png")
