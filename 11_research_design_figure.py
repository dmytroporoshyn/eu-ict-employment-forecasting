"""Research design figure (Ukrainian version) -> output/fig0_research_design.png"""
import os

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

matplotlib.use('Agg')
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

INK = '#1f2a36'
EDGE = '#4e6a85'
FILL = '#eef3f8'
FILL_STAT = '#e6eef7'
FILL_ML = '#f7ebe9'


def box(ax, x, y, w, h, title, body, fill=FILL):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.015",
                                linewidth=1.3, edgecolor=EDGE, facecolor=fill))
    ax.text(x + w / 2, y + h - 0.022, title, ha='center', va='top',
            fontsize=11.5, fontweight='bold', color=INK)
    ax.text(x + w / 2, y + h - 0.062, body, ha='center', va='top',
            fontsize=9.6, color=INK, linespacing=1.4)


def arrow(ax, x0, y0, x1, y1):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='-|>', color=EDGE, lw=1.4, mutation_scale=14,
                                shrinkA=2, shrinkB=2))


fig, ax = plt.subplots(figsize=(10, 8.6))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

X, W, G = 0.04, 0.92, 0.04          # left margin, width, gap between boxes
y = 0.995
h = 0.125
box(ax, X, y - h, W, h, 'Етап 1. Формування панельного датасету',
    'Eurostat, World Bank: 27 країн ЄС × 2004–2024 рр.\n'
    'цільова змінна — частка ІКТ-спеціалістів у зайнятості (%), 8 екзогенних предикторів')
arrow(ax, 0.5, y - h, 0.5, y - h - G)
y -= h + G
h = 0.2
half = (W - 0.02) / 2
box(ax, X, y - h, half, h, 'Етап 2а. Локальні моделі',
    'окрема модель для кожної країни,\nлише власна історія ряду\n\n'
    'статистичні: ARIMA(1,1,1), ETS\nML: Prophet', fill=FILL_STAT)
box(ax, X + half + 0.02, y - h, half, h, 'Етап 2б. Панельна модель',
    'одна модель для всіх 27 країн, 16 ознак:\nлаги, ковзні середні, екзогенні змінні\n\n'
    'ML: дерево рішень → випадковий ліс\n→ XGBoost', fill=FILL_ML)
arrow(ax, X + half / 2, y - h, 0.42, y - h - G)
arrow(ax, X + half * 1.5 + 0.02, y - h, 0.58, y - h - G)
y -= h + G
h = 0.15
box(ax, X, y - h, W, h, 'Етап 3. Оцінювання точності',
    'крос-валідація з розширюваним вікном: 12 точок прогнозу, тестові роки 2013–2024\n'
    '(однакові для всіх моделей), прогноз на 1 рік уперед, 323 спільні прогнози\n'
    "метрики MAE, RMSE, MAPE  •  тест Дібольда-Маріано (об'єднаний і панельний)")
arrow(ax, 0.5, y - h, 0.5, y - h - G)
y -= h + G
h = 0.125
box(ax, X, y - h, W, h, 'Етап 4. Інтерпретація та комбінування',
    'SHAP-декомпозиція моделі XGBoost: внесок авторегресійних та екзогенних ознак\n'
    'ансамбль ARIMA + XGBoost (рівні ваги): поєднання локальної та панельної моделей')
arrow(ax, 0.5, y - h, 0.5, y - h - G)
y -= h + G
h = 0.15
box(ax, X, y - h, W, h, 'Етап 5. Прогноз і сценарії на 2025–2029 рр.',
    'ітеративний прогноз усіма моделями\n'
    'три сценарії: 50%, 100% і 150% приросту базової траєкторії (ARIMA + XGBoost)\n'
    'для кожної країни та для агрегату ЄС-27')

plt.savefig(os.path.join(OUT, "fig0_research_design.png"), bbox_inches='tight', dpi=170)
print("Saved: output/fig0_research_design.png")
