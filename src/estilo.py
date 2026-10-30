"""Estilo único dos gráficos: cores por função, eixos e grade recessivos."""
import warnings

import matplotlib.pyplot as plt

AZUL = "#2a78d6"      # série principal / "no prazo"
LARANJA = "#eb6834"   # segunda série / "atrasado"
CINZA = "#898781"     # referência (linhas de média, baseline)
TINTA = "#0b0b0b"
TINTA_2 = "#52514e"


def aplicar_estilo() -> None:
    # Rodando como script com MPLBACKEND=Agg, plt.show() não tem janela; as figuras
    # são salvas em reports/figuras de qualquer jeito.
    warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive")
    plt.rcParams.update({
        "figure.facecolor": "#fcfcfb",
        "axes.facecolor": "#fcfcfb",
        "axes.edgecolor": "#c3c2b7",
        "axes.labelcolor": TINTA_2,
        "axes.titlecolor": TINTA,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": "#e1e0d9",
        "grid.linewidth": 0.8,
        "xtick.color": CINZA,
        "ytick.color": CINZA,
        "lines.linewidth": 2,
        "legend.frameon": False,
        "savefig.dpi": 120,
        "savefig.bbox": "tight",
    })
