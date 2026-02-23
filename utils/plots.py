import matplotlib.pyplot as plt
import numpy as np


def plot_muestreo_por_mes_barras(
    resumen,
    color_fraude="tab:orange",
    color_normales="tab:green",
    alpha_fraude=0.85,
    alpha_normales=0.85,
    log_scale=True,
    y_bottom=1,
    top_mult=2.3,
    figsize=(12, 6),
    title="Distribución mensual tras undersampling estratificado",
    show_values=True,
    value_fontsize=9,
):
    meses = resumen.index.astype(str)
    x = np.arange(len(meses))
    width = 0.4

    fraudes = resumen["fraudes"].values
    normales = resumen["normales_tomar"].values

    fig, ax = plt.subplots(figsize=figsize)

    bars_f = ax.bar(
        x - width / 2,
        fraudes,
        width=width,
        color=color_fraude,
        alpha=alpha_fraude,
        label="Fraudes",
    )

    bars_n = ax.bar(
        x + width / 2,
        normales,
        width=width,
        color=color_normales,
        alpha=alpha_normales,
        label="Normales muestreados",
    )

    # --- Escala
    if log_scale:
        ax.set_yscale("log")
        ymax = max(np.max(fraudes), np.max(normales)) if len(fraudes) else 1
        ax.set_ylim(bottom=y_bottom, top=ymax * top_mult)
        ax.set_ylabel("Número de transacciones (escala log)")
    else:
        ymax = max(np.max(fraudes), np.max(normales)) if len(fraudes) else 1
        ax.set_ylim(top=ymax * 1.15)
        ax.set_ylabel("Número de transacciones")

    ax.set_title(title)
    ax.set_xlabel("Mes")
    ax.set_xticks(x)
    ax.set_xticklabels(meses, rotation=45)
    ax.legend()

    if show_values:

        def fmt(v):
            return f"{int(v):,}".replace(",", ".")

        offset_mult = 1.28 if log_scale else 1.05

        for b in list(bars_f) + list(bars_n):
            h = b.get_height()
            if h <= 0:
                continue
            y = h * offset_mult
            ax.text(
                b.get_x() + b.get_width() / 2,
                y,
                fmt(h),
                ha="center",
                va="bottom",
                fontsize=value_fontsize,
                clip_on=False,
            )

    plt.tight_layout()
    return fig  