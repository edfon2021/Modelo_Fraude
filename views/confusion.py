# views/confusion.py
import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from db import fetch_df
from views.overview import _ensure_conn

NEG_COLOR = "#2ECC71"   # verde (clase 0)
POS_COLOR = "#F1C40F"   # amarillo (clase 1)


def _fig_to_png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    plt.close(fig)
    return buf.getvalue()


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def _add_bar_labels(ax, fmt="{:,.0f}", fontsize=8, pad=3):
    """
    Pone etiquetas encima de barras (valores).
    Funciona para conteos (TP/FP/FN/TN) y métricas (0-1) si le pasás otro fmt.
    """
    for p in ax.patches:
        h = p.get_height()
        ax.annotate(
            fmt.format(h),
            (p.get_x() + p.get_width() / 2.0, h),
            ha="center",
            va="bottom",
            fontsize=fontsize,
            xytext=(0, pad),
            textcoords="offset points",
        )


def render(cfg: dict):

    st.subheader("4️⃣ Matriz de Confusión + Métricas (modelo actual)")
    st.caption("Cálculos basados en: y_true, y_pred, y_proba (ordenamiento para Top-K).")

    if not cfg.get("conn_ok"):
        st.warning("Primero conectate a la base de datos.")
        return

    pred_table = cfg.get("pred_table", "pred_rf")
    conn = _ensure_conn(cfg)

    sql = f'''
        SELECT
            COALESCE(y_true, 0) AS y_true,
            COALESCE(y_pred, 0) AS y_pred,
            COALESCE(y_proba, 0) AS y_proba
        FROM "{pred_table}";
    '''
    df = fetch_df(conn, sql)

    if df is None or df.empty:
        st.warning(f"No hay datos en {pred_table}.")
        return

    df["y_true"] = pd.to_numeric(df["y_true"], errors="coerce").fillna(0).astype(int)
    df["y_pred"] = pd.to_numeric(df["y_pred"], errors="coerce").fillna(0).astype(int)
    df["y_proba"] = pd.to_numeric(df["y_proba"], errors="coerce").fillna(0.0)

    y_true = df["y_true"].to_numpy()
    y_pred = df["y_pred"].to_numpy()

    # Confusion counts
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())

    total = int(len(df))
    pos = int((y_true == 1).sum())
    neg = total - pos

    # Métricas clásicas
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)         # sensibilidad / TPR
    sensitivity = recall
    specificity = _safe_div(tn, tn + fp)    # TNR
    f1 = _safe_div(2 * precision * recall, precision + recall) if (precision + recall) > 0 else 0.0
    gmean = float(np.sqrt(sensitivity * specificity))

    # --- Top-K
    st.markdown("### 🎯 Top-K (capacidad de revisión)")
    k_pct = st.slider("K% a revisar", 1, 50, 10)
    k = max(1, int(np.ceil(total * (k_pct / 100.0))))

    df_rank = df.sort_values("y_proba", ascending=False).reset_index(drop=True)
    topk = df_rank.iloc[:k]

    # Operativo: de lo revisado, cuánto realmente era fraude
    precision_at_k = _safe_div(int((topk["y_true"] == 1).sum()), k)
    recall_at_k = _safe_div(int((topk["y_true"] == 1).sum()), pos)

    # ----------------------------
    # KPIs
    # ----------------------------
    st.markdown("### 📌 KPIs")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("TP", f"{tp:,}")
    c2.metric("FP", f"{fp:,}")
    c3.metric("FN", f"{fn:,}")
    c4.metric("TN", f"{tn:,}")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Sensibilidad (TPR)", f"{sensitivity*100:,.2f}%")
    m2.metric("Especificidad (TNR)", f"{specificity*100:,.2f}%")
    m3.metric("Precision", f"{precision*100:,.2f}%")
    m4.metric("F1-score", f"{f1*100:,.2f}%")

    m5, m6, m7, m8 = st.columns(4)
    m5.metric("G-Mean", f"{gmean:,.4f}")
    m6.metric(f"Precision@{k_pct}%", f"{precision_at_k*100:,.2f}%")
    m7.metric(f"Recall@{k_pct}%", f"{recall_at_k*100:,.2f}%")
    m8.metric("Soporte (Fraudes/Normales)", f"{pos:,}/{neg:,}")

    # ===============================
    # GRÁFICOS EN UNA SOLA LÍNEA
    # ===============================
    st.markdown("### 📊 Visualizaciones (compactas)")
    col1, col2, col3 = st.columns(3)

    # 1) Confusion bars
    with col1:
        fig1 = plt.figure(figsize=(3.2, 2.0), dpi=110)
        ax1 = fig1.add_subplot(111)

        labels = ["TP", "FP", "FN", "TN"]
        values = [tp, fp, fn, tn]
        colors = [POS_COLOR, NEG_COLOR, POS_COLOR, NEG_COLOR]

        ax1.bar(labels, values, color=colors)
        ax1.set_title("Confusion", fontsize=9)
        ax1.tick_params(labelsize=8)
        ax1.grid(axis="y", alpha=0.2)

        # etiquetas encima (conteos)
        _add_bar_labels(ax1, fmt="{:,.0f}", fontsize=8, pad=2)

        fig1.tight_layout()
        st.image(_fig_to_png(fig1))

    # 2) TPR vs TNR
    with col2:
        fig2 = plt.figure(figsize=(3.2, 2.0), dpi=110)
        ax2 = fig2.add_subplot(111)

        ax2.bar(["TPR", "TNR"], [sensitivity, specificity], color=[POS_COLOR, NEG_COLOR])
        ax2.set_ylim(0, 1)
        ax2.set_title("TPR vs TNR", fontsize=9)
        ax2.tick_params(labelsize=8)
        ax2.grid(axis="y", alpha=0.2)

        # etiquetas encima (porcentaje)
        _add_bar_labels(ax2, fmt="{:.2f}", fontsize=8, pad=2)

        fig2.tight_layout()
        st.image(_fig_to_png(fig2))

    # 3) Top-K
    with col3:
        fig3 = plt.figure(figsize=(3.2, 2.0), dpi=110)
        ax3 = fig3.add_subplot(111)

        ax3.bar([f"P@{k_pct}%", f"R@{k_pct}%"], [precision_at_k, recall_at_k], color=[POS_COLOR, POS_COLOR])
        ax3.set_ylim(0, 1)
        ax3.set_title("Top-K", fontsize=9)
        ax3.tick_params(labelsize=8)
        ax3.grid(axis="y", alpha=0.2)

        # etiquetas encima (porcentaje)
        _add_bar_labels(ax3, fmt="{:.2f}", fontsize=8, pad=2)

        fig3.tight_layout()
        st.image(_fig_to_png(fig3))

    # ===============================
    # MATRIZ DE CONFUSIÓN 2x2 (tabla)
    # ===============================
    st.markdown("### 🧩 Matriz de Confusión (2x2)")
    cm = pd.DataFrame(
        [[tn, fp], [fn, tp]],
        index=["Real: 0 (Normal)", "Real: 1 (Fraude)"],
        columns=["Pred: 0", "Pred: 1"],
    )

    # La mostramos con formato bonito
    st.dataframe(cm, use_container_width=True)

    with st.expander("Ver detalle numérico"):
        st.write(
            {
                "total_tx": total,
                "pos_fraud": pos,
                "neg_normal": neg,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "precision": precision,
                "sensitivity/recall": sensitivity,
                "specificity": specificity,
                "f1": f1,
                "gmean": gmean,
                "k_pct": k_pct,
                "k": k,
                "precision_at_k": precision_at_k,
                "recall_at_k": recall_at_k,
            }
        )