# views/roc.py
import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, roc_auc_score

from db import fetch_df
from views.overview import _ensure_conn


# ----------------------------------------
# Utils
# ----------------------------------------
def _fig_to_png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    return buf.getvalue()


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def _load_pred_table(conn, table_name: str) -> pd.DataFrame:
    sql = f'''
        SELECT
            COALESCE(y_true, 0) AS y_true,
            COALESCE(y_proba, 0) AS y_proba
        FROM "{table_name}";
    '''
    df = fetch_df(conn, sql)

    if df is None or df.empty:
        return pd.DataFrame()

    df["y_true"] = pd.to_numeric(df["y_true"], errors="coerce").fillna(0).astype(int)
    df["y_proba"] = pd.to_numeric(df["y_proba"], errors="coerce").fillna(0.0)
    df["y_proba"] = df["y_proba"].clip(0.0, 1.0)

    return df


def _roc_metrics(df: pd.DataFrame):
    y_true = df["y_true"].to_numpy()
    y_score = df["y_proba"].to_numpy()

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    auc = float(roc_auc_score(y_true, y_score))
    gini = 2 * auc - 1

    # Youden's J statistic (TPR - FPR)
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    best_threshold = float(thresholds[best_idx])
    best_tpr = float(tpr[best_idx])
    best_fpr = float(fpr[best_idx])

    return {
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
        "auc": auc,
        "gini": gini,
        "best_threshold": best_threshold,
        "best_tpr": best_tpr,
        "best_fpr": best_fpr
    }


# ----------------------------------------
# View
# ----------------------------------------
def render(cfg: dict):

    # Oculta errores rojos frontend (por estabilidad)
    st.markdown(
        """
        <style>
          div[data-testid="stException"] { display: none !important; }
          div[role="alert"] { display: none !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("1️⃣ ROC Comparativa")
    st.caption("Comparación de capacidad discriminatoria entre modelos (RF vs GBT).")

    if not cfg.get("conn_ok"):
        st.warning("Primero conectate a la base de datos en el panel lateral.")
        return

    st.markdown("### ⚙️ Configuración")

    col1, col2 = st.columns(2)
    with col1:
        pred_table_a = st.text_input("Tabla modelo A", value=cfg.get("pred_table", "pred_rf"))
    with col2:
        pred_table_b = st.text_input("Tabla modelo B (opcional)", value=cfg.get("pred_table_b", ""))

    conn = _ensure_conn(cfg)

    df_a = _load_pred_table(conn, pred_table_a) if pred_table_a else pd.DataFrame()
    df_b = _load_pred_table(conn, pred_table_b) if pred_table_b else pd.DataFrame()

    if df_a.empty:
        st.warning(f"No hay datos en {pred_table_a}. Ejecutá primero el modelo en Overview.")
        return

    metrics_a = _roc_metrics(df_a)
    metrics_b = _roc_metrics(df_b) if not df_b.empty else None

    # ----------------------------------------
    # KPIs
    # ----------------------------------------
    st.markdown("### 📌 KPIs ROC")

    cols = st.columns(4)
    cols[0].metric(f"AUC ({pred_table_a})", f"{metrics_a['auc']:.4f}")
    cols[1].metric(f"Gini ({pred_table_a})", f"{metrics_a['gini']:.4f}")
    cols[2].metric(f"Best Thr (Youden)", f"{metrics_a['best_threshold']:.4f}")
    cols[3].metric(f"TPR@Best", f"{metrics_a['best_tpr']*100:.2f}%")

    if metrics_b:
        st.markdown("### 📌 KPIs Modelo B")
        cols_b = st.columns(4)
        cols_b[0].metric(f"AUC ({pred_table_b})", f"{metrics_b['auc']:.4f}")
        cols_b[1].metric(f"Gini ({pred_table_b})", f"{metrics_b['gini']:.4f}")
        cols_b[2].metric(f"Best Thr (Youden)", f"{metrics_b['best_threshold']:.4f}")
        cols_b[3].metric(f"TPR@Best", f"{metrics_b['best_tpr']*100:.2f}%")

    # ----------------------------------------
    # ROC Curve
    # ----------------------------------------
    st.markdown("### 📉 Curva ROC")

    fig = plt.figure(figsize=(6.5, 4))
    ax = fig.add_subplot(111)

    ax.plot(
        metrics_a["fpr"],
        metrics_a["tpr"],
        label=f"{pred_table_a} (AUC={metrics_a['auc']:.3f})"
    )

    if metrics_b:
        ax.plot(
            metrics_b["fpr"],
            metrics_b["tpr"],
            label=f"{pred_table_b} (AUC={metrics_b['auc']:.3f})"
        )

    # Línea random
    ax.plot([0, 1], [0, 1], linestyle="--")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")

    st.image(_fig_to_png(fig), use_container_width=False)

    # ----------------------------------------
    # Explicación profesional
    # ----------------------------------------
    with st.expander("Interpretación ejecutiva"):
        st.write("""
- **AUC** mide capacidad discriminatoria total.
    - 0.5 → random
    - 0.7 → aceptable
    - 0.8 → bueno
    - 0.9+ → excelente
- **Gini = 2*AUC - 1**, métrica usada en banca.
- El **Best Threshold (Youden)** maximiza TPR - FPR.
- En fraude, ROC es útil, pero PR suele ser más informativa por el desbalance.
        """)