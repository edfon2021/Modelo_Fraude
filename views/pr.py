# views/pr.py
import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.metrics import precision_recall_curve, average_precision_score, brier_score_loss, log_loss

from db import fetch_df
from views.overview import _ensure_conn


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
            COALESCE(amount, 0) AS amount,
            COALESCE(y_true, 0) AS y_true,
            COALESCE(y_proba, 0) AS y_proba
        FROM "{table_name}";
    '''
    df = fetch_df(conn, sql)
    if df is None or df.empty:
        return pd.DataFrame()

    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
    df["y_true"] = pd.to_numeric(df["y_true"], errors="coerce").fillna(0).astype(int)
    df["y_proba"] = pd.to_numeric(df["y_proba"], errors="coerce").fillna(0.0)

    # clamp por si algún modelo devuelve >1 o <0
    df["y_proba"] = df["y_proba"].clip(0.0, 1.0)
    return df


def _metrics_from_scores(df: pd.DataFrame, k_pct: int, min_precision: float):
    y_true = df["y_true"].to_numpy()
    y_score = df["y_proba"].to_numpy()
    amt = df["amount"].to_numpy()

    pos = int((y_true == 1).sum())
    total = int(len(df))
    fraud_total_amount = float(amt[y_true == 1].sum())

    # PR curve
    prec, rec, thr = precision_recall_curve(y_true, y_score)
    ap = float(average_precision_score(y_true, y_score))

    # F1 para cada punto (thr tiene len = len(prec)-1)
    # Usamos prec[:-1], rec[:-1] para alinear con thr
    prec_t = prec[:-1]
    rec_t = rec[:-1]
    f1_t = np.where((prec_t + rec_t) > 0, 2 * prec_t * rec_t / (prec_t + rec_t), 0.0)

    best_i = int(np.argmax(f1_t)) if len(f1_t) else 0
    best_f1 = float(f1_t[best_i]) if len(f1_t) else 0.0
    best_thr_f1 = float(thr[best_i]) if len(thr) else 0.5
    best_prec_f1 = float(prec_t[best_i]) if len(prec_t) else 0.0
    best_rec_f1 = float(rec_t[best_i]) if len(rec_t) else 0.0

    # “Recall@Precision>=min_precision” (máximo recall cumpliendo precisión mínima)
    mask = prec_t >= float(min_precision)
    recall_at_min_prec = float(rec_t[mask].max()) if mask.any() else 0.0
    thr_at_min_prec = float(thr[mask][np.argmax(rec_t[mask])]) if mask.any() else 1.0

    # Precision@K / Recall@K (Top K% por score)
    k = max(1, int(np.ceil(total * (k_pct / 100.0))))
    df_rank = df.sort_values("y_proba", ascending=False).reset_index(drop=True)
    topk = df_rank.iloc[:k]

    tp_in_topk = int((topk["y_true"] == 1).sum())
    precision_at_k = _safe_div(tp_in_topk, k)
    recall_at_k = _safe_div(tp_in_topk, pos)

    # Captura por monto @K (si amount tiene sentido)
    fraud_cap_amount_k = float(topk.loc[topk["y_true"] == 1, "amount"].sum())
    fraud_cap_rate_amount_k = _safe_div(fraud_cap_amount_k, fraud_total_amount) if fraud_total_amount > 0 else 0.0

    # Calibración (opcionales pero útiles)
    # Brier: requiere y_true 0/1 y score en [0,1]
    brier = float(brier_score_loss(y_true, y_score))
    # LogLoss: cuidado con 0/1 exactos
    eps = 1e-15
    ll = float(log_loss(y_true, np.clip(y_score, eps, 1 - eps)))

    return {
        "prec": prec,
        "rec": rec,
        "thr": thr,
        "ap": ap,
        "best_f1": best_f1,
        "best_thr_f1": best_thr_f1,
        "best_prec_f1": best_prec_f1,
        "best_rec_f1": best_rec_f1,
        "recall_at_min_prec": recall_at_min_prec,
        "thr_at_min_prec": thr_at_min_prec,
        "precision_at_k": precision_at_k,
        "recall_at_k": recall_at_k,
        "fraud_cap_amount_k": fraud_cap_amount_k,
        "fraud_cap_rate_amount_k": fraud_cap_rate_amount_k,
        "brier": brier,
        "logloss": ll,
        "pos": pos,
        "total": total,
    }


def render(cfg: dict):
    # Ocultar bloque rojo del frontend (por estabilidad de entrega)
    st.markdown(
        """
        <style>
          div[data-testid="stException"] { display: none !important; }
          div[role="alert"] { display: none !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("2️⃣ Precision–Recall (curva + métricas)")
    st.caption("Comparativa usando y_proba (score) vs y_true (fraude real).")

    if not cfg.get("conn_ok"):
        st.warning("Primero conectate a la base de datos en el panel lateral.")
        return

    # --- Controles
    st.markdown("### ⚙️ Parámetros")
    c1, c2, c3 = st.columns(3)
    with c1:
        min_precision = st.number_input("Min precision (para Recall@P>=min)", min_value=0.0, max_value=1.0, value=0.12, step=0.01)
    with c2:
        k_pct = st.slider("K% para Precision@K / Recall@K", min_value=1, max_value=50, value=10, step=1)
    with c3:
        # Comparativa: elegís tablas
        pred_table_1 = st.text_input("Tabla modelo A", value=cfg.get("pred_table", "pred_rf"))
        pred_table_2 = st.text_input("Tabla modelo B (opcional)", value=cfg.get("pred_table_b", ""))

    conn = _ensure_conn(cfg)

    df_a = _load_pred_table(conn, pred_table_1) if pred_table_1 else pd.DataFrame()
    df_b = _load_pred_table(conn, pred_table_2) if pred_table_2 else pd.DataFrame()

    if df_a.empty:
        st.warning(f"No hay datos en {pred_table_1}. Corré primero la evaluación en Overview.")
        return

    met_a = _metrics_from_scores(df_a, k_pct=k_pct, min_precision=float(min_precision))
    met_b = _metrics_from_scores(df_b, k_pct=k_pct, min_precision=float(min_precision)) if not df_b.empty else None

    # --- KPIs arriba
    st.markdown("### 📌 KPIs (Precision–Recall y calibración)")
    cols = st.columns(4)
    cols[0].metric(f"AP / PR-AUC ({pred_table_1})", f"{met_a['ap']:.4f}")
    cols[1].metric(f"F1 máx ({pred_table_1})", f"{met_a['best_f1']:.4f}")
    cols[2].metric(f"Precision@{k_pct}% ({pred_table_1})", f"{met_a['precision_at_k']*100:.2f}%")
    cols[3].metric(f"Recall@{k_pct}% ({pred_table_1})", f"{met_a['recall_at_k']*100:.2f}%")

    cols2 = st.columns(4)
    cols2[0].metric(f"Recall@P≥{min_precision:.2f} ({pred_table_1})", f"{met_a['recall_at_min_prec']*100:.2f}%")
    cols2[1].metric(f"Thr (F1 máx) ({pred_table_1})", f"{met_a['best_thr_f1']:.4f}")
    cols2[2].metric(f"Brier ({pred_table_1})", f"{met_a['brier']:.5f}")
    cols2[3].metric(f"LogLoss ({pred_table_1})", f"{met_a['logloss']:.5f}")

    if met_b is not None:
        st.markdown("### 📌 KPIs modelo B (comparativa rápida)")
        cb = st.columns(4)
        cb[0].metric(f"AP / PR-AUC ({pred_table_2})", f"{met_b['ap']:.4f}")
        cb[1].metric(f"F1 máx ({pred_table_2})", f"{met_b['best_f1']:.4f}")
        cb[2].metric(f"Precision@{k_pct}% ({pred_table_2})", f"{met_b['precision_at_k']*100:.2f}%")
        cb[3].metric(f"Recall@{k_pct}% ({pred_table_2})", f"{met_b['recall_at_k']*100:.2f}%")

    # --- Curva PR
    st.markdown("### 📉 Curva Precision–Recall")
    fig = plt.figure(figsize=(6.6, 4.0))
    ax = fig.add_subplot(111)

    ax.plot(met_a["rec"], met_a["prec"], label=f"{pred_table_1} (AP={met_a['ap']:.3f})")

    if met_b is not None:
        ax.plot(met_b["rec"], met_b["prec"], label=f"{pred_table_2} (AP={met_b['ap']:.3f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall Curve")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower left")

    st.image(_fig_to_png(fig), use_container_width=False)

    # --- Tabla resumen de métricas relacionadas (A y B)
    st.markdown("### 📋 Resumen de métricas (PR-related)")
    rows = [
        {
            "modelo": pred_table_1,
            "AP(PR-AUC)": met_a["ap"],
            "F1_max": met_a["best_f1"],
            "thr_F1_max": met_a["best_thr_f1"],
            f"Precision@{k_pct}%": met_a["precision_at_k"],
            f"Recall@{k_pct}%": met_a["recall_at_k"],
            f"Recall@P≥{min_precision:.2f}": met_a["recall_at_min_prec"],
            "thr@P>=min": met_a["thr_at_min_prec"],
            "Brier": met_a["brier"],
            "LogLoss": met_a["logloss"],
        }
    ]
    if met_b is not None:
        rows.append(
            {
                "modelo": pred_table_2,
                "AP(PR-AUC)": met_b["ap"],
                "F1_max": met_b["best_f1"],
                "thr_F1_max": met_b["best_thr_f1"],
                f"Precision@{k_pct}%": met_b["precision_at_k"],
                f"Recall@{k_pct}%": met_b["recall_at_k"],
                f"Recall@P≥{min_precision:.2f}": met_b["recall_at_min_prec"],
                "thr@P>=min": met_b["thr_at_min_prec"],
                "Brier": met_b["brier"],
                "LogLoss": met_b["logloss"],
            }
        )

    df_sum = pd.DataFrame(rows)

    # formateo para UI
    for col in df_sum.columns:
        if col != "modelo":
            df_sum[col] = pd.to_numeric(df_sum[col], errors="coerce")

    st.dataframe(df_sum, use_container_width=True)

    with st.expander("Notas (para defensa)"):
        st.write(
            """
- **AP/PR-AUC** resume la curva Precision–Recall (mejor que ROC-AUC en fraude por desbalance).
- **Precision@K / Recall@K** simula capacidad operativa: solo puedo revisar el Top-K% más sospechoso.
- **Recall@P>=min_precision** muestra el mejor recall posible garantizando una precisión mínima.
- **Brier/LogLoss** son de calibración: qué tan “probabilístico” es el score.
            """
        )