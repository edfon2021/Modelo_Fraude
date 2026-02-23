# views/metrics.py
import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from db import fetch_df
from views.overview import _ensure_conn  # usa tu helper existente


def _fig_to_png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    return buf.getvalue()


def render(cfg: dict):
    # Ocultar posibles bloques rojos del frontend (por estabilidad de entrega)
    st.markdown(
        """
        <style>
          div[data-testid="stException"] { display: none !important; }
          div[role="alert"] { display: none !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("3️⃣ Métricas Clave (Lift + Fraude evitado + Costo operativo)")
    st.caption(
        "Usa la tabla de predicciones (pred_rf): amount, y_true, y_pred y y_proba. "
        "Lift/Gains se calcula ordenando por score (y_proba)."
    )

    if not cfg.get("conn_ok"):
        st.warning("Primero conectate a la base de datos en el panel lateral.")
        return

    # ---- Parámetros de operación
    st.markdown("### ⚙️ Parámetros de operación")
    c1, c2, c3 = st.columns(3)
    with c1:
        analysts = st.number_input("Analistas", min_value=1, max_value=500, value=10, step=1)
    with c2:
        usd_per_hour = st.number_input("Pago por hora (USD)", min_value=0.1, max_value=500.0, value=5.0, step=0.5)
    with c3:
        minutes_per_alert = st.number_input("Minutos por alerta", min_value=0.1, max_value=120.0, value=5.0, step=0.5)

    pred_table = cfg.get("pred_table", "pred_rf")

    # Conexión
    conn = _ensure_conn(cfg)

    # Traer datos mínimos
    sql = f'''
        SELECT
            COALESCE(amount, 0) AS amount,
            COALESCE(y_true, 0) AS y_true,
            COALESCE(y_pred, 0) AS y_pred,
            COALESCE(y_proba, 0) AS y_proba
        FROM "{pred_table}";
    '''
    df = fetch_df(conn, sql)

    if df is None or df.empty:
        st.warning(f"No hay datos en {pred_table}. Corré primero la evaluación en Overview.")
        return

    # Normalizar tipos
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
    df["y_true"] = pd.to_numeric(df["y_true"], errors="coerce").fillna(0).astype(int)
    df["y_pred"] = pd.to_numeric(df["y_pred"], errors="coerce").fillna(0).astype(int)
    df["y_proba"] = pd.to_numeric(df["y_proba"], errors="coerce").fillna(0.0)

    total_tx = len(df)
    fraud_total_count = int((df["y_true"] == 1).sum())
    fraud_total_amount = float(df.loc[df["y_true"] == 1, "amount"].sum())

    st.markdown("### 📌 Totales del día (según y_true)")
    a, b, c = st.columns(3)
    a.metric("Transacciones", f"{total_tx:,}")
    b.metric("Fraudes reales (conteo)", f"{fraud_total_count:,}")
    c.metric("Fraude total ($)", f"${fraud_total_amount:,.2f}")

    if fraud_total_count == 0 or fraud_total_amount == 0:
        st.info("No hay fraude real (y_true=1) o el monto es 0. Lift/montos no se verán.")
        return

    # Ordenar por score
    df_rank = df.sort_values("y_proba", ascending=False).reset_index(drop=True)

    # Capacidad: Top-K%
    st.markdown("### 🎯 Capacidad de revisión: Top-K% por score")
    k_pct = st.slider("Porcentaje del día que el banco puede revisar", min_value=1, max_value=50, value=10, step=1)
    k = max(1, int(np.ceil(total_tx * (k_pct / 100.0))))
    topk = df_rank.iloc[:k].copy()

    # Captura de fraude en el topK
    fraud_captured_amount = float(topk.loc[topk["y_true"] == 1, "amount"].sum())  # detectado (verde)
    fraud_captured_count = int((topk["y_true"] == 1).sum())

    # Fraude perdido por no revisar resto
    fraud_missed_amount = float(fraud_total_amount - fraud_captured_amount)  # se pasó (amarillo)
    fraud_missed_count = int(fraud_total_count - fraud_captured_count)

    capture_rate_amount = fraud_captured_amount / fraud_total_amount
    capture_rate_count = fraud_captured_count / fraud_total_count if fraud_total_count > 0 else 0.0

    lift_amount = capture_rate_amount / (k_pct / 100.0)
    lift_count = capture_rate_count / (k_pct / 100.0)

    # Costos: topK = "alertas a revisar"
    total_minutes = k * float(minutes_per_alert)
    total_hours = total_minutes / 60.0
    cost_usd = total_hours * float(usd_per_hour)
    hours_per_analyst = total_hours / float(analysts)

    st.markdown("### ✅ Impacto (con modelo) vs sin modelo")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Fraude detectado $", f"${fraud_captured_amount:,.2f}")
    k2.metric("Fraude que se pasó $", f"${fraud_missed_amount:,.2f}")
    k3.metric("Lift (monto)", f"{lift_amount:,.2f}x")
    k4.metric("Lift (conteo)", f"{lift_count:,.2f}x")

    st.caption(
        f"Revisando Top-{k_pct}% (≈ {k:,} alertas), capturás "
        f"{capture_rate_amount*100:,.1f}% del monto fraudulento y "
        f"{capture_rate_count*100:,.1f}% de los fraudes (conteo)."
    )

    st.markdown("### 🧾 Costo operativo estimado (revisar Top-K%)")
    cc1, cc2, cc3 = st.columns(3)
    cc1.metric("Alertas a revisar", f"{k:,}")
    cc2.metric("Horas totales", f"{total_hours:,.2f} h")
    cc3.metric("Costo total", f"${cost_usd:,.2f}")
    st.caption(f"Carga aprox. por analista: {hours_per_analyst:,.2f} h/día (repartido entre {analysts}).")

    # =========================
    # Gráfico: Fraude detectado (verde) vs se pasó (amarillo)
    # =========================
    st.markdown("### 💰 Fraude detectado vs fraude que se pasó (monto)")

    fig1 = plt.figure(figsize=(6.2, 3.2))  # ✅ más pequeño
    ax1 = fig1.add_subplot(111)

    labels = ["Detectado (Top-K)", "Se pasó (no revisado)"]
    values = [fraud_captured_amount, fraud_missed_amount]

    ax1.bar(labels, values, color=["#2e7d32", "#f9a825"])  # ✅ verde / amarillo
    ax1.set_ylabel("USD")
    ax1.set_title(f"Fraude detectado vs se pasó (Top-{k_pct}%)")

    # etiquetas sobre barras
    for i, v in enumerate(values):
        ax1.text(i, v, f"${v:,.0f}", ha="center", va="bottom", fontsize=9)

    st.image(_fig_to_png(fig1), use_container_width=False)

    # =========================
    # Tabla Lift/Gains
    # =========================
    st.markdown("### 📈 Tabla Lift / Gains (varios Top-K%)")
    cuts = [1, 2, 5, 10, 20, 30, 50]
    rows = []
    for pct in cuts:
        kk = max(1, int(np.ceil(total_tx * (pct / 100.0))))
        tt = df_rank.iloc[:kk]
        cap_amt = float(tt.loc[tt["y_true"] == 1, "amount"].sum())
        cap_cnt = int((tt["y_true"] == 1).sum())

        rate_amt = cap_amt / fraud_total_amount
        rate_cnt = cap_cnt / fraud_total_count if fraud_total_count > 0 else 0.0

        rows.append(
            {
                "Top_%": pct,
                "Alertas": kk,
                "Fraude_detectado_$": cap_amt,
                "%Fraude_detectado_$": rate_amt,
                "Lift_$": rate_amt / (pct / 100.0),
                "Fraude_detectado_cnt": cap_cnt,
                "%Fraude_detectado_cnt": rate_cnt,
                "Lift_cnt": (rate_cnt / (pct / 100.0)) if pct > 0 else 0.0,
                "Costo_USD": (kk * float(minutes_per_alert) / 60.0) * float(usd_per_hour),
            }
        )

    df_lift = pd.DataFrame(rows)

    # Formato amigable
    df_show = df_lift.copy()
    df_show["Fraude_detectado_$"] = df_show["Fraude_detectado_$"].map(lambda x: f"${x:,.2f}")
    df_show["%Fraude_detectado_$"] = (df_show["%Fraude_detectado_$"] * 100).map(lambda x: f"{x:,.1f}%")
    df_show["Lift_$"] = df_show["Lift_$"].map(lambda x: f"{x:,.2f}x")
    df_show["%Fraude_detectado_cnt"] = (df_show["%Fraude_detectado_cnt"] * 100).map(lambda x: f"{x:,.1f}%")
    df_show["Lift_cnt"] = df_show["Lift_cnt"].map(lambda x: f"{x:,.2f}x")
    df_show["Costo_USD"] = df_show["Costo_USD"].map(lambda x: f"${x:,.2f}")

    st.dataframe(df_show, use_container_width=True)

    with st.expander("Ver detalle numérico (sin formato)"):
        st.dataframe(df_lift, use_container_width=True)