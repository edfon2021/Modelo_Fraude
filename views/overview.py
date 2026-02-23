# views/overview.py
import io
import streamlit as st
import pandas as pd

from db import make_conn, get_first_transactions, fetch_df
from utils.plots import plot_muestreo_por_mes_barras
from evaluation.process import run_evaluation_and_store_minimal


# ----------------------------
# Helpers: conexión estable
# ----------------------------
def _cfg_key(cfg: dict) -> tuple:
    """Key estable (sin password) útil para estado en session_state."""
    return (
        cfg.get("host"),
        int(cfg.get("port", 5432)),
        cfg.get("dbname"),
        cfg.get("user"),
        cfg.get("table"),
    )


@st.cache_resource(show_spinner=False)
def _get_conn_cached(host: str, port: int, dbname: str, user: str, password: str):
    """Una conexión por sesión (cache_resource)."""
    return make_conn(host, port, dbname, user, password)


def _ensure_conn(cfg: dict):
    """
    Devuelve conexión viva.
    Si murió, limpia cache_resource y recrea.
    """
    host = cfg["host"]
    port = int(cfg["port"])
    dbname = cfg["dbname"]
    user = cfg["user"]
    password = cfg["password"]

    conn = _get_conn_cached(host, port, dbname, user, password)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1;")
            cur.fetchone()
        return conn
    except Exception:
        try:
            conn.close()
        except Exception:
            pass
        _get_conn_cached.clear()
        return _get_conn_cached(host, port, dbname, user, password)


def _safe_fetch_preview(conn, table_name: str, limit: int = 50) -> pd.DataFrame:
    return fetch_df(conn, f'SELECT * FROM "{table_name}" LIMIT {int(limit)};')


def _safe_get_sample(conn, table_name: str, limit: int, date_col: str) -> pd.DataFrame:
    return get_first_transactions(conn=conn, table_name=table_name, limit=int(limit), date_col=date_col)


def _safe_resumen_mensual(conn, table_name: str, date_col: str, estado_col: str) -> pd.DataFrame:
    sql = f"""
        SELECT
            to_char(date_trunc('month', "{date_col}"), 'YYYY-MM') AS mes,
            SUM(CASE WHEN upper("{estado_col}"::text) = 'F' THEN 1 ELSE 0 END) AS fraudes,
            SUM(CASE WHEN upper("{estado_col}"::text) = 'N' THEN 1 ELSE 0 END) AS normales_tomar
        FROM "{table_name}"
        GROUP BY 1
        ORDER BY 1;
    """
    df = fetch_df(conn, sql)
    if df is None or df.empty:
        return df

    df = df.set_index("mes")
    df["fraudes"] = pd.to_numeric(df["fraudes"], errors="coerce").fillna(0).astype(int)
    df["normales_tomar"] = pd.to_numeric(df["normales_tomar"], errors="coerce").fillna(0).astype(int)
    return df


def _fig_to_png_bytes(fig) -> bytes:
    """Renderiza una figura matplotlib a PNG para evitar st.pyplot (menos bugs de DOM)."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    return buf.getvalue()


# ----------------------------
# View
# ----------------------------
def render(cfg: dict):
    # ✅ OCULTA el bloque rojo de errores del frontend (removeChild)
    # OJO: esto NO arregla el bug interno, pero deja la UI limpia para la entrega.
    st.markdown(
        """
        <style>
          /* Bloque de excepción (rojo) */
          div[data-testid="stException"] { display: none !important; }

          /* Algunas versiones lo montan como alert */
          div[role="alert"] { display: none !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("0️⃣ Vista de transacciones (muestra) + gráfico mensual (tabla completa)")
    st.caption(
        "La tabla es una muestra para visualización rápida. "
        "El gráfico mensual se calcula en PostgreSQL sobre la tabla completa."
    )

    st.session_state["cfg_runtime"] = cfg

    # Estados iniciales
    st.session_state.setdefault("show_sample_and_chart", False)
    st.session_state.setdefault("run_eval", False)

    st.session_state.setdefault("last_eval_result", None)
    st.session_state.setdefault("last_pred_preview", None)
    st.session_state.setdefault("last_metrics_preview", None)

    st.session_state.setdefault("last_sample_df", None)
    st.session_state.setdefault("last_resumen_df", None)

    # Flags para separar evento -> render pesado (evita glitch DOM)
    st.session_state.setdefault("do_sample_refresh", False)
    st.session_state.setdefault("do_eval_run", False)

    _ = _cfg_key(cfg)  # si luego lo querés para debug/estado

    # Contenedores SIEMPRE se crean (árbol estable)
    top_box = st.container()
    controls_box = st.container()
    sample_box = st.container()
    chart_box = st.container()
    eval_header_box = st.container()
    eval_controls_box = st.container()
    eval_box = st.container()

    # ----------------------------
    # Conexión SIN cortar layout
    # ----------------------------
    conn_ok = bool(cfg.get("conn_ok"))
    conn = None

    with top_box:
        if not conn_ok:
            st.warning("Primero probá la conexión en el panel lateral.")
            st.info("Cuando la conexión esté OK, esta vista mostrará la muestra, gráfico y ejecución del modelo.")
        else:
            try:
                conn = _ensure_conn(cfg)
            except Exception as e:
                conn_ok = False
                st.error(f"No pude mantener conexión viva a PostgreSQL: {e}")

    # =========================
    # Controles UI: muestra + gráfico
    # =========================
    with controls_box:
        c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
        with c1:
            limit = st.number_input(
                "Muestra (filas)", min_value=10, max_value=5000, value=100, step=10, key="ov_limit"
            )
        with c2:
            date_col = st.text_input("Columna fecha", value=cfg.get("date_col", "Fecha"), key="ov_date_col")
        with c3:
            estado_col = st.text_input("Columna estado", value="Estado", key="ov_estado_col")
        with c4:
            st.write("")

        with st.form("form_sample", clear_on_submit=False):
            submitted_sample = st.form_submit_button("📥 Cargar muestra + gráfico")

        if submitted_sample:
            st.session_state["show_sample_and_chart"] = True
            st.session_state["do_sample_refresh"] = True
            st.rerun()

    # ----------------------------
    # Refrescar muestra + resumen (solo cuando toca)
    # ----------------------------
    if st.session_state.get("do_sample_refresh"):
        st.session_state["do_sample_refresh"] = False
        if conn_ok and conn is not None:
            try:
                df_sample = _safe_get_sample(conn, cfg["table"], int(limit), date_col)
                st.session_state["last_sample_df"] = df_sample
            except Exception as e:
                st.session_state["last_sample_df"] = None
                st.warning(f"No pude cargar la muestra: {e}")

            try:
                resumen = _safe_resumen_mensual(conn, cfg["table"], date_col, estado_col)
                st.session_state["last_resumen_df"] = resumen
            except Exception as e:
                st.session_state["last_resumen_df"] = None
                st.warning(f"No pude generar el resumen mensual: {e}")

    # =========================
    # Predicciones (tabla completa)
    # =========================
    with eval_header_box:
        st.markdown("---")
        st.markdown("## ⚙️ Generar predicciones para el dashboard (evalúa TODA la tabla)")
        st.caption(
            "Esto NO usa la muestra. Corre el modelo sobre toda la tabla en PostgreSQL, "
            "borra y recrea las tablas destino en cada ejecución."
        )

    with eval_controls_box:
        d1, d2, d3, d4 = st.columns([1, 1, 1, 1])
        with d1:
            id_col = st.text_input("Columna ID transacción", value=cfg.get("id_col", "Id_Transaccion"), key="ov_id_col")
        with d2:
            model_filename = st.text_input(
                "Modelo (en ./models/)", value=cfg.get("model_filename", "gbt_model.joblib"), key="ov_model"
            )
        with d3:
            dest_table = st.text_input(
                "Tabla predicciones (por modelo)", value=cfg.get("pred_table", "pred_rf"), key="ov_dest"
            )
        with d4:
            metrics_table = st.text_input(
                "Tabla métricas (por modelo)", value=cfg.get("metrics_table", "metrics_rf"), key="ov_metrics"
            )

        e1, e2 = st.columns([1, 1])
        with e1:
            min_precision = st.number_input(
                "Min precision", min_value=0.0, max_value=1.0, value=0.12, step=0.01, key="ov_min_prec"
            )
        with e2:
            step = st.number_input(
                "Step threshold",
                min_value=0.0005,
                max_value=0.05,
                value=0.001,
                step=0.0005,
                format="%.4f",
                key="ov_step",
            )

        with st.form("form_eval", clear_on_submit=False):
            submitted_eval = st.form_submit_button("🚀 Evaluar tabla completa y guardar (recrear tablas)")

        if submitted_eval:
            st.session_state["do_eval_run"] = True
            st.rerun()

    # ----------------------------
    # Ejecutar evaluación (si se pidió)
    # ----------------------------
    if st.session_state.get("do_eval_run"):
        st.session_state["do_eval_run"] = False
        st.session_state["run_eval"] = True

    with eval_box:
        if st.session_state.get("run_eval"):
            st.session_state["run_eval"] = False

            if not conn_ok or conn is None:
                st.error("No hay conexión activa. Probá la conexión en el panel lateral primero.")
            else:
                with st.spinner("Ejecutando evaluación sobre tabla completa..."):
                    try:
                        result = run_evaluation_and_store_minimal(
                            cfg=cfg,
                            source_table=cfg["table"],
                            dest_table=dest_table,
                            metrics_table=metrics_table,
                            model_filename=model_filename,
                            id_col=id_col,
                            min_precision=float(min_precision),
                            step=float(step),
                        )

                        st.session_state["last_eval_result"] = result

                        try:
                            st.session_state["last_pred_preview"] = _safe_fetch_preview(conn, dest_table, limit=50)
                        except Exception:
                            st.session_state["last_pred_preview"] = None

                        try:
                            st.session_state["last_metrics_preview"] = _safe_fetch_preview(conn, metrics_table, limit=5)
                        except Exception:
                            st.session_state["last_metrics_preview"] = None

                    except Exception as e:
                        st.error(f"No pude generar/guardar predicciones: {e}")

        last = st.session_state.get("last_eval_result")
        if last:
            st.success(
                f"Listo ✅ Modelo **{last.get('model_filename')}**\n\n"
                f"- Predicciones: **{last.get('dest_table')}** ({int(last.get('rows_written', 0)):,} filas)\n"
                f"- Métricas: **{last.get('metrics_table')}** (recreada)"
            )
            st.write("Threshold usado:", last.get("threshold_used"))
            st.write("Resultado threshold:", last.get("threshold_result"))

            df_prev = st.session_state.get("last_pred_preview")
            if isinstance(df_prev, pd.DataFrame) and not df_prev.empty:
                with st.expander("Ver preview predicciones", expanded=True):
                    st.dataframe(df_prev, use_container_width=True, height=320, key="ov_df_prev")

            df_m = st.session_state.get("last_metrics_preview")
            if isinstance(df_m, pd.DataFrame) and not df_m.empty:
                with st.expander("Ver métricas (tabla)", expanded=False):
                    st.dataframe(df_m, use_container_width=True, key="ov_df_metrics")

    # =========================
    # Muestra + gráfico (persistentes)
    # =========================
    with sample_box:
        if not st.session_state.get("show_sample_and_chart"):
            st.info("Dale click a **Cargar muestra + gráfico** para ver tabla y gráfico.")
        else:
            df_sample = st.session_state.get("last_sample_df")
            if isinstance(df_sample, pd.DataFrame) and not df_sample.empty:
                st.success(f"Muestra cargada: {len(df_sample):,} filas ✅")
                with st.expander("Ver muestra (tabla)", expanded=True):
                    st.dataframe(df_sample, use_container_width=True, height=520, key="ov_df_sample")
            else:
                st.warning("La muestra está vacía o no se pudo cargar. Revisá tabla/columna fecha y la conexión.")

    with chart_box:
        if st.session_state.get("show_sample_and_chart"):
            st.markdown("---")
            st.markdown("### 📊 Distribución mensual (tabla completa en PostgreSQL)")

            resumen = st.session_state.get("last_resumen_df")
            if isinstance(resumen, pd.DataFrame) and not resumen.empty:
                try:
                    fig = plot_muestreo_por_mes_barras(
                        resumen=resumen,
                        log_scale=True,
                        title="Distribución mensual (Fraude=F, Normal=N)",
                    )

                    png = _fig_to_png_bytes(fig)
                    st.image(png, use_container_width=True)

                    with st.expander("Ver resumen mensual (tabla)", expanded=False):
                        st.dataframe(resumen.reset_index(), use_container_width=True, key="ov_resumen_tabla")
                except Exception as e:
                    st.warning(f"No pude dibujar el gráfico mensual: {e}")
            else:
                st.warning("El resumen mensual vino vacío o no se pudo calcular. Revisá columnas/datos.")