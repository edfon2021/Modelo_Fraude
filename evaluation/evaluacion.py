# views/evaluacion.py
import streamlit as st
from evaluation.process import run_evaluation_and_store


def render(cfg: dict):
    st.subheader("🧪 Evaluación de modelo")
    st.caption(
        "Corre evaluación completa (predict_proba + umbral óptimo) y guarda resultados.\n"
        "Se crea una tabla por modelo y se borra/recrea en cada corrida."
    )

    if not cfg.get("conn_ok"):
        st.warning("Primero probá la conexión en el panel lateral.")
        return

    col1, col2 = st.columns([1, 1])

    with col1:
        model_key = st.selectbox("Modelo a evaluar", options=["RF", "GBT"])

        if model_key == "RF":
            model_filename = "rf_model.joblib"
            dest_table = "pred_rf"
        else:
            model_filename = "gbt_model.joblib"
            dest_table = "pred_gbt"

        min_precision = st.number_input("min_precision", 0.0, 1.0, 0.12, 0.01)
        step = st.number_input("step (grid)", 0.0001, 0.05, 0.001, 0.0005, format="%.4f")

    with col2:
        source_table = st.text_input("Tabla fuente (df_test) en Postgres", value=cfg.get("table", "transacciones"))
        st.text_input("Tabla destino (auto)", value=dest_table, disabled=True)

    st.markdown("---")

    if st.button("🚀 Evaluar", type="primary"):
        with st.spinner("Ejecutando evaluación… (esto recrea la tabla destino)"):
            try:
                result = run_evaluation_and_store(
                    cfg=cfg,
                    source_table=source_table,
                    dest_table=dest_table,
                    model_filename=model_filename,
                    min_precision=float(min_precision),
                    step=float(step),
                )

                st.success("Evaluación completada ✅")
                st.write("**Modelo:**", result["model_path"])
                st.write("**Tabla fuente:**", result["source_table"])
                st.write("**Tabla destino:**", result["dest_table"])
                st.write("**Filas guardadas:**", result["rows_written"])

                thr = result["threshold_result"]
                st.write("**Umbral usado:**", result["threshold_used"])
                st.write("**Precision (split1):**", thr.get("precision"))
                st.write("**Recall (split1):**", thr.get("recall"))

                st.info("Guardado: Vars_finales + Estado + y_pred_adj + y_proba")

            except Exception as e:
                st.error(f"Falló la evaluación: {e}")