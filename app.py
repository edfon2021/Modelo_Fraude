import streamlit as st
from db import test_connection

from views import overview, roc, pr, metrics, confusion

st.set_page_config(page_title="Fraude Emisor - Dashboard", layout="wide")

# ===== Sidebar: conexión =====
st.sidebar.title("⚙️ Configuración")

with st.sidebar.expander("Conexión PostgreSQL", expanded=True):
    host = st.text_input("Host", value="sistinvact01.postgres.database.azure.com")
    port = st.number_input("Port", value=5432, step=1)
    dbname = st.text_input("Database", value="Transacciones")
    user = st.text_input("User", value="az_user01")
    password = st.text_input("Password", type="password", value="Canelo32")
    table = st.text_input("Tabla", value="df_test")
    date_col = st.text_input("Columna fecha", value="Fecha")

    btn_test = st.button("✅ Probar conexión")

if "conn_ok" not in st.session_state:
    st.session_state.conn_ok = False
if "conn_msg" not in st.session_state:
    st.session_state.conn_msg = ""

if btn_test:
    ok, msg = test_connection(host, int(port), dbname, user, password)
    st.session_state.conn_ok = ok
    st.session_state.conn_msg = msg
    if ok:
        st.sidebar.success(msg)
    else:
        st.sidebar.error(msg)

# ===== Sidebar: menú =====
st.sidebar.markdown("---")
menu = st.sidebar.radio(
    "📌 Paneles",
    [
        "0) Vista transacciones (100)",
        "1) ROC Comparativa",
        "2) Precision-Recall Comparativa",
        "3) Métricas Clave (tabla visual)",
        "4) Matrices de Confusión (lado a lado)",
    ],
)

cfg = {
    "host": host,
    "port": int(port),
    "dbname": dbname,
    "user": user,
    "password": password,
    "table": table,
    "date_col": date_col,
    "conn_ok": st.session_state.conn_ok,
}

# ===== Main =====
st.title("📊 Dashboard - Detección de Fraude (Emisor)")
st.caption("Arquitectura modular: app.py (contenedor) + db.py (BD) + pages/*.py (paneles)")

if st.session_state.conn_msg:
    (st.success if st.session_state.conn_ok else st.error)(st.session_state.conn_msg)

st.markdown("---")

# ===== Router =====
if menu.startswith("0)"):
    overview.render(cfg)
elif menu.startswith("1)"):
    roc.render(cfg)
elif menu.startswith("2)"):
    pr.render(cfg)
elif menu.startswith("3)"):
    metrics.render(cfg)
else:
    confusion.render(cfg)