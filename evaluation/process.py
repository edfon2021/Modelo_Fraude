# evaluation/process.py
import os
import re
import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from psycopg2.extras import execute_values

from db import make_conn


# ===== Vars_finales "quemadas" por ahora =====
VARS_FINALES = [
    "countryMerchant",
    "logo",
    "mcc",
    "trancurrencyCode",
    "posEntryMode4",
    "tx_count_last_5m",
    "tx_count_last_15m",
    "tx_count_last_1h",
    "sum_amount_last_1h",
    "count_tx_same_mcc_last_1h",
    "count_tx_same_country_last_1h",
    "ratio_same_amount",
    "count_same_amount_last_1h",
    "chbillingAmount",  
]


def _assert_safe_identifier(name: str, what: str = "identifier"):
    """
    Protección básica: letras, números, underscore. Debe empezar con letra o underscore.
    Evita inyección en nombres de tablas/columnas.
    """
    if not isinstance(name, str) or not name:
        raise ValueError(f"{what} inválido (vacío).")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        raise ValueError(
            f"{what} inválido: '{name}'. Usá solo [A-Za-z0-9_] y que no empiece con número."
        )


def preparar_datos_full(X, y, target_pos_label="F", test_size=0.2):
    """
    Procesa X (features) e y (target) para evaluación/entrenamiento.
    - Convierte categorías de X a números (LabelEncoder por columna).
    - Convierte target (y) a 0 y 1.
    - Genera versiones escaladas (por si luego usás RN).
    """
    X_prep = X.copy()
    y_prep = y.copy()

    # 1) TARGET a 0/1
    if y_prep.dtype == "object" or getattr(y_prep.dtype, "name", "") == "category":
        y_norm = y_prep.astype(str).str.strip().str.upper()
        pos_norm = str(target_pos_label).strip().upper()
        y_prep = y_norm.map({pos_norm: 1}).fillna(0).astype(int)
    else:
        y_prep = pd.to_numeric(y_prep, errors="coerce").fillna(0).astype(int)

    # 2) Features: LabelEncoder en categóricas
    cat_cols = X_prep.select_dtypes(["category", "object"]).columns
    for col in cat_cols:
        le = LabelEncoder()
        X_prep[col] = le.fit_transform(X_prep[col].astype(str))

    # 3) Split estratificado
    X_train, X_test, y_train, y_test = train_test_split(
        X_prep, y_prep, test_size=test_size, random_state=42, stratify=y_prep
    )

    # 4) Escalamiento (por compatibilidad)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test


def best_threshold_max_recall_with_min_precision(y_true, y_proba, min_precision=0.12, step=0.001):
    thresholds = np.arange(0.0, 1.0 + step, step)
    best = {"threshold": None, "precision": -1.0, "recall": -1.0}

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        p = precision_score(y_true, y_pred, zero_division=0)
        r = recall_score(y_true, y_pred, zero_division=0)

        if p >= min_precision:
            if r > best["recall"]:
                best = {"threshold": float(t), "precision": float(p), "recall": float(r)}
            elif r == best["recall"] and p > best["precision"]:
                best = {"threshold": float(t), "precision": float(p), "recall": float(r)}

    return best


def _read_df_full_table(conn, source_table: str) -> pd.DataFrame:
    _assert_safe_identifier(source_table, "source_table")
    return pd.read_sql(f'SELECT * FROM "{source_table}";', conn)


def _recreate_predictions_table_minimal(conn, dest_table: str):
    """
    Borra y recrea SIEMPRE la tabla de predicciones (mínima) + monto.
    """
    _assert_safe_identifier(dest_table, "dest_table")
    ddl_drop = f'DROP TABLE IF EXISTS "{dest_table}";'
    ddl_create = f"""
    CREATE TABLE "{dest_table}" (
        tx_id TEXT,
        amount DOUBLE PRECISION,
        y_true INTEGER,
        y_pred INTEGER,
        y_proba DOUBLE PRECISION
    );
    """
    with conn.cursor() as cur:
        cur.execute(ddl_drop)
        cur.execute(ddl_create)
    conn.commit()


def _recreate_metrics_table(conn, metrics_table: str):
    """
    Borra y recrea SIEMPRE la tabla de métricas (1 fila por run) con métricas extra.
    """
    _assert_safe_identifier(metrics_table, "metrics_table")
    ddl_drop = f'DROP TABLE IF EXISTS "{metrics_table}";'
    ddl_create = f"""
    CREATE TABLE "{metrics_table}" (
        run_ts TIMESTAMP DEFAULT NOW(),
        source_table TEXT,
        model_filename TEXT,
        model_path TEXT,
        threshold_used DOUBLE PRECISION,

        precision DOUBLE PRECISION,
        recall DOUBLE PRECISION,
        f1 DOUBLE PRECISION,
        accuracy DOUBLE PRECISION,

        tp BIGINT,
        fp BIGINT,
        fn BIGINT,
        tn BIGINT,

        rows_written BIGINT,
        alert_rate DOUBLE PRECISION,

        fraud_amount_total DOUBLE PRECISION,
        fraud_amount_captured DOUBLE PRECISION,
        fraud_amount_missed DOUBLE PRECISION,
        fp_amount DOUBLE PRECISION
    );
    """
    with conn.cursor() as cur:
        cur.execute(ddl_drop)
        cur.execute(ddl_create)
    conn.commit()


def _insert_df(conn, table: str, df_out: pd.DataFrame):
    _assert_safe_identifier(table, "table")
    cols = list(df_out.columns)
    cols_sql = ",".join([f'"{c}"' for c in cols])
    insert_sql = f'INSERT INTO "{table}" ({cols_sql}) VALUES %s'
    rows = [tuple(x) for x in df_out.to_numpy()]
    with conn.cursor() as cur:
        execute_values(cur, insert_sql, rows, page_size=5000)
    conn.commit()


def run_evaluation_and_store_minimal(
    cfg: dict,
    source_table: str,
    dest_table: str,
    metrics_table: str,
    model_filename: str,
    id_col: str = "id",
    min_precision: float = 0.12,
    step: float = 0.001,
):
    """
    Evalúa TODA la tabla source_table en PostgreSQL.

    Genera y guarda (borrando/recreando SIEMPRE):
      - dest_table: tx_id + amount + y_true(0/1) + y_pred + y_proba
      - metrics_table: 1 fila con threshold + métricas + montos
    """
    _assert_safe_identifier(source_table, "source_table")
    _assert_safe_identifier(dest_table, "dest_table")
    _assert_safe_identifier(metrics_table, "metrics_table")
    _assert_safe_identifier(id_col, "id_col")

    model_path = os.path.join(os.getcwd(), "models", model_filename)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No existe el modelo en: {model_path}")

    model = joblib.load(model_path)

    conn = make_conn(cfg["host"], cfg["port"], cfg["dbname"], cfg["user"], cfg["password"], False)
    try:
        df_test = _read_df_full_table(conn, source_table)

        needed = [id_col] + VARS_FINALES + ["Estado"]
        missing = [c for c in needed if c not in df_test.columns]
        if missing:
            raise ValueError(f"Faltan columnas en {source_table}: {missing}")

        X_val = df_test[VARS_FINALES].copy()
        y_val = df_test["Estado"].copy()

        # Split estratificado: usamos el "primer split" como conjunto de evaluación
        X_val1, _, _, _, y_val1, _ = preparar_datos_full(X_val, y_val, target_pos_label="F", test_size=0.2)

        # Probabilidades
        y_proba = model.predict_proba(X_val1)[:, 1]

        res = best_threshold_max_recall_with_min_precision(y_val1, y_proba, min_precision=min_precision, step=step)
        threshold = res.get("threshold")
        if threshold is None:
            threshold = 1.0

        y_pred = (y_proba >= threshold).astype(int)

        # Alinear IDs y montos con el índice de X_val1 (importante)
        if hasattr(X_val1, "index") and isinstance(X_val1.index, pd.Index):
            idx = X_val1.index
            ids = df_test.loc[idx, id_col].astype(str).values
            amounts = pd.to_numeric(df_test.loc[idx, "chbillingAmount"], errors="coerce").fillna(0.0).values
        else:
            ids = df_test[id_col].astype(str).iloc[: len(y_proba)].values
            amounts = (
                pd.to_numeric(df_test["chbillingAmount"], errors="coerce")
                .fillna(0.0)
                .iloc[: len(y_proba)]
                .values
            )

        y_true_arr = np.asarray(y_val1, dtype=int)
        y_pred_arr = y_pred.astype(int)
        amt = np.asarray(amounts, dtype=float)

        # Confusion
        tp = int(((y_true_arr == 1) & (y_pred_arr == 1)).sum())
        fp = int(((y_true_arr == 0) & (y_pred_arr == 1)).sum())
        fn = int(((y_true_arr == 1) & (y_pred_arr == 0)).sum())
        tn = int(((y_true_arr == 0) & (y_pred_arr == 0)).sum())

        precision = float(res.get("precision", 0.0))
        recall = float(res.get("recall", 0.0))
        f1 = float((2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0)
        accuracy = float((tp + tn) / max(1, (tp + tn + fp + fn)))
        alert_rate = float((tp + fp) / max(1, len(y_true_arr)))

        fraud_amount_total = float(amt[y_true_arr == 1].sum())
        fraud_amount_captured = float(amt[(y_true_arr == 1) & (y_pred_arr == 1)].sum())
        fraud_amount_missed = float(amt[(y_true_arr == 1) & (y_pred_arr == 0)].sum())
        fp_amount = float(amt[(y_true_arr == 0) & (y_pred_arr == 1)].sum())

        # Output pred table
        df_out = pd.DataFrame(
            {
                "tx_id": ids,
                "amount": amt.astype(float),
                "y_true": y_true_arr.astype(int),
                "y_pred": y_pred_arr.astype(int),
                "y_proba": y_proba.astype(float),
            }
        )

        _recreate_predictions_table_minimal(conn, dest_table)
        _insert_df(conn, dest_table, df_out)

        # Metrics table
        _recreate_metrics_table(conn, metrics_table)
        df_metrics = pd.DataFrame(
            [
                {
                    "source_table": source_table,
                    "model_filename": model_filename,
                    "model_path": model_path,
                    "threshold_used": float(threshold),
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "accuracy": accuracy,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "tn": tn,
                    "rows_written": int(len(df_out)),
                    "alert_rate": alert_rate,
                    "fraud_amount_total": fraud_amount_total,
                    "fraud_amount_captured": fraud_amount_captured,
                    "fraud_amount_missed": fraud_amount_missed,
                    "fp_amount": fp_amount,
                }
            ]
        )
        _insert_df(conn, metrics_table, df_metrics)

        return {
            "model_path": model_path,
            "model_filename": model_filename,
            "source_table": source_table,
            "dest_table": dest_table,
            "metrics_table": metrics_table,
            "threshold_result": res,
            "threshold_used": float(threshold),
            "rows_written": int(len(df_out)),
        }

    finally:
        try:
            conn.close()
        except Exception:
            pass