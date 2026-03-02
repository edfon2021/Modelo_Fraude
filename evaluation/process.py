# evaluation/process.py
import os
import re
import time
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


def _log(msg: str):
    print(f"[process.py] {msg}", flush=True)


def _assert_safe_identifier(name: str, what: str = "identifier"):
    if not isinstance(name, str) or not name:
        raise ValueError(f"{what} inválido (vacío).")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        raise ValueError(
            f"{what} inválido: '{name}'. Usá solo [A-Za-z0-9_] y que no empiece con número."
        )


def _set_timeouts(cur, lock_timeout_ms: int = 5000, statement_timeout_ms: int = 120000):
    cur.execute(f"SET lock_timeout = '{int(lock_timeout_ms)}ms';")
    cur.execute(f"SET statement_timeout = '{int(statement_timeout_ms)}ms';")


def preparar_datos_full(X, y, target_pos_label="F", test_size=0.2):
    _log("preparar_datos_full -> inicio")
    t0 = time.perf_counter()

    X_prep = X.copy()
    y_prep = y.copy()

    _log(f"preparar_datos_full -> X shape original: {X_prep.shape}")
    _log(f"preparar_datos_full -> y len original: {len(y_prep)}")

    # 1) TARGET a 0/1
    if y_prep.dtype == "object" or getattr(y_prep.dtype, "name", "") == "category":
        y_norm = y_prep.astype(str).str.strip().str.upper()
        pos_norm = str(target_pos_label).strip().upper()
        y_prep = y_norm.map({pos_norm: 1}).fillna(0).astype(int)
    else:
        y_prep = pd.to_numeric(y_prep, errors="coerce").fillna(0).astype(int)

    _log(
        f"preparar_datos_full -> target listo. Positivos: {int((y_prep == 1).sum())}, "
        f"Negativos: {int((y_prep == 0).sum())}"
    )

    # 2) Features: LabelEncoder en categóricas
    cat_cols = X_prep.select_dtypes(["category", "object"]).columns.tolist()
    _log(f"preparar_datos_full -> columnas categóricas: {cat_cols}")

    for col in cat_cols:
        _log(f"preparar_datos_full -> LabelEncoding columna: {col}")
        le = LabelEncoder()
        X_prep[col] = le.fit_transform(X_prep[col].astype(str))

    # 3) Split estratificado
    _log("preparar_datos_full -> antes de train_test_split")
    X_train, X_test, y_train, y_test = train_test_split(
        X_prep, y_prep, test_size=test_size, random_state=42, stratify=y_prep
    )
    _log(
        f"preparar_datos_full -> después de split | "
        f"X_train={X_train.shape}, X_test={X_test.shape}, "
        f"y_train={len(y_train)}, y_test={len(y_test)}"
    )

    # 4) Escalamiento (por compatibilidad)
    _log("preparar_datos_full -> antes de StandardScaler.fit_transform")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    _log("preparar_datos_full -> X_train_scaled OK")

    X_test_scaled = scaler.transform(X_test)
    _log("preparar_datos_full -> X_test_scaled OK")

    dt = time.perf_counter() - t0
    _log(f"preparar_datos_full -> fin en {dt:.3f}s")

    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test


def best_threshold_max_recall_with_min_precision(y_true, y_proba, min_precision=0.12, step=0.001):
    _log("best_threshold_max_recall_with_min_precision -> inicio")
    t0 = time.perf_counter()

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

    dt = time.perf_counter() - t0
    _log(f"best_threshold_max_recall_with_min_precision -> fin en {dt:.3f}s | best={best}")
    return best


def _read_df_full_table(conn, source_table: str) -> pd.DataFrame:
    _assert_safe_identifier(source_table, "source_table")
    sql = f'SELECT * FROM "{source_table}";'
    _log(f"_read_df_full_table -> ejecutando: {sql}")
    t0 = time.perf_counter()
    df = pd.read_sql(sql, conn)
    dt = time.perf_counter() - t0
    _log(f"_read_df_full_table -> OK shape={df.shape} en {dt:.3f}s")
    return df


def _clear_table(conn, table_name: str):
    """
    Vacía la tabla existente usando DELETE.
    No toca estructura.
    """
    _assert_safe_identifier(table_name, "table_name")
    sql = f'DELETE FROM "{table_name}";'
    _log(f"_clear_table -> vaciando {table_name}")
    t0 = time.perf_counter()
    with conn.cursor() as cur:
        _set_timeouts(cur)
        cur.execute(sql)
    conn.commit()
    _log(f"_clear_table -> OK en {time.perf_counter() - t0:.3f}s")


def _insert_df(conn, table: str, df_out: pd.DataFrame):
    _assert_safe_identifier(table, "table")
    _log(f"_insert_df -> inicio tabla={table} shape={df_out.shape}")
    _log(f"_insert_df -> dtypes originales:\n{df_out.dtypes}")

    cols = list(df_out.columns)
    cols_sql = ",".join([f'"{c}"' for c in cols])
    insert_sql = f'INSERT INTO "{table}" ({cols_sql}) VALUES %s'

    # coerción defensiva
    df_tmp = df_out.copy()
    for c in df_tmp.columns:
        if pd.api.types.is_integer_dtype(df_tmp[c]):
            df_tmp[c] = pd.to_numeric(df_tmp[c], errors="coerce").fillna(0).astype(int)
        elif pd.api.types.is_float_dtype(df_tmp[c]):
            df_tmp[c] = (
                pd.to_numeric(df_tmp[c], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
                .astype(float)
            )
        else:
            df_tmp[c] = df_tmp[c].astype(str)

    _log(f"_insert_df -> dtypes tras coerción:\n{df_tmp.dtypes}")
    _log(f"_insert_df -> head:\n{df_tmp.head(3)}")

    rows = [tuple(x) for x in df_tmp.to_numpy()]
    _log(f"_insert_df -> rows preparadas: {len(rows)}")

    t0 = time.perf_counter()
    with conn.cursor() as cur:
        _set_timeouts(cur)
        _log(f"_insert_df -> antes execute_values tabla={table}")
        execute_values(cur, insert_sql, rows, page_size=5000)
        _log(f"_insert_df -> execute_values OK tabla={table}")
    conn.commit()
    _log(f"_insert_df -> commit OK tabla={table} en {time.perf_counter() - t0:.3f}s")


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
    Evalúa la tabla source_table y guarda resultados en tablas ya existentes:
      - dest_table
      - metrics_table

    Estrategia:
      - NO crea tablas
      - NO las borra
      - solo hace DELETE + INSERT
    """
    _log("====================================================")
    _log("run_evaluation_and_store_minimal -> INICIO")
    t_global = time.perf_counter()

    _assert_safe_identifier(source_table, "source_table")
    _assert_safe_identifier(dest_table, "dest_table")
    _assert_safe_identifier(metrics_table, "metrics_table")
    _assert_safe_identifier(id_col, "id_col")

    _log(
        f"Parámetros -> source_table={source_table}, dest_table={dest_table}, "
        f"metrics_table={metrics_table}, model_filename={model_filename}, "
        f"id_col={id_col}, min_precision={min_precision}, step={step}"
    )

    model_path = os.path.join(os.getcwd(), "models", model_filename)
    _log(f"Model path -> {model_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No existe el modelo en: {model_path}")

    _log("1) Antes de joblib.load")
    t0 = time.perf_counter()
    model = joblib.load(model_path)
    _log(f"1) Después de joblib.load en {time.perf_counter() - t0:.3f}s | tipo modelo={type(model)}")

    _log("2) Antes de make_conn")
    conn = make_conn(cfg["host"], cfg["port"], cfg["dbname"], cfg["user"], cfg["password"], False)
    _log("2) Después de make_conn")

    try:
        _log("3) Antes de leer tabla completa")
        df_test = _read_df_full_table(conn, source_table)
        _log(f"3) Después de leer tabla -> shape={df_test.shape}")
        _log(f"3.1) Columnas disponibles: {list(df_test.columns)}")

        needed = [id_col] + VARS_FINALES + ["Estado"]
        missing = [c for c in needed if c not in df_test.columns]
        _log(f"4) Validando columnas requeridas -> needed={needed}")
        if missing:
            raise ValueError(f"Faltan columnas en {source_table}: {missing}")
        _log("4) Columnas requeridas OK")

        X_val = df_test[VARS_FINALES].copy()
        y_val = df_test["Estado"].copy()
        _log(f"5) X_val shape={X_val.shape}, y_val len={len(y_val)}")

        _log("6) Antes de preparar_datos_full")
        X_val1, _, _, _, y_val1, _ = preparar_datos_full(
            X_val, y_val, target_pos_label="F", test_size=0.2
        )
        _log(f"6) Después de preparar_datos_full -> X_val1 shape={X_val1.shape}, y_val1 len={len(y_val1)}")

        _log("7) Antes de predict_proba")
        t0 = time.perf_counter()
        y_proba = model.predict_proba(X_val1)[:, 1]
        _log(f"7) Después de predict_proba en {time.perf_counter() - t0:.3f}s -> len(y_proba)={len(y_proba)}")

        _log("8) Antes de best_threshold_max_recall_with_min_precision")
        res = best_threshold_max_recall_with_min_precision(
            y_val1, y_proba, min_precision=min_precision, step=step
        )
        _log(f"8) Después de threshold -> res={res}")

        threshold = res.get("threshold")
        if threshold is None:
            threshold = 1.0
            _log("8.1) No hubo threshold válido, se usa 1.0")

        y_pred = (y_proba >= threshold).astype(int)
        _log(f"9) y_pred generado -> positivos={int((y_pred == 1).sum())}, negativos={int((y_pred == 0).sum())}")

        _log("10) Alineando IDs y montos")
        if hasattr(X_val1, "index") and isinstance(X_val1.index, pd.Index):
            idx = X_val1.index
            _log(f"10.1) Índice detectado -> len(idx)={len(idx)}")
            ids = df_test.loc[idx, id_col].astype(str).values
            amounts = pd.to_numeric(df_test.loc[idx, "chbillingAmount"], errors="coerce").fillna(0.0).values
        else:
            _log("10.2) Sin índice usable, usando slice por longitud")
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

        _log(
            f"10.3) Arrays alineados -> ids={len(ids)}, y_true={len(y_true_arr)}, "
            f"y_pred={len(y_pred_arr)}, amt={len(amt)}"
        )

        _log("11) Calculando métricas base")
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

        _log(f"11.1) tp={tp}, fp={fp}, fn={fn}, tn={tn}")
        _log(
            f"11.2) precision={precision}, recall={recall}, f1={f1}, "
            f"accuracy={accuracy}, alert_rate={alert_rate}"
        )

        _log("12) Construyendo df_out")
        df_out = pd.DataFrame(
            {
                "tx_id": ids,
                "amount": amt.astype(float),
                "y_true": y_true_arr.astype(int),
                "y_pred": y_pred_arr.astype(int),
                "y_proba": y_proba.astype(float),
            }
        )
        _log(f"12.1) df_out shape={df_out.shape}")
        _log(f"12.2) df_out head:\n{df_out.head(3)}")

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
        _log(f"12.3) df_metrics:\n{df_metrics}")

        _log("13) Antes de vaciar tabla de predicciones")
        _clear_table(conn, dest_table)
        _log("13) Después de vaciar tabla de predicciones")

        _log("14) Antes de insertar predicciones")
        _insert_df(conn, dest_table, df_out)
        _log("14) Después de insertar predicciones")

        _log("15) Antes de vaciar tabla métricas")
        _clear_table(conn, metrics_table)
        _log("15) Después de vaciar tabla métricas")

        _log("16) Antes de insertar métricas")
        _insert_df(conn, metrics_table, df_metrics)
        _log("16) Después de insertar métricas")

        dt_global = time.perf_counter() - t_global
        _log(f"run_evaluation_and_store_minimal -> FIN OK en {dt_global:.3f}s")
        _log("====================================================")

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

    except Exception as e:
        try:
            conn.rollback()
        except Exception:
            pass
        _log(f"run_evaluation_and_store_minimal -> ERROR: {repr(e)}")
        raise

    finally:
        _log("finally -> cerrando conexión")
        try:
            conn.close()
            _log("finally -> conexión cerrada")
        except Exception as e:
            _log(f"finally -> error cerrando conexión: {repr(e)}")