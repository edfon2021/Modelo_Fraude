import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor

def make_conn(host, port, dbname, user, password, dict_cursor: bool = True):
    return psycopg2.connect(
        host=host,
        port=port,
        dbname=dbname,
        user=user,
        password=password,
        cursor_factory=RealDictCursor if dict_cursor else None
    )

def fetch_df(conn, sql: str, params=None) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
    return pd.DataFrame(rows)

def test_connection(host: str, port: int, dbname: str, user: str, password: str) -> tuple[bool, str]:
    try:
        conn = make_conn(host, port, dbname, user, password)
        conn.close()
        return True, "Conexión OK ✅"
    except Exception as e:
        return False, f"Error conexión: {e}"

def get_first_transactions(conn, table_name: str, limit: int = 100, date_col: str = "Fecha") -> pd.DataFrame:
    sql = f'SELECT * FROM {table_name} ORDER BY "{date_col}" ASC LIMIT %s;'
    return fetch_df(conn, sql, (limit,))



def get_resumen_mensual(conn, table_name: str, date_col: str = "Fecha", estado_col: str = "Estado") -> pd.DataFrame:
    """
    Devuelve un resumen mensual de fraudes vs normales desde Postgres (NO desde la muestra).
    Asume Estado: 'F' = fraude, 'N' = normal.
    """
    sql = f"""
        SELECT
            to_char(date_trunc('month', "{date_col}"), 'YYYY-MM') AS mes,
            SUM(CASE WHEN upper("{estado_col}"::text) = 'F' THEN 1 ELSE 0 END) AS fraudes,
            SUM(CASE WHEN upper("{estado_col}"::text) = 'N' THEN 1 ELSE 0 END) AS normales_tomar
        FROM {table_name}
        GROUP BY 1
        ORDER BY 1;
    """
    return fetch_df(conn, sql)