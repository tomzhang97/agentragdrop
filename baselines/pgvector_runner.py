
# Minimal pgvector baseline stub. Requires a running Postgres with pgvector.
# pip install psycopg2-binary
import numpy as np
import psycopg2

def search_pgvector(conn_str: str, table: str, q_emb: np.ndarray, k=5):
    conn = psycopg2.connect(conn_str)
    cur = conn.cursor()
    cur.execute(f"SELECT text FROM {table} ORDER BY embedding <-> %s LIMIT %s;", (list(q_emb), k))
    rows = [r[0] for r in cur.fetchall()]
    cur.close()
    conn.close()
    return rows
