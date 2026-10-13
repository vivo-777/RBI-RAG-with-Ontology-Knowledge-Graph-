import sqlite3
import os

DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "qdrant_data", "facts.db"))

def get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    return sqlite3.connect(DB_PATH)

def init_db():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS structured_facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id TEXT,
            filename TEXT,
            page_number INTEGER,
            entity TEXT,
            metric TEXT,
            value TEXT,
            unit TEXT,
            as_of_date TEXT,
            table_row_id INTEGER,
            extraction_confidence REAL,
            raw_context TEXT
        )
    ''')
    conn.commit()
    conn.close()

def insert_fact(fact_data: dict):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO structured_facts (
            document_id, filename, page_number, entity, metric, value, unit, as_of_date, table_row_id, extraction_confidence, raw_context
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        fact_data.get("document_id"),
        fact_data.get("filename"),
        fact_data.get("page_number"),
        fact_data.get("entity", ""),
        fact_data.get("metric", ""),
        fact_data.get("value", ""),
        fact_data.get("unit", ""),
        fact_data.get("as_of_date", ""),
        fact_data.get("table_row_id", 0),
        fact_data.get("extraction_confidence", 1.0),
        fact_data.get("raw_context", "")
    ))
    conn.commit()
    conn.close()

def get_all_facts():
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM structured_facts')
    results = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return results

init_db()
