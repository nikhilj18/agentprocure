# database/db_connect.py
# ─────────────────────────────────────────────────────────────
# PURPOSE: One place to handle all database connections.
# Every other module imports from here — they never write
# their own connection code. This is called the DRY principle:
# Don't Repeat Yourself.
# ─────────────────────────────────────────────────────────────

import psycopg2
import pandas as pd
from dotenv import load_dotenv
import os

# This reads your .env file and loads DB_HOST, DB_PORT, etc.
# into the program's environment.
load_dotenv()


def get_connection():
    """
    Returns a live connection to the agentprocure PostgreSQL database.
    Uses credentials from the .env file — never hardcoded.
    """
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD")
    )
    return conn


def run_query(sql, params=None):
    """
    Runs a SELECT query and returns a pandas DataFrame.

    Args:
        sql    (str):   The SQL SELECT statement.
        params (tuple): Values to safely inject into the query.
                        Always use params instead of string formatting
                        to prevent SQL injection attacks.

    Returns:
        pd.DataFrame: Query results as a table.

    Example:
        df = run_query(
            "SELECT * FROM po_history WHERE part_no = %s",
            params=("RES-0402-10K",)
        )
    """
    conn = get_connection()
    try:
        df = pd.read_sql(sql, conn, params=params)
    finally:
        conn.close()
    return df


def run_insert(sql, params):
    """
    Runs an INSERT, UPDATE, or DELETE statement.
    Commits changes so they are saved permanently.
    Rolls back automatically if an error occurs.

    Args:
        sql    (str):   The SQL statement with %s placeholders.
        params (tuple): Values to fill in the placeholders.

    Example:
        run_insert(
            "INSERT INTO component_master (part_no, description) VALUES (%s, %s)",
            ("RES-0402-10K", "10K Ohm Resistor 0402")
        )
    """
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(sql, params)
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        cursor.close()
        conn.close()


def run_insert_many(sql, params_list):
    """
    Inserts multiple rows efficiently in one database call.
    Much faster than calling run_insert() in a loop.

    Args:
        sql         (str):        SQL INSERT with %s placeholders.
        params_list (list[tuple]): List of value tuples, one per row.

    Example:
        rows = [
            ("RES-0402-10K", "10K Resistor"),
            ("CAP-0805-100N", "100nF Capacitor"),
        ]
        run_insert_many(
            "INSERT INTO component_master (part_no, description) VALUES (%s, %s)",
            rows
        )
    """
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.executemany(sql, params_list)
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        cursor.close()
        conn.close()


def table_exists(table_name):
    """
    Returns True if a table with the given name exists in the database.
    Useful before running schema.sql to avoid re-creating existing tables.
    """
    df = run_query(
        "SELECT 1 FROM information_schema.tables WHERE table_schema='public' AND table_name=%s",
        params=(table_name,)
    )
    return len(df) > 0


# ── SELF-TEST ────────────────────────────────────────────────
# Run this file directly to test: python database/db_connect.py

if __name__ == "__main__":
    print("\n🔍 Testing database connection...")
    print("=" * 50)

    try:
        df = run_query(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'public' ORDER BY table_name"
        )

        if df.empty:
            print("⚠️  Connected but no tables found.")
            print("    Have you run schema.sql yet?")
            print("    Command: psql -U postgres -d agentprocure -f database/schema.sql")
        else:
            print(f"✅ Connected to '{os.getenv('DB_NAME')}' on {os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}")
            print(f"\n📋 Tables in database ({len(df)} found):")
            for t in df['table_name']:
                print(f"   ✔ {t}")

            # Check all 6 expected tables exist
            expected = {
                'component_master', 'supplier_master',
                'approved_vendor_list', 'po_history',
                'bom_input', 'sourcing_recommendations'
            }
            found = set(df['table_name'])
            missing = expected - found

            if missing:
                print(f"\n⚠️  Missing tables: {missing}")
                print("   Re-run: psql -U postgres -d agentprocure -f database/schema.sql")
            else:
                print("\n🎉 All 6 required tables are present. Phase 1 database setup complete!")

    except Exception as e:
        print(f"\n❌ Connection failed!")
        print(f"   Error: {e}")
        print("\n🔧 Troubleshooting:")
        print("   1. Is PostgreSQL running?")
        print("      Windows: Check Services app for 'postgresql-x64-15'")
        print("      Mac:     Run 'brew services list' or check System Preferences")
        print("   2. Is the password in .env correct?")
        print("   3. Did you create the database?")
        print("      Run: psql -U postgres -c \"CREATE DATABASE agentprocure;\"")