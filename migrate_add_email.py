# migrate_add_email.py  (run once)
import sqlite3, os, sys

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
db = os.path.join(BASE_DIR, "users.db")

print("Using DB:", db)
conn = sqlite3.connect(db)
cur = conn.cursor()

# check if email column exists
cur.execute("PRAGMA table_info(users)")
cols = [r[1] for r in cur.fetchall()]
if "email" in cols:
    print("email column already exists. nothing to do.")
    conn.close()
    sys.exit(0)

# safe approach: create new table, copy data, rename
try:
    cur.execute("""
    CREATE TABLE users_new (
        id INTEGER PRIMARY KEY,
        username TEXT UNIQUE,
        password TEXT,
        email TEXT
    )
    """)
    cur.execute("INSERT INTO users_new (id, username, password) SELECT id, username, password FROM users")
    conn.commit()
    cur.execute("DROP TABLE users")
    cur.execute("ALTER TABLE users_new RENAME TO users")
    conn.commit()
    print("Migration completed. 'email' column added.")
except Exception as e:
    print("Migration failed:", e)
finally:
    conn.close()



