import sqlite3

# Database file name
DB_NAME = "skin.db"

# SQL Statements to create tables
TABLES = [
    """
    CREATE TABLE IF NOT EXISTS Patient (
        pid INTEGER PRIMARY KEY AUTOINCREMENT,
        dob DATE NOT NULL,
        gender TEXT CHECK (gender IN ('Male', 'Female', 'Other')) NOT NULL,
        first_name TEXT NOT NULL,
        last_name TEXT,
        doc_id INTEGER,
        FOREIGN KEY (doc_id) REFERENCES Doctor(doc_id) ON DELETE SET NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS Patient_Info (
        pid INTEGER PRIMARY KEY,
        address TEXT NOT NULL,
        city TEXT NOT NULL,
        phone_no TEXT NOT NULL,
        email TEXT,
        FOREIGN KEY (pid) REFERENCES Patient(pid) ON DELETE CASCADE
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS Doctor (
        doc_id INTEGER PRIMARY KEY AUTOINCREMENT,
        first_name TEXT NOT NULL,
        last_name TEXT,
        clinic_name TEXT,
        city TEXT NOT NULL,
        specialty TEXT NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS Doctor_Info (
        doc_id INTEGER PRIMARY KEY,
        prescription TEXT NOT NULL,
        FOREIGN KEY (doc_id) REFERENCES Doctor(doc_id) ON DELETE CASCADE
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS Appointment (
        app_id INTEGER PRIMARY KEY AUTOINCREMENT,
        date DATE NOT NULL,
        time TIME NOT NULL,
        pid INTEGER,
        doc_id INTEGER,
        FOREIGN KEY (pid) REFERENCES Patient(pid) ON DELETE CASCADE,
        FOREIGN KEY (doc_id) REFERENCES Doctor(doc_id) ON DELETE CASCADE
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS Lesion (
        lesion_id INTEGER PRIMARY KEY AUTOINCREMENT,
        previous_prescription TEXT,
        image_file_name TEXT NOT NULL,
        pid INTEGER,
        report_id INTEGER,
        FOREIGN KEY (pid) REFERENCES Patient(pid) ON DELETE CASCADE,
        FOREIGN KEY (report_id) REFERENCES AI_Doctor(rep_id) ON DELETE CASCADE
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS Record (
        record_id INTEGER PRIMARY KEY AUTOINCREMENT,
        age INTEGER GENERATED ALWAYS AS (strftime('%Y', 'now') - strftime('%Y', dob)) STORED,
        medical_history TEXT NOT NULL,
        insured BOOLEAN,
        notes TEXT NOT NULL,
        pid INTEGER,
        rep_id INTEGER,
        FOREIGN KEY (pid) REFERENCES Patient(pid) ON DELETE CASCADE,
        FOREIGN KEY (rep_id) REFERENCES AI_Doctor(rep_id) ON DELETE CASCADE
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS Record_Info (
        record_id INTEGER,
        allergy TEXT,
        FOREIGN KEY (record_id) REFERENCES Record(record_id) ON DELETE CASCADE
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS AI_Doctor (
        rep_id INTEGER PRIMARY KEY AUTOINCREMENT,
        diagnosis TEXT NOT NULL,
        severity_level TEXT CHECK (severity_level IN ('Low', 'Med', 'High')) NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS AI_Doctor_Info (
        rep_id INTEGER PRIMARY KEY,
        prescription TEXT NOT NULL,
        FOREIGN KEY (rep_id) REFERENCES AI_Doctor(rep_id) ON DELETE CASCADE
    );
    """
]

# Function to create tables in SQLite3
def create_tables():
    try:
        # Connect to SQLite3 database (creates file if not exists)
        conn = sqlite3.connect(DB_NAME)
        cur = conn.cursor()
        
        # Execute table creation queries
        for table_sql in TABLES:
            cur.execute(table_sql)
        
        # Commit and close connection
        conn.commit()
        cur.close()
        conn.close()
        print("All tables created successfully in SQLite3.")

    except Exception as e:
        print(f"Error: {e}")

# Run the function
if __name__ == "__main__":
    create_tables()