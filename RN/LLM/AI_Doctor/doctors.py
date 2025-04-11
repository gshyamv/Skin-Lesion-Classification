import csv
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import SQLAlchemyError

# --- Database Credentials ---
DATABASE_USER = "root"
DATABASE_PASSWORD = "devarajan#8"
DATABASE_HOST = "localhost"
DATABASE_PORT = "8808"
DATABASE_NAME = "myappss"

# Construct the Database URL
DATABASE_URL = f"mysql+pymysql://{DATABASE_USER}:{DATABASE_PASSWORD}@{DATABASE_HOST}:{DATABASE_PORT}/{DATABASE_NAME}"

# --- Create Engine and Session ---
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# --- Define the Base Model ---
Base = declarative_base()

# --- Doctor Model ---
class Doctor(Base):
    __tablename__ = 'doctor'
    doc_id = Column(Integer, primary_key=True, autoincrement=True)
    first_name = Column(String(50))
    last_name = Column(String(50))
    clinic_name = Column(String(100))
    city = Column(String(50))
    specialty = Column(String(50))
    years_of_experience = Column(Integer)

# Create tables in the database if they do not exist
Base.metadata.create_all(bind=engine)

# --- Helper Function to Clean and Convert Numeric Fields ---
def clean_numeric(value):
    """
    Clean a string by stripping whitespace and removing any backticks,
    then return an integer if possible.
    """
    cleaned = value.strip().replace("`", "")
    if cleaned == "":
        raise ValueError("Empty numeric value")
    return int(cleaned)

# --- Function to Insert Doctors from CSV ---
def insert_doctors_from_csv(csv_file):
    session = SessionLocal()
    try:
        with open(csv_file, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            row_count = 0
            skipped = 0
            for row in reader:
                row_count += 1
                try:
                    # Clean and convert numeric fields.
                    doc_id_val = clean_numeric(row['DocID'])
                    years_val = clean_numeric(row['YearsofExperience'])
                except ValueError as ve:
                    # Log error and skip this row
                    print(f"Skipping row {row_count}: Error converting numeric field: {ve} - Row data: {row}")
                    skipped += 1
                    continue

                # Create a new Doctor instance using cleaned row data.
                doctor = Doctor(
                    doc_id=doc_id_val,
                    first_name=row['FirstName'].strip(),
                    last_name=row['LastName'].strip(),
                    clinic_name=row['ClinicHospitalName'].strip(),
                    city=row['City'].strip(),
                    specialty=row['Specialty'].strip(),
                    years_of_experience=years_val
                )
                session.add(doctor)
            session.commit()
        print(f"✅ All doctors inserted successfully! Processed: {row_count}, Skipped: {skipped}")
    except SQLAlchemyError as e:
        session.rollback()
        print(f"❌ A database error occurred: {e}")
    except Exception as ex:
        session.rollback()
        print(f"❌ Unexpected error: {ex}")
    finally:
        session.close()

# --- Main Execution ---
if __name__ == "__main__":
    CSV_FILE = r"C:\Users\rdeva\Downloads\SEM3\SEM4\RN\LLM\AI_Doctor\DocID-FirstName-LastName-ClinicHospitalName-City-Specialty-YearsofExperience.csv"
    insert_doctors_from_csv(CSV_FILE)
