# server.py

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Query, Body, Form
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, String, Date, Time, ForeignKey, Boolean, Text, Enum as SqlEnum
from sqlalchemy.orm import sessionmaker, relationship, Session, declarative_base
from contextlib import asynccontextmanager
import pymysql
import enum
import base64
from werkzeug.utils import secure_filename
from datetime import date, datetime
from pydantic import BaseModel, EmailStr
from LLM.AI_Doctor.temp_function import get_random_diagnosis
from LLM.AI_Doctor.Untitled import test
from LLM.AI_Doctor.format_summary import replace_newline_with_br, replace_t_with_tab

# --- Setup MySQL with PyMySQL ---
pymysql.install_as_MySQLdb()
DATABASE_URL = "mysql+pymysql://root:devarajan#8@localhost:8808/myappss"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# --- Enum for Lesion Types ---
class LesionType(enum.Enum):
    MELANOMA = "Melanoma"
    NEVUS = "Nevus"
    BASAL_CELL_CARCINOMA = "Basal Cell Carcinoma"
    ACTINIC_KERATOSIS = "Actinic Keratosis"
    BENIGN_KERATOSIS = "Benign Keratosis"
    DERMATOFIBROMA = "Dermatofibroma"
    VASCULAR_LESION = "Vascular Lesion"

# --- Models ---
class Patient(Base):
    __tablename__ = 'patient'
    pid = Column(Integer, primary_key=True, autoincrement=True)
    dob = Column(Date)
    gender = Column(String(10))
    first_name = Column(String(50), nullable=False)
    last_name = Column(String(50), nullable=False)
    doc_id = Column(Integer, ForeignKey('doctor.doc_id'))
    
    doctor = relationship('Doctor', back_populates='patients')
    patient_info = relationship('PatientInfo', back_populates='patient', uselist=False)
    appointments = relationship('Appointment', back_populates='patient')
    lesions = relationship('Lesion', back_populates='patient')
    records = relationship('Record', back_populates='patient')

class PatientInfo(Base):
    __tablename__ = 'patient_info'
    pid = Column(Integer, ForeignKey('patient.pid', ondelete="CASCADE"), primary_key=True)
    address = Column(String(255))
    phone_no = Column(String(20))
    email = Column(String(100), unique=True)
    city = Column(String(100))
    
    patient = relationship('Patient', back_populates='patient_info')

class Doctor(Base):
    __tablename__ = 'doctor'
    doc_id = Column(Integer, primary_key=True, autoincrement=True)
    first_name = Column(String(50))
    last_name = Column(String(50))
    clinic_name = Column(String(100))
    city = Column(String(50))
    specialty = Column(String(50))
    years_of_experience = Column(Integer)
    
    doctor_info = relationship('DoctorInfo', back_populates='doctor', uselist=False)
    patients = relationship('Patient', back_populates='doctor')
    appointments = relationship('Appointment', back_populates='doctor')
    lesions = relationship('Lesion', back_populates='doctor')

class DoctorInfo(Base):
    __tablename__ = 'doctor_info'
    doc_id = Column(Integer, ForeignKey('doctor.doc_id'), primary_key=True)
    prescription = Column(String(255))
    
    doctor = relationship('Doctor', back_populates='doctor_info')

class Appointment(Base):
    __tablename__ = 'appointment'
    app_id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date)
    time = Column(Time)
    pid = Column(Integer, ForeignKey('patient.pid'))
    doc_id = Column(Integer, ForeignKey('doctor.doc_id'))
    
    patient = relationship('Patient', back_populates='appointments')
    doctor = relationship('Doctor', back_populates='appointments')

class Lesion(Base):
    __tablename__ = 'lesion'
    lesion_id = Column(Integer, primary_key=True, autoincrement=True)
    previous_prescription = Column(String(255))
    image_file_name = Column(String(255))
    lesion_type = Column(SqlEnum(LesionType), nullable=False)
    pid = Column(Integer, ForeignKey('patient.pid'))
    report_id = Column(Integer, ForeignKey('record.record_id'))
    doc_id = Column(Integer, ForeignKey('doctor.doc_id'))

    patient = relationship('Patient', back_populates='lesions')
    doctor = relationship('Doctor', back_populates='lesions')

class AIDoctor(Base):
    __tablename__ = 'ai_doctor'
    rep_id = Column(Integer, primary_key=True, autoincrement=True)
    diagnosis = Column(Text)
    severity_level = Column(String(50))
    
    ai_doctor_info = relationship('AIDoctorInfo', back_populates='ai_doctor', uselist=False)
    records = relationship('Record', back_populates='ai_doctor')

class AIDoctorInfo(Base):
    __tablename__ = 'ai_doctor_info'
    rep_id = Column(Integer, ForeignKey('ai_doctor.rep_id'), primary_key=True)
    prescription = Column(Text)
    
    ai_doctor = relationship('AIDoctor', back_populates='ai_doctor_info')

class Record(Base):
    __tablename__ = 'record'
    record_id = Column(Integer, primary_key=True, autoincrement=True)
    age = Column(Integer)
    medical_history = Column(Text)
    insured = Column(Boolean)
    notes = Column(Text)
    pid = Column(Integer, ForeignKey('patient.pid'))
    rep_id = Column(Integer, ForeignKey('ai_doctor.rep_id'))
    
    record_info = relationship('RecordInfo', back_populates='record', cascade="all, delete-orphan")
    patient = relationship('Patient', back_populates='records')
    ai_doctor = relationship('AIDoctor', back_populates='records')

class RecordInfo(Base):
    __tablename__ = 'record_info'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    record_id = Column(Integer, ForeignKey('record.record_id'))
    allergy = Column(String(255))
    
    record = relationship('Record', back_populates='record_info')

class Image(Base):
    __tablename__ = 'image'
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    data = Column(Text, nullable=False)
    content_type = Column(String(100), nullable=False)

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Starting up... Creating tables if not exist")
    Base.metadata.create_all(bind=engine)
    yield
    print("🛑 Shutting down... Cleanup if needed")

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class RegisterRequest(BaseModel):
    email: EmailStr
    firstName: str
    lastName: str = ""
    dob: date
    gender: str
    address: str
    phone_no: str
    city: str

class UpdateUserRequest(BaseModel):
    email: EmailStr
    firstName: str = None
    lastName: str = None
    dob: date = None
    gender: str = None
    address: str = None
    phone_no: str = None
    city: str = None
    medical_history: str = ""
    insured: bool = False
    notes: str = ""

@app.post("/register")
def register(payload: RegisterRequest, db: Session = Depends(get_db)):
    email = payload.email.strip()
    existing = db.query(PatientInfo).filter_by(email=email).first()
    if existing:
        raise HTTPException(status_code=400, detail="User already exists")

    new_patient = Patient(
        dob=payload.dob,
        gender=payload.gender,
        first_name=payload.firstName,
        last_name=payload.lastName
    )
    db.add(new_patient)
    db.commit()
    db.refresh(new_patient)

    new_patient_info = PatientInfo(
        pid=new_patient.pid,
        address=payload.address,
        phone_no=payload.phone_no,
        email=email,
        city=payload.city
    )
    db.add(new_patient_info)
    db.commit()

    return {"message": "User registered successfully", "pid": new_patient.pid}

@app.post("/updateUser")
def update_user(payload: UpdateUserRequest, db: Session = Depends(get_db)):
    email = payload.email.strip()
    patient_info = db.query(PatientInfo).filter_by(email=email).first()
    if not patient_info:
        if not (payload.firstName and payload.dob and payload.address and payload.phone_no and payload.city):
            raise HTTPException(status_code=400, detail="Missing compulsory fields for new user")
        
        new_patient = Patient(
            dob=payload.dob,
            gender=payload.gender or "Not Specified",
            first_name=payload.firstName,
            last_name=payload.lastName or ""
        )
        db.add(new_patient)
        db.commit()
        db.refresh(new_patient)
        
        new_patient_info = PatientInfo(
            pid=new_patient.pid,
            address=payload.address,
            phone_no=payload.phone_no,
            email=email,
            city=payload.city
        )
        db.add(new_patient_info)
        db.commit()
        
        today = date.today()
        age = today.year - payload.dob.year - ((today.month, today.day) < (payload.dob.month, payload.dob.day))
        
        record = Record(
            pid=new_patient.pid,
            age=age,
            medical_history=payload.medical_history,
            insured=payload.insured,
            notes=payload.notes
        )
        db.add(record)
        db.commit()
        db.refresh(record)

        allergies_text = payload.notes.strip() if payload.notes else ""
        allergies_list = [allergy.strip() for allergy in allergies_text.split(',') if allergy.strip()]
        
        for allergy in allergies_list:
            record_info = RecordInfo(
                record_id=record.record_id,
                allergy=allergy
            )
            db.add(record_info)
        
        if not allergies_list:
            record_info = RecordInfo(
                record_id=record.record_id,
                allergy=""
            )
            db.add(record_info)
            
        db.commit()

        return {"message": "New user created and details saved", "pid": new_patient.pid}

    patient = db.query(Patient).filter_by(pid=patient_info.pid).first()
    if not patient:
        raise HTTPException(status_code=404, detail="User not found")

    if payload.firstName is not None:
        patient.first_name = payload.firstName
    if payload.lastName is not None:
        patient.last_name = payload.lastName
    if payload.dob is not None:
        patient.dob = payload.dob
    if payload.gender is not None:
        patient.gender = payload.gende
    if payload.address is not None:
        patient_info.address = payload.address
    if payload.phone_no is not None:
        patient_info.phone_no = payload.phone_no
    if payload.city is not None:
        patient_info.city = payload.city

    record = db.query(Record).filter_by(pid=patient.pid).first()
    if not record:
        record = Record(
            pid=patient.pid,
            medical_history=payload.medical_history,
            insured=payload.insured,
            notes=payload.notes
        )
        db.add(record)
    else:
        if payload.medical_history is not None:
            record.medical_history = payload.medical_history
        record.insured = payload.insured
        if payload.notes is not None:
            record.notes = payload.notes

    db.commit()
    return {"message": "User details updated successfully"}

@app.get("/getDetails")
def get_details(email: str = Query(...), db: Session = Depends(get_db)):
    email = email.strip()
    patient_info = db.query(PatientInfo).filter_by(email=email).first()
    if not patient_info:
        raise HTTPException(status_code=404, detail="User not found")
    
    patient = db.query(Patient).filter_by(pid=patient_info.pid).first()
    record = db.query(Record).filter_by(pid=patient.pid).first()
    
    patient_city = patient_info.city
    print("Retrieved city from DB:", patient_city)
    
    result = {
        "patient": {
            "firstName": patient.first_name,
            "lastName": patient.last_name,
            "dob": str(patient.dob),
            "gender": patient.gender,
        },
        "contact": {
            "address": patient_info.address,
            "phone_no": patient_info.phone_no,
            "email": patient_info.email,
            "city": patient_city
        },
        "record": {
            "medical_history": record.medical_history if record else "",
            "insured": record.insured if record else False,
            "notes": record.notes if record else ""
        }
    }
    return {"details": result}

@app.post("/upload")
def upload_image(
    photo: UploadFile = File(...),
    prescription: str = Form(...),
    db: Session = Depends(get_db)
):
    if not photo:
        raise HTTPException(status_code=400, detail="No file uploaded")
    filename = secure_filename(photo.filename)
    content_type = photo.content_type
    file_bytes = photo.file.read()
    image_data = base64.b64encode(file_bytes).decode("utf-8")
    new_image = Image(name=filename, data=image_data, content_type=content_type)
    db.add(new_image)
    db.commit()
    db.refresh(new_image)
    
    skin_lession = get_random_diagnosis()
    AI_diagnosis = test(skin_lession)
    AI_diagnosis = replace_newline_with_br(AI_diagnosis)
    AI_diagnosis = replace_t_with_tab(AI_diagnosis)
    
    new_ai_doctor = AIDoctor(
        diagnosis=AI_diagnosis,
        severity_level="Medium"
    )
    db.add(new_ai_doctor)
    db.commit()
    db.refresh(new_ai_doctor)

    ai_doctor_info = AIDoctorInfo(
        rep_id=new_ai_doctor.rep_id,
        prescription=AI_diagnosis
    )
    db.add(ai_doctor_info)
    db.commit()
    
    latest_record = db.query(Record).order_by(Record.record_id.desc()).first()
    if latest_record:
        latest_record.rep_id = new_ai_doctor.rep_id
        db.commit()
        
        new_lesion = Lesion(
            image_file_name=filename,
            lesion_type=LesionType.NEVUS,  
            pid=latest_record.pid,
            report_id=latest_record.record_id,
            doc_id=None,
            previous_prescription=prescription  
        )
        db.add(new_lesion)
        db.commit()
    else:
        print("Warning: No patient record found to update with AI diagnosis")
    
    return {"message": "Image uploaded successfully", "image_id": new_image.id, "diagnosis": AI_diagnosis}

@app.get("/getDoctors")
def get_doctors(city: str = Query(...), db: Session = Depends(get_db)):
    if not city:
        raise HTTPException(status_code=400, detail="City parameter is required")
    print("Searching doctors for city:", city)
    doctors = db.query(Doctor).filter(Doctor.city.ilike(f"%{city}%")).all()
    
    if not doctors:
        print(f"No doctors found in {city}, falling back to Chennai")
        doctors = db.query(Doctor).filter(Doctor.city.ilike("%chennai%")).all()
    doctor_list = []
    for doc in doctors:
        doctor_list.append({
            "doc_id": doc.doc_id,
            "first_name": doc.first_name,
            "last_name": doc.last_name,
            "clinic_name": doc.clinic_name,
            "city": doc.city,
            "specialty": doc.specialty,
            "years_of_experience": doc.years_of_experience
        })
    return {"doctors": doctor_list}

@app.post("/bookAppointment")
def book_appointment(
    doctorId: int = Body(...),
    date: str = Body(...),
    time: str = Body(...),
    patientEmail: str = Body(...),
    db: Session = Depends(get_db)
):
    print(f"Received appointment request: doctor={doctorId}, date={date}, time={time}, email={patientEmail}")
    
    patient_info = db.query(PatientInfo).filter_by(email=patientEmail).first()
    if not patient_info:
        raise HTTPException(status_code=404, detail=f"Patient with email {patientEmail} not found")
    
    patient_id = patient_info.pid
    
    if isinstance(date, str):
        try:
            appointment_date = datetime.strptime(date, "%Y-%m-%d").date()
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
    else:
        appointment_date = date
    
    try:
        time_obj = datetime.strptime(time, "%I:%M %p").time()
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid time format. Use HH:MM AM/PM")
    
    existing_appointment = db.query(Appointment).filter(
        Appointment.doc_id == doctorId,
        Appointment.date == appointment_date,
        Appointment.time == time_obj
    ).first()
    
    if existing_appointment:
        raise HTTPException(status_code=409, detail="This time slot is already booked")
    
    new_appointment = Appointment(
        date=appointment_date,
        time=time_obj,
        pid=patient_id,
        doc_id=doctorId
    )
    
    try:
        db.add(new_appointment)
        db.commit()
        db.refresh(new_appointment)
        
        patient = db.query(Patient).filter_by(pid=patient_id).first()
        if patient:
            patient.doc_id = doctorId
        lesions = db.query(Lesion).filter_by(pid=patient_id).all()
        for lesion in lesions:
            lesion.doc_id = doctorId
        db.commit()
        
        return {
            "message": "Appointment booked successfully", 
            "appointment_id": new_appointment.app_id,
            "details": {
                "doctor_id": doctorId,
                "patient_id": patient_id,
                "date": date,
                "time": time
            }
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to book appointment: {str(e)}")

@app.get("/getAvailableSlots")
def get_available_slots(doctor_id: int = Query(...), date: str = Query(...), db: Session = Depends(get_db)):
    try:
        appointment_date = datetime.strptime(date, "%Y-%m-%d").date()
        booked_appointments = db.query(Appointment).filter(
            Appointment.doc_id == doctor_id,
            Appointment.date == appointment_date
        ).all()
        booked_times = [appointment.time.strftime("%I:%M %p") for appointment in booked_appointments]
        return {"booked_slots": booked_times}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching available slots: {str(e)}")

# New endpoint: Get Appointments by patient email
@app.get("/getAppointments")
def get_appointments(email: str = Query(...), db: Session = Depends(get_db)):
    email = email.strip()
    patient_info = db.query(PatientInfo).filter_by(email=email).first()
    if not patient_info:
        raise HTTPException(status_code=404, detail="Patient not found")
    patient = db.query(Patient).filter_by(pid=patient_info.pid).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient record not found")

    appointments = db.query(Appointment).filter_by(pid=patient.pid).order_by(Appointment.date, Appointment.time).all()
    today_date = date.today()
    upcoming = []
    past = []
    for app in appointments:
        doc = db.query(Doctor).filter_by(doc_id=app.doc_id).first()
        app_data = {
            "appointment_id": app.app_id,
            "date": app.date.strftime("%Y-%m-%d"),
            "time": app.time.strftime("%I:%M %p"),
            "doctor": {
                "first_name": doc.first_name if doc else "",
                "last_name": doc.last_name if doc else "",
                "clinic_name": doc.clinic_name if doc else ""
            }
        }
        if app.date >= today_date:
            upcoming.append(app_data)
        else:
            past.append(app_data)
    return {"upcoming": upcoming, "past": past}

def create_database_if_not_exists():
    import pymysql
    try:
        conn = pymysql.connect(
            host='localhost',
            port=8808,
            user='root',
            password='devarajan#8'
        )
        cursor = conn.cursor()
        cursor.execute("SHOW DATABASES LIKE 'myapp'")
        result = cursor.fetchone()
        if not result:
            print("Creating database 'myapp'...")
            cursor.execute("CREATE DATABASE myapp")
            print("Database created successfully!")
        else:
            print("Database 'myapp' already exists.")
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"Error creating database: {e}")
        exit(1)

if __name__ == "__main__":
    create_database_if_not_exists()
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
