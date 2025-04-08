from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
import base64
from werkzeug.utils import secure_filename
import os
import pymysql

# Replace pymysql with mysqlclient if you prefer
pymysql.install_as_MySQLdb()

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Configure MySQL database connection with custom port and hardcoded password
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:devarajan#8@localhost:8808/myapp'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    name = db.Column(db.String(120), nullable=False)
    first_name = db.Column(db.String(120))
    last_name = db.Column(db.String(120))
    gender = db.Column(db.String(20))
    dob = db.Column(db.String(20))

class Image(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120))
    data = db.Column(db.Text(length=4294967295))
    content_type = db.Column(db.String(50))

@app.route("/register", methods=["POST"])
def register():
    try:
        data = request.get_json()
        email = data.get("email")
        name = data.get("name")
        if not email or not name:
            return jsonify({"error": "Email and name are required"}), 400

        # Check if the user already exists
        if User.query.filter_by(email=email).first():
            return jsonify({"error": "User already exists"}), 400

        new_user = User(email=email, name=name)
        db.session.add(new_user)
        db.session.commit()
        return jsonify({"message": "User registered successfully", "user_id": new_user.id}), 201

    except Exception as e:
        print(e)
        return jsonify({"error": "Server error"}), 500

@app.route("/details", methods=["POST"])
def update_details():
    try:
        data = request.get_json()
        email = data.get("email")
        firstName = data.get("firstName")
        lastName = data.get("lastName")
        gender = data.get("gender")
        dob = data.get("dob")

        if not email:
            return jsonify({"error": "Email is required"}), 400

        user = User.query.filter_by(email=email).first()
        if not user:
            return jsonify({"error": "User not found"}), 404

        user.first_name = firstName
        user.last_name = lastName
        user.gender = gender
        user.dob = dob
        db.session.commit()
        return jsonify({"message": "Details updated successfully"}), 200

    except Exception as e:
        print(e)
        return jsonify({"error": "Server error"}), 500

@app.route("/getDetails", methods=["GET"])
def get_details():
    try:
        email = request.args.get("email")
        if not email:
            return jsonify({"error": "Email is required"}), 400
        
        user = User.query.filter_by(email=email).first()
        if not user:
            return jsonify({"error": "User not found"}), 404

        result = {
            "name": user.name,
            "email": user.email,
            "details": {
                "firstName": user.first_name,
                "lastName": user.last_name,
                "gender": user.gender,
                "dob": user.dob
            }
        }
        return jsonify({"details": result}), 200

    except Exception as e:
        print(e)
        return jsonify({"error": "Server error"}), 500

@app.route("/upload", methods=["POST"])
def upload_image():
    try:
        if "photo" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        image_file = request.files["photo"]
        filename = secure_filename(image_file.filename)
        content_type = image_file.content_type

        # Convert image data to base64 string
        image_data = base64.b64encode(image_file.read()).decode("utf-8")

        new_image = Image(name=filename, data=image_data, content_type=content_type)
        db.session.add(new_image)
        db.session.commit()
        return jsonify({"message": "Image uploaded successfully", "image_id": new_image.id}), 200

    except Exception as e:
        print(e)
        return jsonify({"error": "Error uploading image"}), 500

def create_database_if_not_exists():
    """Create the database if it doesn't exist"""
    try:
        # Create a temporary connection to MySQL server without specifying a database
        conn = pymysql.connect(
            host='localhost',
            port=8808,
            user='root',
            password='devarajan#8'
        )
        cursor = conn.cursor()
        
        # Check if database exists
        cursor.execute("SHOW DATABASES LIKE 'myapp'")
        result = cursor.fetchone()
        
        # If database doesn't exist, create it
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
    # Create the database if it doesn't exist
    create_database_if_not_exists()
    
    with app.app_context():
        # Create tables if they don't exist
        db.create_all()
        print("Database tables created/verified.")
    
    print("Starting Flask server...")
    app.run(host="0.0.0.0", port=5000, debug=True)