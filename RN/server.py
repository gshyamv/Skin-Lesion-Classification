from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
import base64
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)  

# MongoDB Connection
MONGO_URI = "mongodb://localhost:27017/imagedb" 
client = MongoClient(MONGO_URI)
db = client["imagedb"]
images_collection = db["images"]

@app.route("/upload", methods=["POST"])
def upload_image():
    try:
        if "photo" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        image_file = request.files["photo"]
        filename = secure_filename(image_file.filename)
        content_type = image_file.content_type

        # Convert image to base64
        image_data = base64.b64encode(image_file.read()).decode("utf-8")

        # Store in MongoDB
        image_doc = {
            "name": filename,
            "data": image_data,
            "content_type": content_type
        }
        image_id = images_collection.insert_one(image_doc).inserted_id

        return jsonify({"message": "Image uploaded successfully", "image_id": str(image_id)}), 200

    except Exception as e:
        print(e)
        return jsonify({"error": "Error uploading image"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
