from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import os
import cv2
import numpy as np
from deepface import DeepFace

app = Flask(__name__)
CORS(app)

# Path to stored suspect sketches
IMAGES_FOLDER = "D:\invasketch\Suspect-Sketch-Recognition-System\images"

def extract_face_embedding(image_path):
    """Extracts DeepFace embeddings using Facenet512"""
    try:
        embedding = DeepFace.represent(img_path=image_path, model_name="Facenet512", enforce_detection=False)
        return np.array(embedding[0]["embedding"])
    except:
        return None  # Return None if face detection fails

def cosine_similarity(embedding1, embedding2):
    """Calculates cosine similarity between two face embeddings"""
    dot_product = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    return (dot_product / (norm1 * norm2)) * 100  # Convert to percentage

@app.route("/upload-real-image", methods=["POST"])
def upload_real_image():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    image = Image.open(file).convert("RGB")
    
    # Save temporary uploaded image
    temp_path = "temp_uploaded_image.jpg"
    image.save(temp_path)

    # Extract face embedding from the uploaded real image
    real_image_embedding = extract_face_embedding(temp_path)
    if real_image_embedding is None:
        return jsonify({"error": "No face detected in the uploaded image"}), 400

    best_match = None
    max_similarity = 0

    for filename in os.listdir(IMAGES_FOLDER):
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            image_path = os.path.join(IMAGES_FOLDER, filename)

            # Extract face embedding from stored sketch
            sketch_embedding = extract_face_embedding(image_path)
            if sketch_embedding is None:
                continue  # Skip if no face detected

            # Calculate similarity
            similarity = cosine_similarity(real_image_embedding, sketch_embedding)

            if similarity > max_similarity:
                max_similarity = similarity
                best_match = filename

    os.remove(temp_path)  # Remove temp file after comparison

    return jsonify({"best_match": best_match, "similarity": max_similarity})

if __name__ == "__main__":
    app.run(debug=False)
