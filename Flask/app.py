from flask import Flask, request, jsonify, send_file
from PIL import Image
from Detector.detector import detect_objects
from Classifier.classify import classify_and_visualize
from Captioning.caption import generate_caption
import os

app = Flask(__name__)

# Set the UPLOAD_FOLDER to the current directory
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the UPLOAD_FOLDER exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/process_image', methods=['POST'])
def process_image():
    try:
        # Get the image file from the request
        image_file = request.files['image']

        # Save the image in the UPLOAD_FOLDER
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_image.jpg')
        image_file.save(image_path)

        # Step 1: Detect Objects
        detection_results = detect_objects(image_path)

        # Step 2: Classify Objects and Visualize
        classification_output_path = classify_and_visualize(image_path, output_folder='static/uploaded_images/classification_output')

        # Step 3: Generate Caption
        caption_result = generate_caption(image_path)

        # Organize the results into the specified output format
        output_format = {
            'detection': detection_results,
            'classification_output': classification_output_path,
            'caption': caption_result,
            'uploaded_image_path': image_path  # Include the path of the uploaded image
        }
        return image_path
        # Return the formatted response
        # return json.dumps(output_format)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_processed_image', methods=['GET'])
def get_processed_image():
    # Example: Return the processed image to the client
    processed_image_path = 'static/uploaded_images/classification_output/temp_image_classified.jpg'
    return send_file(processed_image_path, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)
