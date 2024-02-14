from flask import Flask, request, jsonify
from PIL import Image
from detector import detect_objects
from classify import classify_and_visualize
from caption import generate_caption

app = Flask(__name__)

@app.route('/process_image', methods=['POST'])
def process_image():
    try:
        # Get the image file from the request
        image_file = request.files['image']

        # Save the image temporarily
        image_path = 'static/uploaded_images/temp_image.jpg'
        image_file.save(image_path)

        # Step 1: Detect Objects
        detection_results = detect_objects(image_path)

        # Step 2: Classify Objects and Visualize
        classify_and_visualize(image_path, output_folder='static/uploaded_images/classification_output')

        # Step 3: Generate Caption
        caption_result = generate_caption(image_path)

        # Organize the results into the specified output format
        output_format = {
            'detection': detection_results,
            'classification_output': 'static/uploaded_images/classification_output',
            'caption': caption_result
        }

        # Return the formatted response
        return jsonify(output_format)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
