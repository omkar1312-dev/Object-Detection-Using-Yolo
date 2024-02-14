from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image, ImageDraw
import os

# Load the image
image_path = 'F:\Assignment\Infilect\project\Flask\Captioning\image-14.jpg'
output_folder = 'output'
image = Image.open(image_path)

def generate_caption(image_path):
    image = Image.open(image_path)

    # Initialize the processor and model
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    # Preprocess the image and generate a caption
    inputs = processor(images=image, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)

    # Draw bounding box on the image (for illustration purposes)
    draw = ImageDraw.Draw(image)
    draw.rectangle([(10, 10), (100, 100)], outline="red", width=3)

    # Save the visualized image in the output folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    output_path = os.path.join(output_folder, f"output_{os.path.basename(image_path)}")
    image.save(output_path)

    # Print the generated caption
    print("Generated Caption:", caption)
    print(f"Visualized image with bounding box saved at: {output_path}")
    
    return caption

