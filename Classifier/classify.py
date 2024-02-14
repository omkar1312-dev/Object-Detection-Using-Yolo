import torch
from torchvision import models, transforms
from PIL import Image
import cv2
import os
import numpy as np
# Load the pre-trained ResNet-18 model
model = models.resnet18(pretrained=True)
model.eval()

# Load ImageNet class labels from a local text file
LABELS_FILE = 'F:/Assignment/Infilect/project/Classifier/imagenet_labels.txt'

with open(LABELS_FILE, 'r') as file:
    labels = file.read().splitlines()

# Define the image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function to classify an image and draw bounding box
def classify_and_visualize(image_path, output_folder='output'):
    # Load and preprocess the image
    img = Image.open(image_path)
    img_tensor = transform(img)
    img_tensor = img_tensor.unsqueeze(0)  # Add a batch dimension

    # Make predictions
    with torch.no_grad():
        output = model(img_tensor)

    # Get the predicted label
    _, predicted_idx = torch.max(output, 1)
    predicted_label = labels[predicted_idx.item()]

    # Convert PIL image to OpenCV format
    img_cv2 = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # Draw a red bounding box
    h, w, _ = img_cv2.shape
    cv2.rectangle(img_cv2, (0, 0), (w, h), (0, 0, 255), 2)

    # Save the visualized image in the output folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    output_path = os.path.join(output_folder, f"output_{os.path.basename(image_path)}")
    cv2.imwrite(output_path, img_cv2)

    print(f"Predicted label: {predicted_label}")
    print(f"Visualized image saved at: {output_path}")

# Provide the path to the image you want to classify
image_path = 'F:\Assignment\Infilect\project\Classifier\image-14.jpg'
classify_and_visualize(image_path)
