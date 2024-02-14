import yolov5

# Load pretrained model
model = yolov5.load('yolov5s.pt')

# Set model parameters
model.conf = 0.25  # NMS confidence threshold
model.iou = 0.45  # NMS IoU threshold
model.agnostic = False  # NMS class-agnostic
model.multi_label = False  # NMS multiple labels per box
model.max_det = 1000  # Maximum number of detections per image

# Set image path
img_path = 'F:\Assignment\Infilect\project\Flask\Detector\image-26.jpg'  # Use forward slash or double backslash

def detect_objects(img_path):
    results = model(img_path)

    # Inference with larger input size
    results = model(img_path, size=1280)

    # Inference with test time augmentation
    results = model(img_path, augment=True)

    # Show detection bounding boxes on image
    results.show()

    # Save results into "results/" folder
    results.save(save_dir='results/output')
    
    return results