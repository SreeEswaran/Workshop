from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import cv2
import base64

app = Flask(__name__)
CORS(app)

# Load the COCO-SSD model
model = tf.saved_model.load('http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco/saved_model')

def decode_image(base64_string):
    img_data = base64.b64decode(base64_string)
    np_arr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return img

def encode_image(img):
    _, buffer = cv2.imencode('.jpg', img)
    base64_string = base64.b64encode(buffer).decode('utf-8')
    return base64_string

@app.route('/detect', methods=['POST'])
def detect_objects():
    data = request.json
    img_base64 = data['image']
    img = decode_image(img_base64)

    # Convert image to tensor
    input_tensor = tf.convert_to_tensor(np.expand_dims(img, 0), dtype=tf.uint8)

    # Run detection
    detections = model(input_tensor)

    # Process detections
    detection_boxes = detections['detection_boxes'][0].numpy()
    detection_scores = detections['detection_scores'][0].numpy()
    detection_classes = detections['detection_classes'][0].numpy().astype(np.int64)

    height, width, _ = img.shape
    results = []
    for i in range(len(detection_scores)):
        if detection_scores[i] > 0.5:
            box = detection_boxes[i] * [height, width, height, width]
            results.append({
                'class': int(detection_classes[i]),
                'score': float(detection_scores[i]),
                'box': box.tolist()
            })
            y1, x1, y2, x2 = box.astype(int)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{int(detection_classes[i])}: {detection_scores[i]:.2f}"
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Encode the image with detections
    img_base64_with_detections = encode_image(img)

    return jsonify({'image': img_base64_with_detections, 'detections': results})

if __name__ == '__main__':
    app.run(debug=True)
