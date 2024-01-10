from flask import Flask, render_template, request, redirect, url_for, flash
import os
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
from base64 import b64encode

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with a secret key for flash messages

class ImageProcessingApp:
    def __init__(self):
        self.image_path = None
        self.threshold = 0.8

    def process_image(self):
        if not self.image_path:
            flash("Please select an image.", 'error')
            return None, None

        # Load the source image
        source_path = "P_80204011000000927.tiff"
        source_image = cv2.imread(source_path, cv2.IMREAD_GRAYSCALE)
        if source_image is None:
            flash(f"Unable to read the source image {source_path}", 'error')
            return None, None

        # Load the target image
        target_image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        if target_image is None:
            flash(f"Unable to read the target image {self.image_path}", 'error')
            return None, None

        # Resize the target image to match the dimensions of the source image
        target_image = cv2.resize(target_image, (source_image.shape[1], source_image.shape[0]))

        # Ensure both images have the same depth and type
        target_image = cv2.convertScaleAbs(target_image)

        # Load all template images in the specified folder
        templates_folder = "Logos"
        template_files = [f for f in os.listdir(templates_folder) if f.endswith((".jpg", ".jpeg", ".png", ".tiff"))]

        # Set the threshold for template matching
        threshold = self.threshold

        # Variable to check if any template is detected
        template_detected = False

        # Process each template
        for template_file in template_files:
            template_path = os.path.join(templates_folder, template_file)
            template_name = os.path.splitext(template_file)[0].split("_")[0]

            # Load the template
            template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)

            # Ensure both template and target image have the same depth and type
            template = cv2.convertScaleAbs(template)

            # Apply template matching
            result = cv2.matchTemplate(target_image, template, cv2.TM_CCOEFF_NORMED)
            locations = np.where(result >= threshold)
            locations = list(zip(*locations[::-1]))

            # If any match is found, set template_detected to True and break the loop
            if locations:
                template_detected = True
                break

        # If no template is detected, display a message without showing the image
        if not template_detected:
            flash("Previous report:-Logo not detected", 'info')
            return target_image, None

        # Draw rectangles around the detected areas
        for loc in locations:
            x, y = loc
            h, w = template.shape[:2]
            cv2.rectangle(target_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        return target_image, template_name

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        processing_app = ImageProcessingApp()

    
        if 'image' not in request.files:
            flash("No file selected", 'error')
            return redirect(request.url)

        uploaded_image = request.files['image']
        if uploaded_image.filename == '':
            flash("No file selected", 'error')
            return redirect(request.url)

        processing_app.image_path = r"C:\Users\admin\Desktop\Sample_model\temp_image\temp_image.jpg"
        uploaded_image.save(processing_app.image_path)
        processed_image, output_text = processing_app.process_image()

        if processed_image is not None:
            # Encode the processed image to display in the browser
            _, buffer = cv2.imencode('.jpg', processed_image)
            img_str = b64encode(buffer).decode('utf-8')
            img_data = f'data:image/jpeg;base64,{img_str}'
            return render_template('result.html', img_data=img_data, output_text=output_text)

    # If it's a GET request, handle the restart logic
    elif request.method == 'GET':
        processing_app = ImageProcessingApp()
        processing_app.image_path = None
        flash("Restarted. Please select an image.", 'info')

    return render_template('index.html')


if __name__ == '__main__':
    app.run(ssl_context='adhoc', debug=True)