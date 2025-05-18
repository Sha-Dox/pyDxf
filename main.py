import os
import io
import base64
import numpy as np
import cv2
from PIL import Image
from flask import Flask, request, render_template, jsonify
import ezdxf
import tempfile

app = Flask(__name__)


def process_image(image_data, tolerance=0.5, scale=1.0, deduplicate=True, enhance_contrast=True):
    # Convert image data to numpy array
    img = Image.open(io.BytesIO(image_data))
    img = img.convert('RGB')
    img_array = np.array(img)

    # Get dimensions
    height, width = img_array.shape[:2]

    # Convert to grayscale
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    # Create empty preview image with white background
    preview_img = np.ones((height, width, 3), dtype=np.uint8) * 255

    # Preprocessing based on user options
    if enhance_contrast:
        # Enhance contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

    # Apply bilateral filter to reduce noise while preserving edges
    smoothed = cv2.bilateralFilter(gray, 9, 75, 75)

    # METHOD 1: Adaptive thresholding with smaller block size for finer details
    thresh1 = cv2.adaptiveThreshold(smoothed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 7, 2)

    # METHOD 2: Canny edge detection with hysteresis parameters
    edges = cv2.Canny(smoothed, 20, 50)

    # METHOD 3: Multi-level thresholding
    # Use Otsu's method to find optimal threshold
    otsu_thresh, thresh2 = cv2.threshold(smoothed, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Combine methods with weighted importance
    combined = cv2.bitwise_or(cv2.bitwise_or(thresh1, edges), thresh2)

    # Apply light morphological operations to clean up noise
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)

    # Connect nearby lines to reduce fragmentation
    kernel_close = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_close)

    # Find contours with different methods
    all_contours = []

    # RETR_EXTERNAL for outer contours
    contours1, _ = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    all_contours.extend(contours1)

    # RETR_TREE for hierarchical contours (inner details)
    contours2, _ = cv2.findContours(cleaned, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
    all_contours.extend(contours2)

    # Filter contours
    filtered_contours = []
    min_area = max(3, width * height / 20000)  # More permissive area threshold

    # Function to check if two contours are too similar (to avoid duplicates)
    def contours_too_similar(c1, c2, threshold=0.9):
        if abs(cv2.contourArea(c1) - cv2.contourArea(c2)) / max(cv2.contourArea(c1), cv2.contourArea(c2)) < 0.2:
            # Create mask images
            mask1 = np.zeros((height, width), dtype=np.uint8)
            mask2 = np.zeros((height, width), dtype=np.uint8)
            cv2.drawContours(mask1, [c1], -1, 255, 1)
            cv2.drawContours(mask2, [c2], -1, 255, 1)

            # Calculate intersection and union
            intersection = cv2.bitwise_and(mask1, mask2)
            union = cv2.bitwise_or(mask1, mask2)

            # Calculate IoU
            intersection_area = np.count_nonzero(intersection)
            union_area = np.count_nonzero(union)

            if union_area > 0:
                iou = intersection_area / union_area
                return iou > threshold
        return False

    # Filter by area and deduplicate if requested
    for contour in all_contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            # Check if this contour is too similar to any already filtered contour
            is_duplicate = False
            if deduplicate:
                for existing in filtered_contours:
                    if contours_too_similar(contour, existing):
                        is_duplicate = True
                        break

            if not is_duplicate:
                filtered_contours.append(contour)

    # Apply tolerance for simplification
    simplified_contours = []
    for contour in filtered_contours:
        # Use absolute tolerance in pixels
        epsilon = tolerance
        approx = cv2.approxPolyDP(contour, epsilon, True)
        simplified_contours.append(approx)

    # Create DXF file
    doc = ezdxf.new()
    msp = doc.modelspace()

    # Calculate scaling factors
    max_dimension = max(width, height)
    scale_factor = scale / max_dimension

    # Draw contours on preview
    cv2.drawContours(preview_img, simplified_contours, -1, (0, 0, 255), 1)

    # Add contours to DXF
    for contour in simplified_contours:
        if len(contour) >= 2:  # Need at least 2 points to make a line
            # Create polyline
            points = []
            for point in contour:
                x, y = point[0]
                # Scale and center the points
                x_scaled = (x - width / 2) * scale_factor
                y_scaled = ((height - y) - height / 2) * scale_factor  # Flip Y and center
                points.append((x_scaled, y_scaled))

            # Close the polyline
            msp.add_lwpolyline(points, close=True)

    # Save to temporary file
    temp_file = tempfile.NamedTemporaryFile(suffix='.dxf', delete=False)
    doc.saveas(temp_file.name)

    # Encode preview image to base64
    _, buffer = cv2.imencode('.png', preview_img)
    preview_base64 = base64.b64encode(buffer).decode('utf-8')

    return temp_file.name, preview_base64
    return temp_file.name, preview_base64

    # Create DXF file
    doc = ezdxf.new()
    msp = doc.modelspace()

    # Calculate scaling factors
    height, width = img_array.shape[:2]
    max_dimension = max(width, height)
    scale_factor = scale / max_dimension

    # Add contours to DXF
    for contour in simplified_contours:
        if len(contour) >= 2:  # Need at least 2 points to make a line
            # Create polyline
            points = []
            for point in contour:
                x, y = point[0]
                # Scale and center the points
                x_scaled = (x - width / 2) * scale_factor
                y_scaled = ((height - y) - height / 2) * scale_factor  # Flip Y and center
                points.append((x_scaled, y_scaled))

            # Close the polyline
            msp.add_lwpolyline(points, close=True)

    # Save to temporary file
    temp_file = tempfile.NamedTemporaryFile(suffix='.dxf', delete=False)
    doc.saveas(temp_file.name)

    # Create a visualization for preview
    preview_img = np.zeros((height, width, 3), dtype=np.uint8)
    preview_img.fill(255)  # White background
    cv2.drawContours(preview_img, simplified_contours, -1, (0, 0, 255), 2)

    # Encode preview image to base64
    _, buffer = cv2.imencode('.png', preview_img)
    preview_base64 = base64.b64encode(buffer).decode('utf-8')

    return temp_file.name, preview_base64


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/convert', methods=['POST'])
def convert():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Get parameters
    tolerance = float(request.form.get('tolerance', 0.5))
    scale = float(request.form.get('scale', 1.0))
    deduplicate = request.form.get('deduplicate', 'off') == 'on'
    enhance_contrast = request.form.get('enhance-contrast', 'off') == 'on'

    # Read file data
    image_data = file.read()

    try:
        # Process image
        dxf_path, preview_base64 = process_image(image_data, tolerance, scale, deduplicate, enhance_contrast)

        # Return the DXF file path and preview
        return jsonify({
            'status': 'success',
            'dxf_path': dxf_path,
            'preview': f'data:image/png;base64,{preview_base64}'
        })
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/download/<path:filename>')
def download_file(filename):
    from flask import send_file
    return send_file(filename, as_attachment=True)


if __name__ == '__main__':
    # Create templates folder if it doesn't exist
    if not os.path.exists('templates'):
        os.makedirs('templates')

    # Create template file
    with open('templates/index.html', 'w') as f:
        f.write('''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Image to DXF Converter</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        .container {
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
        }
        .controls {
            flex: 1;
            min-width: 300px;
        }
        .preview {
            flex: 2;
            min-width: 300px;
            border: 1px solid #ccc;
            padding: 10px;
            min-height: 400px;
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        #image-preview {
            max-width: 100%;
            max-height: 350px;
            margin-top: 10px;
        }
        .form-group {
            margin-bottom: 15px;
        }
        label {
            display: block;
            margin-bottom: 5px;
        }
        input, button {
            width: 100%;
            padding: 8px;
            box-sizing: border-box;
        }
        button {
            background-color: #4CAF50;
            color: white;
            border: none;
            cursor: pointer;
            margin-top: 10px;
        }
        button:hover {
            background-color: #45a049;
        }
        .download-btn {
            margin-top: 10px;
            background-color: #2196F3;
        }
        .download-btn:hover {
            background-color: #0b7dda;
        }
        #loading {
            display: none;
            text-align: center;
        }
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #3498db;
            border-radius: 50%;
            width: 30px;
            height: 30px;
            animation: spin 2s linear infinite;
            margin: 10px auto;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <h1>Image to DXF Converter</h1>
    <div class="container">
        <div class="controls">
            <form id="converter-form">
                <div class="form-group">
                    <label for="file">Select Image (PNG or JPEG):</label>
                    <input type="file" id="file" name="file" accept=".png,.jpg,.jpeg" required>
                </div>
                <div class="form-group">
                    <label for="tolerance">Detail Level (0.1-10):</label>
                    <input type="range" id="tolerance" name="tolerance" min="0.1" max="10" step="0.1" value="0.5">
                    <span id="tolerance-value">0.5</span>
                    <p><small>Lower values = more detail</small></p>
                </div>
                <div class="form-group">
                    <label for="scale">Scale (0.1-10):</label>
                    <input type="range" id="scale" name="scale" min="0.1" max="10" step="0.1" value="1.0">
                    <span id="scale-value">1.0</span>
                </div>
                <div class="form-group">
                    <label>Advanced Options:</label>
                    <div class="advanced-options">
                        <div>
                            <input type="checkbox" id="deduplicate" name="deduplicate" checked>
                            <label for="deduplicate">Remove duplicate contours</label>
                        </div>
                        <div>
                            <input type="checkbox" id="enhance-contrast" name="enhance-contrast" checked>
                            <label for="enhance-contrast">Enhance contrast</label>
                        </div>
                    </div>
                </div>
                <button type="submit" id="convert-btn">Convert to DXF</button>
            </form>
            <div id="download-container" style="display: none;">
                <button id="download-btn" class="download-btn">Download DXF</button>
            </div>
        </div>
        <div class="preview">
            <h3>Preview</h3>
            <div id="loading">
                <div class="spinner"></div>
                <p>Processing...</p>
            </div>
            <img id="image-preview" src="" alt="Preview will appear here">
        </div>
    </div>

    <script>
        // Display slider values
        document.getElementById('tolerance').addEventListener('input', function() {
            document.getElementById('tolerance-value').textContent = this.value;
            // If there's already an image loaded, update the preview
            if (document.getElementById('file').files.length > 0) {
                submitForm();
            }
        });

        document.getElementById('scale').addEventListener('input', function() {
            document.getElementById('scale-value').textContent = this.value;
            // If there's already an image loaded, update the preview
            if (document.getElementById('file').files.length > 0) {
                submitForm();
            }
        });

        // Handle file input change
        document.getElementById('file').addEventListener('change', function() {
            if (this.files.length > 0) {
                submitForm();
            }
        });

        // Form submission
        document.getElementById('converter-form').addEventListener('submit', function(e) {
            e.preventDefault();
            submitForm();
        });

        let currentDxfPath = '';

        function submitForm() {
            const form = document.getElementById('converter-form');
            const formData = new FormData(form);

            // Show loading indicator
            document.getElementById('loading').style.display = 'block';
            document.getElementById('image-preview').style.display = 'none';

            fetch('/convert', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                // Hide loading indicator
                document.getElementById('loading').style.display = 'none';
                document.getElementById('image-preview').style.display = 'block';

                if (data.error) {
                    alert('Error: ' + data.error);
                    return;
                }

                // Display preview
                document.getElementById('image-preview').src = data.preview;

                // Show download button
                document.getElementById('download-container').style.display = 'block';
                currentDxfPath = data.dxf_path;
            })
            .catch(error => {
                document.getElementById('loading').style.display = 'none';
                alert('Error: ' + error);
            });
        }

        // Handle download button
        document.getElementById('download-btn').addEventListener('click', function() {
            if (currentDxfPath) {
                window.location.href = '/download/' + currentDxfPath;
            }
        });
    </script>
</body>
</html>
        ''')

    # Run the app
    app.run(debug=True, port=5000)