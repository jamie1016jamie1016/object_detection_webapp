from flask import Flask, request, jsonify, render_template, redirect, url_for
import os
import cv2
from ultralytics import YOLO
from shutil import copyfile
from werkzeug.utils import secure_filename
from PIL import Image, UnidentifiedImageError, ImageDraw, ImageFont

"""
Flask application for object detection and product management.

Features:
- CRUD API for managing products
- Web interface for product management
- YOLOv8 model for detecting objects in uploaded images
- Bounding box overlay on detected objects with product information

Technologies Used: Flask, YOLOv8, OpenCV, PIL, SQL-like product storage.
"""

# Initialize Flask application
app = Flask(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'  # Folder for uploaded images
app.config['STATIC_FOLDER'] = 'static'    # Folder for static files

# Ensure the upload and static directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(os.path.join(app.config['STATIC_FOLDER'], 'uploads'), exist_ok=True)

# In-memory storage for product information
products = {}

# Load YOLO model once at startup for efficiency
yolo_model = YOLO('yolov8n.pt')

# Allowed file extensions for image uploads
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    """
    Check if the file has an allowed extension.

    Args:
        filename (str): File name to check.
    
    Returns:
        bool: True if the file extension is allowed, False otherwise.
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_new_product_id():
    """
    Generate a new product ID.

    Returns:
        str: New product ID.
    """
    if products:
        max_id = max(int(pid) for pid in products.keys())
        return f"{max_id + 1:03d}"
    return "001"

# --------------------------- Root Route --------------------------- #

@app.route('/', methods=['GET'])
def home():
    """Render the home page."""
    return render_template('home.html')

# --------------------------- CRUD API Endpoints --------------------------- #

@app.route('/api/products', methods=['POST'])
def create_product():
    """
    Create a new product entry.

    Returns:
        JSON response with success message and product ID.
    """
    product = request.get_json()
    if 'name' not in product or 'price' not in product:
        return jsonify({'error': 'Missing name or price.'}), 400

    product_id = generate_new_product_id()
    product['id'] = product_id
    products[product_id] = product
    return jsonify({'message': 'Product created successfully.', 'id': product_id}), 201

@app.route('/api/products/<product_id>', methods=['GET'])
def get_product(product_id):
    """
    Retrieve a product by its ID.

    Args:
        product_id (str): Product ID to retrieve.
    
    Returns:
        JSON: Product details or error message.
    """
    product = products.get(product_id)
    if product:
        return jsonify(product), 200
    return jsonify({'error': 'Product not found.'}), 404

@app.route('/api/products/<product_id>', methods=['PUT'])
def update_product(product_id):
    """
    Update an existing product.

    Args:
        product_id (str): Product ID to update.
    
    Returns:
        JSON: Success or error message.
    """
    if product_id in products:
        product = request.get_json()
        if 'name' not in product or 'price' not in product:
            return jsonify({'error': 'Missing name or price.'}), 400
        product['id'] = product_id  # Ensure the ID remains the same
        products[product_id] = product
        return jsonify({'message': 'Product updated successfully.'}), 200
    return jsonify({'error': 'Product not found.'}), 404

@app.route('/api/products/<product_id>', methods=['DELETE'])
def delete_product(product_id):
    """
    Delete a product by its ID.

    Args:
        product_id (str): Product ID to delete.
    
    Returns:
        JSON: Success or error message.
    """
    if product_id in products:
        del products[product_id]
        return jsonify({'message': 'Product deleted successfully.'}), 200
    return jsonify({'error': 'Product not found.'}), 404

# ----------------------- Web Interface Routes ----------------------- #

@app.route('/products', methods=['GET'])
def list_products():
    """Render the product list page."""
    return render_template('products.html', products=products.values())

@app.route('/add_product', methods=['GET', 'POST'])
def add_product():
    """
    Add a new product via web interface.

    Returns:
        HTML: Redirect to product list page or rendered add product page.
    """
    if request.method == 'POST':
        product_id = generate_new_product_id()
        product = {
            'id': product_id,
            'name': request.form['name'],
            'price': float(request.form['price']),
            'in_stock': 'in_stock' in request.form
        }
        products[product_id] = product
        return redirect(url_for('list_products'))
    return render_template('add_product.html')

@app.route('/edit_product/<product_id>', methods=['GET', 'POST'])
def edit_product(product_id):
    """
    Edit an existing product via web interface.

    Args:
        product_id (str): Product ID to edit.
    
    Returns:
        HTML: Redirect to product list page or rendered edit product page.
    """
    product = products.get(product_id)
    if not product:
        return 'Product not found!', 404
    if request.method == 'POST':
        product['name'] = request.form['name']
        product['price'] = float(request.form['price'])
        product['in_stock'] = 'in_stock' in request.form
        return redirect(url_for('list_products'))
    return render_template('edit_product.html', product=product)

@app.route('/delete_product/<product_id>', methods=['GET'])
def delete_product_web(product_id):
    """
    Delete a product via web interface.

    Args:
        product_id (str): Product ID to delete.
    
    Returns:
        HTML: Redirect to product list page or error message.
    """
    if product_id in products:
        del products[product_id]
        return redirect(url_for('list_products'))
    return 'Product not found!', 404

# ----------------------- Image Upload and Processing ----------------------- #

@app.route('/upload', methods=['GET', 'POST'])
def upload_image():
    """
    Upload an image and process it using YOLO for object detection.

    Returns:
        HTML: Rendered upload or display image page with results.
    """
    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('upload.html', error='No image part in the request.')
        file = request.files['image']
        if file.filename == '':
            return render_template('upload.html', error='No selected image.')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Validate image file
            try:
                img = Image.open(filepath)
                img.verify()
            except (UnidentifiedImageError, IOError):
                os.remove(filepath)
                return render_template('upload.html', error="Uploaded file is not a valid image.")
            
            # Resize and process the image
            filepath = resize_image(filepath)
            output_image_path = process_image(filepath)
            if output_image_path is None:
                os.remove(filepath)
                return render_template('upload.html', error="Sorry, we couldn't process the image.")
            else:
                static_output_path = os.path.join(app.config['STATIC_FOLDER'], 'uploads', os.path.basename(output_image_path))
                copyfile(output_image_path, static_output_path)
                return render_template('display_image.html', image_filename=os.path.basename(output_image_path))
        return render_template('upload.html', error="Unsupported file type. Please upload a PNG or JPEG image.")
    return render_template('upload.html')

# --------------------------- Helper Functions --------------------------- #

def process_image(image_path):
    """
    Process image for object detection and overlay product info.

    Args:
        image_path (str): Path to the image.
    
    Returns:
        str: Path to the output image with overlaid product info.
    """
    detections = perform_object_detection(image_path)
    if detections is None:
        return None
    product_infos = get_product_info(detections)
    return overlay_product_info(image_path, product_infos)

def perform_object_detection(image_path):
    """
    Perform object detection using YOLOv8 model.

    Args:
        image_path (str): Path to input image.
    
    Returns:
        list: List of detected objects with class names, bounding boxes, and scores.
    """
    try:
        results = yolo_model(image_path)
        detections = []
        for result in results:
            for box, score, label in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy(), result.boxes.cls.cpu().numpy()):
                class_id = int(label)
                class_name = yolo_model.names[class_id]
                bbox = box.astype(int)
                detections.append({'class_name': class_name, 'bbox': bbox, 'score': score})
        return detections
    except Exception as e:
        print(f"Error during object detection: {e}")
        return None

def get_product_info(detections):
    """
    Get product information for detected objects.

    Args:
        detections (list): List of detected objects.
    
    Returns:
        list: Product information for detected objects.
    """
    product_infos = []
    product_lookup = {product['name'].lower(): product for product in products.values()}
    for detection in detections:
        class_name = detection['class_name'].lower()
        if class_name in product_lookup:
            product = product_lookup[class_name]
            product_info = {
                'class_name': class_name,
                'bbox': detection['bbox'],
                'price': product['price'],
                'in_stock': product['in_stock']
            }
            product_infos.append(product_info)
    return product_infos

def overlay_product_info(image_path, product_infos):
    """
    Overlay bounding boxes and product information onto the image, with different colors for each product type.

    Args:
        image_path (str): Path to the input image.
        product_infos (list): Product information to overlay.
    
    Returns:
        str: Path to the output image.
    """
    # Load image with PIL for better text rendering
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    
    # Define a set of colors to be used for different bounding boxes (RGB)
    colors = [
        (255, 0, 0),   # Red
        (0, 255, 0),   # Green
        (0, 0, 255),   # Blue
        (255, 255, 0), # Yellow
        (255, 165, 0), # Orange
        (0, 255, 255), # Cyan
        (255, 0, 255)  # Magenta
    ]
    
    # Assign a different color to each unique product type
    displayed_products = {}
    color_index = 0
    
    for info in product_infos:
        class_name = info['class_name']
        price = info['price']
        in_stock = 'Yes' if info['in_stock'] else 'No'
        color = colors[color_index % len(colors)]  # Cycle through colors

        if class_name not in displayed_products:
            displayed_products[class_name] = {
                'price': price,
                'in_stock': in_stock,
                'bboxes': [],
                'color': color  # Assign color for this product type
            }
            color_index += 1
        displayed_products[class_name]['bboxes'].append(info['bbox'])

    # Draw bounding boxes and labels
    for class_name, info in displayed_products.items():
        price = info['price']
        in_stock = info['in_stock']
        bboxes = info['bboxes']
        color = info['color']

        # Draw bounding boxes for this product
        for bbox in bboxes:
            xmin, ymin, xmax, ymax = bbox
            draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=3)

        # Choose largest bounding box for text positioning
        largest_bbox = max(bboxes, key=lambda bbox: (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
        xmin, ymin, xmax, ymax = largest_bbox
        
        # Create label with product info
        label = f"{class_name.capitalize()}: ${price}, In Stock: {in_stock}"

        # Load font
        font_size = 20
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except IOError:
            font = ImageFont.load_default()

        # Calculate text size and position
        text_bbox = draw.textbbox((xmin, ymin), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        text_x = xmin
        text_y = ymin - text_height - 5

        # Adjust text position if it goes beyond image boundaries
        if text_x + text_width > image.width:
            text_x = image.width - text_width - 5
        if text_y < 0:
            text_y = ymax + 5
            if text_y + text_height > image.height:
                text_y = image.height - text_height - 5

        # Draw background for text
        draw.rectangle([text_x - 2, text_y - 2, text_x + text_width + 2, text_y + text_height + 2], fill=(255, 255, 255))

        # Draw product info text using the same color as the bounding box
        draw.text((text_x, text_y), label, fill=color, font=font)

    # Save output image
    output_image_filename = 'output_' + os.path.basename(image_path)
    output_image_path = os.path.join(app.config['UPLOAD_FOLDER'], output_image_filename)
    image.save(output_image_path)

    return output_image_path


def resize_image(image_path, max_size=(1024, 1024)):
    """
    Resize the image for processing efficiency.

    Args:
        image_path (str): Path to the image.
        max_size (tuple): Max width and height.
    
    Returns:
        str: Path to the resized image.
    """
    img = Image.open(image_path)
    img.thumbnail(max_size)
    resized_path = image_path.replace(".jpg", "_resized.jpg")
    img.save(resized_path)
    return resized_path

# --------------------------- Run the Application --------------------------- #

if __name__ == '__main__':
    app.run(debug=True)