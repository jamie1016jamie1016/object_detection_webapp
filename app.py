"""
Flask application for object detection and product management.

Features:
- CRUD API for managing products
- Web interface for product management
- YOLOv8 model for detecting objects in uploaded images
- Bounding box overlay on detected objects with product information

Technologies Used: Flask, YOLOv8, OpenCV, PIL, SQL-like product storage.
"""

from flask import Flask, request, jsonify, render_template, redirect, url_for
import os
import cv2
from ultralytics import YOLO
from shutil import copyfile
from werkzeug.utils import secure_filename
from PIL import Image, UnidentifiedImageError, ImageDraw, ImageFont
import random

# Initialize the Flask application
app = Flask(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'  # Folder to store uploaded images
app.config['STATIC_FOLDER'] = 'static'    # Folder for static files

# Ensure the upload and static directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(os.path.join(app.config['STATIC_FOLDER'], 'uploads'), exist_ok=True)

# In-memory database to store product information
products = {}

# Load the YOLO model once at startup for efficiency
yolo_model = YOLO('yolov8n.pt')

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    """
    Check if a file has an allowed extension.

    Args:
        filename (str): Name of the file to check.
    
    Returns:
        bool: True if the file is allowed, False otherwise.
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_new_product_id():
    """
    Generate a new product ID based on existing product IDs.

    Returns:
        str: New product ID.
    """
    if products:
        max_id = max(int(pid) for pid in products.keys())
        return f"{max_id + 1:03d}"
    else:
        return "001"

# --------------------------- Root Route --------------------------- #

@app.route('/', methods=['GET'])
def home():
    """
    Render the home page.s

    Returns:
        HTML: Rendered home page.
    """
    return render_template('home.html')

# --------------------------- CRUD API Endpoints --------------------------- #

# Create a new product
@app.route('/api/products', methods=['POST'])
def create_product():
    """
    Create a new product entry.

    Returns:
        JSON response with a success message and product ID.
    """
    product = request.get_json()
    if 'name' not in product or 'price' not in product:
        return jsonify({'error': 'Missing name or price.'}), 400

    product_id = generate_new_product_id()
    product['id'] = product_id
    products[product_id] = product
    return jsonify({'message': 'Product created successfully.', 'id': product_id}), 201

# Retrieve a product by ID
@app.route('/api/products/<product_id>', methods=['GET'])
def get_product(product_id):
    """
    Retrieve a product by its ID.

    Args:
        product_id (str): ID of the product to retrieve.
    
    Returns:
        JSON: Product details or error message.
    """
    product = products.get(product_id)
    if product:
        return jsonify(product), 200
    else:
        return jsonify({'error': 'Product not found.'}), 404

# Update an existing product
@app.route('/api/products/<product_id>', methods=['PUT'])
def update_product(product_id):
    """
    Update an existing product.

    Args:
        product_id (str): ID of the product to update.
    
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
    else:
        return jsonify({'error': 'Product not found.'}), 404

# Delete a product
@app.route('/api/products/<product_id>', methods=['DELETE'])
def delete_product(product_id):
    """
    Delete a product by its ID.

    Args:
        product_id (str): ID of the product to delete.
    
    Returns:
        JSON: Success or error message.
    """
    if product_id in products:
        del products[product_id]
        return jsonify({'message': 'Product deleted successfully.'}), 200
    else:
        return jsonify({'error': 'Product not found.'}), 404

# ----------------------- Web Interface Routes ----------------------- #

# List all products
@app.route('/products', methods=['GET'])
def list_products():
    """
    List all products.

    Returns:
        HTML: Rendered product list page.
    """
    return render_template('products.html', products=products.values())

# Add a new product
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
    else:
        return render_template('add_product.html')

# Edit an existing product
@app.route('/edit_product/<product_id>', methods=['GET', 'POST'])
def edit_product(product_id):
    """
    Edit an existing product via web interface.

    Args:
        product_id (str): ID of the product to edit.
    
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
    else:
        return render_template('edit_product.html', product=product)

# Delete a product (via web interface)
@app.route('/delete_product/<product_id>', methods=['GET'])
def delete_product_web(product_id):
    """
    Delete a product via web interface.

    Args:
        product_id (str): ID of the product to delete.
    
    Returns:
        HTML: Redirect to product list page or error message.
    """
    if product_id in products:
        del products[product_id]
        return redirect(url_for('list_products'))
    else:
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
            
            # Resize the image to improve detection efficiency
            filepath = resize_image(filepath)
            output_image_path = process_image(filepath)
            if output_image_path is None:
                os.remove(filepath)
                return render_template('upload.html', error="Sorry, we couldn't process the image.")
            else:
                static_output_path = os.path.join(app.config['STATIC_FOLDER'], 'uploads', os.path.basename(output_image_path))
                copyfile(output_image_path, static_output_path)
                return render_template('display_image.html', image_filename=os.path.basename(output_image_path))
        else:
            return render_template('upload.html', error="Unsupported file type. Please upload a PNG or JPEG image.")
    else:
        return render_template('upload.html')

# --------------------------- Helper Functions --------------------------- #

def process_image(image_path):
    """
    Process the image to perform object detection and overlay product information.

    Args:
        image_path (str): Path to the image to process.
    
    Returns:
        str: Path to the output image with overlaid product information.
    """
    detections = perform_object_detection(image_path)
    if detections is None:
        return None
    product_infos = get_product_info(detections)
    output_image_path = overlay_product_info(image_path, product_infos)
    return output_image_path

def perform_object_detection(image_path):
    """
    Perform object detection using YOLOv8 model.

    Args:
        image_path (str): Path to the input image.
    
    Returns:
        list: List of detected objects containing class names, bounding boxes, and scores.
    """
    try:
        results = yolo_model(image_path)
        detections = []
        for result in results:
            # Extract the bounding boxes, scores, and labels
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
        list: List of product information matching detected objects.
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
    Overlay bounding boxes and product information onto the image.

    Args:
        image_path (str): Path to the input image.
        product_infos (list): List of product information to overlay.
    
    Returns:
        str: Path to the output image with overlaid information.
    """
    # Load image using PIL for better text rendering
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    
    # Assign unique colors to each object type
    object_colors = {}
    for info in product_infos:
        class_name = info['class_name'].lower()
        if class_name not in object_colors:
            # Generate a random color
            object_colors[class_name] = tuple(random.choices(range(256), k=3))
    
    # Draw bounding boxes for all detected objects
    for info in product_infos:
        xmin, ymin, xmax, ymax = info['bbox']
        class_name = info['class_name'].lower()
        color = object_colors[class_name]
    
        # Draw bounding box for each detected object
        draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=3)
    
    # Display product info only once per product type
    displayed_products = {}
    for info in product_infos:
        class_name = info['class_name'].lower()
        price = info['price']
        in_stock = 'Yes' if info['in_stock'] else 'No'
    
        # Collect info and all bounding boxes for this product type
        if class_name not in displayed_products:
            displayed_products[class_name] = {
                'price': price,
                'in_stock': in_stock,
                'bboxes': []
            }
        displayed_products[class_name]['bboxes'].append(info['bbox'])
    
    for class_name, info in displayed_products.items():
        price = info['price']
        in_stock = info['in_stock']
        bboxes = info['bboxes']
        color = object_colors[class_name]
        
        # Choose the largest bounding box for text positioning
        largest_bbox = max(bboxes, key=lambda bbox: (bbox[2]-bbox[0])*(bbox[3]-bbox[1]))
        xmin, ymin, xmax, ymax = largest_bbox
        
        # Create the label with product info
        label = f"{class_name.capitalize()}: ${price}, In Stock: {in_stock}"
    
        # Load a truetype or opentype font file, and create a font object
        font_size = 20  # Adjust as needed
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except IOError:
            font = ImageFont.load_default()
    
        # Calculate text size using textbbox method
        text_bbox = draw.textbbox((xmin, ymin), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
    
        # Position text above the bounding box
        text_x = xmin
        text_y = ymin - text_height - 5  # 5 pixels above the bounding box
    
        # Adjust text position if it goes beyond the image boundaries
        if text_x + text_width > image.width:
            text_x = image.width - text_width - 5  # Move text left
        if text_y < 0:
            text_y = ymax + 5  # Position text below the bounding box if above is not possible
            if text_y + text_height > image.height:
                text_y = image.height - text_height - 5  # Adjust if it still goes beyond

        # Draw background rectangle for text for better visibility
        draw.rectangle([text_x - 2, text_y - 2, text_x + text_width + 2, text_y + text_height + 2], fill=(255, 255, 255))
    
        # Draw the product info text
        draw.text((text_x, text_y), label, fill=color, font=font)
    
    # Save the output image
    output_image_filename = 'output_' + os.path.basename(image_path)
    output_image_path = os.path.join(app.config['UPLOAD_FOLDER'], output_image_filename)
    image.save(output_image_path)
    
    return output_image_path

def resize_image(image_path, max_size=(1024, 1024)):
    """
    Resize the image to a maximum size for processing efficiency.

    Args:
        image_path (str): Path to the image to resize.
        max_size (tuple): Maximum width and height for resizing.
    
    Returns:
        str: Path to the resized image.
    """
    img = Image.open(image_path)
    img.thumbnail(max_size)
    resized_path = image_path.replace(".jpg", "_resized.jpg")  # Example filename change
    img.save(resized_path)
    return resized_path

# --------------------------- Run the Application --------------------------- #

if __name__ == '__main__':
    app.run(debug=True)
