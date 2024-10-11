from flask import Flask, request, jsonify, render_template, redirect, url_for
import os
import cv2
from ultralytics import YOLO
from shutil import copyfile
from werkzeug.utils import secure_filename
from PIL import Image, UnidentifiedImageError


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
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --------------------------- Root Route --------------------------- #

@app.route('/', methods=['GET'])
def home():
    """
    Render the home page with navigation buttons.
    """
    return render_template('home.html')

# --------------------------- CRUD API Endpoints --------------------------- #

# Create a new product
@app.route('/api/products', methods=['POST'])
def create_product():
    """
    Create a new product and add it to the in-memory database.
    Expects JSON data with 'name', 'price', and 'in_stock'.
    """
    product = request.get_json()
    # Auto-generate the next product ID
    if products:
        max_id = max(int(pid) for pid in products.keys())
        new_id = f"{max_id + 1:03d}"
    else:
        new_id = "001"
    product_id = new_id
    product['id'] = product_id
    products[product_id] = product
    return jsonify({'message': 'Product created successfully.', 'id': product_id}), 201

# Retrieve a product by ID
@app.route('/api/products/<product_id>', methods=['GET'])
def get_product(product_id):
    """
    Retrieve product information by product ID.
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
    Update product information.
    Expects JSON data with updated 'name', 'price', and 'in_stock'.
    """
    if product_id in products:
        product = request.get_json()
        product['id'] = product_id  # Ensure the ID remains the same
        products[product_id] = product
        return jsonify({'message': 'Product updated successfully.'}), 200
    else:
        return jsonify({'error': 'Product not found.'}), 404

# Delete a product
@app.route('/api/products/<product_id>', methods=['DELETE'])
def delete_product(product_id):
    """
    Delete a product from the in-memory database.
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
    Display a list of all products with options to edit or delete.
    """
    return render_template('products.html', products=products.values())

# Add a new product
@app.route('/add_product', methods=['GET', 'POST'])
def add_product():
    """
    Display a form to add a new product. Handles form submission.
    """
    if request.method == 'POST':
        # Auto-generate the next product ID
        if products:
            max_id = max(int(pid) for pid in products.keys())
            new_id = f"{max_id + 1:03d}"
        else:
            new_id = "001"
        product = {
            'id': new_id,
            'name': request.form['name'],
            'price': float(request.form['price']),
            'in_stock': 'in_stock' in request.form
        }
        products[new_id] = product
        return redirect(url_for('list_products'))
    else:
        return render_template('add_product.html')

# Edit an existing product
@app.route('/edit_product/<product_id>', methods=['GET', 'POST'])
def edit_product(product_id):
    """
    Display a form to edit an existing product. Handles form submission.
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
    Delete a product and redirect to the product list.
    """
    if product_id in products:
        del products[product_id]
        return redirect(url_for('list_products'))
    else:
        return 'Product not found!', 404

# ----------------------- Image Upload and Processing ----------------------- #

@app.route('/upload', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        # Check if an image was uploaded
        if 'image' not in request.files:
            error_message = 'No image part in the request.'
            return render_template('upload.html', error=error_message)
        file = request.files['image']
        if file.filename == '':
            error_message = 'No selected image.'
            return render_template('upload.html', error=error_message)
        if file and allowed_file(file.filename):
            # Secure the filename
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Validate the image file
            try:
                img = Image.open(filepath)
                img.verify()  # Verify that it is an image
            except (UnidentifiedImageError, IOError):
                os.remove(filepath)  # Remove invalid image file
                error_message = "Uploaded file is not a valid image. Please upload a valid PNG or JPEG image."
                return render_template('upload.html', error=error_message)
            
            # Resize the image if it's too large
            resize_image(filepath)
            
            # Process the image
            output_image_path = process_image(filepath)
            if output_image_path is None:
                os.remove(filepath)  # Clean up the uploaded file
                error_message = "Sorry, we couldn't process the image. Please try uploading a different image."
                return render_template('upload.html', error=error_message)
            else:
                # Copy the output image to the static/uploads directory
                static_output_path = os.path.join(app.config['STATIC_FOLDER'], 'uploads', os.path.basename(output_image_path))
                copyfile(output_image_path, static_output_path)
                # Render the result
                return render_template('display_image.html', image_filename=os.path.basename(output_image_path))
        else:
            error_message = "Unsupported file type. Please upload a PNG or JPEG image."
            return render_template('upload.html', error=error_message)
    else:
        # Render the upload form template
        return render_template('upload.html')


# --------------------------- Helper Functions --------------------------- #

def process_image(image_path):
    """
    Perform object detection on the image and overlay product information.
    Returns the path to the output image or None if an error occurs.
    """
    # Perform object detection
    detections = perform_object_detection(image_path)
    if detections is None:
        return None  # Indicate failure

    # Get product information for detected objects
    product_infos = get_product_info(detections)
    # Overlay product info on the image
    output_image_path = overlay_product_info(image_path, product_infos)
    return output_image_path


def perform_object_detection(image_path):
    """
    Use the YOLO model to detect objects in the image.
    Returns a list of detections with class names, bounding boxes, and scores.
    """
    try:
        results = yolo_model(image_path)
        detections = []
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes
            scores = result.boxes.conf.cpu().numpy()  # Confidence scores
            labels = result.boxes.cls.cpu().numpy()   # Class labels
            for i in range(len(boxes)):
                class_id = int(labels[i])
                class_name = yolo_model.names[class_id]
                bbox = boxes[i].astype(int)
                score = scores[i]
                detections.append({'class_name': class_name, 'bbox': bbox, 'score': score})
        return detections
    except Exception as e:
        print(f"Error during object detection: {e}")
        return None


def get_product_info(detections):
    """
    Match detected objects with products in the database.
    Returns a list of product information to be overlaid on the image.
    """
    product_infos = []
    for detection in detections:
        class_name = detection['class_name']
        # Match the detected class name with product names in the database
        for product in products.values():
            if product['name'].lower() == class_name.lower():
                product_info = {
                    'class_name': class_name,
                    'bbox': detection['bbox'],
                    'price': product['price'],
                    'in_stock': product['in_stock']
                }
                product_infos.append(product_info)
                break  # Stop searching after the first match
    return product_infos

def overlay_product_info(image_path, product_infos):
    """
    Overlay bounding boxes and product information onto the image.
    Saves and returns the path to the output image.
    """
    image = cv2.imread(image_path)
    for info in product_infos:
        xmin, ymin, xmax, ymax = info['bbox']
        price = info['price']
        in_stock = 'Yes' if info['in_stock'] else 'No'
        label = f"{info['class_name']}: ${price}, In Stock: {in_stock}"
        # Draw the bounding box
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        # Put the label above the bounding box
        cv2.putText(image, label, (xmin, ymin - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    # Save the output image
    output_image_filename = 'output_' + os.path.basename(image_path)
    output_image_path = os.path.join(app.config['UPLOAD_FOLDER'], output_image_filename)
    cv2.imwrite(output_image_path, image)
    return output_image_path

def resize_image(image_path, max_size=(1024, 1024)):
    """
    Resize the image to a maximum size to prevent memory issues.
    """
    img = Image.open(image_path)
    img.thumbnail(max_size)
    img.save(image_path)


# --------------------------- Run the Application --------------------------- #

if __name__ == '__main__':
    app.run(debug=True)
