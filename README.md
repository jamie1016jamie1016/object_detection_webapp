# Object Detection and CRUD API

## Overview
This project is a full-stack web application utilizing **Python** and **Flask** to integrate image processing, object detection, and CRUD operations. Users can upload images to detect products using a pre-trained YOLO model and manage product information through a RESTful API.

---

## Features
1. **Object Detection**:
   - Uses the YOLO (You Only Look Once) model to detect objects.
   - Draws bounding boxes on the uploaded images to highlight detected products.

2. **CRUD API**:
   - RESTful API built with Flask to manage product information.
   - Supports creating, retrieving, updating, and deleting products.
   - Product data includes name, price, and stock status.

3. **Augmented Reality Simulation**:
   - Overlays product information on detected items.
   - Simulates an AR experience by displaying product details (price, availability) next to detected products.

---

## Requirements
Ensure you have the following installed before starting:
- **Python 3.11.9**
- **pip** (Python package manager)

---

## Installation and Setup

### 1. Clone the Repository
```bash
git clone https://github.com/jamie1016jamie1016/object_detection_webapp.git
cd object_detection_webapp
```

### 2. Set Up the Virtual Environment
Create and activate a virtual environment:
```bash
python -m venv env
source env/bin/activate  # For macOS/Linux
# For Windows: env\Scripts\activate
```

### 3. Install Dependencies
Install the required Python packages:
```bash
pip install -r requirements.txt
```

---

## Running the Application

### 1. Start the Flask Application
```bash
python app.py
```

### 2. Access the Web Application
Open your web browser and navigate to:
```
http://127.0.0.1:5000
```

### 3. Functionality Overview
- **Home Page**:
  - Links to navigate between product management and image upload features.
- **Upload Image**:
  - Upload an image of shelves to run object detection.
- **Products Page**:
  - View, add, update, or delete product information.

---

## API Endpoints

- **Create Product (POST)**:
  ```http
  POST /api/products
  ```
  - Request Body: 
    ```json
    {
      "name": "Product Name",
      "price": 10.0,
      "in_stock": true
    }
    ```

- **Retrieve Product (GET)**:
  ```http
  GET /api/products/<product_id>
  ```

- **Update Product (PUT)**:
  ```http
  PUT /api/products/<product_id>
  ```
  - Request Body: 
    ```json
    {
      "name": "Updated Product Name",
      "price": 12.5,
      "in_stock": false
    }
    ```

- **Delete Product (DELETE)**:
  ```http
  DELETE /api/products/<product_id>
  ```

---

## Technologies Used
- **Python 3.x**: Backend logic and API development.
- **Flask**: Web framework for API and frontend interface.
- **YOLOv8**: Pre-trained object detection model for detecting products.
- **OpenCV**: Image processing and object detection.
- **Pillow (PIL)**: Image resizing and handling various formats.
- **HTML/CSS**: Frontend UI components.

---

## Future Improvements
- Implement a user authentication system for enhanced product management security.
- Add more advanced AR effects, such as 3D animations.
- Improve UI/UX and replace in-memory storage with a persistent database (e.g., SQLite, PostgreSQL).

---

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contact
For questions or inquiries, please contact:
- **Name**: Jamie (Shih-Hsuan Yan)
- **Email**: [jamieysh0910@gmail.com](mailto:jamieysh0910@gmail.com)
- [GitHub Profile](https://github.com/jamie1016jamie1016)