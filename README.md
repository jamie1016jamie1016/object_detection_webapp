
# Full Stack Developer Coding Test - Object Detection and CRUD API

## Overview
This project demonstrates a full-stack application using **Python** and **Flask**, with image processing, object detection, and CRUD operations. It allows users to upload images of shelves, detects products using a pre-trained YOLO model, and manages product information through a RESTful API.

---

## Features
1. **Object Detection:**
   - Detects objects on shelves using YOLO (You Only Look Once) model.
   - Bounding boxes are drawn on the image to highlight detected products.

2. **CRUD API:**
   - Flask-based API to create, retrieve, update, and delete product information.
   - Product data includes name, price, and stock status.

3. **Augmented Reality Simulation:**
   - Overlays product information on the detected objects in the image.
   - Simulates AR by displaying product details (price, availability) next to the detected items.

---

## Requirements
Before you start, ensure you have the following installed:
- **Python 3.11.9**
- **pip** (Python package manager)

---

## Installation and Setup

### 1. Clone the Repository
```bash
cd your-repo-name
git clone https://github.com/jamie1016jamie1016/object_detection_webapp.git
```

### 2. Set up the Virtual Environment
Create and activate a virtual environment:
```bash
python -m venv env
source env/bin/activate  # For macOS/Linux
# For Windows: env\Scripts\activate
```

### 3. Install the Dependencies
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
Open a web browser and go to:
```
http://127.0.0.1:5000
```

### 3. Functionality
- **Home Page:**
  - Provides links to navigate between product management and image upload functionalities.
  
- **Upload Image:**
  - Allows users to upload an image of shelves for object detection.
  
- **Products Page:**
  - Displays all products and provides functionality to add, update, or delete product information.

---

## API Endpoints

- **Create Product (POST):**
  ```http
  POST /api/products
  ```
  - Request Body: 
    ```json
    {
      "id": "001",
      "name": "Product Name",
      "price": 10.0,
      "in_stock": true
    }
    ```

- **Retrieve Product (GET):**
  ```http
  GET /api/products/<product_id>
  ```

- **Update Product (PUT):**
  ```http
  PUT /api/products/<product_id>
  ```

- **Delete Product (DELETE):**
  ```http
  DELETE /api/products/<product_id>
  ```

---

## Technologies Used
- **Python 3.x**: Backend logic and API development.
- **Flask**: Web framework used to build the API and frontend interface.
- **YOLOv8**: Pre-trained object detection model for identifying objects on shelves.
- **OpenCV**: Library for image processing and object detection.
- **Pillow (PIL)**: Image processing library used for image resizing and handling various formats.
- **HTML/CSS**: Frontend UI elements.

---

## Future Improvements
- Implement a user login system for product management.
- Add more advanced AR effects such as 3D animations.
- Improve the UI and integrate a proper database instead of in-memory storage.

---

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contact
For any questions or inquiries, please contact:
- **Name**: Jamie (Shih-Hsuan Yan)
- **Email**: jamieysh0910@gmail.com
- [GitHub Profile](https://github.com/jamie1016jamie1016)
