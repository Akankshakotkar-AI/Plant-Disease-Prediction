# Plant-Disease-Prediction
Plant disease prediction web application that detects plant diseases from leaf images using Python, OpenCV, and Machine Learning.
## Project Structure
```
/plant-disease-detection
    /backend
        app.py           # Flask Server & Disease Knowledge Base
        model.h5         # Pre-trained CNN Model (Place yours here)
        class_names.json # Predicted class index mapping
        requirements.txt # Python dependencies
    /frontend
        index.html       # Landing page & UI structure
        style.css        # Premium nature-themed styling
        script.js        # Camera & API integration
```

## Setup Instructions

### 1. Backend (API)
1. Ensure you have Python installed.
2. Navigate to the `backend` directory:
   ```bash
   cd backend
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. **IMPORTANT**: Place your trained `model.h5` file inside the `backend` folder.
5. Start the server:
   ```bash
   python app.py
   ```
   The API will be available at `http://localhost:5000`.

### 2. Frontend
1. Open `frontend/index.html` in any modern web browser (Chrome, Edge, Firefox).
2. Drag and drop a leaf image or use the 'Use Camera' feature to capture a photo.
3. Click 'Analyze Leaf' to get a detailed health report.

### Supported Plants
- Tomato (10 conditions including healthy)
- Potato (3 conditions including healthy)
- Bell Pepper (2 conditions including healthy)

## Technologies Used
- **Backend**: Flask, TensorFlow, Keras, PIL, NumPy
- **Frontend**: Vanilla JavaScript (ES6+), CSS3 (Modern Glassmorphism), HTML5 Semantic Tags
- **Icons**: Font Awesome 6
- **Fonts**: Google Fonts (Outfit)
