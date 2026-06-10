# 🫁 Smart Detection of Lung Cancer via Image Processing & Symptom Integration

A comprehensive machine learning and computer vision application designed to aid in the early detection and prediction of lung cancer. This project uniquely combines patient symptom data with advanced medical image processing techniques, wrapped in a user-friendly web interface.

## 🚀 Features

* **Symptom-Based Prediction:** Analyzes patient survey data to predict lung cancer risk using machine learning models trained on robust datasets.
* **Advanced Image Processing:** Applies computer vision techniques to medical images (such as X-rays/CT scans) including:
  * Grayscale conversion
  * CLAHE (Contrast Limited Adaptive Histogram Equalization) for enhanced visibility
  * Edge Detection for structural analysis
  * HOG (Histogram of Oriented Gradients) feature extraction
* **Interactive Web Interface:** A Flask-based web application (`app.py`) allowing users to easily upload images and input symptoms for real-time analysis.
* **Data Augmentation:** Includes custom scripts (`data_augment.py`) to artificially expand the training dataset and improve model robustness.

## 🛠️ Tech Stack

* **Language:** Python 3.x
* **Web Framework:** Flask (via `app.py`)
* **Machine Learning & Data Science:** Jupyter Notebooks, Pandas, Scikit-Learn
* **Image Processing:** OpenCV / Scikit-Image (for HOG, CLAHE, Edge Detection)
* **Frontend:** HTML, CSS (`templates/`, `static/`)

## 📂 Project Structure

```text
Smart-Detection-of-Lung-Cancer.../
│
├── Dataset/                     # Raw and processed datasets
│   ├── survey lung cancer.csv
│   └── survey_lung_cancer_500.csv
│
├── Python Code/                 # Exploratory Data Analysis & Model Training
│   └── Lung Cancer Prediction.ipynb
│
├── scripts/                     # Utility scripts
│   └── data_augment.py          # Script for image/data augmentation
│
├── webapp/                      # Web application directory
│   ├── app.py                   # Main Flask backend server
│   ├── static/                  # Static assets
│   │   ├── css/styles.css       # UI Styling
│   │   ├── images/              # Generated visualizations (e.g., confusion_matrix.png)
│   │   └── uploads/             # User-uploaded and processed images
│   └── templates/               # HTML templates
│       ├── base.html
│       ├── index.html
│       └── result.html
│
└── requirements.txt             # Python dependencies
⚙️ Getting Started
Follow these steps to set up the project on your local machine.

Prerequisites
Python 3.8+

Virtual Environment (recommended)

Installation
1. Clone the repository:

Bash
git clone [https://github.com/mansiym13-sketch/Smart-Detection-of-Lung-Cancer-via-Image-Processing-and-Symptom-Integration.git](https://github.com/mansiym13-sketch/Smart-Detection-of-Lung-Cancer-via-Image-Processing-and-Symptom-Integration.git)
cd Smart-Detection-of-Lung-Cancer-via-Image-Processing-and-Symptom-Integration
2. Create and activate a virtual environment:

Bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
3. Install dependencies:

Bash
pip install -r requirements.txt
Running the Web Application
1. Navigate to the webapp directory:

Bash
cd webapp
2. Start the Flask server:

Bash
python app.py
The application will start, typically accessible in your browser at http://127.0.0.1:5000.

🧠 Model Training (Optional)
If you wish to explore the data or retrain the model:

Ensure Jupyter is installed (pip install jupyter).

Navigate to the Python Code/ directory.

Launch Jupyter Notebook and open Lung Cancer Prediction.ipynb.

Bash
jupyter notebook
