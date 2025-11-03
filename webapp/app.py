import os
from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename
from skimage import io, color, exposure, feature, transform, util

app = Flask(__name__)

# Load data and train model at startup
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(PROJECT_ROOT, 'Dataset', 'survey_lung_cancer_500.csv')
FALLBACK_DATA_PATH = os.path.join(PROJECT_ROOT, 'Dataset', 'survey lung cancer.csv')
STATIC_DIR = os.path.join(os.path.dirname(__file__), 'static')
IMG_DIR = os.path.join(STATIC_DIR, 'images')
os.makedirs(IMG_DIR, exist_ok=True)
UPLOAD_DIR = os.path.join(STATIC_DIR, 'uploads')
os.makedirs(UPLOAD_DIR, exist_ok=True)

FEATURE_COLUMNS = [
    'GENDER','AGE','SMOKING','YELLOW_FINGERS','ANXIETY','PEER_PRESSURE',
    'CHRONIC DISEASE','FATIGUE','ALLERGY','WHEEZING','ALCOHOL CONSUMING',
    'COUGHING','SHORTNESS OF BREATH','SWALLOWING DIFFICULTY','CHEST PAIN'
]
TARGET_COLUMN = 'LUNG_CANCER'

model = None
metrics = {}
feature_importances = []

def load_and_train():
    global model, metrics, feature_importances
    csv_path = DATA_PATH if os.path.exists(DATA_PATH) else FALLBACK_DATA_PATH
    df = pd.read_csv(csv_path)

    # Map textual labels if present (robustness)
    if df['GENDER'].dtype == object:
        df['GENDER'] = df['GENDER'].map({'M':1,'F':2})
    if df[TARGET_COLUMN].dtype == object:
        df[TARGET_COLUMN] = df[TARGET_COLUMN].map({'YES':1,'NO':2})

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label=1)
    rec = recall_score(y_test, y_pred, pos_label=1)
    f1 = f1_score(y_test, y_pred, pos_label=1)

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred, labels=[1,2])
    fig, ax = plt.subplots(figsize=(4,4))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(['YES','NO']); ax.set_yticklabels(['YES','NO'])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha='center', va='center', color='black')
    fig.tight_layout()
    cm_path = os.path.join(IMG_DIR, 'confusion_matrix.png')
    fig.savefig(cm_path)
    plt.close(fig)

    # Feature importances
    feature_importances = []
    if hasattr(clf, 'feature_importances_'):
        importances = clf.feature_importances_
        order = np.argsort(importances)[::-1]
        for idx in order[:10]:
            feature_importances.append({
                'feature': FEATURE_COLUMNS[idx],
                'importance': float(importances[idx])
            })

    model = clf
    metrics = {
        'dataset_size': len(df),
        'accuracy': round(float(acc), 4),
        'precision': round(float(prec), 4),
        'recall': round(float(rec), 4),
        'f1': round(float(f1), 4),
        'cm_image': 'static/images/confusion_matrix.png'
    }


def ensure_model():
    global model
    if model is None:
        load_and_train()

@app.route('/')
def index():
    ensure_model()
    return render_template('index.html', metrics=metrics, features=feature_importances)

@app.route('/predict', methods=['POST'])
def predict():
    ensure_model()

    def to_int(val, default=1):
        try:
            return int(val)
        except Exception:
            return default

    form_data = {c: to_int(request.form.get(c)) for c in FEATURE_COLUMNS}

    X_input = pd.DataFrame([form_data], columns=FEATURE_COLUMNS)
    proba = getattr(model, 'predict_proba', None)
    if callable(proba):
        probs = model.predict_proba(X_input)[0]
        # class order assumed [1,2]
        try:
            idx_yes = list(model.classes_).index(1)
        except Exception:
            idx_yes = 0
        symptom_prob = float(probs[idx_yes])
    else:
        # fallback to hard prediction
        symptom_prob = 1.0 if int(model.predict(X_input)[0]) == 1 else 0.0

    # Handle optional image upload
    image_info = None
    image_score = None
    file = request.files.get('image')
    if file and file.filename:
        filename = secure_filename(file.filename)
        save_path = os.path.join(UPLOAD_DIR, filename)
        file.save(save_path)

        # Read and process
        img = io.imread(save_path)
        if img.ndim == 3:
            gray = color.rgb2gray(img)
        else:
            gray = util.img_as_float(img)
        # Resize to consistent size for HOG
        gray_rs = transform.resize(gray, (256, 256), anti_aliasing=True)
        # Contrast enhancement
        clahe = exposure.equalize_adapthist(gray_rs, clip_limit=0.03)
        # Edge detection
        edges = feature.canny(clahe, sigma=1.5)
        # HOG features
        hog_vec, hog_img = feature.hog(clahe, pixels_per_cell=(16,16), cells_per_block=(2,2), visualize=True, block_norm='L2-Hys')

        # Simple image metrics (normalized)
        edge_density = float(edges.mean())  # [0,1]
        contrast = float(clahe.std())       # roughly [0, ~0.5]
        hog_energy = float(np.linalg.norm(hog_vec))

        # Normalize heuristically
        contrast_n = min(contrast / 0.5, 1.0)
        hog_n = min(hog_energy / (np.sqrt(len(hog_vec)) * 0.2), 1.0)
        image_score = float(np.clip(0.4*edge_density + 0.3*contrast_n + 0.3*hog_n, 0, 1))

        # Save previews
        plt.imsave(os.path.join(UPLOAD_DIR, f"{filename}_gray.png"), gray_rs, cmap='gray')
        plt.imsave(os.path.join(UPLOAD_DIR, f"{filename}_clahe.png"), clahe, cmap='gray')
        plt.imsave(os.path.join(UPLOAD_DIR, f"{filename}_edges.png"), edges, cmap='gray')
        plt.imsave(os.path.join(UPLOAD_DIR, f"{filename}_hog.png"), hog_img, cmap='gray')

        image_info = {
            'original': f'static/uploads/{filename}',
            'gray': f'static/uploads/{filename}_gray.png',
            'clahe': f'static/uploads/{filename}_clahe.png',
            'edges': f'static/uploads/{filename}_edges.png',
            'hog': f'static/uploads/{filename}_hog.png',
            'metrics': {
                'edge_density': round(edge_density, 4),
                'contrast': round(contrast, 4),
                'hog_energy': round(hog_energy, 4)
            }
        }

    # Combine symptom probability with image score (if available)
    if image_score is not None:
        combined_risk = round(float(0.7*symptom_prob + 0.3*image_score), 4)
    else:
        combined_risk = round(float(symptom_prob), 4)

    label = 'YES' if combined_risk >= 0.5 else 'NO'

    return render_template(
        'result.html',
        input_data=form_data,
        prediction=label,
        symptom_prob=round(float(symptom_prob),4),
        image_score=(round(float(image_score),4) if image_score is not None else None),
        combined_risk=combined_risk,
        image_info=image_info
    )

@app.route('/api/predict', methods=['POST'])
def api_predict():
    ensure_model()
    data = request.get_json(silent=True) or {}
    def to_int(val, default=1):
        try:
            return int(val)
        except Exception:
            return default
    values = {c: to_int(data.get(c)) for c in FEATURE_COLUMNS}
    X_input = pd.DataFrame([values], columns=FEATURE_COLUMNS)
    pred = model.predict(X_input)[0]
    label = 'YES' if int(pred) == 1 else 'NO'
    return jsonify({
        'prediction': label,
        'raw': int(pred),
        'inputs': values
    })


if __name__ == '__main__':
    # host for local dev
    load_and_train()
    app.run(host='0.0.0.0', port=5000, debug=True)
