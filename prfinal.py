import sys 
import os
import re
from functools import partial
import cv2
import numpy as np
import tensorflow as tf
try:
    import tensorflow_hub as hub
except Exception:
    hub = None
import json
import sqlite3
from datetime import datetime
from PyQt5.QtWidgets import (
    QApplication, QLabel, QPushButton, QVBoxLayout,
    QWidget, QTextEdit, QFileDialog, QHBoxLayout,
    QStackedWidget, QFrame, QLineEdit, QMessageBox, QScrollArea, QGridLayout, QGroupBox,
    QDialog, QTableWidget, QTableWidgetItem, QHeaderView, QSizePolicy, QGraphicsDropShadowEffect, QProgressBar
)
from PyQt5.QtGui import QImage, QPixmap, QTextDocument, QFont, QColor, QPainter, QPen, QBrush, QRadialGradient
from PyQt5.QtCore import Qt, QTimer, QRectF
from PyQt5.QtPrintSupport import QPrinter
from skimage.feature import graycomatrix, graycoprops

# ---------------- CONFIG ----------------
IMG_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 30
LEAF_AREA_THRESHOLD = 15.0
HISTORY_FILE = "scan_history.json"
DB_FILE = "users.db"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPORTS_DIR = os.path.join(BASE_DIR, "downloaded_reports")
REPORT_BG_IMAGE = r"C:\Users\Sarika\.cursor\projects\d-project\assets\c__Users_Sarika_AppData_Roaming_Cursor_User_workspaceStorage_f97d7464d74a81a780fc9b512ee2b018_images_image-2f44f267-ba0b-4b7f-a306-503639751053.png"

os.makedirs(REPORTS_DIR, exist_ok=True)

MODEL_PATH = r"D:\project\model\plant_disease_model_4.h5"
TREATMENT_PATH = r"D:\project\plant_disease_mobile\assets\treatment_expert.json"
LANDING_BG = os.path.join(BASE_DIR, "assets", "landing_bg.png")
LOGIN_BG = os.path.join(BASE_DIR, "assets", "login_bg.png")

# ---------------- DATABASE ----------------
class DatabaseManager:
    def __init__(self):
        self.conn = sqlite3.connect(DB_FILE); self.create_table()
    def create_table(self):
        self.conn.execute("CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, email TEXT, password TEXT)")
        self.conn.commit()
    def register_user(self, u, e, p):
        try: self.conn.execute("INSERT INTO users VALUES (?, ?, ?)", (u, e, p)); self.conn.commit(); return True
        except: return False
    def validate_user(self, u, p):
        cursor = self.conn.execute("SELECT * FROM users WHERE username=? AND password=?", (u, p))
        return cursor.fetchone() is not None
    def reset_password(self, u, e, p):
        cursor = self.conn.execute("SELECT * FROM users WHERE username=? AND email=?", (u, e))
        if cursor.fetchone():
            self.conn.execute("UPDATE users SET password=? WHERE username=?", (p, u))
            self.conn.commit(); return True
        return False

db_mgr = DatabaseManager()

# ---------------- LOAD RESOURCES ----------------
try:
    if hasattr(tf, "keras") and hasattr(tf.keras, "models"):
        custom_objects = {'KerasLayer': hub.KerasLayer} if hub is not None else None
        model = tf.keras.models.load_model(
            MODEL_PATH,
            custom_objects=custom_objects,
            compile=False
        )
    else:
        raise RuntimeError("TensorFlow installation does not provide tf.keras")
except Exception as e:
    print("Model load error:", e)
    model = None
    MODEL_LOAD_ERROR = str(e)
else:
    MODEL_LOAD_ERROR = ""

try:
    with open(TREATMENT_PATH, "r") as f:
        treatment_dict = json.load(f)
except Exception as e:
    print("JSON load error:", e)
    treatment_dict = {}

class_names = [
    'Chilli_cercospora','Chilli_healthy','Chilli_mites_and_trips','Chilli_nutritional',
    'Chilli_powdery mildew','Corn Gray leaf spot','Corn leaf blight','Corn rust leaf',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot','Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight','Corn_(maize)___healthy','Okra Insects_eaten_leaves',
    'Okra Leaf_curl','Okra Leaf_spot','Okra Predisposition','Okra Yellow_vein_mosaic_virus',
    'Okra healthy','Pumpkin Bacterial Leaf Spot','Pumpkin Downy Mildew','Pumpkin Healthy Leaf',
    'Pumpkin Mosaic Disease','Pumpkin Powdery_Mildew'
]

CROP_INDICES = {"Chilli": (0, 5), "Corn": (5, 12), "Okra": (12, 18), "Pumpkin": (18, 23)}

# ---------------- LOGIC ----------------
def extract_features(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB); hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV); gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    R,G,B = cv2.split(rgb); H,S,V = cv2.split(hsv); glcm = graycomatrix(gray, [1], [0], 256, symmetric=True, normed=True)
    return {
        "Mean_R": np.mean(R), "Mean_G": np.mean(G), "Mean_B": np.mean(B),
        "Std_R": np.std(R), "Std_G": np.std(G), "Std_B": np.std(B),
        "Contrast": graycoprops(glcm, 'contrast')[0, 0], "Energy": graycoprops(glcm, 'energy')[0, 0],
        "Homogeneity": graycoprops(glcm, 'homogeneity')[0, 0], "Correlation": graycoprops(glcm, 'correlation')[0, 0],
        "Hue_Range": np.max(H) - np.min(H), "Mean_Luminance": np.mean(V),
        "Entropy": -np.sum((cv2.calcHist([gray],[0],None,[256],[0,256])/gray.size)*np.log2((cv2.calcHist([gray],[0],None,[256],[0,256])/gray.size)+1e-7))
    }

def precise_segmentation(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # --- VISUAL PRESENTATION MASK ---
    # We want the crisp, natural leaf edge, but we DO NOT want to punch black holes
    # where the disease (like white powdery mildew or brown spots) is located.
    green_mask = cv2.inRange(hsv, np.array([20, 25, 25]), np.array([95, 255, 255]))
    
    # RETR_EXTERNAL gets ONLY the true outer boundaries, ignoring internal spots
    contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    visual_mask = np.zeros_like(green_mask)
    
    if contours:
        # Remove tiny background noise specs
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 500]
        # Filling the outer boundary naturally restores internally diseased tissue!
        cv2.drawContours(visual_mask, valid_contours, -1, 255, thickness=cv2.FILLED)
        
    # The image shown on the UI: Perfectly cut out natural leaf, no black holes inside.
    res = cv2.bitwise_and(img, img, mask=visual_mask)
    
    # --- TRUE AGRICULTURAL SEVERITY MATH ---
    # Relax hue bounds to mathematically capture all browns/yellows/white blights
    broad_color_mask = cv2.inRange(hsv, np.array([0, 15, 20]), np.array([179, 255, 255]))
    
    math_contours, _ = cv2.findContours(broad_color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    math_total = np.zeros_like(broad_color_mask)
    
    if math_contours:
        math_contours = sorted(math_contours, key=cv2.contourArea, reverse=True)
        max_area = cv2.contourArea(math_contours[0])
        for cnt in math_contours:
            if cv2.contourArea(cnt) > max_area * 0.05:
                # Convex hull for math ensures we don't miss severely decayed edges
                cv2.drawContours(math_total, [cv2.convexHull(cnt)], -1, 255, thickness=cv2.FILLED)
                
    # Defined strictly healthy green tissue
    strictly_healthy = cv2.inRange(hsv, np.array([30, 25, 25]), np.array([95, 255, 255]))
    damage_mask = cv2.bitwise_and(math_total, cv2.bitwise_not(strictly_healthy))
    
    total_area = np.sum(math_total > 0)
    damage_area = np.sum(damage_mask > 0)
    
    if total_area < 100:
        severity = 0.0
    else:
        linear_severity = (damage_area / total_area) * 100.0
        # Weber-Fechner perception curve for agricultural scoring
        severity = (linear_severity / 100.0) ** 0.5 * 100.0
        
    severity = max(0.0, min(100.0, severity))
    
    return res, severity

def process_image(img, selected_crop):
    if model is None:
        return {
            "disease": f"MODEL BACKEND ERROR ({MODEL_LOAD_ERROR})",
            "confidence": 0,
            "severity": 0,
            "treatment_data": {
                "Causal_Organism": "Unavailable (model backend not loaded)",
                "Symptoms": "Image is loaded, but disease prediction is unavailable.",
                "Management_Cultural": "Use Python 3.10 and install a valid TensorFlow build, then restart.",
                "Management_Chemical": "After TensorFlow is fixed, rerun scan to get disease-specific treatment."
            },
            "image": img,
            "features": {}
        }
    
    original = img.copy()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_batch = np.expand_dims(cv2.resize(img_rgb, IMG_SIZE) / 255.0, axis=0)
    
    # 1. Absolute Model Prediction
    prediction = model.predict(img_batch, verbose=0)
    global_idx = np.argmax(prediction)
    global_winner = class_names[global_idx]
    global_confidence = np.max(prediction) * 100
    
    # 2. Local Crop-Specific Prediction
    start, end = CROP_INDICES[selected_crop]
    crop_probs = prediction[0][start:end]
    local_idx = np.argmax(crop_probs)
    local_winner = class_names[start + local_idx]
    local_confidence = np.max(crop_probs) * 100
    
    def get_fam(name):
        for f in ["Chilli", "Corn", "Okra", "Pumpkin"]:
            if f.lower() in name.lower(): return f
        return "Unknown"

    global_fam = get_fam(global_winner)
    
    # 3. Leaf Validation (HSV Greenery Detection)
    hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
    leaf_mask = cv2.inRange(hsv, np.array([20, 30, 30]), np.array([95, 255, 255]))
    leaf_ratio = (np.sum(leaf_mask > 0) / leaf_mask.size) * 100
    
    # Segment for visual severity
    segmented, severity = precise_segmentation(original)
    
    # 4. Strict Decision Logic (Rejection System)
    diagnosis = local_winner
    treatment_key = local_winner
    final_confidence = local_confidence
    is_rejected = False
    
    # Sequence of rejection:
    if leaf_ratio < LEAF_AREA_THRESHOLD:
        diagnosis = "DOES NOT WORK (NOT A PLANT/LEAF)"
        treatment_key = "Unknown"
        final_confidence = 0
        is_rejected = True
    elif global_confidence < 30:
        diagnosis = "DOES NOT WORK (UNKNOWN PLANT)"
        treatment_key = "Unknown"
        is_rejected = True
    elif global_fam != selected_crop:
        # Important: If it looks like a different plant entirely, call it Unknown
        diagnosis = f"DOES NOT WORK (Looks like {global_fam})"
        treatment_key = global_winner
        final_confidence = global_confidence
    elif local_confidence < CONFIDENCE_THRESHOLD:
        diagnosis = "DOES NOT WORK (UNKNOWN/BLURRY)"
        treatment_key = "Unknown"
        is_rejected = True

    default_treatment = {
        "Causal_Organism": "Not available in treatment database",
        "Symptoms": "No symptom notes available for this class.",
        "Management_Cultural": "Use preventive care and monitor progression.",
        "Management_Chemical": "Consult local agronomist for crop-safe treatment."
    }
    
    def enrich(t, label):
        # Specific Enrichment for better suggestions
        extra = {
            "Corn": {"Med": "Propiconazole (1ml/L) or Copper Oxychloride.", "Fert": "Apply Urea (High Nitrogen) and Phosphorus-rich fertilizer."},
            "Chilli": {"Med": "Mancozeb (2g/L) for leaf spots or Imidacloprid for pests.", "Fert": "Apply Potassium-rich NPK for better fruit quality."},
            "Okra": {"Med": "Spray Neem Oil or Dimethoate (1.5ml/L) for virus vectors.", "Fert": "Apply Balanced NPK with Calcium/Magnesium micro-nutrients."},
            "Pumpkin": {"Med": "Sulfur-based fungicides or Carbendazim (1g/L).", "Fert": "High Phosphorus and Potassium for healthy vine growth."},
            "Healthy": {"Med": "No medicine required. Keep monitored.", "Fert": "Apply Organic Compost and balanced organic nutrients."}
        }
        fam = get_fam(label)
        mode = "Healthy" if "healthy" in label.lower() else fam
        data = extra.get(mode, {"Med": "Consult agronomist.", "Fert": "Balanced NPK."})
        t["Medicine"] = data["Med"]
        t["Fertilizer"] = data["Fert"]
        return t

    full_treatment = enrich(treatment_dict.get(treatment_key, default_treatment).copy(), treatment_key)
    
    if is_rejected:
        severity = 0.0
        full_treatment = {
            "Causal_Organism": "-",
            "Symptoms": "-",
            "Management_Cultural": "-",
            "Management_Chemical": "-",
            "Medicine": "-",
            "Fertilizer": "-"
        }
    
    return {
        "disease": diagnosis, 
        "confidence": final_confidence, 
        "severity": severity, 
        "treatment_data": full_treatment, 
        "original": original,
        "segmented": segmented, 
        "heatmap": generate_visual_heatmap(original),
        "features": extract_features(original)
    }

# ---------------- STYLES ----------------
THEMES = {
    "midnight": {
        "bg_1": "#000000",
        "bg_2": "#050a05",
        "card": "rgba(10, 30, 10, 0.65)",
        "glass": "rgba(0, 168, 107, 0.15)",
        "text": "#ffffff",
        "muted_text": "#a0eaba",
        "accent": "#00ff88",
        "accent_hover": "#33ffa3",
        "danger": "#ff3333",
        "danger_hover": "#cc0000",
        "border": "rgba(0, 255, 136, 0.3)",
        "header_bg": "rgba(0, 20, 0, 0.8)",
        "header_text": "#00ff88",
        "input_bg": "rgba(10, 20, 10, 0.85)",
        "input_text": "#ffffff"
    },
    "slate": {
        "bg_1": "#f2f5fb",
        "bg_2": "#e9eef7",
        "card": "#ffffff",
        "text": "#1f2b43",
        "muted_text": "#54607a",
        "accent": "#1b5e20",
        "accent_hover": "#144618",
        "danger": "#d44e43",
        "danger_hover": "#be3b30",
        "border": "#cad4ea",
        "header_bg": "#1f2b43",
        "header_text": "#ffffff",
        "input_bg": "#ffffff",
        "input_text": "#192236"
    }
}


def build_stylesheet(theme_name):
    t = THEMES.get(theme_name, THEMES["midnight"])
    return f"""
QWidget {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 {t['bg_1']}, stop:1 {t['bg_2']});
    color: {t['text']};
    font-family: 'Segoe UI', 'Verdana', sans-serif;
}}
QLabel#Header {{
    background: {t['header_bg']};
    color: {t['header_text']};
    font-size: 31px;
    font-weight: 700;
    letter-spacing: 1px;
    padding: 18px;
    border-radius: 16px;
}}
QFrame#FeatureCard, QFrame#Card {{
    background: {t['card']};
    border-radius: 20px;
    border: 1px solid {t['border']};
}}
QLabel#CardTitle {{
    font-size: 22px;
    font-weight: 700;
    color: {t['text']};
}}
QLabel#CardText {{
    font-size: 15px;
    color: {t['muted_text']};
}}
QPushButton {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 {t['accent']}, stop:1 {t['accent_hover']});
    color: #ffffff;
    padding: 12px 16px;
    font-size: 15px;
    font-weight: 700;
    border-radius: 12px;
    border: 1px solid transparent;
}}
QPushButton:hover {{
    background: {t['accent_hover']};
    border: 1px solid {t['border']};
}}
QPushButton:pressed {{
    padding-top: 13px;
}}
QPushButton#SecondaryBtn, QPushButton#ThemeBtn {{
    background: {t['card']};
    color: {t['text']};
    border: 1px solid {t['border']};
}}
QPushButton#DangerBtn {{
    background: {t['danger']};
    color: #ffffff;
}}
QPushButton#DangerBtn:hover {{
    background: {t['danger_hover']};
}}
QPushButton#SnapBtn {{
    font-size: 16px;
}}
QLineEdit {{
    padding: 11px 13px;
    font-size: 15px;
    border-radius: 10px;
    border: 1px solid {t['border']};
    background: {t['input_bg']};
    color: {t['input_text']};
}}
QLineEdit:focus {{
    border: 2px solid {t['accent']};
}}
QTextEdit {{
    background: {t['card']};
    color: {t['text']};
    border: 1px solid {t['border']};
    border-radius: 12px;
    padding: 8px;
}}
QGroupBox {{
    border: 1px solid {t['border']};
    border-radius: 12px;
    margin-top: 8px;
    font-size: 15px;
    font-weight: 600;
    color: {t['text']};
    background: {t['card']};
}}
QGroupBox::title {{
    left: 10px;
    padding: 0 6px;
}}
QTableWidget {{
    background: {t['card']};
    border-radius: 10px;
    border: 1px solid {t['border']};
    gridline-color: {t['border']};
}}
QHeaderView::section {{
    background: {t['accent']};
    color: #ffffff;
    border: none;
    padding: 6px;
    font-weight: 700;
}}
QLabel#AuthHint {{
    padding: 8px 10px;
    border-radius: 10px;
    background: {t['card']};
    border: 1px dashed {t['border']};
    color: {t['muted_text']};
    font-size: 13px;
}}
QLabel#AuthHint[mode="error"] {{
    color: #f5aaa0;
    border: 1px solid #a74a43;
}}
QLabel#AuthHint[mode="ok"] {{
    color: #8ae2c2;
    border: 1px solid #2f8065;
}}
QProgressBar {{
    border: 1px solid {t['border']};
    border-radius: 8px;
    background: {t['card']};
    height: 10px;
}}
QProgressBar::chunk {{
    border-radius: 7px;
    background: {t['accent']};
}}
"""


def apply_card_shadow(widget):
    shadow = QGraphicsDropShadowEffect(widget)
    shadow.setBlurRadius(20)
    shadow.setOffset(0, 5)
    shadow.setColor(QColor(0, 255, 136, 40))
    widget.setGraphicsEffect(shadow)

class AnalysisQuadrant(QFrame):
    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background: rgba(0, 20, 10, 0.8); border: 1px solid rgba(0, 255, 136, 0.2); border-radius: 12px;")
        l = QVBoxLayout(self)
        l.setContentsMargins(0, 0, 0, 0)
        l.setSpacing(0)
        
        self.header = QLabel(title)
        self.header.setFixedHeight(28)
        self.header.setStyleSheet("background: rgba(0, 255, 136, 0.1); border-bottom: 1px solid rgba(0, 255, 136, 0.2); color: #00ff88; font-size: 11px; font-weight: bold; padding-left: 10px; border-top-left-radius: 12px; border-top-right-radius: 12px;")
        l.addWidget(self.header)
        
        self.content = QLabel("")
        self.content.setAlignment(Qt.AlignCenter)
        self.content.setStyleSheet("border: none; background: transparent;")
        l.addWidget(self.content, 1)

class DiagnosticGrid(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.l = QGridLayout(self)
        self.l.setContentsMargins(10, 10, 10, 10)
        self.l.setSpacing(15)
        
        self.q1 = AnalysisQuadrant("ORIGINAL CAPTURE")
        self.q2 = AnalysisQuadrant("LEAF SEGMENTATION")
        self.q3 = AnalysisQuadrant("GRAD-CAM ANALYSIS")
        self.q4 = AnalysisQuadrant("DIAGNOSTIC SUMMARY")
        
        self.l.addWidget(self.q1, 0, 0)
        self.l.addWidget(self.q2, 0, 1)
        self.l.addWidget(self.q3, 1, 0)
        self.l.addWidget(self.q4, 1, 1)
        
        # Setup Summary layout
        self.summary_l = QHBoxLayout(self.q4.content)
        self.summary_l.setAlignment(Qt.AlignCenter)
        self.summary_l.setSpacing(20)
        
        self.conf_gauge = CircularGauge("CONFIDENCE")
        self.sev_gauge = CircularGauge("SEVERITY")
        
        self.summary_l.addWidget(self.conf_gauge)
        self.summary_l.addWidget(self.sev_gauge)

    def update_results(self, original, segmented, heatmap, confidence, severity):
        def to_px(img):
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, c = rgb.shape
            q = QImage(rgb.data, w, h, w*c, QImage.Format_RGB888)
            return QPixmap.fromImage(q.copy()).scaled(self.q1.content.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            
        self.q1.content.setPixmap(to_px(original))
        self.q2.content.setPixmap(to_px(segmented))
        self.q3.content.setPixmap(to_px(heatmap))
        
        self.conf_gauge.setValue(confidence)
        self.sev_gauge.setValue(severity)

def generate_visual_heatmap(img):
    # 1. Structural Awareness (Leaf only)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    l_m = cv2.inRange(hsv, np.array([20, 20, 20]), np.array([105, 255, 255]))
    l_m = cv2.morphologyEx(l_m, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))

    # 2. Hybrid Attention Capture (Texture + Color)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    energy = np.uint8(np.absolute(cv2.Laplacian(gray, cv2.CV_64F)))
    energy = cv2.GaussianBlur(energy, (11, 11), 0)
    
    m1 = cv2.inRange(hsv, np.array([10, 30, 30]), np.array([30, 255, 255]))
    m2 = cv2.inRange(hsv, np.array([90, 30, 30]), np.array([110, 255, 255]))
    color_dev = cv2.bitwise_or(m1, m2)
    color_dev = cv2.GaussianBlur(color_dev, (11, 11), 0)

    combined = cv2.addWeighted(energy, 0.7, color_dev, 0.5, 0)
    combined = cv2.bitwise_and(combined, combined, mask=l_m)
    
    attn = cv2.normalize(combined, None, 0, 255, cv2.NORM_MINMAX)
    attn = cv2.threshold(attn, 30, 255, cv2.THRESH_TOZERO)[1]

    # 3. Classic Rainbow (JET) Color Mapping
    # High-intensity red = Heat, Low-intensity blue = Cool
    heatmap_c = cv2.applyColorMap(attn, cv2.COLORMAP_JET)

    # 4. HUI Synthesis (Glowing Grid Overlay)
    # Dim the original image for contrast
    dimmed = cv2.addWeighted(img, 0.5, np.zeros_like(img), 0.5, 0)
    
    # Composite: Dimmed image + Heatmap
    alpha = 0.5
    final = cv2.addWeighted(dimmed, 1.0, heatmap_c, alpha, 0)
    
    # Mask final back to leaf structure only
    final = cv2.bitwise_and(final, final, mask=l_m)
    
    # Add background (black out everything non-leaf in this view for focus)
    bg = np.zeros_like(img)
    mask_inv = cv2.bitwise_not(l_m)
    res = cv2.bitwise_or(final, cv2.bitwise_and(bg, bg, mask=mask_inv))
    
    return res

class InteractiveButton(QPushButton):
    def __init__(self, text="", object_name=None):
        super().__init__(text)
        if object_name:
            self.setObjectName(object_name)
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumHeight(46)
        self._shadow = QGraphicsDropShadowEffect(self)
        self._shadow.setBlurRadius(16)
        self._shadow.setOffset(0, 4)
        self._shadow.setColor(QColor(18, 34, 56, 35))
        self.setGraphicsEffect(self._shadow)

    def enterEvent(self, event):
        self._shadow.setBlurRadius(24)
        self._shadow.setOffset(0, 8)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._shadow.setBlurRadius(16)
        self._shadow.setOffset(0, 4)
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        self._shadow.setOffset(0, 2)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        self._shadow.setOffset(0, 8 if self.underMouse() else 4)
        super().mouseReleaseEvent(event)


def auth_message(label, text, mode="normal"):
    label.setText(text)
    label.setProperty("mode", mode)
    label.style().unpolish(label)
    label.style().polish(label)


def password_strength(password):
    score = 0
    if len(password) >= 8:
        score += 25
    if re.search(r"[A-Z]", password) and re.search(r"[a-z]", password):
        score += 25
    if re.search(r"\d", password):
        score += 25
    if re.search(r"[^\w]", password):
        score += 25
    if score < 40:
        return score, "Weak"
    if score < 75:
        return score, "Medium"
    return score, "Strong"

# ---------------- PAGES ----------------
class HistoryWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent); self.setWindowTitle("System Diagnostic Logs"); self.setMinimumSize(1000, 650);
        self.setStyleSheet("background: #000; color: #fff;")
        l = QVBoxLayout(self); 
        
        header = QLabel("CENTRAL DIAGNOSTIC LOGS"); header.setStyleSheet("font-size: 24px; color: #00ff88; font-weight: bold; margin-bottom: 10px;")
        l.addWidget(header)
        
        self.table = QTableWidget(); self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["[ SELECT ]", "TIMESTAMP", "DIAGNOSIS", "SEVERITY", "STATUS"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setStyleSheet("QTableWidget { background: #050a05; gridline-color: #00ff88; border: 1px solid #00ff88; } QTableWidget::item { padding: 10px; }")
        l.addWidget(self.table)
        
        btn_box = QHBoxLayout()
        self.btn_del = QPushButton("DELETE SELECTED RECORDS"); self.btn_del.setMinimumHeight(50)
        self.btn_del.setStyleSheet("background: #ff3333; color: white; font-weight: bold; border-radius: 8px;")
        self.btn_del.clicked.connect(self.delete_selected)
        btn_box.addWidget(self.btn_del)
        
        self.btn_ref = QPushButton("REFRESH"); self.btn_ref.setMinimumHeight(50)
        self.btn_ref.setStyleSheet("background: #00ff88; color: black; font-weight: bold; border-radius: 8px;")
        self.btn_ref.clicked.connect(self.load_data)
        btn_box.addWidget(self.btn_ref)
        l.addLayout(btn_box)
        
        self.load_data()

    def load_data(self):
        self.table.setRowCount(0)
        if not os.path.exists(HISTORY_FILE): return
        try:
            with open(HISTORY_FILE, "r") as f:
                self.history_data = json.load(f)
                self.table.setRowCount(len(self.history_data))
                # Show in reverse order (newest first)
                for i, e in enumerate(reversed(self.history_data)):
                    # Checkbox Column
                    ck = QTableWidgetItem(""); ck.setCheckState(Qt.Unchecked); self.table.setItem(i, 0, ck)
                    self.table.setItem(i, 1, QTableWidgetItem(e.get("timestamp","")))
                    self.table.setItem(i, 2, QTableWidgetItem(e.get("disease","")))
                    self.table.setItem(i, 3, QTableWidgetItem(f"{e.get('severity',0):.2f}%"))
                    self.table.setItem(i, 4, QTableWidgetItem(e.get("status","")))
        except: pass

    def delete_selected(self):
        if not os.path.exists(HISTORY_FILE): return
        to_keep = []
        # Since we displayed in reverse, we need to map back carefully
        rev_data = list(reversed(self.history_data))
        indices_to_remove = []
        for i in range(self.table.rowCount()):
            if self.table.item(i, 0).checkState() == Qt.Checked:
                indices_to_remove.append(i)
        
        if not indices_to_remove:
            QMessageBox.information(self, "No Selection", "Please check at least one record to delete.")
            return

        reply = QMessageBox.question(self, "Confirm Delete", f"Are you sure you want to delete {len(indices_to_remove)} record(s)?", QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.No: return

        # Re-build history data by excluding the reversed indices that were checked
        new_history = []
        for i, item in enumerate(rev_data):
            if i not in indices_to_remove:
                new_history.append(item)
        
        # Original was reversed of rev_data. So we just reverse new_history back to store
        with open(HISTORY_FILE, "w") as f:
            json.dump(list(reversed(new_history)), f, indent=4)
        
        self.load_data()
        QMessageBox.information(self, "Success", "Selected records purged from database.")

class ComparisonPage(QWidget):
    def __init__(self, parent):
        super().__init__(); self.p = parent; self.s1 = 0; self.s2 = 0; self.img1 = None; self.img2 = None
        l = QVBoxLayout(self); l.setContentsMargins(0,0,0,0); l.setSpacing(0)
        
        # System Header (Unified)
        h = QFrame(); h.setFixedHeight(70); h.setStyleSheet("background: #0b1220; border-bottom: 2px solid #2a3b5c;"); hl = QHBoxLayout(h)
        title = QLabel("LEAF HEALTH COMPARISON"); title.setAlignment(Qt.AlignCenter); title.setStyleSheet("font-size:26px; font-weight:bold; color:white;"); hl.addWidget(title, 1)
        row_btn = QHBoxLayout(); hl.addLayout(row_btn)
        for t, n, f in [("← BACK TO CROPS","SecondaryBtn",lambda: self.p.setCurrentIndex(4)), ("EXIT","DangerBtn",app.quit)]:
            b = QPushButton(t); b.setObjectName(n); b.clicked.connect(f); row_btn.addWidget(b)
        l.addWidget(h)

        body = QVBoxLayout(); body.setContentsMargins(20,20,20,20); body.setSpacing(12); l.addLayout(body, 1)
        
        # Comparison Viewport
        self.viewport = QFrame(); self.viewport.setStyleSheet("background:#0b1220; border:2px solid #00ff88; border-radius:20px;"); vl = QVBoxLayout(self.viewport)
        vl.addWidget(QLabel("DIAGNOSTIC COMPARISON: PRESENT VS PAST", alignment=Qt.AlignCenter, styleSheet="font-size: 24px; font-weight: bold; color: #00ff88; margin-top: 15px;"))
        
        row = QHBoxLayout(); row.setContentsMargins(20,10,20,10); row.setSpacing(20); vl.addLayout(row)
        for i in [1, 2]:
            v = QVBoxLayout(); row.addLayout(v)
            lbl = QLabel("SELECT IMAGE"); lbl.setMinimumSize(360, 260); lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("background:#0f1c34; border:2px solid #2a3b5c; border-radius:15px; font-size:18px; color:#95a4bf;")
            btn = QPushButton(f"UPLOAD {'PRESENT' if i==1 else 'PAST'} SCAN")
            btn.clicked.connect(partial(self.up, i)); btn.setMinimumHeight(50); btn.setStyleSheet("background:#00ff88; color:#000; font-weight:bold; border-radius:10px;")
            v.addWidget(lbl); v.addWidget(btn); setattr(self, f"lbl{i}", lbl)
        
        self.res = QLabel("READY FOR ANALYSIS"); self.res.setAlignment(Qt.AlignCenter); self.res.setWordWrap(True)
        self.res.setStyleSheet("font-size:24px; font-weight:bold; color:#00ff88; background:#0f1c34; border:1px solid #2a3b5c; border-radius:12px; margin:10px 20px; padding:20px;")
        vl.addWidget(self.res); body.addWidget(self.viewport)

    def up(self, i):
        p, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.jpg *.jpeg *.png *.webp *.bmp)")
        if not p: return
        img = cv2.imread(p)
        if img is None:
            QMessageBox.warning(self, "Invalid Image", "Could not read the selected file.")
            return

        # Check if it is a leaf
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        leaf_mask = cv2.inRange(hsv, np.array([20, 30, 30]), np.array([95, 255, 255]))
        leaf_ratio = (np.sum(leaf_mask > 0) / leaf_mask.size) * 100
        if leaf_ratio < 15.0: # LEAF_AREA_THRESHOLD
            QMessageBox.warning(self, "Not a Leaf", "The uploaded image does not appear to be a valid plant leaf. Comparison canceled.")
            return

        try:
            seg, sev = precise_segmentation(img)
        except Exception as exc:
            QMessageBox.warning(self, "Processing Error", str(exc))
            return

        # Contiguous BGR buffer required for QImage lifetime and layout
        seg = np.ascontiguousarray(seg)
        h, w = seg.shape[:2]
        bytes_per_line = w * 3
        q = QImage(seg.data, w, h, bytes_per_line, QImage.Format_BGR888).copy()

        target_lbl = getattr(self, f"lbl{i}")
        tw = max(1, target_lbl.width())
        th = max(1, target_lbl.height())
        if tw <= 1 or th <= 1:
            tw, th = 360, 260
        target_lbl.setPixmap(
            QPixmap.fromImage(q).scaled(tw, th, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

        setattr(self, f"s{i}", sev)
        setattr(self, f"img{i}", img)

        if self.img1 is not None and self.img2 is not None:
            diff = self.s1 - self.s2

            if diff > 1:
                color = "#ff3333"
                st = f"ALERT: INCREASED FROM {self.s2:.2f}% TO {self.s1:.2f}% (+{diff:.2f}%)"
            elif diff < -1:
                color = "#00ff88"
                st = f"GOOD: DECREASE LEVEL FROM {self.s2:.2f}% TO {self.s1:.2f}% ({diff:.2f}%)"
            else:
                color = "#ffcc00"
                st = f"INFO: STABLE AT {self.s1:.1f}%"

            self.res.setText(st)
            self.res.setStyleSheet(f"""
                font-size: 24px;
                font-weight: bold;
                color: {color};
                padding: 20px;
                background: #0f1c34;
                border-radius: 12px;
                margin: 10px 20px;
                border: 1px solid #2a3b5c;
            """)
class LandingPage(QWidget):
    def __init__(self, parent):
        super().__init__(); self.p = parent
        self.bg = QPixmap(LANDING_BG)
        # Force a fixed style that ignores the theme toggle
        self.setStyleSheet("background: transparent;")
        l = QVBoxLayout(self); l.setContentsMargins(0, 60, 0, 60); l.setSpacing(0)
        
        # Top Title Panel
        t_box = QFrame(); t_box.setStyleSheet("background: transparent; border: none; margin: 0 40px;")
        tl = QVBoxLayout(t_box)
        title = QLabel("PLANT HEALTH MONITORING SYSTEM USING LEAVES")
        title.setAlignment(Qt.AlignCenter); title.setStyleSheet("font-size: 44px; font-weight: 900; color: white; border: none; background: transparent;")
        tl.addWidget(title); l.addWidget(t_box, 0, Qt.AlignTop | Qt.AlignHCenter)
        
        l.addStretch()

        # Bottom Glass Panel for Buttons
        glass = QFrame(); glass.setMinimumHeight(120); glass.setFixedWidth(640)
        glass.setStyleSheet("background: rgba(255, 255, 255, 25); border: 1px solid rgba(255, 255, 255, 45); border-radius: 25px;")
        gl = QHBoxLayout(glass); gl.setContentsMargins(30, 20, 30, 20); gl.setSpacing(30)
        
        b1 = InteractiveButton("GET STARTED"); b1.setStyleSheet("background: #00ff88; color: #000; font-size: 20px; font-weight: bold; border-radius: 12px;")
        b1.clicked.connect(lambda: parent.setCurrentIndex(1))
        
        b2 = InteractiveButton("EXIT", "DangerBtn"); b2.setStyleSheet("background: #ff5252; color: #fff; font-size: 20px; font-weight: bold; border-radius: 12px;")
        b2.clicked.connect(app.quit)
        
        gl.addWidget(b1); gl.addWidget(b2); l.addWidget(glass, 0, Qt.AlignBottom | Qt.AlignHCenter)

    def paintEvent(self, event):
        painter = QPainter(self)
        if not self.bg.isNull():
            painter.drawPixmap(self.rect(), self.bg.scaled(self.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation))
        else:
            # Fallback gradient if image is missing
            grad = QRadialGradient(self.rect().center(), self.width())
            grad.setColorAt(0, QColor("#1b5e20")); grad.setColorAt(1, QColor("#0b1220"))
            painter.fillRect(self.rect(), grad)


class LoginPage(QWidget):
    def __init__(self, parent):
        super().__init__(); self.p = parent; self.bg = QPixmap(LOGIN_BG)
        self.setStyleSheet("background: transparent;")
        l = QVBoxLayout(self); l.setContentsMargins(0,0,0,0); l.setAlignment(Qt.AlignCenter)
        
        # Center Login Card (Glassmorphic)
        card = QFrame(); card.setMinimumSize(540, 680); card.setMaximumWidth(600)
        card.setStyleSheet("background: rgba(0, 40, 20, 180); border: 2px solid #00ff88; border-radius: 25px;")
        cl = QVBoxLayout(card); cl.setContentsMargins(40, 30, 40, 40); cl.setSpacing(15); l.addWidget(card, 0, Qt.AlignCenter)
        
        # Stylized Header Box
        h_box = QFrame(); h_box.setStyleSheet("background: rgba(0, 255, 136, 15); border: 2px solid #00ff88; border-radius: 12px; padding: 10px;")
        hl = QVBoxLayout(h_box); title = QLabel("USER LOGIN")
        title.setAlignment(Qt.AlignCenter); title.setStyleSheet("font-size: 32px; font-weight: bold; color: #00ff88; background: transparent; border: none;")
        hl.addWidget(title); cl.addWidget(h_box)

        sub = QLabel("Secure access to your plant diagnostics dashboard")
        sub.setWordWrap(True); sub.setStyleSheet("color: rgba(255, 255, 255, 180); font-size: 14px; background: transparent; border: none;"); cl.addWidget(sub)
        
        in_style = "background: rgba(0, 20, 10, 150); border: 1px solid rgba(0, 255, 136, 80); color: white; padding: 12px; border-radius: 10px; font-size: 16px;"

        self.u = QLineEdit(); self.u.setPlaceholderText("Username"); self.u.setStyleSheet(in_style); cl.addWidget(self.u)
        
        p_row = QHBoxLayout(); self.p = QLineEdit(); self.p.setPlaceholderText("Password"); self.p.setEchoMode(QLineEdit.Password); self.p.setStyleSheet(in_style); p_row.addWidget(self.p)
        self.toggle_btn = QPushButton("SHOW"); self.toggle_btn.setMinimumHeight(45); self.toggle_btn.setMaximumWidth(100)
        self.toggle_btn.setStyleSheet("background: rgba(0,0,0,100); color: white; border: 1px solid #00ff88; border-radius: 8px; font-weight: bold;")
        self.toggle_btn.clicked.connect(self.toggle_password); p_row.addWidget(self.toggle_btn); cl.addLayout(p_row)

        self.msg = QLabel("Enter your credentials to continue.")
        self.msg.setStyleSheet("color: #00ff88; font-size: 13px; background: transparent; border: none;"); cl.addWidget(self.msg)
        
        cl.addSpacing(10)
        b_sign = QPushButton("SIGN IN")
        b_sign.setStyleSheet("background: #00ff88; color: #000; font-weight: bold; font-size: 18px; border-radius: 12px; padding: 14px;")
        b_sign.clicked.connect(lambda: self.auth(parent)); cl.addWidget(b_sign)
        
        sec_style = "background: transparent; border: 1px solid rgba(0, 255, 136, 120); color: white; font-weight: bold; padding: 10px; border-radius: 10px;"
        for t, f in [("NEW REGISTRATION", lambda: parent.setCurrentIndex(2)), ("CONTINUE AS GUEST", lambda: parent.go_to_crops("Guest")), ("RECOVER PASSWORD", lambda: parent.setCurrentIndex(3))]:
            b = QPushButton(t); b.setStyleSheet(sec_style); b.clicked.connect(f); cl.addWidget(b)
        
        b_exit = QPushButton("EXIT")
        b_exit.setStyleSheet("background: #ff5252; color: #fff; font-weight: bold; font-size: 18px; border-radius: 12px; padding: 12px;")
        b_exit.clicked.connect(app.quit); cl.addWidget(b_exit)

    def paintEvent(self, event):
        painter = QPainter(self)
        if not self.bg.isNull():
            painter.drawPixmap(self.rect(), self.bg.scaled(self.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation))
        else:
            painter.fillRect(self.rect(), QColor("#0b1220"))

    def validate_inputs(self):
        if not self.u.text().strip():
            auth_message(self.msg, "Username is required.", "error")
            return False
        if len(self.p.text()) < 4:
            auth_message(self.msg, "Password looks too short.", "error")
            return False
        return True

    def toggle_password(self):
        showing = self.p.echoMode() == QLineEdit.Normal
        self.p.setEchoMode(QLineEdit.Password if showing else QLineEdit.Normal)
        self.toggle_btn.setText("SHOW" if showing else "HIDE")

    def auth(self, parent):
        if not self.validate_inputs(): return
        if db_mgr.validate_user(self.u.text().strip(), self.p.text()):
            auth_message(self.msg, "Signed in successfully.", "ok")
            parent.go_to_crops(self.u.text().strip())
        else:
            auth_message(self.msg, "Invalid username or password.", "error")

class SignupPage(QWidget):
    def __init__(self, parent):
        super().__init__()
        l = QVBoxLayout(self); l.setContentsMargins(20,20,20,20); l.setAlignment(Qt.AlignCenter)
        f = QFrame(); f.setObjectName("Card"); f.setMinimumSize(500, 620); f.setMaximumWidth(760); f.setStyleSheet("background: #0f1c34; border: 2px solid #00ff88; border-radius: 20px;"); fl = QVBoxLayout(f); l.addWidget(f); apply_card_shadow(f)
        fl.setSpacing(10)
        fl.addSpacing(8); fl.addWidget(QLabel("CREATE ACCOUNT", styleSheet="font-size: 36px; font-weight: bold; color: #00ff88;"), 0, Qt.AlignCenter)
        self.u = QLineEdit(); self.u.setPlaceholderText("Username"); fl.addWidget(self.u)
        self.e = QLineEdit(); self.e.setPlaceholderText("Email Address"); fl.addWidget(self.e)
        p_row = QHBoxLayout(); self.p = QLineEdit(); self.p.setPlaceholderText("Password"); self.p.setEchoMode(QLineEdit.Password); p_row.addWidget(self.p)
        self.p_toggle = InteractiveButton("SHOW", "SecondaryBtn"); self.p_toggle.setMinimumHeight(40); self.p_toggle.setMaximumWidth(110); self.p_toggle.clicked.connect(self.toggle_password); p_row.addWidget(self.p_toggle); fl.addLayout(p_row)
        self.cp = QLineEdit(); self.cp.setPlaceholderText("Confirm Password"); self.cp.setEchoMode(QLineEdit.Password); fl.addWidget(self.cp)
        self.str_lbl = QLabel("Password strength: Weak"); self.str_bar = QProgressBar(); self.str_bar.setTextVisible(False); self.str_bar.setRange(0, 100); self.str_bar.setValue(0); fl.addWidget(self.str_lbl); fl.addWidget(self.str_bar)
        self.msg = QLabel("Use at least 8 chars with mixed types."); auth_message(self.msg, "Use at least 8 chars with mixed types.", "normal"); fl.addWidget(self.msg)
        self.p.textChanged.connect(self.update_strength); self.cp.textChanged.connect(self.check_match)
        fl.addSpacing(8); b = InteractiveButton("REGISTER NOW"); b.setStyleSheet("background: #00ff88; color: #000; font-weight: bold;"); b.clicked.connect(self.reg); fl.addWidget(b)
        back = InteractiveButton("CANCEL", "SecondaryBtn"); back.clicked.connect(lambda: parent.setCurrentIndex(1)); fl.addWidget(back)

    def toggle_password(self):
        showing = self.p.echoMode() == QLineEdit.Normal
        mode = QLineEdit.Password if showing else QLineEdit.Normal
        self.p.setEchoMode(mode)
        self.cp.setEchoMode(mode)
        self.p_toggle.setText("SHOW" if showing else "HIDE")

    def update_strength(self):
        score, level = password_strength(self.p.text())
        self.str_bar.setValue(score)
        auth_message(self.str_lbl, f"Password strength: {level}", "ok" if score >= 75 else "normal")

    def check_match(self):
        if self.cp.text() and self.cp.text() != self.p.text():
            auth_message(self.msg, "Passwords do not match.", "error")
            return False
        auth_message(self.msg, "Passwords match.", "ok")
        return True

    def reg(self):
        u = self.u.text().strip(); e = self.e.text().strip(); p = self.p.text()
        if len(u) < 3:
            auth_message(self.msg, "Username must be at least 3 characters.", "error")
            return
        if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", e):
            auth_message(self.msg, "Enter a valid email address.", "error")
            return
        score, _ = password_strength(p)
        if score < 50:
            auth_message(self.msg, "Password is too weak.", "error")
            return
        if not self.check_match():
            return
        if db_mgr.register_user(u, e, p):
            QMessageBox.information(self, "Success", "Account Created!")
            self.parent().setCurrentIndex(1)
        else:
            auth_message(self.msg, "Username already exists. Try another one.", "error")

class RecoveryPage(QWidget):
    def __init__(self, parent):
        super().__init__()
        l = QVBoxLayout(self); l.setContentsMargins(20,20,20,20); l.setAlignment(Qt.AlignCenter)
        f = QFrame(); f.setObjectName("Card"); f.setMinimumSize(500, 600); f.setMaximumWidth(760); f.setStyleSheet("background: #0f1c34; border: 2px solid #00ff88; border-radius: 20px;"); fl = QVBoxLayout(f); l.addWidget(f); apply_card_shadow(f)
        fl.addSpacing(12); fl.addWidget(QLabel("RESET PASSWORD", styleSheet="font-size: 36px; font-weight: bold; color: #00ff88;"), 0, Qt.AlignCenter)
        self.u = QLineEdit(); self.u.setPlaceholderText("Username"); fl.addWidget(self.u)
        self.e = QLineEdit(); self.e.setPlaceholderText("Recovery Email"); fl.addWidget(self.e)
        p_row = QHBoxLayout(); self.p = QLineEdit(); self.p.setPlaceholderText("New Password"); self.p.setEchoMode(QLineEdit.Password); p_row.addWidget(self.p)
        self.p_toggle = InteractiveButton("SHOW", "SecondaryBtn"); self.p_toggle.setMinimumHeight(40); self.p_toggle.setMaximumWidth(110); self.p_toggle.clicked.connect(self.toggle_password); p_row.addWidget(self.p_toggle); fl.addLayout(p_row)
        self.msg = QLabel("Enter your account details."); auth_message(self.msg, "Enter your account details.", "normal"); fl.addWidget(self.msg)
        fl.addSpacing(12); b = InteractiveButton("UPDATE PASSWORD"); b.setStyleSheet("background: #00ff88; color: #000; font-weight: bold;"); b.clicked.connect(self.rst); fl.addWidget(b)
        back = InteractiveButton("BACK", "SecondaryBtn"); back.clicked.connect(lambda: parent.setCurrentIndex(1)); fl.addWidget(back)

    def toggle_password(self):
        showing = self.p.echoMode() == QLineEdit.Normal
        self.p.setEchoMode(QLineEdit.Password if showing else QLineEdit.Normal)
        self.p_toggle.setText("SHOW" if showing else "HIDE")

    def rst(self):
        u = self.u.text().strip(); e = self.e.text().strip(); p = self.p.text()
        if not u or not e or not p:
            auth_message(self.msg, "All fields are required.", "error")
            return
        score, _ = password_strength(p)
        if score < 50:
            auth_message(self.msg, "Choose a stronger password.", "error")
            return
        if db_mgr.reset_password(u, e, p):
            QMessageBox.information(self, "Success", "Updated!")
            self.parent().setCurrentIndex(1)
        else:
            auth_message(self.msg, "Username/email not found.", "error")

class CropSelectionPage(QWidget):
    def __init__(self, parent):
        super().__init__(); self.p = parent; l = QVBoxLayout(self); l.setContentsMargins(0,0,0,0); l.setSpacing(0)
        
        # Header (Unified with Dashboard)
        h = QFrame(); h.setFixedHeight(70); h.setStyleSheet("background: #0b1220; border-bottom: 2px solid #2a3b5c;"); hl = QHBoxLayout(h)
        title = QLabel("PLANT HEALTH MONITORING SYSTEM USING LEAVES"); title.setAlignment(Qt.AlignCenter); title.setStyleSheet("font-size:26px; font-weight:bold; color:white;"); hl.addWidget(title, 1)
        row_btn = QHBoxLayout(); hl.addLayout(row_btn)
        for t, n, f in [("LOGOUT","SecondaryBtn",lambda: self.p.setCurrentIndex(1)), ("EXIT","DangerBtn",app.quit)]:
            b = QPushButton(t); b.setObjectName(n); b.clicked.connect(f); row_btn.addWidget(b)
        l.addWidget(h)

        body = QVBoxLayout(); body.setContentsMargins(20,20,20,20); body.setSpacing(12); l.addLayout(body, 1)
        
        # Mission Control Viewport
        self.viewport = QFrame(); self.viewport.setStyleSheet("background:#0b1220; border:2px solid #00ff88; border-radius:20px;"); vl = QVBoxLayout(self.viewport)
        msg_lbl = QLabel("MISSION PREPARATION: SELECT TARGET CROP"); msg_lbl.setAlignment(Qt.AlignCenter); msg_lbl.setStyleSheet("font-size: 28px; font-weight: bold; color: #00ff88; margin-top: 20px;"); vl.addWidget(msg_lbl)
        
        grid_frame = QFrame(); grid_l = QGridLayout(grid_frame); grid_l.setSpacing(25); grid_l.setContentsMargins(50,30,50,50)
        crops = ["Corn", "Chilli", "Pumpkin", "Okra"]
        for i, c in enumerate(crops):
            btn = SidebarButton(c, c.capitalize()); btn.setMinimumHeight(140); btn.setText(c.upper()); btn.clicked.connect(partial(self.p.go_to_analysis, c.capitalize())); grid_l.addWidget(btn, i//2, i%2)
        vl.addWidget(grid_frame, 1); body.addWidget(self.viewport)

        # Footer Command Bar
        f = QHBoxLayout(); f.setContentsMargins(20,0,20,20); f.setSpacing(20); l.addLayout(f)
        b_cmp = QPushButton("COMPARE LEAVES ANALYSIS"); b_ana = QPushButton("DIAGNOSTIC RECORDS")
        for b, func in [(b_cmp, lambda: self.p.setCurrentIndex(6)), (b_ana, lambda: HistoryWindow(self).exec_())]:
            b.setMinimumHeight(65); b.clicked.connect(func); b.setStyleSheet("background:#00ff88; color:#000; font-weight:bold; border-radius:12px; font-size:17px;"); f.addWidget(b, 1)

class SidebarButton(QPushButton):
    def __init__(self, text, crop_name, parent=None):
        super().__init__(text, parent)
        self.crop_name = crop_name
        self.setCheckable(True)
        self.setMinimumHeight(100)
        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet("""
            QPushButton {
                background: #0f1c34;
                color: #95a4bf;
                border: 2px solid #2a3b5c;
                border-radius: 12px;
                font-size: 18px;
                font-weight: bold;
                margin: 4px;
            }
            QPushButton:hover { background: #1a2a47; border-color: #00ff88; color: #ffffff; }
            QPushButton:checked {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #006400, stop:1 #00a86b);
                color: #ffffff;
                border: 2px solid #00ff88;
            }
        """)

class CircularGauge(QWidget):
    def __init__(self, title="", parent=None):
        super().__init__(parent)
        self.value = 0; self.title = title
        self.setMinimumSize(140, 140)
    def setValue(self, val):
        self.value = val; self.update()
    def paintEvent(self, event):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing); w = self.width(); h = self.height(); s = min(w, h)
        rect = QRectF(10, 10, s-20, s-20)
        
        # Track
        p.setPen(QPen(QColor(0, 255, 136, 30), 10)); p.drawEllipse(rect)
        
        # Outer Glow
        p.setPen(QPen(QColor(0, 255, 136, 15), 14)); p.drawArc(rect, 90*16, -int(self.value*3.6*16))
        
        # Main Arc
        p.setPen(QPen(QColor("#00ff88"), 10, Qt.SolidLine, Qt.RoundCap)); p.drawArc(rect, 90*16, -int(self.value*3.6*16))
        
        # Text
        p.setPen(QColor("#ffffff")); p.setFont(QFont("Segoe UI", 18, QFont.Bold)); p.drawText(rect.adjusted(0, -10, 0, -10), Qt.AlignCenter, f"{int(self.value)}%")
        p.setPen(QColor("#00ff88")); p.setFont(QFont("Segoe UI", 8, QFont.Bold)); p.drawText(rect.adjusted(0, 25, 0, 25), Qt.AlignCenter, self.title)
class AnalysisPage(QWidget):
    def __init__(self, parent):
        super().__init__(); self.p = parent; self.crop = ""; self.cur_res = None; self.cap = None; self.timer = QTimer(); self.timer.timeout.connect(self.update_frame)
        l = QVBoxLayout(self); l.setContentsMargins(0,0,0,0); l.setSpacing(0)
        
        # Header
        h = QFrame(); h.setFixedHeight(70); h.setStyleSheet("background: #0b1220; border-bottom: 2px solid #2a3b5c;"); hl = QHBoxLayout(h)
        title = QLabel("PLANT HEALTH MONITORING SYSTEM USING LEAVES"); title.setAlignment(Qt.AlignCenter); title.setStyleSheet("font-size:26px; font-weight:bold; color:white;"); hl.addWidget(title, 1)
        row_btn = QHBoxLayout(); hl.addLayout(row_btn)
        for t, n, f in [("BACK TO CROPS","SecondaryBtn",lambda: self.stop_all(4)), ("EXIT","DangerBtn",app.quit)]:
            b = QPushButton(t); b.setObjectName(n); b.clicked.connect(f); row_btn.addWidget(b)
        l.addWidget(h)

        body = QHBoxLayout(); body.setContentsMargins(12,12,12,12); body.setSpacing(12); l.addLayout(body, 1)
        
        # Sidebar
        self.sidebar = QFrame(); self.sidebar.setFixedWidth(240); self.sidebar.setStyleSheet("background:#0f1c34; border-radius:15px; border:1px solid #2a3b5c;"); sl = QVBoxLayout(self.sidebar)
        sl.addWidget(QLabel("CROP SELECTION", styleSheet="color:#00ff88; font-weight:bold; font-size:12px; margin-bottom:10px;"))
        self.crop_btns = []
        for c in ["CORN", "CHILLI", "PUMPKIN", "OKRA"]:
            b = SidebarButton(c, c.capitalize()); b.clicked.connect(partial(self.set_crop, b)); sl.addWidget(b); self.crop_btns.append(b)
        sl.addStretch(); body.addWidget(self.sidebar)

        # Center Viewport
        self.viewport = QFrame(); self.viewport.setStyleSheet("background:#000000; border:2px solid #00ff88; border-radius:15px;"); vl = QVBoxLayout(self.viewport)
        
        # Status Header (The 'SCAN ABORTED: INTERFERENCE' style bar)
        self.status_bar = QLabel("SYSTEM READY: STANDBY"); self.status_bar.setFixedHeight(35); self.status_bar.setAlignment(Qt.AlignCenter)
        self.status_bar.setStyleSheet("background: rgba(0, 255, 136, 0.1); border: 1px solid #00ff88; color: #00ff88; font-weight: bold; font-size: 14px; border-radius: 5px; margin: 5px;")
        vl.addWidget(self.status_bar)

        self.img_lbl = QLabel("READY TO SCAN"); self.img_lbl.setAlignment(Qt.AlignCenter); vl.addWidget(self.img_lbl)
        
        # Grid Results (Hidden initially)
        self.result_grid = DiagnosticGrid()
        self.result_grid.setVisible(False)
        vl.addWidget(self.result_grid, 1)

        self.btn_snap = QPushButton("CAPTURE & ANALYZE", objectName="SnapBtn"); self.btn_snap.setVisible(False); self.btn_snap.clicked.connect(self.snap); vl.addWidget(self.btn_snap)
        body.addWidget(self.viewport, 2)

        # Right Hub
        self.hub = QFrame(); self.hub.setFixedWidth(340); self.hub.setStyleSheet("background:#0f1c34; border-radius:15px; border:1px solid #2a3b5c;"); rl = QVBoxLayout(self.hub)
        rl.addWidget(QLabel("DIAGNOSIS HUB", styleSheet="color:#00ff88; font-weight:bold; font-size:12px;"))
        self.res_lbl = QLabel("SELECT CROP & UPLOAD"); self.res_lbl.setWordWrap(True); self.res_lbl.setStyleSheet("font-size:22px; font-weight:bold; color:white; margin:10px 0;"); rl.addWidget(self.res_lbl)
        rl.addWidget(QLabel("INFECTION SEVERITY", alignment=Qt.AlignCenter, styleSheet="color:#00ff88; font-size:11px;"))
        self.gauge = CircularGauge(); rl.addWidget(self.gauge, 0, Qt.AlignCenter)
        rl.addWidget(QLabel("NATURAL & ORGANIC SOLUTIONS", styleSheet="color:#00ff88; font-weight:bold; font-size:12px; margin-top:15px;"))
        self.txt = QTextEdit(); self.txt.setReadOnly(True); rl.addWidget(self.txt); body.addWidget(self.hub)

        # Footer
        f = QHBoxLayout(); f.setContentsMargins(12,0,12,12); f.setSpacing(12); l.addLayout(f)
        self.b_cam = QPushButton("LIVE CAMERA"); self.b_up = QPushButton("FILE UPLOAD"); self.b_rep = QPushButton("DOWNLOAD REPORT"); self.b_ana = QPushButton("ANALYSIS")
        for b, func in [(self.b_cam, self.start_cam), (self.b_up, self.up), (self.b_rep, self.report), (self.b_ana, self.show_analysis)]:
            b.setMinimumHeight(60); b.clicked.connect(func); b.setStyleSheet("background:#00ff88; color:#000; font-weight:bold; border-radius:12px; font-size:16px;"); f.addWidget(b);
        self.b_rep.setEnabled(False)

    def set_crop(self, btn):
        for b in self.crop_btns: b.setChecked(False)
        btn.setChecked(True); self.crop = btn.crop_name
        self.clear_ui()

    def clear_ui(self):
        self.img_lbl.clear(); self.img_lbl.setText("READY TO SCAN"); self.img_lbl.setVisible(True)
        self.result_grid.setVisible(False); self.status_bar.setText("SYSTEM READY: STANDBY")
        self.status_bar.setStyleSheet("background: rgba(0, 255, 136, 0.1); border: 1px solid #00ff88; color: #00ff88; font-weight: bold; font-size: 14px; border-radius: 5px; margin: 5px;")
        self.res_lbl.setText("DIAGNOSIS..."); self.gauge.setValue(0); self.txt.clear(); self.b_rep.setEnabled(False)
    def stop_all(self, idx):
        self.timer.stop(); self.clear_ui(); 
        if self.cap: self.cap.release(); self.cap = None
        self.btn_snap.setVisible(False); self.p.setCurrentIndex(idx)
    def start_cam(self):
        if not self.crop: QMessageBox.warning(self, "Crop Not Selected", "Please select a crop from the sidebar first."); return
        if self.cap: self.cap.release(); self.cap = None
        self.clear_ui()
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if self.cap.isOpened(): self.img_lbl.clear(); self.btn_snap.setVisible(True); self.timer.start(30)
        else: QMessageBox.critical(self, "Error", "Camera failed.")
    def update_frame(self):
        ret, f = self.cap.read()
        if ret:
            q = QImage(cv2.cvtColor(f, cv2.COLOR_BGR2RGB).data, f.shape[1], f.shape[0], f.shape[1]*3, QImage.Format_RGB888)
            self.img_lbl.setPixmap(QPixmap.fromImage(q).scaled(self.img_lbl.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
    def snap(self):
        self.timer.stop(); ret, f = self.cap.read(); self.cap.release(); self.cap = None; self.btn_snap.setVisible(False)
        if ret: self.process(f)
    def up(self):
        if not self.crop: QMessageBox.warning(self, "Crop Not Selected", "Please select a crop from the sidebar first."); return
        self.clear_ui()
        p, _ = QFileDialog.getOpenFileName(self, "Open Leaf", "", "Images (*.jpg *.jpeg *.png *.webp *.bmp)")
        if p:
            img = cv2.imread(p)
            if img is None: QMessageBox.warning(self, "Error", "Invalid image file."); return
            self.process(img)
    def process(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        leaf_mask = cv2.inRange(hsv, np.array([10, 20, 20]), np.array([105, 255, 255]))
        leaf_ratio = (np.sum(leaf_mask > 0) / leaf_mask.size) * 100
        if leaf_ratio < 5.0:
            QMessageBox.warning(self, "Invalid Scan", "This image does not contain a recognizable leaf. Live camera and file upload only allow leaf images. Human images are not permitted.")
            self.clear_ui()
            return
            
        res = process_image(img, self.crop); self.cur_res = res
        
        # Switch View: Hide Single Image, Show Grid
        self.img_lbl.setVisible(False)
        self.result_grid.setVisible(True)
        self.status_bar.setText(f"DIAGNOSTIC COMPLETE: {res['disease'].upper()}")
        
        if "INVALID" in res['disease'] or "LOW CONFIDENCE" in res['disease']:
            self.status_bar.setStyleSheet("background: rgba(255, 51, 51, 0.2); border: 1px solid #ff3333; color: #ff3333; font-weight: bold; font-size: 14px; border-radius: 5px; margin: 5px;")
            if "INVALID" in res['disease']: self.status_bar.setText("SCAN ABORTED: INTERFERENCE DETECTED")
        else:
            self.status_bar.setStyleSheet("background: rgba(0, 255, 136, 0.1); border: 1px solid #00ff88; color: #00ff88; font-weight: bold; font-size: 14px; border-radius: 5px; margin: 5px;")

        self.result_grid.update_results(res['original'], res['segmented'], res['heatmap'], res['confidence'], res['severity'])
        self.res_lbl.setText(f"{self.crop.upper()}: {res['disease'].upper()}")
        self.gauge.setValue(res['severity'])
        
        # Color highlighting: Red for disease, Green for healthy, Orange for warnings
        if "INVALID" in res['disease'] or "LOW CONFIDENCE" in res['disease']:
            color = "#FFB300" # Orange for warnings
        else:
            color = "#FF5252" if "healthy" not in res['disease'].lower() else "#00FF88"
            
        self.res_lbl.setStyleSheet(f"font-size: 22px; font-weight: bold; color: {color}; margin: 10px 0;")
        t = res['treatment_data']
        
        if "INVALID" in res['disease']:
            html = "<div style='font-size:16px; color:#FFB300;'><b>SYSTEM WARNING:</b><br>The captured image does not appear to be a plant leaf.<br><br><b>Action Required:</b><br>Please point the camera clearly at a leaf and ensure good lighting.</div>"
        elif "LOW CONFIDENCE" in res['disease']:
            html = "<div style='font-size:16px; color:#FFB300;'><b>SYSTEM WARNING:</b><br>Diagnosis confidence is too low to provide a safe recommendation.<br><br><b>Action Required:</b><br>Try capturing from a different angle or distance.</div>"
        else:
            html = f"""<div style='font-size:15px; color:white;'>
                <b style='color:#00ff88;'>SYMPTOMS:</b><br>{t.get('Symptoms','N/A')}<br><br>
                <b style='color:#00ff88;'>NATURAL & ORGANIC SOLUTIONS:</b><br>{t.get('Management_Cultural','N/A')}<br><br>
                <b style='color:#00ff88;'>EXPERT MEDICINE GUIDE:</b><br>{t.get('Medicine','Contact local expert')}<br><br>
                <b style='color:#00ff88;'>NUTRIENT & FERTILIZER GUIDE:</b><br>{t.get('Fertilizer','Standard organic NPK')}<br><br>
                <b style='color:#00ff88;'>CHEMICAL CONTROL:</b><br>{t.get('Management_Chemical','Consult agronomist')}
            </div>"""
            
        self.txt.setHtml(html); self.b_rep.setEnabled(True if "INVALID" not in res['disease'] else False)
        s = "HEALTHY" if "healthy" in res['disease'].lower() else "DISEASED"
        log = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "disease": res['disease'], "severity": res['severity'], "status": s}; h = []
        if os.path.exists(HISTORY_FILE):
            try:
                with open(HISTORY_FILE, "r") as fl: h = json.load(fl)
            except: pass
        h.append(log)
        with open(HISTORY_FILE, "w") as fl: json.dump(h, fl, indent=4)
    def show_analysis(self):
        if not self.cur_res: QMessageBox.warning(self, "No Data", "Please scan a leaf first."); return
        f = self.cur_res['features']; txt = "<b>SCIENTIFIC FEATURE ANALYSIS</b><br><br>"
        for k, v in f.items(): txt += f"<b>{k}:</b> {v if isinstance(v, str) else f'{float(v):.4f}'}<br>"
        QMessageBox.information(self, "Analysis Details", txt)
    def report(self):
        
        if not self.cur_res:
            QMessageBox.warning(self, "Error", "No analysis data available.")
            return

        d = self.cur_res

    # Timestamp for unique filename
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        disease_name = d['disease'].replace(" ", "_")
        default_name = f"Report_{disease_name}_{timestamp}.pdf"

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Report",
            default_name,
            "PDF (*.pdf)"
         )

        if not path:
            return

        printer = QPrinter(QPrinter.HighResolution)
        printer.setOutputFormat(QPrinter.PdfFormat)
        printer.setOutputFileName(path)

        doc = QTextDocument()

    # Extract treatment safely
        t = d.get('treatment_data', {})

        bg_uri = ""
        if os.path.exists(REPORT_BG_IMAGE):
            bg_uri = "file:///" + REPORT_BG_IMAGE.replace("\\", "/")

        bg_style = ""
        if bg_uri:
            bg_style = f"background-image: url('{bg_uri}'); background-size: cover; background-repeat: no-repeat; background-position: center;"
        html = f"""
        <div style="font-family: 'Segoe UI', Arial; background-color: #000000; color: #ffffff; padding: 50px;">
            <div style="border: 2px solid #00ff88; padding: 30px; border-radius: 15px;">
                <h1 style="text-align: center; color: #00ff88; font-size: 38pt; margin-bottom: 5px;">
                    PLANT HEALTH DIAGNOSTIC REPORT
                </h1>
                <p style="text-align: center; color: #00ff88; font-size: 14pt; margin-top: 0; letter-spacing: 2px;">
                    // AI DIAGNOSTIC SECURED //
                </p>
                
                <div style="text-align: right; color: #a0eaba; font-size: 10pt; margin-bottom: 30px;">
                    TIMESTAMP: {now.strftime("%Y-%m-%d %H:%M:%S")}<br>
                    SYSTEM ID: AGPROTECH-V4
                </div>

                <hr style="border: 0; border-top: 1px solid rgba(0, 255, 136, 0.3); margin-bottom: 30px;">

                <table width="100%" cellpadding="15">
                    <tr>
                        <td width="60%" style="background: rgba(0, 255, 136, 0.05); border: 1px solid rgba(0, 255, 136, 0.2); border-radius: 10px;">
                            <h2 style="color: #00ff88; margin-top: 0;">PRIMARY DIAGNOSIS</h2>
                            <p style="font-size: 24pt; font-weight: bold; margin: 10px 0;">{d['disease'].upper()}</p>
                            <p style="color: #a0eaba;">CROP TYPE: {self.crop.upper()}</p>
                        </td>
                        <td width="40%" style="background: rgba(0, 255, 136, 0.05); border: 1px solid rgba(0, 255, 136, 0.2); border-radius: 10px; text-align: center;">
                            <h2 style="color: #00ff88; margin-top: 0;">ANALYTICS</h2>
                            <div style="font-size: 22pt; font-weight: bold; color: #ffffff;">{d['confidence']:.1f}%</div>
                            <div style="color: #00ff88; font-size: 9pt;">CONFIDENCE LEVEL</div>
                            <div style="margin-top: 15px; font-size: 22pt; font-weight: bold; color: #ffffff;">{d['severity']:.1f}%</div>
                            <div style="color: #00ff88; font-size: 9pt;">INFECTION SEVERITY</div>
                        </td>
                    </tr>
                </table>

                <div style="margin-top: 30px; padding: 20px; background: rgba(0, 255, 136, 0.05); border: 1px solid rgba(0, 255, 136, 0.2); border-radius: 10px;">
                    <h2 style="color: #00ff88; margin-top: 0;">SCIENTIFIC RECOMMENDATIONS</h2>
                    <p style="color: #a0eaba;"><b>CAUSAL ORGANISM:</b><br>{t.get('Causal_Organism','None')}</p>
                    <p style="color: #ffffff;"><b>SYMPTOMS:</b><br>{t.get('Symptoms','N/A')}</p>
                    <p style="color: #00ff88; font-weight: bold;"><b>NATURAL & ORGANIC MANAGEMENT:</b><br>{t.get('Management_Cultural','N/A')}</p>
                    <p style="color: #00ff88;"><b>EXPERT MEDICINE GUIDE:</b><br>{t.get('Medicine','Contact local expert')}</p>
                    <p style="color: #00ff88;"><b>NUTRIENT & FERTILIZER GUIDE:</b><br>{t.get('Fertilizer','Standard organic NPK')}</p>
                    <p style="color: #ff3333; font-weight: bold;"><b>EXPERT CHEMICAL CONTROL:</b><br>{t.get('Management_Chemical','N/A')}</p>
                </div>

                <div style="margin-top: 40px; text-align: center; color: rgba(0, 255, 136, 0.4); font-size: 9pt;">
                    THIS IS AN AI-GENERATED DIAGNOSTIC DOCUMENT. FOR CRITICAL CROPS, VERIFY WITH AN AGRONOMY EXPERT.
                </div>
            </div>
        </div>

        """
        doc.setHtml(html)
        doc.print_(printer)

        archived_path = os.path.join(REPORTS_DIR, default_name)
        if os.path.abspath(path) != os.path.abspath(archived_path):
            archived_printer = QPrinter(QPrinter.HighResolution)
            archived_printer.setOutputFormat(QPrinter.PdfFormat)
            archived_printer.setOutputFileName(archived_path)
            doc.print_(archived_printer)

        QMessageBox.information(self, "Success", f"Report saved.\nPrimary: {path}\nArchive: {archived_path}")
class MainApp(QStackedWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Plant Health Monitoring System Using Leaves")
        self.setMinimumSize(1200, 780)
        self.theme_name = "midnight"
        self.setStyleSheet(build_stylesheet(self.theme_name))

        self.pg_lnd = LandingPage(self)
        self.pg_log = LoginPage(self)
        self.pg_reg = SignupPage(self)
        self.pg_rec = RecoveryPage(self)
        self.pg_cho = CropSelectionPage(self)
        self.pg_ana = AnalysisPage(self)
        self.pg_cmp = ComparisonPage(self)

        for p in [self.pg_lnd, self.pg_log, self.pg_reg, self.pg_rec, self.pg_cho, self.pg_ana, self.pg_cmp]:
            self.addWidget(p)

        # Start at Landing Page
        self.setCurrentIndex(0)

    def resizeEvent(self, event):
        super().resizeEvent(event)

    def toggle_theme(self):
        self.theme_name = "slate" if self.theme_name == "midnight" else "midnight"
        self.setStyleSheet(build_stylesheet(self.theme_name))

    def go_to_crops(self, u):
        self.pg_ana.set_crop(self.pg_ana.crop_btns[0])
        self.setCurrentIndex(5)
    def go_to_analysis(self, c):
        self.pg_ana.clear_ui()
        for b in self.pg_ana.crop_btns:
            if b.crop_name.lower() == c.lower():
                self.pg_ana.set_crop(b); break
        self.setCurrentIndex(5)

if __name__ == "__main__":
    app = QApplication(sys.argv); window = MainApp(); window.showMaximized(); sys.exit(app.exec_())
