import os
from flask import Flask, render_template, request, redirect, url_for
import cv2
import torch
import numpy as np
import pickle
from skimage.feature import hog
import pathlib
import platform

# --- Flask Uygulamasını ve Modelleri Başlatma ---
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Modelleri sadece bir kere, uygulama başlarken yükle
print("Modeller ve gerekli fonksiyonlar hazırlanıyor...")

# Windows için PosixPath yaması
if platform.system() == 'Windows':
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath

try:
    yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=False, verbose=False)
    print("✅ YOLOv5 modeli yüklendi.")
    with open("ocr_model_rfc.pkl", "rb") as f:
        rfc_model = pickle.load(f)
    print("✅ Random Forest OCR modeli yüklendi.")
except Exception as e:
    print(f"HATA: Modeller yüklenemedi. Gerekli dosyaların olduğundan emin olun.\n{e}")
finally:
    if platform.system() == 'Windows':
        pathlib.PosixPath = temp

# --- Yardımcı Fonksiyonlar (Daha önceki kodlarımızdan) ---
def extract_hog_features(char_img):
    if len(char_img.shape) > 2 and char_img.shape[2] == 3:
        gray_img = cv2.cvtColor(char_img, cv2.COLOR_BGR2GRAY)
    else: gray_img = char_img
    resized_img = cv2.resize(gray_img, (32, 32))
    hog_features = hog(resized_img, orientations=9, pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2), transform_sqrt=True, block_norm='L2-Hys')
    return hog_features.reshape(1, -1)

def segment_characters(plate_image):
    H, W = plate_image.shape[:2]
    plate_gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
    th_img = cv2.adaptiveThreshold(plate_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    kernel = np.ones((3, 3), np.uint8)
    th_img = cv2.morphologyEx(th_img, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, hierarchy = cv2.findContours(th_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    possible_chars_with_coords = []
    for i, contour in enumerate(contours):
        if hierarchy[0][i][3] == -1:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = h / w
            if aspect_ratio < 1.5 or aspect_ratio > 4.0: continue
            if h < H * 0.4 or h > H * 0.9: continue
            char_image = plate_image[y:y+h, x:x+w]
            possible_chars_with_coords.append(((x, y, w, h), char_image))
    possible_chars_with_coords.sort(key=lambda item: item[0][0])
    return [item[1] for item in possible_chars_with_coords]

# --- Flask Rotaları (Sayfalar) ---
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Plaka tanıma işlemini yap
            img = cv2.imread(filepath)
            results_data = {'original_image': filepath}

            detections = yolo_model(img).pandas().xyxy[0]
            if not detections.empty:
                best_detection = detections.sort_values('confidence', ascending=False).iloc[0]
                xmin, ymin, xmax, ymax = map(int, [best_detection['xmin'], best_detection['ymin'], best_detection['xmax'], best_detection['ymax']])
                plate_crop = img[ymin:ymax, xmin:xmax]
                
                # Kesilen plakayı da kaydet
                plate_filename = 'plate_' + filename
                plate_filepath = os.path.join(app.config['UPLOAD_FOLDER'], plate_filename)
                cv2.imwrite(plate_filepath, plate_crop)
                results_data['plate_image'] = plate_filepath

                character_images = segment_characters(plate_crop)
                if character_images:
                    plate_text = ""
                    for char_img in character_images:
                        hog_features = extract_hog_features(char_img)
                        prediction = rfc_model.predict(hog_features)[0]
                        plate_text += str(prediction)
                    results_data['plate_text'] = plate_text
                else:
                    results_data['plate_text'] = "Karakter Bulunamadı"
            else:
                results_data['plate_text'] = "Plaka Bulunamadı"

            return render_template('index.html', results=results_data)

    return render_template('index.html', results=None)

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    # HF ortamında port Dockerfile ile yönetilir, yerelde test için debug=True kalabilir.
    app.run(host='0.0.0.0', port=7860)