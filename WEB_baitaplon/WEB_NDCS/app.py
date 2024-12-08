from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import base64
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.svm import LinearSVC
from keras.datasets import mnist
from sklearn.metrics import accuracy_score

# Tải dữ liệu MNIST
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Trích xuất đặc trưng HOG từ dữ liệu huấn luyện
X_train_feature = []
for i in range(len(X_train)):
    feature = hog(X_train[i], orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), block_norm="L2")
    X_train_feature.append(feature)
X_train_feature = np.array(X_train_feature, dtype=np.float32)

# Trích xuất đặc trưng HOG từ dữ liệu kiểm tra
X_test_feature = []
for i in range(len(X_test)):
    feature = hog(X_test[i], orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), block_norm="L2")
    X_test_feature.append(feature)
X_test_feature = np.array(X_test_feature, dtype=np.float32)

# Huấn luyện mô hình LinearSVC
model = LinearSVC(C=10, max_iter=5000)
model.fit(X_train_feature, y_train)
y_pre = model.predict(X_test_feature)
print("Test Accuracy:", accuracy_score(y_test, y_pre))

# Tạo ứng dụng Flask
app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/recognize', methods=['POST'])
def recognize():
    if 'file' not in request.files:
        return jsonify({'status': False, 'message': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'status': False, 'message': 'No file selected'})
    
    # Đọc và kiểm tra ảnh
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return jsonify({'status': False, 'message': 'Cannot decode image.'})
    
    # Lưu ảnh gốc để kiểm tra
    cv2.imwrite('uploaded_image.jpg', image)

    # Tiền xử lý
    im_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    im_blur = cv2.GaussianBlur(im_gray, (5, 5), 0)
    _, thre = cv2.threshold(im_blur, 90, 255, cv2.THRESH_BINARY_INV)
    cv2.imwrite('threshold_image.jpg', thre)  # Lưu ảnh threshold để kiểm tra

    # Phát hiện Contour
    contours, _ = cv2.findContours(thre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Contours found: {len(contours)}")  # Debug: Số lượng contour
    
    if len(contours) == 0:
        return jsonify({'status': False, 'message': 'No digits found in the image.'})

    # Sắp xếp các contour và trích xuất chữ số
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
    prediction_result = []
    for idx, contour in enumerate(contours[:3]):  # Xử lý tối đa 3 chữ số
        (x, y, w, h) = cv2.boundingRect(contour)
        roi = thre[y:y + h, x:x + w]
        roi = np.pad(roi, (20, 20), 'constant', constant_values=(0, 0))
        roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
        roi = cv2.dilate(roi, (3, 3))
        cv2.imwrite(f'roi_{idx}.jpg', roi)  # Debug: Lưu ROI

        # Trích xuất đặc trưng HOG
        roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), block_norm="L2")
        try:
            nbr = model.predict(np.array([roi_hog_fd], np.float32))
            prediction_result.append(int(nbr[0]))
        except Exception as e:
            print(f"Prediction error: {e}")
            return jsonify({'status': False, 'message': 'Error during prediction.'})

        # Vẽ kết quả lên ảnh gốc
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(image, str(int(nbr[0])), (x, y), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)

    # Mã hóa ảnh kết quả thành Base64
    _, img_encoded = cv2.imencode('.jpg', image)
    img_base64 = base64.b64encode(img_encoded).decode('utf-8')

    return jsonify({
        'prediction': prediction_result,
        'status': True,
        'image': img_base64
    })

if __name__ == '__main__':
    app.run(debug=True)
