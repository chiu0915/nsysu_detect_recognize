import cv2
import numpy as np
import pytesseract

# 載入模型和設定
weightsPath = "D:/CarPlateRecognition/custom.weights"
configPath = "D:/CarPlateRecognition/yolov4-tiny-custom.cfg"
net = cv2.dnn.readNet(configPath, weightsPath)
classes = ["license_plate"]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# 設定車牌辨識引擎
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
custom_config = r'--oem 3 --psm 6'

# 讀取影片檔案
cap = cv2.VideoCapture("D:/CarPlateRecognition/cars.mp4")

# 計算汽車與機車的數量
car_count = 0
motorbike_count = 0

while True:
    # 讀取影格
    ret, frame = cap.read()

    if ret:
        # 偵測物件
        height, width, channels = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        # 解析偵測結果
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5 and class_id == 0:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

                    # 計算汽車與機車的數量
                    label = str(classes[class_id])
                    if "car" in label:
                        car_count += 1
                    elif "motorcycle" in label:
                        motorbike_count += 1

        # 套用非最大值抑制 (Non-Maximum Suppression, NMS) 篩選重疊物件
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        # 車牌辨識
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                roi = frame[y:y + h, x:x + w]
                plate_text = pytesseract.image_to_string(roi, config=custom_config)
                print("車牌號碼：", plate_text)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
         # 繪製車輛與機車的辨識框
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                color = (0, 0, 255)  # 車輛為紅色，機車為藍色
                if "motorcycle" in label:
                    color = (255, 0, 0)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # 在視窗上顯示汽車和機車的數量
        cv2.putText(frame, "Cars: {}".format(car_count), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "Motorbikes: {}".format(motorbike_count), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # 顯示偵測結果
        cv2.imshow("Video", frame)

        # 按下 q 鍵結束
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# 釋放資源並關閉視窗
cap.release()
cv2.destroyAllWindows()
