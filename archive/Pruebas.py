from ultralytics import YOLO
import cv2
import easyocr
import matplotlib.pyplot as plt

model = YOLO("runs/detect/car_plate_detection4/weights/best.pt")

reader = easyocr.Reader(["en"])

def detect_and_read_plate(image_path):
    results = model(image_path)
    
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            
            cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            plate_img = img[y1:y2, x1:x2]
            
            text_results = reader.readtext(plate_img)
            if text_results:
                plate_text = text_results[0][1]
                confidence = text_results[0][2]
                
                cv2.putText(img_rgb, f"{plate_text}", 
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.7, (0, 255, 0), 2)
                print(f"Texto detectado: {plate_text} | Confianza: {confidence:.2f}")
            else:
                print("No se detectó texto en la placa")
    
    plt.figure(figsize=(12, 8))
    plt.imshow(img_rgb)
    plt.axis("off")
    plt.show()

test_image = "C:/Users/Alex/Downloads/archive/images/train/Cars288.png"
detect_and_read_plate(test_image)