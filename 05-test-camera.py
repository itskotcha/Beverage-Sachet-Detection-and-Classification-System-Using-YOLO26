import cv2
from ultralytics import YOLO

def main():
    # 1. ระบุ Absolute Path ของ weights ที่เทรนได้ล่าสุด
    MODEL_PATH = r"C:\Users\Victus\Downloads\Beverage-Sachet-Detection-and-Classification-System-Using-YOLO26\tt\runs\detect\train-3\weights\best.pt"
    model = YOLO(MODEL_PATH)

    # 2. เปิดใช้งาน WebCam (0 คือกล้องหลักของเครื่อง)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ไม่สามารถเปิดกล้อง WebCam ได้")
        return

    print("เปิดกล้องสำเร็จ! กด 'q' บนหน้าต่างวิดีโอเพื่อปิดโปรแกรม")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ไม่สามารถรับสัญญาณภาพจากกล้องได้")
            break

        # 3. รัน Predict เฟรมต่อเฟรม
        results = model.predict(source=frame, conf=0.5, verbose=False)

        # 4. วาด Bounding Box และ Label ลงบนเฟรม
        annotated_frame = results[0].plot()

        # แสดงผลภาพจากกล้อง
        cv2.imshow("YOLO Beverage Sachet Detection", annotated_frame)

        # กด 'q' เพื่อออกจากโปรแกรม
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()