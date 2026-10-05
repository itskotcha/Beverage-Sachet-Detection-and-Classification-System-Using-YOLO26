import os
import glob
from ultralytics import YOLO

# 1. โหลดโมเดล Weights จากโฟลเดอร์ที่เทรนเสร็จล่าสุด (train-3)
MODEL_PATH = r"C:\Users\Victus\Downloads\Beverage-Sachet-Detection-and-Classification-System-Using-YOLO26\tt\runs\detect\train-3\weights\best.pt"
model = YOLO(MODEL_PATH)

# 2. ระบุโฟลเดอร์รูปภาพที่จะใช้ทดสอบ (หรือเปลี่ยนเป็น path รูปภาพเดี่ยวๆ เช่น r"C:\...\frame\images\0101.jpg")
IMAGE_DIR = r"C:\Users\Victus\Downloads\Beverage-Sachet-Detection-and-Classification-System-Using-YOLO26\frame\images"

# ค้นหารูปภาพแรกในโฟลเดอร์มาทดสอบอัตโนมัติ
sample_images = glob.glob(os.path.join(IMAGE_DIR, "*.jpg")) + glob.glob(os.path.join(IMAGE_DIR, "*.png"))

if sample_images:
    test_image = sample_images[0]  # ดึงรูปแรกมาทดสอบ
    print(f"กำลังทดสอบกับรูปภาพ: {test_image}")

    # 3. นำโมเดลไปทดสอบ
    results = model.predict(
        source=test_image,
        conf=0.5,       # กำหนดค่า Confidence ขั้นต่ำที่ 50%
        save=True       # บันทึกภาพผลลัพธ์ลงในโฟลเดอร์ runs/detect/predict
    )

    # 4. แสดงผลลัพธ์
    results[0].show()
else:
    print(f"ไม่พบไฟล์รูปภาพในโฟลเดอร์: {IMAGE_DIR}")