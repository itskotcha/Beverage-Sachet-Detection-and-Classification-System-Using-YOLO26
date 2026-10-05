import os
import glob
from ultralytics import YOLO

# 1. ระบุ Absolute Path ของ Weights ล่าสุด
MODEL_PATH = r"C:\Users\Victus\Downloads\Beverage-Sachet-Detection-and-Classification-System-Using-YOLO26\tt\runs\detect\train-3\weights\best.pt"
model = YOLO(MODEL_PATH)

# 2. ระบุไฟล์วิดีโอตรงๆ (เลือกใช้ test.MOV หรือ train.MOV)
VIDEO_SOURCE = "test.MOV" 

# หากหาไม่เจอ จะสแกนหาไฟล์วิดีโอตระกูล .MOV, .mp4, .avi ทั้งหมดในโฟลเดอร์
if not os.path.exists(VIDEO_SOURCE):
    videos = glob.glob("*.MOV") + glob.glob("*.mov") + glob.glob("*.mp4") + glob.glob("*.avi")
    if videos:
        VIDEO_SOURCE = videos[0]
        print(f"พบไฟล์วิดีโอในโปรเจกต์: {VIDEO_SOURCE}")
    else:
        print("ไม่พบไฟล์วิดีโอ สลับไปใช้งาน WebCam (Device 0)")
        VIDEO_SOURCE = 0

print(f"กำลังรันการทดสอบด้วยวิดีโอ: {VIDEO_SOURCE}")

# 3. รัน Predict แบบ stream=True เพื่อไม่ให้กิน RAM
results = model.predict(
    source=VIDEO_SOURCE,
    conf=0.5,       # Confidence ขั้นต่ำ 50%
    show=True,      # แสดงหน้าต่างวิดีโอขณะประมวลผล
    save=True,      # บันทึกวิดีโอผลลัพธ์ลงโฟลเดอร์ runs/detect/predict
    stream=True     # ประมวลผลแบบ Generator ป้องกัน RAM เต็ม
)

# วนลูปอ่านผลลัพธ์แต่ละเฟรม
for r in results:
    pass