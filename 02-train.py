from ultralytics import YOLO

if __name__ == '__main__':

    # โหลดโมเดล
    model = YOLO('yolo26n.pt')

    # เริ่มเทรนพร้อมตั้งค่า Data Augmentation
    results = model.train(
        # ระบุ Absolute Path ของ data.yaml ให้ชัดเจน
        data=r'C:\Users\Victus\Downloads\Beverage-Sachet-Detection-and-Classification-System-Using-YOLO26\dataset\data.yaml',
        device='cpu',
        
        # --- พารามิเตอร์สำหรับ Data Augmentation หลัก ---
        degrees=15.0,        # สุ่มหมุนภาพ -15 ถึง +15 องศา
        shear=5.0,           # สุ่มบิดภาพแบบเฉียงด้านขนาน
        perspective=0.001,   # เพิ่มความลึก

        # --- การพลิกภาพ (Flip) ---
        fliplr=0.5,          # โอกาส 50% พลิกภาพซ้าย-ขวา
        flipud=0.0,          # พลิกภาพบน-ล่าง

        # --- พารามิเตอร์ขั้นสูง ---
        mosaic=1.0,          # นำภาพ 4 ภาพมาตัดแปรรวมกัน
        mixup=0.1,           # ซ้อนภาพ 2 ภาพเข้าด้วยกันแบบโปร่งแสง

        # ปิด Mosaic ในช่วง 10 epochs สุดท้าย
        close_mosaic=10
    )