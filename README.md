# 🧃 Beverage Sachet Detection and Classification System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO-orange)](https://github.com/ultralytics/ultralytics)
[![Label Studio](https://img.shields.io/badge/Label_Studio-Data_Labeling-green)](https://labelstud.io/)

โปรเจกต์พัฒนาระบบปัญญาประดิษฐ์ด้วยโมเดล **YOLO** (ผ่านไลบรารี Ultralytics) สำหรับการตรวจจับ (Detection) และจำแนก (Classification) ซองเครื่องดื่มจากภาพถ่าย วิดีโอ และกล้องเว็บแคมแบบ Real-time โดยใช้ **Label Studio** ในการจัดการและสร้าง Dataset (Bounding Box)

---

## 🔗 Resources

- 📁 [ดาวน์โหลด Video Dataset, Test Image และ Video Test (Google Drive)](https://drive.google.com/drive/folders/1_Jhw7aTe7rR3EKcoR8h8A-xBMU_yMFO4)
- 🏷️ [ดาวน์โหลดและติดตั้ง Label Studio](https://labelstud.io)

---

## 📂 Project Structure

```text
Beverage-Sachet-Detection-and-Classification-System/
├── README.md
├── dataset/                    # โฟลเดอร์จัดเก็บข้อมูลสำหรับเทรนโมเดล
│   ├── images/                 # รูปภาพ (train/val)
│   ├── labels/                 # ไฟล์ Annotation .txt (train/val)
│   ├── classes.txt             # รายชื่อคลาส
│   └── data.yaml               # ไฟล์ Config ของ Dataset
├── env/                        # Virtual Environment (ถูก Ignore ไม่นำขึ้น Git)
├── yolo26n.pt                  # Pre-trained / Trained Model
├── requirements.txt            # รายการ Dependencies
├── 01-export_dataset.py        # สคริปต์แปลง JSON จาก Label Studio เป็น YOLO Format
├── 02-train.py                 # สคริปต์สำหรับ Train โมเดล
├── 03-test_image.py            # สคริปต์ทดสอบโมเดลด้วยรูปภาพ
├── 04-test_video.py            # สคริปต์ทดสอบโมเดลด้วยวิดีโอ
└── 05-test-camera.py           # สคริปต์ทดสอบโมเดลผ่านกล้องเว็บแคม
```

---

# 🚀 Getting Started (การติดตั้ง)

## 1. Clone Project & สร้าง Virtual Environment

แนะนำให้รันโปรเจกต์ใน Virtual Environment เพื่อป้องกันเวอร์ชันของไลบรารีชนกัน

### Windows

```bash
python -m venv env
.\env\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv env
source env/bin/activate
```

เมื่อ Activate สำเร็จ จะมีคำว่า `(env)` นำหน้า Command line

---

## 2. ติดตั้ง Dependencies

อัปเกรด `pip` และติดตั้งแพ็กเกจที่จำเป็นทั้งหมดผ่านไฟล์ `requirements.txt`

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -U ultralytics opencv-python matplotlib
```

---

## 3. ติดตั้ง PyTorch (สำหรับผู้ใช้งานการ์ดจอ NVIDIA GPU)

เพื่อให้การ Train และ Detect รวดเร็วขึ้น แนะนำให้รันบน GPU (CUDA 12.1):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

> **หมายเหตุ:** หากใช้งาน macOS หรือไม่มีการ์ดจอ NVIDIA ให้ข้ามขั้นตอนนี้ (ระบบจะใช้ CPU ตามปกติ)

---

# 🛠️ Data Preparation & Labeling (การเตรียมข้อมูล)

## 1. แยกเฟรมภาพจากวิดีโอ (Extract Frames)

ติดตั้ง FFmpeg (สำหรับ Windows) ผ่าน Command Prompt:

```bash
winget install ffmpeg
```
สร้างโฟลเดอร์ frame/images

<img width="304" height="156" alt="yolo1" src="https://github.com/user-attachments/assets/2984a821-0e33-48ba-90a5-50fadb838b69" />

จากนั้นแยกเฟรมรูปภาพจากวิดีโอด้วยคำสั่ง:

```bash
ffmpeg -i train_candy2.mp4 -vf fps=2 frame/images/%04d.jpg
```

### คำอธิบาย

- `train_candy2.mp4`: ชื่อไฟล์วิดีโอต้นทาง
- `-vf fps=2`: ดึงภาพออกมา 2 เฟรมต่อ 1 วินาที (ปรับเปลี่ยนได้ตามความเหมาะสม)
- `frame/images/%04d.jpg`: บันทึกภาพลงโฟลเดอร์โดยรันเลขรหัส เช่น `0001.jpg`, `0002.jpg`

---

## 2. การใช้งาน Label Studio

เปิด Command Prompt หน้าต่างใหม่ (ต้อง Activate Environment ก่อน) และตั้งค่า Path สำหรับ Local Files

### Windows

```cmd
set NLTK_DISABLE_IMPORT_SECURITY=1
set LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
set LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=C:\Users\Victus\Downloads
```

เปลี่ยน `C:\Users\Victus\Downloads` เป็น Path ที่เก็บโฟลเดอร์ภาพของคุณ

เริ่มใช้งาน Label Studio (ห้ามปิดหน้าต่าง CMD นี้ระหว่างใช้งาน):

```bash
label-studio start
```

---

## 3. การตั้งค่า Bounding Box & Classes
<img width="552" height="297" alt="image" src="https://github.com/user-attachments/assets/8a486782-3586-4410-9cfc-ba7110226e89" />

ในหน้า Labeling Setup ของ Label Studio ให้เลือก Code และกำหนด Classes ดังนี้ (แก้ไขได้ตามวัตถุจริง):

```xml
<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="label" toName="image">
    <Label value="NESCAFE" background="#8d4844"/>
    <Label value="NESTEA" background="#0d3fd3"/>
    <Label value="OVALTINT" background="#e40c0c"/>
    <Label value="MATCTA" background="#05ad11"/>
    <Label value="BIRDY" background="#c4ec32"/>
  </RectangleLabels>
</View>
```


<img width="508" height="393" alt="image" src="https://github.com/user-attachments/assets/cb9d307f-b73d-4f7a-95bd-fba0e7f40d2a" />
<img width="1917" height="971" alt="image" src="https://github.com/user-attachments/assets/8aa4a394-c3f7-46de-af6e-9925675b7418" />
<img width="1917" height="906" alt="image" src="https://github.com/user-attachments/assets/add862cc-1d83-45cc-91df-fe42740e160a" />

เมื่อสร้าง Bounding Box เสร็จแล้ว ให้กด Export ข้อมูลออกมาในรูปแบบ JSON แล้วนำไปวางไว้ในโฟลเดอร์โปรเจกต์

---
# การทดสอบและผลลัพธ์
---
# 1. Dataset Conversion (การแปลงข้อมูล)

เปิดไฟล์ `01-export_dataset.py` และตรวจสอบแก้ไข Path ให้ตรงกับเครื่องของคุณ:

```python
IMAGES_DIR = Path(r"C:\Path\To\Your\frame\images") # Path รูปภาพต้นฉบับ
OUTPUT_DIR = Path(r"C:\Path\To\Your\dataset")      # Path ที่ต้องการให้สร้าง Dataset
TRAIN_SPLIT = 0.8  # แบ่งข้อมูล Train 80% / Val 20%
```

รันสคริปต์เพื่อสร้าง Dataset ในรูปแบบที่ YOLO รองรับ:

```bash
python 01-export_dataset.py
```

<img width="1627" height="205" alt="image" src="https://github.com/user-attachments/assets/edbe304a-aaeb-4f73-ba8a-2ef934fec157" />


---

# 2. Training the Model (การสอนโมเดล)

โปรเจกต์นี้ตั้งค่าพื้นฐานสำหรับการ Train ไว้ใน `02-train.py` ดังนี้:

- **Epochs:** 50
- **Image Size:** 640
- **Optimizer:** MuSGD
- **Data Augmentation:** เช่น `degrees`, `fliplr`, `mosaic` เพื่อเพิ่มประสิทธิภาพ

เริ่มเทรนโมเดลด้วยคำสั่ง:

```bash
python 02-train.py
```

<img width="1638" height="347" alt="image" src="https://github.com/user-attachments/assets/a08624b9-6e14-4c34-83ac-a7020c81a000" />
<img width="1641" height="785" alt="image" src="https://github.com/user-attachments/assets/c5898d34-5bf2-4dd5-8cc8-78f1e3295202" />
<img width="1637" height="566" alt="image" src="https://github.com/user-attachments/assets/90293ca9-6972-4374-acca-3e22e6f69491" />
<img width="1637" height="597" alt="image" src="https://github.com/user-attachments/assets/321be0da-db78-4ce7-9174-81d2b7e3ff6c" />


---

# 3. Testing & Inference (การทดสอบโมเดล)

คุณสามารถทดสอบโมเดลที่เทรนเสร็จแล้วได้ 3 รูปแบบ โดยรูปภาพผลลัพธ์จะถูกบันทึกอัตโนมัติในโฟลเดอร์:

```text
runs/detect/predict/
```

## 1. ทดสอบด้วยรูปภาพ (Image)

แก้ไขชื่อไฟล์ภาพใน `03-test_image.py` แล้วรัน:

```bash
python 03-test_image.py
```
<img width="1642" height="198" alt="image" src="https://github.com/user-attachments/assets/148c5e12-6ade-4b97-a6ba-efcea1b63574" />
<img width="1917" height="1078" alt="image" src="https://github.com/user-attachments/assets/2606c4aa-5a17-46f2-88c7-ed70584f5513" />


## 2. ทดสอบด้วยวิดีโอ (Video)

แก้ไขชื่อไฟล์วิดีโอใน `04-test_video.py` (เช่น `video_candy1.MOV`) แล้วรัน:

```bash
python 04-test_video.py
```

<img width="1917" height="1078" alt="image" src="https://github.com/user-attachments/assets/93d718c0-a9e9-4706-aacc-40ccafd44631" />


## 3. ทดสอบด้วยกล้องเว็บแคม (Real-time Camera)

รันสคริปต์ด้านล่างเพื่อเปิดกล้อง (กดปุ่ม `q` บนคีย์บอร์ดเพื่อออกจากโปรแกรม):

```bash
python 05-test-camera.py
```

<img width="1917" height="1078" alt="image" src="https://github.com/user-attachments/assets/1da57d2d-adab-4a20-8303-78cb6c62d77c" />

---
