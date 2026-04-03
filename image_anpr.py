import firebase_admin
from firebase_admin import credentials, db

# ---------------------------
# FIREBASE INIT (SAFE)
# ---------------------------
cred = credentials.Certificate("config/firebase_key.json")

if not firebase_admin._apps:
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://parkvahan-7d679-default-rtdb.firebaseio.com/'
    })

# ---------------------------
# IMPORTS
# ---------------------------
import cv2
import requests
from ultralytics import YOLO

# ---------------------------
# API TOKEN
# ---------------------------
API_TOKEN = "6cad7ac65e495a14d5353925b4f33efa4cc3bfe3"

# ---------------------------
# LOAD MODEL
# ---------------------------
model = YOLO("best.pt")

# ---------------------------
# LOAD IMAGE
# ---------------------------
image_path = r"C:\Users\gurpj\OneDrive\Desktop\testimage\testimage3.png"
img = cv2.imread(image_path)

if img is None:
    print("❌ Image not found")
    exit()

# ---------------------------
# VEHICLE DETECTION (BEST BOX ONLY)
# ---------------------------
results = model(img)

best_box = None
best_conf = 0
vehicle_class = "Vehicle"

for r in results:
    boxes = r.boxes.xyxy.cpu().numpy()
    classes = r.boxes.cls.cpu().numpy()
    scores = r.boxes.conf.cpu().numpy()

    for box, cls, conf in zip(boxes, classes, scores):
        if conf > best_conf:
            best_conf = conf
            best_box = box
            vehicle_class = model.names[int(cls)]

# Draw only best detection
if best_box is not None:
    x1, y1, x2, y2 = map(int, best_box)

    cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 3)

    cv2.putText(
        img,
        vehicle_class.upper(),
        (x1 + 10, y1 + 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0,255,0),
        3,
        cv2.LINE_AA
    )

# ---------------------------
# PLATE RECOGNITION API
# ---------------------------
plate_text = "NOT DETECTED"

try:
    with open(image_path, 'rb') as fp:
        response = requests.post(
            'https://api.platerecognizer.com/v1/plate-reader/',
            files={'upload': fp},
            headers={'Authorization': 'Token ' + API_TOKEN}
        )

    data = response.json()

    if data['results']:
        plate_text = data['results'][0]['plate'].upper()
        box = data['results'][0]['box']

        px, py = box['xmin'], box['ymin']
        pw, ph = box['xmax'], box['ymax']

        cv2.rectangle(img, (px, py), (pw, ph), (0,0,255), 3)

        cv2.putText(
            img,
            plate_text,
            (px, py - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,0,255),
            2,
            cv2.LINE_AA
        )

except Exception as e:
    print("❌ API Error:", e)
    plate_text = "ERROR"

# ---------------------------
# PRINT RESULT
# ---------------------------
print("----------- ANPR RESULT -----------")
print("Vehicle Type :", vehicle_class)
print("Number Plate :", plate_text)
print("-----------------------------------")

# ---------------------------
# STORE IN FIREBASE
# ---------------------------
try:
    if plate_text not in ["NOT DETECTED", "ERROR"]:

        ref = db.reference("vehicles")

        ref.push({
            "plate_number": plate_text,
            "vehicle_type": vehicle_class,
            "status": "entry"
        })

        print("✅ Data stored in Firebase")

    else:
        print("⚠️ Plate not stored (not detected)")

except Exception as e:
    print("❌ Firebase Error:", e)

# ---------------------------
# SHOW RESULT
# ---------------------------
cv2.imshow("ANPR Result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()