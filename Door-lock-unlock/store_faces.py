import cv2
import pickle
import numpy as np
import os
import time
import RPi.GPIO as GPIO
from mfrc522 import SimpleMFRC522
from RPLCD.i2c import CharLCD

# ✅ Initialize LCD
lcd = CharLCD(i2c_expander='PCF8574', address=0x27, port=1, cols=16, rows=2, dotsize=8)

# ✅ Welcome Message
lcd.clear()
lcd.write_string("HI,Welcome to")
lcd.crlf()
lcd.write_string("Face Register")
time.sleep(2)
lcd.clear()

# ✅ Setup GPIO
GPIO.setwarnings(False)
if GPIO.getmode() is None:
    GPIO.setmode(GPIO.BCM)

# ✅ Setup RFID Reader and LED
reader = SimpleMFRC522()
LED_PIN = 18
GPIO.setup(LED_PIN, GPIO.OUT)
GPIO.output(LED_PIN, GPIO.LOW)

# ✅ Open Camera
video = cv2.VideoCapture(0)
if not video.isOpened():
    lcd.write_string("Camera Error!")
    print("Error: Could not open camera.")
    exit()

# ✅ Load Haarcascade
facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# ✅ Paths
os.makedirs('data', exist_ok=True)
faces_file = 'data/faces_data.pkl'
names_file = 'data/names.pkl'
rfid_file = 'data/rfid_data.pkl'

# ✅ Scan RFID Card
lcd.write_string("Scan your CARD")
print("Scan your RFID card...")

try:
    card_id, card_text = reader.read()
    print(f"RFID Card ID: {card_id}")
    lcd.clear()

    # Check if already registered
    if os.path.exists(rfid_file):
        with open(rfid_file, 'rb') as f:
            stored_rfid = pickle.load(f)
        if card_id in stored_rfid:
            lcd.write_string("Already")
            lcd.crlf()
            lcd.write_string("Registered!")
            print("Face already registered for this RFID.")
            time.sleep(3)
            lcd.clear()
            GPIO.cleanup()
            exit()

    lcd.write_string("Enter Your Name:")
    name = input("Enter Your Name: ")

except Exception as e:
    lcd.clear()
    lcd.write_string("RFID Error!")
    print(f"RFID Error: {e}")
    exit()

# ✅ Face Capture Prompt
lcd.clear()
lcd.write_string("Look at Camera")
lcd.crlf()
lcd.write_string("Stay - light off")
print("Put your face towards the camera and stay until the light turns off...")

# ✅ LED ON
GPIO.output(LED_PIN, GPIO.HIGH)

faces_data = []
rfid_data = []
names_data = []
i = 0
max_images = 50

while True:
    ret, frame = video.read()
    if not ret:
        print("Error: Could not capture frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w]
        resized_img = cv2.resize(crop_img, (50, 50))

        if len(faces_data) < max_images and i % 5 == 0:
            faces_data.append(resized_img)
            rfid_data.append(card_id)
            names_data.append(name)

        i += 1
        cv2.putText(frame, str(len(faces_data)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 1)

    cv2.imshow("Frame", frame)
    k = cv2.waitKey(1)
    if k == ord('q') or len(faces_data) == max_images:
        break

video.release()
cv2.destroyAllWindows()
GPIO.output(LED_PIN, GPIO.LOW)

# ✅ Saving Data
lcd.clear()
lcd.write_string("Saving Data...")
print("Saving data...")

faces_data = np.asarray(faces_data).reshape(len(faces_data), 50, 50, 3)

# ✅ Save Names
if not os.path.exists(names_file):
    with open(names_file, 'wb') as f:
        pickle.dump(names_data, f)
else:
    with open(names_file, 'rb') as f:
        stored_names = pickle.load(f)
    stored_names.extend(names_data)
    with open(names_file, 'wb') as f:
        pickle.dump(stored_names, f)

# ✅ Save RFID Data
if not os.path.exists(rfid_file):
    with open(rfid_file, 'wb') as f:
        pickle.dump(rfid_data, f)
else:
    with open(rfid_file, 'rb') as f:
        stored_rfid = pickle.load(f)
    stored_rfid.extend(rfid_data)
    with open(rfid_file, 'wb') as f:
        pickle.dump(stored_rfid, f)

# ✅ Save Faces
if not os.path.exists(faces_file):
    with open(faces_file, 'wb') as f:
        pickle.dump(faces_data, f)
else:
    with open(faces_file, 'rb') as f:
        stored_faces = pickle.load(f)
        stored_faces = stored_faces.reshape(-1, 50, 50, 3)
    stored_faces = np.vstack((stored_faces, faces_data))
    with open(faces_file, 'wb') as f:
        pickle.dump(stored_faces, f)

# ✅ Done
lcd.clear()
lcd.crlf()
lcd.write_string("Successfully!")
print("Data saved successfully!")
GPIO.cleanup()
