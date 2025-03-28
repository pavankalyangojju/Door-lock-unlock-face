# --- IMPORTS ---
import cv2
import pickle
import numpy as np
import os
import time
import pyttsx3
import RPi.GPIO as GPIO
import smbus2
import threading
from mfrc522 import SimpleMFRC522
from sklearn.neighbors import KNeighborsClassifier
from telegram.ext import Updater, CommandHandler

# --- GPIO SETUP ---
GPIO.setmode(GPIO.BCM)
BUZZER_PIN = 17
SERVO_PIN = 26
LED_PIN = 19
FAN_PIN = 13

GPIO.setup(BUZZER_PIN, GPIO.OUT)
GPIO.setup(SERVO_PIN, GPIO.OUT)
GPIO.setup(LED_PIN, GPIO.OUT)
GPIO.setup(FAN_PIN, GPIO.OUT)

GPIO.output(BUZZER_PIN, GPIO.LOW)
GPIO.output(LED_PIN, GPIO.LOW)
GPIO.output(FAN_PIN, GPIO.LOW)

servo = GPIO.PWM(SERVO_PIN, 50)
servo.start(0)

# --- LCD SETUP ---
LCD_ADDRESS = 0x27
LCD_WIDTH = 16
LCD_CHR = 1
LCD_CMD = 0
LINE_1 = 0x80
LINE_2 = 0xC0
LCD_BACKLIGHT = 0x08
ENABLE = 0b00000100
bus = smbus2.SMBus(1)

def lcd_byte(bits, mode):
    high_bits = mode | (bits & 0xF0) | LCD_BACKLIGHT
    low_bits = mode | ((bits << 4) & 0xF0) | LCD_BACKLIGHT
    bus.write_byte(LCD_ADDRESS, high_bits)
    lcd_toggle_enable(high_bits)
    bus.write_byte(LCD_ADDRESS, low_bits)
    lcd_toggle_enable(low_bits)

def lcd_toggle_enable(bits):
    time.sleep(0.0005)
    bus.write_byte(LCD_ADDRESS, bits | ENABLE)
    time.sleep(0.0005)
    bus.write_byte(LCD_ADDRESS, bits & ~ENABLE)
    time.sleep(0.0005)

def lcd_init():
    lcd_byte(0x33, LCD_CMD)
    lcd_byte(0x32, LCD_CMD)
    lcd_byte(0x06, LCD_CMD)
    lcd_byte(0x0C, LCD_CMD)
    lcd_byte(0x28, LCD_CMD)
    lcd_byte(0x01, LCD_CMD)
    time.sleep(0.005)

def lcd_display(message, line):
    lcd_byte(line, LCD_CMD)
    message = message.ljust(LCD_WIDTH, " ")
    for char in message:
        lcd_byte(ord(char), LCD_CHR)

def lcd_clear():
    lcd_display("", LINE_1)
    lcd_display("", LINE_2)

# --- LCD STARTUP MESSAGE ---
lcd_init()
lcd_display("Hi, welcome", LINE_1)
lcd_display("Door Lock Unlock", LINE_2)
time.sleep(3)
lcd_clear()

# --- Other Modules ---
engine = pyttsx3.init()
reader = SimpleMFRC522()

def speak(text):
    engine.say(text)
    engine.runAndWait()

# --- Servo Control ---
def unlock_door():
    print("Servo: Unlocking (180°)")
    servo.ChangeDutyCycle(12.0)
    time.sleep(1)
    servo.ChangeDutyCycle(0)

def lock_door():
    print("Servo: Locking (0°)")
    servo.ChangeDutyCycle(2.5)
    time.sleep(1)
    servo.ChangeDutyCycle(0)

def buzzer_and_lcd_message(name):
    for _ in range(2):
        GPIO.output(BUZZER_PIN, GPIO.HIGH)
        time.sleep(1)
        GPIO.output(BUZZER_PIN, GPIO.LOW)
        time.sleep(0.5)
    lcd_clear()
    lcd_display("Access Granted", LINE_1)
    lcd_display(f"Welcome {name}", LINE_2)
    speak(f"Access granted, welcome {name}")
    time.sleep(3)
    lcd_clear()

# --- Load Face Data ---
video = cv2.VideoCapture(0)
if not video.isOpened():
    print("Error: Camera not found.")
    exit()

facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

try:
    with open('data/names.pkl', 'rb') as w:
        LABELS = pickle.load(w)
    with open('data/faces_data.pkl', 'rb') as f:
        FACES = pickle.load(f)
    with open('data/rfid_data.pkl', 'rb') as r:
        RFID_LIST = pickle.load(r)
except Exception as e:
    print("Error loading data:", e)
    exit()

FACES = np.array(FACES).reshape(FACES.shape[0], -1)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

# --- Telegram Bot Setup ---
TELEGRAM_TOKEN = '7038070025:AAHOoUWmqVPvFmmITJKpbWVGcdwzLDmcVJI'

def led_on(update, context):
    GPIO.output(LED_PIN, GPIO.HIGH)
    update.message.reply_text("LED turned ON")

def led_off(update, context):
    GPIO.output(LED_PIN, GPIO.LOW)
    update.message.reply_text("LED turned OFF")

def fan_on(update, context):
    GPIO.output(FAN_PIN, GPIO.HIGH)
    update.message.reply_text("Fan turned ON")

def fan_off(update, context):
    GPIO.output(FAN_PIN, GPIO.LOW)
    update.message.reply_text("Fan turned OFF")

def start_bot():
    updater = Updater(TELEGRAM_TOKEN, use_context=True)
    dp = updater.dispatcher
    dp.add_handler(CommandHandler("led_on", led_on))
    dp.add_handler(CommandHandler("led_off", led_off))
    dp.add_handler(CommandHandler("fan_on", fan_on))
    dp.add_handler(CommandHandler("fan_off", fan_off))
    updater.start_polling()

bot_thread = threading.Thread(target=start_bot)
bot_thread.daemon = True
bot_thread.start()

# --- MAIN LOOP ---
while True:
    lcd_clear()
    lcd_display("Put Face in Front", LINE_1)
    lcd_display("of Camera", LINE_2)

    ret, frame = video.read()
    if not ret:
        print("Error: Frame not captured.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w]
        resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)

        try:
            output = knn.predict(resized_img)
            recognized_name = output[0]
            print("Recognized:", recognized_name)
        except Exception as e:
            print("Prediction Error:", e)
            continue

        # Face recognized → show welcome, turn ON LED & Fan
        lcd_clear()
        lcd_display("Door is Open", LINE_1)
        lcd_display("Welcome", LINE_2)
        speak("Door is open, welcome")
        GPIO.output(LED_PIN, GPIO.HIGH)
        GPIO.output(FAN_PIN, GPIO.HIGH)

        unlock_door()
        time.sleep(10)
        lock_door()

        lcd_clear()
        lcd_display("Open Door", LINE_1)
        lcd_display("Put RFID Card", LINE_2)
        speak("Please show your RFID card")

        try:
            card_id, card_text = reader.read()
            print("RFID:", card_id)

            try:
                user_index = LABELS.index(recognized_name)
                expected_rfid = RFID_LIST[user_index]

                if str(card_id) != str(expected_rfid):
                    lcd_clear()
                    lcd_display("Card doesn't match", LINE_1)
                    lcd_display("Try again", LINE_2)
                    speak("RFID does not match the recognized person")
                    time.sleep(3)
                    continue

            except ValueError:
                lcd_clear()
                lcd_display("Unknown Person", LINE_1)
                lcd_display("Access Denied", LINE_2)
                speak("Access denied")
                time.sleep(3)
                continue

            unlock_door()
            time.sleep(5)
            lock_door()

            buzzer_and_lcd_message(recognized_name)

            video.release()
            time.sleep(5)
            video = cv2.VideoCapture(0)

        except Exception as e:
            lcd_clear()
            lcd_display("RFID Error!", LINE_1)
            print(f"RFID Error: {e}")
            time.sleep(2)
            continue

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        lcd_clear()
        break

# --- Cleanup ---
video.release()
cv2.destroyAllWindows()
servo.stop()
GPIO.cleanup()
