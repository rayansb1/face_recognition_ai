import os
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from collections import Counter

# Load images and train the model
def load_images_from_dataset(dataset_path='dataset'):
    X, y = [], []
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    for person_name in os.listdir(dataset_path):
        person_folder = os.path.join(dataset_path, person_name)
        if not os.path.isdir(person_folder):
            continue

        image_count = 0  # Counter for each person's images
        for image_name in os.listdir(person_folder):
            img_path = os.path.join(person_folder, image_name)
            img = cv2.imread(img_path)
            if img is None:
                print(f"[!] Could not read image: {img_path}")
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            if len(faces) == 0:
                print(f"[!] No face detected in image: {img_path}")
                continue

            for (x, y_, w, h) in faces:
                face = gray[y_:y_+h, x:x+w]
                face = cv2.resize(face, (100, 100))
                X.append(face.flatten())
                y.append(person_name)
                image_count += 1
                break  # Use only one face per image

        print(f"[✓] Loaded {image_count} image(s) for {person_name}")

    return np.array(X), np.array(y)

# Load data and train the model
X, y = load_images_from_dataset()
print("\n🔍 Image count per person:", Counter(y))

if len(set(y)) < 2:
    print("\n❌ You need at least two people to train the model properly.")
    exit()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = KNeighborsClassifier(n_neighbors=1)  # Reduce number of neighbors
model.fit(X_train, y_train)

print(f"\n✅ Model accuracy: {model.score(X_test, y_test) * 100:.2f}%")

# Open webcam and start recognition
cap = cv2.VideoCapture(1)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

print("\n🎥 Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (100, 100)).flatten().reshape(1, -1)
        pred = model.predict(face_resized)[0]

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, pred, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()