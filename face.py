import cv2
import os

#  camera
camera = cv2.VideoCapture(0)  
# use 0 default camera

# dir image
dataset_folder = 'dataset'
os.makedirs(dataset_folder, exist_ok=True)

# face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# face img
def capture_faces(person_id, person_name, university_name):
    person_folder_name = f"{person_id}_{person_name}_{university_name}"
    person_folder = os.path.join(dataset_folder, person_folder_name)
    if os.path.exists(person_folder):
        print(f"Folder for {person_folder_name} already exists. This user is already registered.")
        return

    os.makedirs(person_folder)

    count = 0  # Count capture image
    while count < 12:  # capture 12 images  person
        # read camera
        ret, frame = camera.read()
        if not ret:
            print("Error: Failed to capture image from camera.")
            break

        #  grayscale  face detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # one face  detect
        if len(faces) == 1:
            for (x, y, w, h) in faces:
                face_roi = frame[y:y+h, x:x+w]  # extract  face region
                face_image_path = os.path.join(person_folder, f'face_{count}.jpg')
                cv2.imwrite(face_image_path, face_roi)
                print(f"Saved: {face_image_path}")
                count += 1
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)


        cv2.imshow('Capture Faces', frame)


        key = cv2.waitKey(1)
        if key == ord('q'):  # q to exit
            break

    print(f"Captured {count} images for {person_name}")

# Take input from the user for the person's ID, name, and university name
person_id = input("Enter the ID of the person: ")
person_name = input("Enter the name of the person: ")
university_name = input("Enter the university name of the person: ")

# Capture faces for the specified person if the folder doesn't already exist
print(f"Capturing faces for {person_name} from {university_name}...")
capture_faces(person_id, person_name, university_name)

# Release the camera and close all OpenCV windows
camera.release()
cv2.destroyAllWindows()
