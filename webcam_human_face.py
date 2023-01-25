import cv2

# Get a reference to the webcam
webcam = cv2.VideoCapture(0)

# Load the human face cascade file
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

while True:
    # Get the current frame from the webcam
    ret, frame = webcam.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Draw a rectangle around each face
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Show the frame
    cv2.imshow("Webcam", frame)

    # Check if the user pressed the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam
webcam.release()

# Destroy the window
cv2.destroyAllWindows()
