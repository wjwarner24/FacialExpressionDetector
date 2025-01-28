import cv2
import numpy as np
import tensorflow as tf

# load the model
model = tf.keras.models.load_model('facial_expression_model2.keras')

# Load pre-trained face detection model (Haar Cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Emotion classes
classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Function to preprocess the cropped image
def preprocess_roi(image, bbox, target_size=(48, 48)):
    x1, y1, x2, y2 = bbox
    cropped = image[y1:y2, x1:x2]
    resized = cv2.resize(cropped, target_size)
    normalized = resized / 255.0
    return np.expand_dims(normalized, axis=0)

# Open the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Read frame from webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break
    
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Draw bounding boxes around detected faces
    for (x, y, w, h) in faces:
        side_length = max(w, h)
        x_start = x
        y_start = y
    
        # Center the square by adjusting coordinates
        x_start = max(x - (side_length - w) // 2, 0)
        y_start = max(y - (side_length - h) // 2, 0)

        # Calculate the bottom-right corner
        x_end = min(x_start + side_length, gray.shape[1]) 
        y_end = min(y_start + side_length, gray.shape[0])

        # Draw the square bounding box
        cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)
        
        # Get the region of interest
        roi = preprocess_roi(gray, (x_start, y_start, x_end, y_end))
        
        #predict the emotion
        prediction = model.predict(roi)
        emotion = classes[np.argmax(prediction)]
        
        # Display the emotion label
        cv2.putText(frame, emotion, (x_start, y_start - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        
    # Window settings
    cv2.namedWindow('Real-Time Face Detection', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('Real-Time Face Detection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

    
    cv2.imshow('Real-Time Face Detection', cv2.resize(frame,(1600,960),interpolation = cv2.INTER_CUBIC))
    
    # Break the loop if window closed or q is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if cv2.getWindowProperty('Real-Time Face Detection', cv2.WND_PROP_VISIBLE) < 1:
        break

# Exit the webcam
cap.release()
cv2.destroyAllWindows()
