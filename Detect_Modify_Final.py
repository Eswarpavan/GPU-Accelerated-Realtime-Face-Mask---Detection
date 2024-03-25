# importing libraries

import os # To interact with operating system
import numpy as np # to work with arrays
import imutils # It is a set of convenience functions to make basic image processing operations 
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input # To preprocess the input
from tensorflow.keras.preprocessing.image import img_to_array # To convert PIL image to numpy array
from tensorflow.keras.models import load_model # To load the pre-trained model
from imutils.video import VideoStream # To access the video stram
import time # To measure the execution time
import cv2 # To perform computer vision tasks

# CUDA_VISIBLE_DEVICES = 0 , which is used by NVIDIA CUDA to select a specific GPU, it is set to use the first GPU device available
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# Detect faces in input frame and classify mask or no mask
def detect_and_predict_mask(frame, faceNet, maskNet):
    # Height and width of the input frame
    (h, w) = frame.shape[:2]
    # Input image preprocessinng
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
                                 (104.0, 177.0, 123.0))

    # Set the input to the faceNet and then pass it through the network to detect faces 
    faceNet.setInput(blob)
    detections = faceNet.forward()
    print(detections.shape)

    # initializing cropped_faces,bounding_box_coordinates, predictions for each detected face as face_detected_predicitons
    cropped_faces = []
    bounding_box_coordinates = []
    face_detected_predictions = []

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # If confidence is >50% it calculates the bounding box coordinates for the detected faces,
        # and scales them back to the original image size and clips them to ensure they are within the bounds of the image
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # Extract the cropped face from the source
            face = frame[startY:endY, startX:endX]
            # Convert color space from BGR to RGB
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            # Resizing it to (224,224)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            # Preprocessing it for input into the neural network
            face = preprocess_input(face)

            # Cropped faces and bounding box coordinates are appended to their corresponding lists
            cropped_faces.append(face)
            bounding_box_coordinates.append((startX, startY, endX, endY))

    # Check whether faces were detected and cropped from source 
    if len(cropped_faces) > 0:
        # If detected converting cropped faces list into NumPy array of type float32
        cropped_faces = np.array(cropped_faces, dtype="float32")
        # Used maskNet neural network to predict whether each face is wearing a mask
        face_detected_predictions = maskNet.predict(cropped_faces, batch_size=32)

    # Returned a tuple containing the bounding box coordinates for each detected faces and the corresponding mask detection predictions
    # if no faces were detected an empty list is returned for bothe bounding box coordinates and mask detection predictions 
    return (bounding_box_coordinates, face_detected_predictions) if len(face_detected_predictions) > 0 else ([], [])

# Loading two pre-trained neural networks used for detecting faces and masks.

# Architecture of the face detection neural network
prototxtPath = r"pre-trained-weights\deploy.prototxt"

# Stores the pre-trained weights for face detection neural network
weightsPath = r"pre-trained-weights\res10_300x300_ssd_iter_140000.caffemodel"

# Loading the neural network architecture and pre-trained weights from these files into the faceNet object
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# Loading our pre-trained model using keras library
maskNet = load_model("facemask_detector_MobileNetV2.model")

print("[INFO] starting video stream...")
# Starting a video stream
Video_stream = VideoStream(src=0).start()

while True:
    # Reading frames from the video stream
	frame = Video_stream.read()

    # Recordthe start time of the face detected and prediction process
	start_time = time.time()
	(bounding_box_coordinates, face_detected_predictions) = detect_and_predict_mask(frame, faceNet, maskNet)
	print("Elapsed time: {:.5f}".format(time.time() - start_time))
    
	for (bounding_box, mask_pred) in zip(bounding_box_coordinates, face_detected_predictions):
        # Bounding_box stores the coordinates of the bounding box around the face
		(startX, startY, endX, endY) = bounding_box
        
        # Extract probabilities of wearing a mask or not wearing a mask
		(mask, withoutMask) = mask_pred

        # Prediction conditions
		label = "Mask" if mask > withoutMask else "No Mask"
  
        # Bounding box color will be green if label is "Mask" and 
        # Bounding box color will be red if label is "No Mask"
		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        # Adding mask detection probability as a percentage to label
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

        # Display the label above the bounding box
		cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

        # Draw the bounding box around the face
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    # Display the output
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
    # If the key pressed is letter 'q' the loop will be broken and program exits
	if key == ord("q"):
		break

# Close all windows created by OpenCV 
cv2.destroyAllWindows()

# Stopping the video strezm
Video_stream.stop()
