import sys
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import time
import numpy as np

sys.dont_write_bytecode = True

print(f"Python version: {sys.version}")
print(f"OpenCV version: {cv2.__version__}")
print(f"MediaPipe version: {mp.__version__}")

annotated_image_global = np.zeros((640, 640, 3), np.uint8)  # Prepares a blank image to hold image that cv2 will show
mp_drawing_object = mp.solutions.drawing_utils
mp_hands_object = mp.solutions.hands


# Callback function for HandLandmarkerOptions object
def result_callback_func(result, mp_image, frame_ms):
    hand_landmarks_list = result.hand_landmarks  # A list containing multiple lists of NormalizedLandmark objects
    mp_image = mp_image.numpy_view()  # Turns the MediaPipe image into a numpy ndarray
    annotated_image = np.copy(mp_image)  # Copy of image to be used that will be actually annotated on

    # Iterates through each NormalizedLandmark object
    for index in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[index]  # Stores a single list of NormalizedLandmark objects

        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()  # Initializes a pb2 protobuf object so that it
        # is compatible with the draw_landmarks() function later on
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
        ])

        # Draws the lists' normalized landmarks and their connections onto annotated_image
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style()
        )
    # Fixes annotated_image so that it is mirrored properly and is represented in RGB not BGR
    annotated_image = cv2.flip(annotated_image, 1)
    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

    # Assigns the corresponding annotated_image_global to annotated_image
    global annotated_image_global
    annotated_image_global = annotated_image
    return


# Configuring options for hand landmark detection. Model is 'hand_landmarker.task'
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
# Creates an instance of HandLandmarker options and configures it. Uses the base_options, detect 2 hands,
# sets mode to live stream mode, and defines the callback function which is required when using live stream mode
options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
    result_callback=result_callback_func
)
try:
    # Creates the instance of HandLandmarker that will perform the hand landmark detection
    my_HandLandmarker_object = mp.tasks.vision.HandLandmarker.create_from_options(options)
except Exception as e:
    print(f"Error creating HandLandmarker: {e}")
    exit(1)


videoCaptureObject = cv2.VideoCapture(0)
# cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
# cv2.setUseOptimized(False)
# cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
if not videoCaptureObject.isOpened():
    print("Error: Could not open video capture")
    exit(1)

last_timestamp = 0
try:
    while videoCaptureObject.isOpened():
        ret, frame = videoCaptureObject.read()

        if not ret:
            print("Error: could not read frame")
            break

        frame_ms = videoCaptureObject.get(cv2.CAP_PROP_POS_MSEC)
        frame_ms_as_int = int(str(frame_ms).replace('.', ''))

        if frame_ms_as_int <= last_timestamp:
            frame_ms_as_int = last_timestamp + 1

        last_timestamp = frame_ms_as_int

        fixed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        flipped_frame = cv2.flip(frame, 1)
        cv2.imshow("frame", annotated_image_global)
        print(f"Displayed annotated image global")
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=fixed_frame)

        print(f"Processing frame at timestamp {frame_ms_as_int}")
        try:
            my_HandLandmarker_object.detect_async(mp_image, frame_ms_as_int)
            time.sleep(0.01)
        except Exception as e:
            print(f"Error in detect_async: {e}")
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except Exception as e:
    print(f"error in main loop: {e}")
videoCaptureObject.release()
cv2.destroyAllWindows()

#TODO:
# How to get the specific coordinate from a landmark like the index finger or something?
# Use tkinter to overlay. need to create a tkinter window to display the live stream feed on.
