import sys
import cv2
import mediapipe as mp
from mediapipe.tasks import python
import time

sys.dont_write_bytecode = True

print(f"Python version: {sys.version}")
print(f"OpenCV version: {cv2.__version__}")
print(f"MediaPipe version: {mp.__version__}")


mp_drawing_object=mp.solutions.drawing_utils
mp_hands_object=mp.solutions.hands
def result_callback_func(result, mp_image, frame_ms):

    try:
        print (f"Callback received for timestamp {frame_ms}")
        annotated_image = mp_image.numpy_view().copy()
        if result.hand_landmarks:
            for hand_landmarks in result.hand_landmarks:
                mp_drawing_object.draw_landmarks(
                    image=annotated_image,
                    landmark_list=hand_landmarks,
                    connections=mp_hands_object.HAND_CONNECTIONS)
        resized_image = cv2.resize(annotated_image, (640, 480))
        cv2.imshow("frame", cv2.cvtColor(resized_image, cv2.COLOR_RGB2BGR))
        print("Frame displayed")
    except Exception as e:
        print(f"Error in callback: {e}")
        import traceback
        traceback.print_exc()

    # try:
    #     print(f"Callback received for timestamp {frame_ms}")
    #     if result.hand_landmarks:
    #         for hand_landmarks in result.hand_landmarks:
    #             mp_drawing_object.draw_landmarks(mp_image, hand_landmarks, mp_hands_object.HAND_CONNECTIONS)
    #
    #     cv2.imshow("frame", cv2.cvtColor(mp_image.numpy_view(), cv2.COLOR_RGB2BGR))
    #     print("Frame displayed")
    # except Exception as e:
    #     print(f"Error in callback: {e}")



base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
    result_callback=result_callback_func
)
try:
    my_HandLandmarker_object = mp.tasks.vision.HandLandmarker.create_from_options(options)
except Exception as e:
    print(f"Error creating HandLandmarker: {e}")
    exit(1)



videoCaptureObject = cv2.VideoCapture(0)
cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
cv2.setUseOptimized(False)
cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
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
        frame_ms_as_int = int(str(frame_ms).replace('.',''))

        if frame_ms_as_int <= last_timestamp:
            frame_ms_as_int = last_timestamp + 1

        last_timestamp = frame_ms_as_int

        fixed_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
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
