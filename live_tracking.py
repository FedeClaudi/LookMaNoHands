import mediapipe as mp
import time
import cv2
import numpy as np

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# https://developers.google.com/mediapipe/solutions/vision/face_landmarker/python#live-stream

def draw_landmarks_on_image(rgb_image, detection_result):
  face_landmarks_list = detection_result.face_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected faces to visualize.
  for idx in range(len(face_landmarks_list)):
    face_landmarks = face_landmarks_list[idx]

    # Draw the face landmarks.
    face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    face_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
    ])

    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_contours_style())
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_IRISES,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp.solutions.drawing_styles
          .get_default_face_mesh_iris_connections_style())

  return annotated_image

# define a global variable to store the results
results = None

# Create a face landmarker instance with the live stream mode:
def print_result(new_result: FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global results
    results = new_result
    

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="face_landmarker.task"),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True,
    )



with FaceLandmarker.create_from_options(options) as landmarker:
    # Set up video capture from default camera
    cap = cv2.VideoCapture(0)

    # set video fps to 60
    cap.set(cv2.CAP_PROP_FPS, 60)
    print("Starting")
    # Set up FPS counter
    frames = 0
    start_time = time.time()

    #   The landmarker is initialized. Use it here.
    while True:
        ret, frame = cap.read()
        # Display frame
        frames += 1

        # Calculate FPS
        if frames > 10:
            fps = round(frames / (time.time() - start_time), 2)
        else:
            fps = 0
        frame = cv2.putText(frame, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
        # Convert the frame received from OpenCV to a MediaPipe’s Image object.
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        # Send live image data to perform face landmarking.
        # The results are accessible via the `result_callback` provided in
        # the `FaceLandmarkerOptions` object.
        # The face landmarker must be created with the live stream mode.\
        frame_timestamp_ms = int(round(time.time() * 1000))
        landmarker.detect_async(mp_image, frame_timestamp_ms)

        if results is not None:
            if len(results.face_landmarks) == 0:
                print("No face detected")
                continue
                
            frame = draw_landmarks_on_image(frame, results)


            # check if a recognized action is being made
            for blend in results.face_blendshapes[0]:
                if blend.score > 0.5 and  "eye" not in blend.category_name:
                    print(blend.category_name, blend.score)

        # show frame
        cv2.imshow("frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        