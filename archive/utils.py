import mediapipe as mp
import numpy as np

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

N_FEATURES = 15

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode


def get_landmarks_idxs(connections):
    return list(set(np.vstack(connections).ravel()))


LEFT_PUPIL = 468
RIGHT_PUPIL = 473

NOSE = get_landmarks_idxs(mp.solutions.face_mesh.FACEMESH_NOSE)
MOUTH = get_landmarks_idxs(mp.solutions.face_mesh.FACEMESH_LIPS)
OVAL = get_landmarks_idxs(mp.solutions.face_mesh.FACEMESH_FACE_OVAL)
LEFT_EYE = get_landmarks_idxs(mp.solutions.face_mesh.FACEMESH_LEFT_EYE)
RIGHT_EYE = get_landmarks_idxs(mp.solutions.face_mesh.FACEMESH_RIGHT_EYE)
LEFT_IRIS = get_landmarks_idxs(mp.solutions.face_mesh.FACEMESH_LEFT_IRIS)
RIGHT_IRIS = get_landmarks_idxs(mp.solutions.face_mesh.FACEMESH_RIGHT_IRIS)

face_landmarks = [*NOSE, *MOUTH, *OVAL, *LEFT_EYE, *RIGHT_EYE]
eyes_landmark = [*LEFT_IRIS, *RIGHT_IRIS]


def extract_landmark_coords(face_landmarker, coords):
    if len(face_landmarker.face_landmarks) == 0:
        # set coords to nans
        coords = np.full((478, 3), np.nan)
    else:
        for (i, marker) in enumerate(face_landmarker.face_landmarks[0]):
            coords[i, :] = [marker.x, marker.y, marker.z]

    return coords


def extract_features(coords):
    center = np.mean(coords, axis=0)
    center[2] = np.max(coords[:, 2])

    # center coords
    coords -= center

    # get the position of the eyes, mouth and nose
    nose = coords[NOSE].mean(axis=0)
    left_eye = coords[LEFT_EYE].mean(axis=0)
    right_eye = coords[RIGHT_EYE].mean(axis=0)
    left_iris = coords[LEFT_IRIS].mean(axis=0)
    right_iris = coords[RIGHT_IRIS].mean(axis=0)
    left_pupil = coords[LEFT_PUPIL]
    right_pupil = coords[RIGHT_PUPIL]

    # stack it into a vector
    features = np.array(
        [*nose, *left_eye, *right_eye, *left_iris, *right_iris]
    )
    return features



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
