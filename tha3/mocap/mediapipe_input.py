import cv2
import mediapipe as mp
import numpy as np
import math

from tha3.mocap.ifacialmocap_constants import BLENDSHAPE_NAMES, HEAD_BONE_X, HEAD_BONE_Y, HEAD_BONE_Z, HEAD_BONE_QUAT

# Mapping from MediaPipe FaceMesh landmarks to ARKit/iFacialMocap blendshape names
# This is a simplified initial mapping and will need significant refinement.
# Indices are from https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png

FACEMESH_LIPS_OUTER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
FACEMESH_LIPS_INNER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]

FACEMESH_LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
FACEMESH_RIGHT_EYE = [362, 382, 381, 380, 373, 374, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

FACEMESH_LEFT_EYEBROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
FACEMESH_RIGHT_EYEBROW = [336, 296, 334, 293, 300, 285, 295, 282, 283, 276]

FACEMESH_NOSE_TIP = 1
FACEMESH_CHIN = 152

# Landmark indices for head pose estimation
# These are just some points on the face that are relatively stable.
# A proper 3D-2D point correspondence solver (like cv2.solvePnP) is needed for accurate head pose.
FACEMESH_POSE_LANDMARKS = [
    1,  # Nose tip
    33,  # Left eye inner corner
    263,  # Right eye inner corner
    61,  # Left mouth corner
    291,  # Right mouth corner
    152, # Chin
    10, # Forehead top
    168 # Nose bridge center
]


def euler_to_quaternion(roll, pitch, yaw):
    """
    Convert Euler angles (in radians) to quaternion.
    Roll (X), Pitch (Y), Yaw (Z)
    """
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return [qx, qy, qz, qw]


class MediaPipeWebcamInput:
    def __init__(self, webcam_index=0, image_width=640, image_height=480):
        self.webcam_index = webcam_index
        self.image_width = image_width
        self.image_height = image_height

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.cap = None
        self.latest_pose_data = self._create_default_pose()
        self.raw_landmarks = None

        # For head pose: 3D model points (approximate canonical face model)
        # These points need to be in the same order as FACEMESH_POSE_LANDMARKS
        # and represent their average positions in a generic 3D head model.
        # This is a placeholder and needs to be accurate for good results.
        self.model_points_3d = np.array([
            [0.0, 0.0, 0.0],         # Nose tip
            [-2.3, 1.3, -1.0],       # Left eye inner corner
            [2.3, 1.3, -1.0],        # Right eye inner corner
            [-1.5, -1.5, -0.8],      # Left mouth corner
            [1.5, -1.5, -0.8],       # Right mouth corner
            [0.0, -3.3, -0.5],       # Chin
            [0.0, 4.0, -1.5],        # Forehead top
            [0.0, 1.7, -1.6]         # Nose bridge center
        ], dtype=np.float32) * 25 # Scale factor, adjust as needed


    def _create_default_pose(self):
        data = {}
        for blendshape_name in BLENDSHAPE_NAMES:
            data[blendshape_name] = 0.0
        data[HEAD_BONE_X] = 0.0
        data[HEAD_BONE_Y] = 0.0
        data[HEAD_BONE_Z] = 0.0
        data[HEAD_BONE_QUAT] = [0.0, 0.0, 0.0, 1.0]
        # Add other bones (eyes) if we decide to track them
        return data

    def start_capture(self):
        if self.cap is not None:
            self.stop_capture()
        self.cap = cv2.VideoCapture(self.webcam_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.image_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.image_height)
        if not self.cap.isOpened():
            print(f"Error: Could not open webcam index {self.webcam_index}")
            self.cap = None
            return False
        print("Webcam capture started.")
        return True

    def stop_capture(self):
        if self.cap:
            self.cap.release()
            self.cap = None
        cv2.destroyAllWindows()
        print("Webcam capture stopped.")

    def get_current_pose(self):
        return self.latest_pose_data.copy()

    def process_frame(self, show_video=True):
        if self.cap is None or not self.cap.isOpened():
            return None, None

        success, image = self.cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            return None, None

        image.flags.writeable = False
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)
        image.flags.writeable = True

        current_pose = self._create_default_pose()

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                self.raw_landmarks = face_landmarks # Store for potential drawing/debugging

                # --- Head Pose Estimation ---
                image_points_2d = np.array([
                    (face_landmarks.landmark[i].x * self.image_width, face_landmarks.landmark[i].y * self.image_height)
                    for i in FACEMESH_POSE_LANDMARKS
                ], dtype=np.float32)

                focal_length = self.image_width
                center = (self.image_width / 2, self.image_height / 2)
                camera_matrix = np.array([
                    [focal_length, 0, center[0]],
                    [0, focal_length, center[1]],
                    [0, 0, 1]
                ], dtype=np.float32)

                # Assuming no lens distortion
                dist_coeffs = np.zeros((4, 1), dtype=np.float32)

                try:
                    (success_pnp, rotation_vector, translation_vector) = cv2.solvePnP(
                        self.model_points_3d, image_points_2d, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_SQPNP # SOLVEPNP_ITERATIVE can also be used
                    )

                    if success_pnp:
                        # Convert rotation vector to rotation matrix
                        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

                        # Decompose rotation matrix to Euler angles
                        # sy = math.sqrt(rotation_matrix[0,0] * rotation_matrix[0,0] +  rotation_matrix[1,0] * rotation_matrix[1,0])
                        # singular = sy < 1e-6
                        # if not singular:
                        #     x_rad = math.atan2(rotation_matrix[2,1] , rotation_matrix[2,2])
                        #     y_rad = math.atan2(-rotation_matrix[2,0], sy)
                        #     z_rad = math.atan2(rotation_matrix[1,0], rotation_matrix[0,0])
                        # else:
                        #     x_rad = math.atan2(-rotation_matrix[1,2], rotation_matrix[1,1])
                        #     y_rad = math.atan2(-rotation_matrix[2,0], sy)
                        #     z_rad = 0
                        # Pitch, Yaw, Roll
                        # This order might need adjustment depending on the target system's coordinate frame.
                        # Typically, MediaPipe/OpenCV might give: Y: yaw, X: pitch, Z: roll
                        # The iFacialMocap constants are HEAD_BONE_X, HEAD_BONE_Y, HEAD_BONE_Z
                        # Let's assume X=Pitch, Y=Yaw, Z=Roll for now.
                        # Rotation angles from solvePnP are in radians.
                        # Convert to degrees for easier interpretation if needed, but iFacialMocap might expect radians or specific degrees.
                        # The values are often scaled. iFacialMocap uses values up to +/- 90 for head rotations.

                        # Let's try to get Euler angles more directly and carefully
                        # R = Pitch, P = Yaw, Y = Roll
                        #eulerAngles = cv2.decomposeProjectionMatrix(np.hstack((rotation_matrix, translation_vector.reshape(3,1))))[6]
                        # pitch_rad = math.radians(eulerAngles[0,0])
                        # yaw_rad   = math.radians(eulerAngles[1,0])
                        # roll_rad  = math.radians(eulerAngles[2,0])

                        # Alternative Euler angle extraction (more robust for some cases)
                        P = math.atan2(-rotation_matrix[2, 0], math.sqrt(rotation_matrix[0, 0]**2 + rotation_matrix[1, 0]**2))
                        Y = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
                        R = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])

                        # Convert to degrees and scale for iFacialMocap (common range is -90 to 90 or -1 to 1)
                        # This scaling factor might need adjustment.
                        # iFacialMocap seems to use degrees for X, Y, Z head rotations.
                        # The order of X, Y, Z in iFacialMocap might be Pitch, Yaw, Roll respectively.
                        # Pitch (around X-axis), Yaw (around Y-axis), Roll (around Z-axis)
                        current_pose[HEAD_BONE_X] = math.degrees(P) * 1.0  # Pitch
                        current_pose[HEAD_BONE_Y] = math.degrees(Y) * 1.0  # Yaw
                        current_pose[HEAD_BONE_Z] = math.degrees(R) * 1.0  # Roll

                        # Convert Euler to Quaternion for HEAD_BONE_QUAT
                        # Ensure Euler angles are in radians for this conversion
                        current_pose[HEAD_BONE_QUAT] = euler_to_quaternion(R, P, Y)


                        # For drawing on image (optional)
                        (nose_end_point2D, jacobian) = cv2.projectPoints(
                            np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)

                        if show_video:
                            for p in image_points_2d:
                                cv2.circle(image, (int(p[0]), int(p[1])), 3, (0,0,255), -1)

                            p1 = (int(image_points_2d[0][0]), int(image_points_2d[0][1])) # Nose tip
                            p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
                            cv2.line(image, p1, p2, (255,0,0), 2)

                except Exception as e:
                    print(f"Error in solvePnP: {e}")
                    pass # Keep default head pose if PnP fails


                # --- Blendshape Calculation (Simplified Examples) ---
                # These require careful calibration and mapping from landmarks to blendshape values (0.0 - 1.0)

                # JAW_OPEN: Compare distance between upper and lower lip landmarks
                upper_lip_y = face_landmarks.landmark[FACEMESH_LIPS_INNER[5]].y # Landmark 13 (top of inner lip)
                lower_lip_y = face_landmarks.landmark[FACEMESH_LIPS_INNER[15]].y # Landmark 14 (bottom of inner lip)
                jaw_open_dist = abs(lower_lip_y - upper_lip_y)
                # Normalize: This needs calibration based on a neutral face and fully open mouth.
                # Let's assume a neutral distance of 0.01 and max of 0.1 (these are normalized landmark coords)
                jaw_open_normalized = (jaw_open_dist - 0.01) / (0.1 - 0.01)
                current_pose["jawOpen"] = np.clip(jaw_open_normalized * 2.0, 0.0, 1.0) # Scale up a bit

                # EYE_BLINK_LEFT / EYE_BLINK_RIGHT
                # Compare vertical distance between upper and lower eyelid landmarks
                # Left Eye
                left_eye_top_y = face_landmarks.landmark[FACEMESH_LEFT_EYE[7]].y # Approx. 159
                left_eye_bottom_y = face_landmarks.landmark[FACEMESH_LEFT_EYE[2]].y # Approx. 145
                left_eye_dist = abs(left_eye_bottom_y - left_eye_top_y)
                # Normalize: Assume open eye is ~0.05, closed is ~0.005
                left_blink_normalized = 1.0 - ((left_eye_dist - 0.005) / (0.05 - 0.005))
                current_pose["eyeBlinkLeft"] = np.clip(left_blink_normalized, 0.0, 1.0)

                # Right Eye
                right_eye_top_y = face_landmarks.landmark[FACEMESH_RIGHT_EYE[7]].y # Approx. 386
                right_eye_bottom_y = face_landmarks.landmark[FACEMESH_RIGHT_EYE[2]].y # Approx. 374
                right_eye_dist = abs(right_eye_bottom_y - right_eye_top_y)
                right_blink_normalized = 1.0 - ((right_eye_dist - 0.005) / (0.05 - 0.005))
                current_pose["eyeBlinkRight"] = np.clip(right_blink_normalized, 0.0, 1.0)

                # MOUTH_SMILE_LEFT / MOUTH_SMILE_RIGHT (very basic)
                # Compare horizontal position of mouth corners relative to some neutral width or nose anchor
                # This is a very crude approximation. True smile involves cheek movement and upper lip raising.
                nose_x = face_landmarks.landmark[FACEMESH_NOSE_TIP].x
                left_mouth_corner_x = face_landmarks.landmark[FACEMESH_LIPS_OUTER[0]].x # Landmark 61
                right_mouth_corner_x = face_landmarks.landmark[FACEMESH_LIPS_OUTER[10]].x # Landmark 291

                # Consider the y position of mouth corners relative to their neutral position
                # A smile usually pulls corners up and out.
                # For simplicity, just check horizontal stretch for now.
                # This needs calibration for neutral mouth width.
                # Let's assume neutral mouth width is 0.15 in normalized coords. Max stretch 0.25
                mouth_width = right_mouth_corner_x - left_mouth_corner_x
                smile_normalized = (mouth_width - 0.12) / (0.22 - 0.12) # Calibrate these values

                # A more direct way: if mouth corners go up
                left_mouth_corner_y = face_landmarks.landmark[FACEMESH_LIPS_OUTER[0]].y
                right_mouth_corner_y = face_landmarks.landmark[FACEMESH_LIPS_OUTER[10]].y
                # Compare to a neutral y, e.g., y of landmark 0 or 13
                upper_lip_center_y = face_landmarks.landmark[0].y # Landmark 0 (lip center top)

                # Simplified: if corner Y is less than lip center Y (higher on screen)
                # This needs a threshold and scaling.
                # A simple smile might just be based on how much the corners pull outwards.
                # Let's try to estimate if the corners have moved outwards from a resting pose.
                # For now, a placeholder:
                # current_pose["mouthSmileLeft"] = np.clip(smile_normalized, 0.0, 1.0)
                # current_pose["mouthSmileRight"] = np.clip(smile_normalized, 0.0, 1.0)

                # A slightly better smile: check if corners are higher than the center of the mouth horizontal line
                mouth_center_y = (face_landmarks.landmark[FACEMESH_LIPS_OUTER[0]].y + face_landmarks.landmark[FACEMESH_LIPS_OUTER[10]].y) / 2.0
                # Compare with y of point 13 (upper lip center) and 14 (lower lip center)
                lip_center_y_avg = (face_landmarks.landmark[13].y + face_landmarks.landmark[14].y) / 2.0

                smile_left_val = (lip_center_y_avg - left_mouth_corner_y) / 0.02 # Normalize by some expected max movement
                smile_right_val = (lip_center_y_avg - right_mouth_corner_y) / 0.02

                current_pose["mouthSmileLeft"] = np.clip(smile_left_val, 0.0, 1.0)
                current_pose["mouthSmileRight"] = np.clip(smile_right_val, 0.0, 1.0)


                # Update the latest pose
                self.latest_pose_data = current_pose

                if show_video:
                    # Draw the face mesh annotations on the image.
                    self.mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style())
                    self.mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style())
                    self.mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_IRISES,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_iris_connections_style())
                break # Process only the first detected face

        if show_video:
            # Flip the image horizontally for a selfie-view display.
            cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1))
            if cv2.waitKey(5) & 0xFF == 27: # ESC key
                self.stop_capture() # Allow closing window to stop capture
                return None, "STOP"


        return image, self.get_current_pose()

    def __del__(self):
        self.stop_capture()
        if self.face_mesh:
            self.face_mesh.close()

if __name__ == '__main__':
    # Example Usage
    tracker = MediaPipeWebcamInput()
    if not tracker.start_capture():
        exit()

    print("Processing frames... Press ESC in the video window to stop.")
    while True:
        processed_image, pose_data = tracker.process_frame(show_video=True)

        if pose_data == "STOP": # User pressed ESC
            print("Stopping via ESC key.")
            break
        if processed_image is None and pose_data is None : # Error or end of stream
            print("Stopping due to no image or data.")
            break

        if pose_data:
            print(f"Head Yaw: {pose_data.get(HEAD_BONE_Y, 0.0):.2f}, "
                  f"Pitch: {pose_data.get(HEAD_BONE_X, 0.0):.2f}, "
                  f"Roll: {pose_data.get(HEAD_BONE_Z, 0.0):.2f}, "
                  f"JawOpen: {pose_data.get('jawOpen', 0.0):.2f}, "
                  f"BlinkL: {pose_data.get('eyeBlinkLeft', 0.0):.2f}, "
                  f"BlinkR: {pose_data.get('eyeBlinkRight', 0.0):.2f}, "
                  f"SmileL: {pose_data.get('mouthSmileLeft', 0.0):.2f}, "
                  f"SmileR: {pose_data.get('mouthSmileRight', 0.0):.2f}")
            # print(f"Head Quat: {pose_data.get(HEAD_BONE_QUAT)}")

    tracker.stop_capture()
    print("Exited main loop.")
