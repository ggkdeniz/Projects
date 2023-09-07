import cv2
import mediapipe as mp


def main():
    # Initialize OpenCV and MediaPipe Hands
    cap = cv2.VideoCapture(0)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2)

    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=2)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the image with MediaPipe Hands
        results_hands = hands.process(rgb_frame)

        if results_hands.multi_hand_landmarks:
            for hand_landmarks in results_hands.multi_hand_landmarks:
                # Draw hand landmarks on the frame
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                for point in hand_landmarks.landmark:
                    x, y = int(point.x * frame.shape[1]), int(point.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

        # Process the image with MediaPipe Face Detection
        results_face = face_detection.process(rgb_frame)

        if results_face.detections:
            for detection in results_face.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                    int(bboxC.width * iw), int(bboxC.height * ih)
                cv2.rectangle(frame, bbox, (0, 255, 0), 2)
                cv2.putText(frame, f'Face: {int(detection.score[0] * 100)}%', (bbox[0], bbox[1] - 20),
                            cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

                # Convert face region to RGB and process with FaceMesh
                face_rgb = cv2.cvtColor(frame[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]],
                                        cv2.COLOR_BGR2RGB)
                face_results = face_mesh.process(face_rgb)

                if face_results.multi_face_landmarks:
                    for face_landmarks in face_results.multi_face_landmarks:
                        # Draw face landmarks on the face region
                        landmarks = face_landmarks.landmark
                        # Convert landmarks to image coordinates
                        ih, iw, _ = frame[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]].shape
                        for landmark in landmarks:
                            x, y = int(landmark.x * iw), int(landmark.y * ih)
                            cv2.circle(frame[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]], (x, y), 2,
                                       (0, 255, 0), -1)

        cv2.imshow('Hand and Face Detection', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    mp_drawing = mp.solutions.drawing_utils
    main()
