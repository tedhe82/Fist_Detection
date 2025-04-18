import cv2
import mediapipe as mp
import numpy as np

class FistDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.fist_count = 0
        self.last_fist_state = False

    def detect_fist(self, image):
        # Convert the BGR image to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image and detect hands
        results = self.hands.process(rgb_image)
        
        current_fist_detected = False
        
        # Check if any hands are detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                self.mp_draw.draw_landmarks(
                    image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                # Get the coordinates of the thumb and index finger
                thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
                index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                
                # Calculate the distance between thumb and index finger
                distance = np.sqrt((thumb_tip.x - index_tip.x)**2 + 
                                 (thumb_tip.y - index_tip.y)**2)
                
                # If the distance is small, it's likely a fist
                if distance < 0.1:
                    current_fist_detected = True
                    # Draw a rectangle around the fist
                    h, w, c = image.shape
                    x_min = int(min(thumb_tip.x, index_tip.x) * w)
                    y_min = int(min(thumb_tip.y, index_tip.y) * h)
                    x_max = int(max(thumb_tip.x, index_tip.x) * w)
                    y_max = int(max(thumb_tip.y, index_tip.y) * h)
                    
                    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    cv2.putText(image, "Fist Detected", (x_min, y_min - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Update fist count if a new fist is detected
        if current_fist_detected and not self.last_fist_state:
            self.fist_count += 1
        
        self.last_fist_state = current_fist_detected
        
        # Display the fist count
        cv2.putText(image, f"Fist Count: {self.fist_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return image

    def reset_count(self):
        self.fist_count = 0

def main():
    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    detector = FistDetector()
    
    while True:
        # Read frame from webcam
        success, image = cap.read()
        if not success:
            print("Failed to read from webcam")
            break
        
        # Flip the image horizontally for a later selfie-view display
        image = cv2.flip(image, 1)
        
        # Detect fists
        image = detector.detect_fist(image)
        
        # Display the image
        cv2.imshow("Fist Detection", image)
        
        # Check for key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            detector.reset_count()
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 