import cv2
import mediapipe as mp
import numpy as np
from tkinter import *
from shapely.geometry import LineString
import cvzone

SIN_225 = 0.38268343236
COS_225 = 0.92387953251

# tick=str(emoji.emojize(':red-heart:'))
class HandDetector():
    
# =========================================================================================================
    def __init__(self, static_image_mode=False, max_num_hands=10, min_detection_confidence=0.5):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=static_image_mode, 
                                    max_num_hands=max_num_hands, 
                                    min_detection_confidence=min_detection_confidence)
        self.heart = cv2.imread(r'D:\hand\heart_sign_detection\263739024_253383533446094_6241243490509535356_n.png', cv2.IMREAD_UNCHANGED)

# =========================================================================================================
    def compile(self, image):
        image_RBG = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(image_RBG)
# =========================================================================================================
    def _is_opened_finger(self, finger, ratio=0.8):
        head_to_tail_length = np.linalg.norm(finger[0] - finger[3])
        sum_length = sum([np.linalg.norm(finger[i] - finger[i+1]) for i in range(3)])

        return sum_length*ratio < head_to_tail_length

    def _is_crossed_fingers(self, finger1, finger2):
        line1 = LineString([finger1[0], finger1[3]])
        line2 = LineString([finger2[0], finger2[3]])

        return line1.intersects(line2)

    def _check_heart_sign(self, landmark_list):
        thumb = landmark_list[1:5]
        index = landmark_list[5:9]
        middle = landmark_list[9:13]
        ring = landmark_list[13:17]
        pinky = landmark_list[17:21]

        return self._is_opened_finger(thumb) and self._is_opened_finger(index) and not self._is_opened_finger(middle) and not self._is_opened_finger(ring) and not self._is_opened_finger(pinky) and self._is_crossed_fingers(thumb, index)

    def _check_fuck_sign(self, landmark_list):
        # thumb = landmark_list[1:5]
        index = landmark_list[5:9]
        middle = landmark_list[9:13]
        ring = landmark_list[13:17]
        pinky = landmark_list[17:21]

        return not self._is_opened_finger(index) and self._is_opened_finger(middle) and not self._is_opened_finger(ring) and not self._is_opened_finger(pinky)

    def _check_spider_sign(self, landmark_list):
        thumb = landmark_list[1:5]
        index = landmark_list[5:9]
        middle = landmark_list[9:13]
        ring = landmark_list[13:17]
        pinky = landmark_list[17:21]

        return self._is_opened_finger(thumb) and self._is_opened_finger(index) and not self._is_opened_finger(middle) and not self._is_opened_finger(ring) and self._is_opened_finger(pinky)
        
# =========================================================================================================
    def _create_landmark_list_of_a_hand(self, hand, image_height, image_width):
        landmark_list = []
        for landmark in hand.landmark:
            x, y = landmark.x * image_width, landmark.y * image_height
            landmark_list.append([x, y])
        return np.array(landmark_list)

    def _create_bounding_box_of_a_hand(self, landmark_list, image_height, image_width, padding_rate=0.1):
        left = np.min(landmark_list[:, 0])
        right = np.max(landmark_list[:, 0])
        up = np.min(landmark_list[:, 1])
        down = np.max(landmark_list[:, 1])

        left = max(0, int(left-image_width*padding_rate))
        right = min(image_width, int(right+image_width*padding_rate))
        up = max(0, int(up-image_height*padding_rate))
        down = min(image_height, int(down+image_height*padding_rate))

        return (left, up, right, down)

    def _draw_heart_at_thumb(self, image, landmark_list):
        tip_thumb_x, tip_thumb_y = landmark_list[4]
        cv2.putText(image, '<3', (int(tip_thumb_x), int(tip_thumb_y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 5)
        return image
# =========================================================================================================
    def create_hand_bone_list_from_image(self, image, draw=True):
        list_landmark_list = []
        image_height, image_width, _ = image.shape
        if self.results.multi_hand_landmarks:
            for hand in self.results.multi_hand_landmarks:
                landmark_list = self._create_landmark_list_of_a_hand(hand, image_height, image_width)
                list_landmark_list.append(landmark_list)
                if draw:
                    self.mp_drawing.draw_landmarks(image, hand, self.mp_hands.HAND_CONNECTIONS)
            return (image, list_landmark_list)
        return (image, None)

    def create_bounding_box_list_from_image(self, image, draw=True):
        bounding_box_list = []
        image_height, image_width, _ = image.shape
        if self.results.multi_hand_landmarks:
            for hand in self.results.multi_hand_landmarks:
                landmark_list = self._create_landmark_list_of_a_hand(hand, image_height, image_width)
                left, up, right, down = self._create_bounding_box_of_a_hand(landmark_list, image_height, image_width)
                bounding_box_list.append((left, up, right, down))
                if draw:
                    cv2.rectangle(image, (left, up), (right, down), (0, 255, 0), 2)
            return (image, bounding_box_list)
        return (image, None)

    def detect_heart_sign_in_image(self, image, draw=True):
        flag = False
        image_height, image_width, _ = image.shape
        if self.results.multi_hand_landmarks:
            for hand in self.results.multi_hand_landmarks:
                landmark_list = self._create_landmark_list_of_a_hand(hand, image_height, image_width)
                if self._check_heart_sign(landmark_list):
                    flag = True
                    cv2.putText(image, '<3 <3 <3 <3 <3', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 5)
                    if draw:
                        left, up, right, down = self._create_bounding_box_of_a_hand(landmark_list, image_height, image_width)
                        cv2.rectangle(image, (left, up), (right, down), (0, 255, 0), 2)

        return (image, flag)


    def detect_fuck_sign_in_image(self, image, draw=True):
        flag = False
        image_height, image_width, _ = image.shape
        if self.results.multi_hand_landmarks:
            for hand in self.results.multi_hand_landmarks:
                landmark_list = self._create_landmark_list_of_a_hand(hand, image_height, image_width)
                if self._check_fuck_sign(landmark_list):
                    flag = True
                    cv2.putText(image, 'Fuckkkkk Youuuu', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 5)
                    if draw:
                        left, up, right, down = self._create_bounding_box_of_a_hand(landmark_list, image_height, image_width)
                        cv2.rectangle(image, (left, up), (right, down), (0, 255, 0), 2)

        return (image, flag)


    def detect_spider_sign_in_image(self, image, draw=True):
        flag = False
        image_height, image_width, _ = image.shape
        if self.results.multi_hand_landmarks:
            for hand in self.results.multi_hand_landmarks:
                landmark_list = self._create_landmark_list_of_a_hand(hand, image_height, image_width)
                if self._check_spider_sign(landmark_list):
                    flag = True
                    cv2.putText(image, 'Great power...', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.putText(image, '... comes with great responsibility', (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    if draw:
                        left, up, right, down = self._create_bounding_box_of_a_hand(landmark_list, image_height, image_width)
                        cv2.rectangle(image, (left, up), (right, down), (0, 255, 0), 2)

        return (image, flag)

    def detect_heart_sign_in_image_moving(self, image, index):
        flag = False
        image_height, image_width, _ = image.shape
        if self.results.multi_hand_landmarks:
            for hand in self.results.multi_hand_landmarks:
                landmark_list = self._create_landmark_list_of_a_hand(hand, image_height, image_width)
                if self._check_heart_sign(landmark_list):
                    flag = True
                    # cv2.putText(image, '<3 <3 <3 <3 <3', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 5)
                    left, up, right, down = self._create_bounding_box_of_a_hand(landmark_list, image_height, image_width)
                    cx, cy = left+(right-left)//2, up+(down-up)//2
                    radius = int(np.sqrt((left-right)**2 + (down-up)**2)/3)
                    list_point = np.array([
                        [cx, cy-radius],
                        [int(cx+radius*SIN_225), int(cy-radius*COS_225)],
                        [int(cx+radius/np.sqrt(2)), int(cy-radius/np.sqrt(2))],
                        [int(cx+radius*COS_225), int(cy-radius*SIN_225)],

                        [cx+radius, cy],
                        [int(cx+radius*COS_225), int(cy+radius*SIN_225)],
                        [int(cx+radius/np.sqrt(2)), int(cy+radius/np.sqrt(2))],
                        [int(cx+radius*SIN_225), int(cy+radius*COS_225)],

                        [cx, cy+radius],
                        [int(cx-radius*SIN_225), int(cy+radius*COS_225)],
                        [int(cx-radius/np.sqrt(2)), int(cy+radius/np.sqrt(2))],
                        [int(cx-radius*COS_225), int(cy+radius*SIN_225)],

                        [cx-radius, cy],
                        [int(cx-radius*COS_225), int(cy-radius*SIN_225)],
                        [int(cx-radius/np.sqrt(2)), int(cy-radius/np.sqrt(2))],
                        [int(cx-radius*SIN_225), int(cy-radius*COS_225)],
                    ])

                    new_size = int(radius/3)
                    heart = cv2.resize(self.heart, (new_size, new_size))

                    image = cvzone.overlayPNG(image, heart, pos=list_point[(index)%16])
                    image = cvzone.overlayPNG(image, heart, pos=list_point[(index+4)%16])
                    image = cvzone.overlayPNG(image, heart, pos=list_point[(index+8)%16])
                    image = cvzone.overlayPNG(image, heart, pos=list_point[(index+12)%16])


                    # image = self._draw_heart_at_thumb(image, landmark_list)

        return (image, flag)


cap = cv2.VideoCapture(0)
hand_detector = HandDetector()
index=0

while True:
    success, image = cap.read()
    if not success:
        print('not')
        continue

    image = cv2.flip(image, 1)
    hand_detector.compile(image)
    # image, _ = hand_detector.create_hand_bone_list_from_image(image)
    # image, _ = hand_detector.detect_heart_sign_in_image(image)
    # image, _ = hand_detector.detect_fuck_sign_in_image(image)
    # image, _ = hand_detector.detect_spider_sign_in_image(image)
    try:
        image, _ = hand_detector.detect_heart_sign_in_image_moving(image, index%16)
    except:
        continue
    index += 1

    # # status = hand_detector.check_heart_sign_in_image(image)

    # # if status:
    # #     start_time = time.time()
    # #     end_time = time.time()

    cv2.imshow('Hand detector', image)
    if cv2.waitKey(5) & 0xFF == 27:
        break