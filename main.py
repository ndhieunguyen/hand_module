import cv2
from hand_module import HandDetector

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
        image, _ = hand_detector.detect_heart_sign_in_image_moving(image, index%20)
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