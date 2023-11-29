import cv2

image_path = "Training/01.원천데이터/TS_13.제주/JJ_BF01_M0001_1952745_1.jpg"
image = cv2.imread(image_path)
if image is None:
    print("이미지를 로드하는 데 실패했습니다.")
else:
    cv2.imshow("Loaded Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
