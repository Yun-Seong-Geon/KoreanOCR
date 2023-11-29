import matplotlib.pyplot as plt
import keras_ocr.tools
from keras.models import load_model
import tensorflow as tf

# 저장된 모델을 불러오는 함수
def load_finetuned_model(model_path):
    """ Fine-tuned 모델 불러오기 """
    # Custom 객체가 필요한 경우, custom_objects 매개변수에 전달
    # 예: custom_objects={'CustomLayer': CustomLayer}
    model = load_model(model_path, compile=False)
    return model

def load_and_preprocess_image(image_path):
    """ 이미지를 로드하고 전처리 """
    image = keras_ocr.tools.read(image_path)  # 이미지 로드
    # 추가 전처리가 필요하다면 여기에 추가
    return image

def predict_and_visualize(image_path, model):
    """ 이미지에서 텍스트 인식 후 결과 시각화 """
    image = load_and_preprocess_image(image_path)

    # recognizer 객체 생성 및 모델 할당
    recognizer = keras_ocr.recognition.Recognizer()
    recognizer.model = model

    # 이미지를 리스트로 래핑하여 모델에 전달
    prediction_groups = recognizer.recognize(image)

    print(prediction_groups)
    # 결과 시각화
    fig, ax = plt.subplots()
    if prediction_groups[0]:  # 예측값이 존재하는 경우에만 시각화
        # drawAnnotations 함수에 맞게 예측 결과 형식 변환
        formatted_predictions = [(word, box) for word, box in zip(prediction_groups[0][0], prediction_groups[0][1])]
        keras_ocr.tools.drawAnnotations(image=image, predictions=formatted_predictions, ax=ax)
    else:
        ax.imshow(image)
        ax.set_title("No predictions")
    plt.show()
# 저장된 모델 경로
model_path = 'recognizer_borndigital.h5'

# Fine-tuned 모델 불러오기
finetuned_model = load_finetuned_model(model_path)

# 테스트 이미지 경로 설정
test_image_path = 'Training/01.원천데이터/TS_13.제주/JJ_BF01_M0001_1952745_1.jpg'

# 테스트 이미지에 대한 예측 및 시각화
predict_and_visualize(test_image_path, finetuned_model)

### keras-ocr의 경우 영어만을 인식하는 오픈소스 인공지능이므로 언어를 한국어 몇개를 넣는 다고 해서 제대로 학습되지않음 
### 필요할 경우 easyocr을 사용하고 이를 naver의 finetuning 프로그램을 사용해서 추가학습을 해야한다고 판단.
### 