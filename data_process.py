#data_load.py
import keras_ocr
import json
import os
from PIL import Image
import matplotlib.pyplot as plt
import setting
import sklearn.model_selection
import imgaug
import tensorflow as tf
import numpy as np

tf.compat.v1.disable_eager_execution()
tf.compat.v1.experimental.output_all_intermediates(True)
# 이미지와 라벨링 데이터를 로드하는 함수
def load_data(image_dir : str, json_dir : str) -> list:
    """
    이미지의 경로와 Json파일의 경로를 입력받습니다.
    
    이후 경로에 있는 파일들을 읽어 하나의 데이터로 출력합니다

    Keyword arguments:
    
    image_dir : str / 이미지 주소
    json_dir : str / Json파일 주소
    Return: list / Json파일내용과 이미지경로들이 합쳐진 데이터 리스트
    
    """  
    
    data = []
    for json_file in os.listdir(json_dir):
        
        if json_file == '.DS_Store':  # .DS_Store 파일은 건너뛰기
            continue
        json_path = os.path.join(json_dir, json_file)
        
        try:
            with open(json_path, 'r', encoding='utf-8') as file:
                json_data = json.load(file)
        except UnicodeDecodeError:
            try:
                with open(json_path, 'r', encoding='ISO-8859-1') as file:
                    json_data = json.load(file)
            except Exception as e:
                print(f"Error reading file {json_path}: {e}")
                continue

        image_file = json_data['meta']['file_name']
        image_path = os.path.join(image_dir, image_file)
        annotations = json_data['annotations']
        for annotation in annotations:
            text = annotation['ocr']['text']
            x = annotation['ocr']['x']
            y = annotation['ocr']['y']
            width = annotation['ocr']['width']
            height = annotation['ocr']['height']
            data.append((image_path, text, (x, y, x + width, y + height)))
    return data



hangul_alphabet ="!\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~가각간갇갈감갑값강갖같갚갛개객걀거걱건걷걸검겁것겉게겨격겪견결겹경곁계고곡곤곧골곰곱곳공과관광괜괴굉교구국군굳굴굵굶굽궁권귀규균그극근글긁금급긋긍기긴길김깅깊까깎깐깔깜깝깥깨꺼꺾껍껏껑께껴꼬꼭꼴꼼꼽꽂꽃꽉꽤꾸꿀꿈뀌끄끈끊끌끓끔끗끝끼낌나낙낚난날낡남납낫낭낮낯낱낳내냄냉냐냥너넉널넓넘넣네넥넷녀녁년념녕노녹논놀놈농높놓놔뇌뇨누눈눕뉘뉴늄느늑는늘늙능늦늬니닐님다닥닦단닫달닭닮담답닷당닿대댁댐더덕던덜덤덥덧덩덮데델도독돈돌돕동돼되된두둑둘둠둡둥뒤뒷드득든듣들듬듭듯등디딩딪따딱딴딸땀땅때땜떠떡떤떨떻떼또똑뚜뚫뚱뛰뜨뜩뜯뜰뜻띄라락란람랍랑랗래랜램랫략량러럭런럴럼럽럿렁렇레렉렌려력련렬렵령례로록론롬롭롯료루룩룹룻뤄류륙률륭르른름릇릎리릭린림립릿마막만많말맑맘맙맛망맞맡맣매맥맨맵맺머먹먼멀멈멋멍멎메멘멩며면멸명몇모목몰몸몹못몽묘무묵묶문묻물뭄뭇뭐뭣므미민믿밀밉밌및밑바박밖반받발밝밟밤밥방밭배백뱀뱃뱉버번벌범법벗베벤벼벽변별볍병볕보복볶본볼봄봇봉뵈뵙부북분불붉붐붓붕붙뷰브블비빌빗빚빛빠빨빵빼뺨뻐뻔뻗뼈뽑뿌뿐쁘쁨사삭산살삶삼상새색샌생서석섞선설섬섭섯성세센셈셋션소속손솔솜솟송솥쇄쇠쇼수숙순술숨숫숲쉬쉽슈스슨슬슴습슷승시식신싣실싫심십싱싶싸싹쌀쌍쌓써썩썰썹쎄쏘쏟쑤쓰쓸씀씌씨씩씬씹씻아악안앉않알앓암압앗앙앞애액야약얇양얗얘어억언얹얻얼엄업없엇엉엌엎에엔엘여역연열엷염엽엿영옆예옛오옥온올옮옳옷와완왕왜왠외왼요욕용우욱운울움웃웅워원월웨웬위윗유육율으윽은을음응의이익인일읽잃임입잇있잊잎자작잔잖잘잠잡장잦재쟁저적전절젊점접젓정젖제젠젯져조족존졸좀좁종좋좌죄주죽준줄줌줍중쥐즈즉즌즐즘증지직진질짐집짓징짙짚짜짝짧째쨌쩌쩍쩐쪽쫓쭈쭉찌찍찢차착찬찮찰참창찾채책챔챙처척천철첫청체쳐초촉촌총촬최추축춘출춤춥춧충취츠측츰층치칙친칠침칭카칸칼캐캠커컨컬컴컵컷켓켜코콜콤콩쾌쿠퀴크큰클큼키킬타탁탄탈탑탓탕태택탤터턱털텅테텍텔템토톤톱통퇴투툼퉁튀튜트특튼튿틀틈티틱팀팅파팎판팔패팩팬퍼퍽페펴편펼평폐포폭표푸푹풀품풍퓨프플픔피픽필핏핑하학한할함합항해핵핸햄햇행향허헌험헤헬혀현혈협형혜호혹혼홀홍화확환활황회획횟효후훈훌훔훨휘휴흉흐흑흔흘흙흡흥흩희흰히힘"
recognizer = keras_ocr.recognition.Recognizer(alphabet=hangul_alphabet, weights=None)
recognizer.compile()
# 이미지 폴더와 JSON 폴더 경로 설정

# 기존 코드의 데이터 로딩
training_data = load_data(setting.image_dir, setting.json_dir)

print(training_data)
# keras_ocr 형식에 맞게 데이터를 변환합니다.
# 각 항목은 (이미지_파일_경로, 단어_박스, 단어) 튜플의 형태로 구성됩니다.
# 박스 데이터 변환
converted_training_data = [
    (image_path, np.array([[x, y], [x + width, y], [x + width, y + height], [x, y + height]], dtype='float32'), text)
    for image_path, text, (x, y, width, height) in training_data
]
print(converted_training_data)

# 데이터를 훈련 및 검증 세트로 분할
train_labels, validation_labels = sklearn.model_selection.train_test_split(
    converted_training_data, test_size=0.2, random_state=42)

# 데이터 생성기 및 배치 생성기 설정
batch_size = 8
augmenter = imgaug.augmenters.Sequential([
    imgaug.augmenters.GammaContrast(gamma=(0.25, 3.0)),
])

# keras_ocr 데이터 생성기
(training_image_gen, training_steps), (validation_image_gen, validation_steps) = [
    (
        keras_ocr.datasets.get_recognizer_image_generator(
            labels=labels,
            height=recognizer.model.input_shape[1],
            width=recognizer.model.input_shape[2],
            alphabet=recognizer.alphabet,
            augmenter=augmenter if is_augment else None
        ),
        len(labels) // batch_size
    ) for labels, is_augment in [(train_labels, True), (validation_labels, False)]
]

# 배치 생성기 설정
training_gen, validation_gen = [
    recognizer.get_batch_generator(
        image_generator=image_generator,
        batch_size=batch_size
    )
    for image_generator in [training_image_gen, validation_image_gen]
]

batch_images, batch_texts = next(training_gen)

# 배치 데이터의 형식 확인
print("Batch images shape:", batch_images)
print("Batch texts:", batch_texts)


image, text = next(training_image_gen)
print('text:', text)
plt.imshow(image)


callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, restore_best_weights=False),
    tf.keras.callbacks.ModelCheckpoint('recognizer_borndigital.h5', monitor='val_loss', save_best_only=True),
    tf.keras.callbacks.CSVLogger('recognizer_borndigital.csv')
]
recognizer.training_model.fit(
    x = training_gen,
    steps_per_epoch=training_steps,
    validation_steps=validation_steps,
    validation_data=validation_gen,
    callbacks=callbacks,
    epochs=1000,
)