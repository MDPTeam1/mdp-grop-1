from keras.models import load_model  # keras를 사용하기 위해선 TensorFlow가 필요
import cv2  # Install opencv-python
import numpy as np

# 과학적 표기법을 알기 쉽게 변환 (ex: 1.e-10 -> 0.0000000001 )
np.set_printoptions(suppress=True)

# 모델 불러오기
model = load_model("keras_Model.h5", compile=False)

# 라벨 불러오기
class_names = open("labels.txt", "r").readlines()

# 컴퓨터의 기본 카메라와 상호 작용하고 프레임을 캡처하는 데 사용할 수 있는 비디오 캡처 개체를 생성 (기본카메라는 0)
camera = cv2.VideoCapture(0)

while True:
    # 웹캠의 이미지를 가져옴
    ret, image = camera.read()

    # 원래 이미지의 크기를 (높이 224, 너비 224) 픽셀로 조정
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    # 웹캠 이미지를 표시
    cv2.imshow("Webcam Image", image)

    # 이미지를 넘파이 배열로 변경
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

    # 이미지 배열을 정규화하여-1부터 1까지의 범위로 변환
    image = (image / 127.5) - 1

    # 모델 예측
    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # 예측 및 정확도 인쇄
    print("초콜릿:", class_name[2:], end="")
    print("정확도:", str(np.round(confidence_score * 100))[:-2], "%")

    # 키보드 입력 값을 변수에 저장
    keyboard_input = cv2.waitKey(1)

    # ESC버튼을 누르면 While문 나감
    if keyboard_input == 27:
        break

camera.release()
cv2.destroyAllWindows()
