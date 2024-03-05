
# 필요한 경우 Pillow 라이브러리 설치 (이미지 처리를 위해)
# !pip install Pillow
# 이미지 처리를 위한 Pillow 라이브러리는 대부분의 경우 기본적으로 설치되어 있다.

# 필요한 경우 keras 라이브러리 설치 (딥러닝 모델을 위해)
# !pip install keras
# 딥러닝 모델을 사용하기 위해 keras 라이브러리를 설치해야 할 수 있다.

# 필요한 경우 tensorflow 라이브러리 설치 (딥러닝 모델을 위해)
# !pip install tensorflow
# 딥러닝 모델을 사용하기 위해 tensorflow 라이브러리를 설치해야 할 수 있다.
from PIL import Image
import numpy as np
import pandas as pd
from FeatureExtractor import fe

def dataload():
    data=pd.read_csv('pokemon.csv')

    features = []
    img_paths = []

    # Save Image Feature Vector with Database Images
    for i in data["Name"]:
        try:
            image_path = f"C:/Users/chunjae/Documents/images/images/{i}.png"
            img_paths.append(image_path)

            # Extract Features
            feature = fe.extract(img=Image.open(image_path))

            features.append(feature)

            # Save the Numpy array (.npy) on designated path
            feature_path = "./images/m" + str(i) + ".npy"
            np.save(feature_path, feature)
        except Exception as e:
            print('예외가 발생했습니다.', e)

    return features,img_paths