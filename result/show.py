import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from FeatureExtractor import fe
from data_load import dataload
#from data_load import img_paths

def show():
    #다른그림 넣어도 되고 포문돌려도 되고
    img = Image.open("C:/Users/chunjae/Documents/images/images/abomasnow.png")
    # Extract its features
    query = fe.extract(img)

    features,img_paths = dataload()
    # img_paths = dataload()
    # Calculate the similarity (distance) between images
    dists = np.linalg.norm(features - query, axis=1)

    # Extract 30 images that have lowest distance
    ids = np.argsort(dists)[:30]

    scores = [(dists[id], img_paths[id], id) for id in ids]
    # Visualize the result
    axes=[]
    fig=plt.figure(figsize=(8,8))
    for a in range(5*6):
        score = scores[a]
        axes.append(fig.add_subplot(5, 6, a+1))
        subplot_title=str(round(score[0],2)) + "/m" + str(score[2]+1)
        axes[-1].set_title(subplot_title)  
        plt.axis('off')
        plt.imshow(Image.open(score[1]))
    fig.tight_layout()
    plt.show()
