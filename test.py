import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from read_xml import *
import cv2

model = load_model("vggtraintop_kfold_model.h5")
model.summary()
s = 0
for i  in range(33):
    images = []
    image = cv2.imread("data/maligns/"+str(i+1)+".png")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image,(128,128))
    image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 15)
    images.append(image)
    print(np.floor(model.predict(np.array(images))))
    s = s + np.floor(model.predict(np.array(images)))

print(s)

"""image = cv2.imread("data/maligns/1.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image,(128,128))
image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 15)
images.append(image)

print(np.array(images).shape)
print(model.predict(np.array(images)))"""

