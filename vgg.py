import matplotlib.pyplot as plt
import pandas as pd
from read_xml import *
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Flatten,Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
from sklearn.metrics import confusion_matrix
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense


images,target = collect_data()
images = np.array(images)

X_train,X_test,y_train,y_test = train_test_split(images,target,test_size=0.2,shuffle=True,random_state=42)

INPUT_SHAPE = (150, 150, 3)

base_model = VGG16(weights = 'imagenet', include_top = False, input_shape = INPUT_SHAPE)
for layer in base_model.layers[:25]:
    layer.trainable = False

x = Flatten()(base_model.output)
x = Dense(1000)(x)
x = Activation("relu")(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation = 'sigmoid')(x)
model = Model(inputs = base_model.input, outputs = predictions)
model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])

from sklearn.model_selection import KFold
kfold = KFold(n_splits=10, shuffle=True, random_state=42)

confs = []
scores = []
preds = []
actual = []

from sklearn import metrics
print(model.summary()) 

from tensorflow.keras.callbacks import EarlyStopping
earlystopping = EarlyStopping(monitor ="val_loss", 
                                        mode ="min", patience = 5, 
                                        restore_best_weights = True)

for train, test in kfold.split(images, target):
    x_train = images[train]
    y_train = target[train]
    x_test = images[test]
    y_test = target[test]

    model.fit(x_train,y_train,
              validation_data=(x_test,y_test),verbose=1,
              epochs=40,callbacks =[earlystopping])
    pred = model.predict(x_test)
    prediction=np.round(pred).reshape(y_test.shape)
    score = metrics.accuracy_score(y_test, prediction)
    print(confusion_matrix(y_test,prediction))
    confs.append(confusion_matrix(y_test,prediction))
    actual.append(y_test)
    preds.append(prediction)
    scores.append(score)
    
print(np.average(np.array(scores)))
model.save("vggtraintop_kfold_model.h5")
np.save("confusions_vgg.npy",np.array(confs))
np.save("scores_vgg.npy",np.array(scores))
np.save("prediction_vgg.npy",np.array(preds))
np.save("actual_vgg.npy",np.array(actual))
