from read_xml import *
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import  ResNet50
from sklearn.metrics import confusion_matrix
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

images,df = collect_data()
images,target = collect_data()
images = np.array(images)

confs = []
scores = []

INPUT_SHAPE = (224, 224, 3)

train_aug = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

base_model = ResNet50(weights = 'imagenet', include_top = False, input_shape = INPUT_SHAPE)

for layer in base_model.layers[-26:]:
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

    model.fit(train_aug.flow(x_train,y_train),
              validation_data=(x_test,y_test),verbose=1,
              epochs=1,callbacks =[earlystopping])
    pred = model.predict(x_test)

    prediction=np.round(pred).reshape(y_test.shape)
    score = metrics.accuracy_score(y_test, prediction)
    scores.append(score)
    confs.append(confusion_matrix(y_test,prediction))
    print(confusion_matrix(y_test,prediction))


"""model.save("resnet_aug_1000_kfold_model.h5")
np.save("confusions.npy",np.array(confs))
np.save("scores.npy",np.array(scores))"""


