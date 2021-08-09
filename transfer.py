from read_xml import *
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten, AveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications import  ResNet152
from sklearn.metrics import confusion_matrix
from tensorflow.keras.optimizers import SGD

from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense

images,df = collect_data()
images,target = collect_data()
images = np.array(images)

X_train,X_test,y_train,y_test = train_test_split(images,target,test_size=0.2,shuffle=True,random_state=42)
INPUT_SHAPE = (128, 128, 3)


base_model = ResNet152(weights = 'imagenet', include_top = False, input_shape = INPUT_SHAPE)
for layer in base_model.layers:
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
cvscores = []
oos_y = []
oos_pred = []
scores = []

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
              epochs=15,callbacks =[earlystopping])
    pred = model.predict(x_test)
    oos_y.append(y_test)

    prediction=np.round(pred).reshape(y_test.shape)
    """oos_pred.append(pred)  
    viz = plot_roc_curve(model, x_test, y_test,
                         name='ROC fold',
                         alpha=0.3, lw=1, ax=ax)
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)"""

    score = metrics.accuracy_score(y_test, prediction)
    print(confusion_matrix(y_test,prediction))
    scores.append(score)

print(np.average(np.array(scores)))
model.save("simple1000_kfold_model.h5")



"""history = head_model.fit(X_train, 
                         y_train, 
                         batch_size = 64, 
                         verbose = 1, 
                         epochs = 15,      
                         validation_data=(X_test,y_test),
                         shuffle = False
                     )

y_pred = head_model.predict(X_test, batch_size=32)

prediction=np.round(y_pred).reshape(y_test.shape)
from sklearn.metrics import confusion_matrix

print(prediction)
print(y_test)
print(confusion_matrix(y_test,prediction))"""

