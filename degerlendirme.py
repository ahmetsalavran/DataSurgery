import numpy as np
import matplotlib.pyplot as plt
a = np.load("confusions_vgg.npy")

tn = []
tp = []
fn = []
fp = []
accs = []
precs = []
recall= []
f_score=[]


for i in range(10):
    tn.append(a[i][0][0])
    tp.append(a[i][1][1])
    fn.append(a[i][1][0])
    fp.append(a[i][0][1])
    accs.append((tn[i]+tp[i])/(tn[i]+tp[i]+fn[i]+fp[i]))
    precs.append(tp[i]/(tp[i]+fp[i]))
    recall.append(tp[i]/(tp[i]+fn[i]))
    f_score.append(2*((precs[i]*recall[i])/(precs[i]+recall[i])))

print("Accuracy:",np.array(accs).mean())#accuracy neredeyse %90
print("Precision:",np.array(precs).mean())
"""bu deger olması gereknden biraz yüksek bunun sebebide benign
verilerindeki azlık sonucu her ne kadar yükseltmeye çalışsakta malign verilerinin baskınlığından ötürü
o sınıf baskın çıkıyor.
"""
print("Recall:",np.array(recall).mean())
print("F_score:",np.array(f_score).mean())

from sklearn.metrics import plot_confusion_matrix
x1 = 0
x2 = 0
x3 = 0
x4 = 0
for i in range(10):
    x1 = x1 + a[i][0][0]
    x2 = x2 + a[i][0][1]
    x3 = x3 + a[i][1][0]
    x4 = x4 + a[i][1][1]
conf_matrix = np.round(np.array([[x1,x2],[x3,x4]])/10)

#print(conf_matrix)

fig, ax = plt.subplots(figsize=(7.5, 7.5))
ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
 
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()

