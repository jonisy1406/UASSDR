#import library
import pandas as pd
import os
import tensorflow as tf
import pathlib
import numpy as np
from keras.models import load_model
from sklearn.metrics import multilabel_confusion_matrix

#memasukan model
Mod=input('masukan model: ')
model = load_model(Mod)

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

#data tes(predik)
ket = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
       1,1,1,1,1,1,1,1,
       2,2,2,2,2,2,2,2,2,2,2,2,
       3,3,3,3,3,3,3,3,3,3,3
       ]

#direktori data test(predik)
test_dir = input('masukan direktori file test: ')
test_dir = pathlib.Path(test_dir)

#proses prediksi
kawung_score=[]
megamendung_score=[]
parang_score=[]
truntum_score=[]
pred=[]
Test=[]
img_height = 150
img_width = 150
for i in os.listdir(test_dir):
  name = f"{test_dir}\\{i}"
  img = tf.keras.utils.load_img(
    name, target_size=(img_height, img_width)
    )
  img_array = tf.keras.utils.img_to_array(img)
  img_array = tf.expand_dims(img_array, 0) # Create a batch)
  predictions = model.predict(img_array)
  score = tf.nn.softmax(predictions[0])
  kawung_score.append(float(score[0])*100)
  megamendung_score.append(float(score[1])*100)
  parang_score.append(float(score[2])*100)
  truntum_score.append(float(score[3])*100)
  pred.append(np.argmax(score))

hasil=pd.DataFrame({'Kawung':kawung_score,'Mega Mendung':megamendung_score,'Parang':parang_score,'Truntum':truntum_score})
hasil.insert(0, "file test", np.arange(1, 51,1).tolist(), True)
tag_name = ['Kawung','Mega Mendung','Parang', 'Truntum']
matrix_confusion= multilabel_confusion_matrix(pred, ket)

True_N = matrix_confusion[:, 0, 0]
True_P = matrix_confusion[:, 1, 1]
False_N = matrix_confusion[:, 1, 0]
False_P = matrix_confusion[:, 0, 1]
Kinerja = pd.DataFrame({"True-Positif":True_P,"True-Negatif":True_N,"False-Positif":False_P,"False-Negatif":False_N}, index=tag_name).T
Kinerja['Total']=[True_P.sum(),True_N.sum(),False_P.sum(),False_N.sum()]
Kinerja.index.name = 'Kinerja'

TP = Kinerja['Total']['True-Positif']
TN = Kinerja['Total']['True-Negatif']
FP = Kinerja['Total']['False-Positif']
FN = Kinerja['Total']['False-Negatif']

# Overall accuracy
ACC = (TP+TN)/(TP+FP+FN+TN)
# Sensitivity, hit rate, recall, or true positive rate
# TPR = TP/(TP+FN)
# # Specificity or true negative rate
TNR = TN/(TN+FP) 
# Precision or positive predictive value
PPV = TP/(TP+FP)
# # Negative predictive value
# NPV = TN/(TN+FN)
# # Fall out or false positive rate
# FPR = FP/(FP+TN)
# # False negative rate
# FNR = FN/(TP+FN)
# # False discovery rate
# FDR = FP/(TP+FP)
# F1-Score
F1_Score= 2*TP/(2*TP +FP +FN)


#hasil
print('''
PENGENALAN CORAK BATIK
Sampel: a
Metode: Standar (TensorFlow)
''')
print('-'*6)
print(hasil)
print('-'*6)
print(Kinerja)
print('-'*6)
print("Accuracy   ",str(round(ACC*100,2))+'%')
print("Precision  ",str(round(PPV*100,2))+'%')
print("Selectivity",str(round(TNR*100,2))+'%')
print("F1-Score   ",str(round(F1_Score*100,2))+'%')
