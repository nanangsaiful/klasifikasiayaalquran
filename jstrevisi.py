
import nltk
nltk.download('stopwords')
nltk.download('punkt')
# -*- codi ng: utf-8 -*-
"""
Created on Tue Jun 05 18:26:43 2018

@author: nanang saiful
"""
from keras import callbacks
from keras import preprocessing
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential,Model,load_model
from keras.layers import Dense,Dropout

from keras.callbacks import ModelCheckpoint,EarlyStopping
import pandas as pd
import re
import numpy as np
from sklearn.model_selection import KFold,train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import hamming_loss,classification_report,f1_score,zero_one_loss
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from imblearn.over_sampling import SMOTE,RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import pickle 
import time
start = time.time()

from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
class Metrics(Callback):
    def on_train_begin(self, logs={}):
     self.val_f1s = []
     self.val_recalls = []
     self.val_precisions = []
     
    def on_epoch_end(self, epoch, logs={}):
     val_predict = (np.asarray(self.model.predict(self.model.validation_data[0]))).round()
     val_targ = self.model.validation_data[1]
     _val_f1 = f1_score(val_targ, val_predict)
     _val_recall = recall_score(val_targ, val_predict)
     _val_precision = precision_score(val_targ, val_predict)
     self.val_f1s.append(_val_f1)
     self.val_recalls.append(_val_recall)
     self.val_precisions.append(_val_precision)
     print (" — val_f1: %f — val_precision: %f — val_recall %f" %(_val_f1, _val_precision, _val_recall))
     return
 
metrics = Metrics()

dokumen=pd.ExcelFile('datasetalquran.xlsx')
df=dokumen.parse('Sheet1')
kelas={}



print ("preposesing")
#lower
df['Terjemahan']=df['Terjemahan'].apply(lambda x: x.lower())
#mengambil huruf saja
df['Terjemahan']=df['Terjemahan'].apply(lambda x: re.sub('[^a-zA-z\s]','',x))
#stemming
ps=PorterStemmer()
df['Terjemahan']=df['Terjemahan'].apply(lambda x: " ".join([ps.stem(y) for y in x.split(" ")]))

#stopword removal
stop_word= set (stopwords.words("english"))
stop_word.remove('did')
df['Terjemahan']=df['Terjemahan'].apply(lambda x: " ".join([item for item in x.split(" ") if item not in stop_word]))

kf=KFold(n_splits=5)
k=0;
#indextukar=np.random.permutation(len(df))
#np.save('hasilindextukar',[indextukar,kf])
#df=df.iloc[indextukar]
tohammingloss=[]
hm=[]
report=[]
f1mi=[]
f1ma=[]
loss01=[]
for train, test in kf.split(df):
    akur=[]
    allpredik=pd.DataFrame()
    train_data = np.array(df)[train]
    test_data = np.array(df)[test]
    X_train=pd.Series(np.resize(train_data[:,[3]],(train_data[:,[0]].size,)), dtype='str')
    X_test=pd.Series(np.resize(test_data[:,[3]],(test_data[:,[0]].size,)), dtype='str')
    allytest=test_data[:,[4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]]
    tokenizer = Tokenizer( split=' ')
    tokenizer.fit_on_texts(X_train)
    X_train1 = tokenizer.texts_to_matrix(X_train, mode='tfidf')
    X_test=tokenizer.texts_to_matrix(X_test, mode='tfidf')
    labeltrain=[]
    allhistory=[]
    for kelas in range(15):       
        Y_train=train_data[:,[kelas+4]]
        Y_test=test_data[:,[kelas+4]]
    #   menggunakan tfidf
#        sm = SMOTE(random_state=42)
#        X_train, Y_train = sm.fit_resample(X_train1, Y_train.ravel())
        rm=RandomUnderSampler()
        X_train, Y_train = rm.fit_resample(X_train1, Y_train)
        indextukar=np.random.permutation(len(X_train))
        X_train=X_train[indextukar]
        Y_train=Y_train[indextukar]
        print(X_train.shape,Y_train.shape)
        print(X_test.shape,Y_test.shape)
        
        print ("save tokenizer")
#        with open('F:/1301154154_NANANG_NITIP/nanang/1/tokenizer'+str(k)+str(kelas)+'.pickle', 'wb') as handle:
        with open('tokenizer'+str(k)+str(kelas)+'.pickle', 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        
        #mlp
        print("build model")
        model = Sequential()
#        model.add(Dense(75, activation='sigmoid', input_shape=(X_train.shape[1],)))
        model.add(Dense(75, activation='sigmoid', input_shape=(X_train.shape[1],)))
        model.add(Dense(50, activation='sigmoid'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss = 'binary_crossentropy', optimizer='adam',metrics = ["accuracy"])
    #    print(model.summary())
        
        print("training")
        batch_size = 1
#        ckpt=ModelCheckpoint('F:/1301154154_NANANG_NITIP/nanang/1/model'+str(k)+str(kelas)+'.h5', monitor="val_loss",verbose=1,save_best_only=True)
        ckpt=ModelCheckpoint('model'+str(k)+str(kelas)+'.h5', monitor="val_acc",verbose=1,save_best_only=True)
        es=EarlyStopping(monitor="val_acc",verbose=1,patience=2)
        history=model.fit(X_train, Y_train[:].ravel(), epochs = 10, batch_size=batch_size, verbose = 2,validation_data=(X_test,Y_test),callbacks=[ckpt,es])
        
        print("save model")

        # save model
        print("Saved model to disk")

#        model=load_model('F:/1301154154_NANANG_NITIP/nanang/1/model'+str(k)+str(kelas)+'.h5')
        model=load_model('model'+str(k)+str(kelas)+'.h5')
        predik=np.where(model.predict(X_test) > 0.5, 1, 0)
        allpredik[str(kelas+1)]=predik.reshape(len(Y_test))
        report.append(classification_report(Y_test.astype(dtype='int32'),predik))
        allhistory.append(history)
    p=[]
    for pr  in np.array(allpredik):
        if sum(pr)==0:
            p.append(np.append(pr,1).tolist())
        else:
            p.append(np.append(pr,0).tolist())
    predik=np.asarray(p,dtype="int")
    print("Hammming loss: %.4f" % (hamming_loss(allytest.astype(dtype='int32'),predik)))
    f1ma.append(f1_score(allytest.astype(dtype='int32'),predik, average='macro'))
    f1mi.append(f1_score(allytest.astype(dtype='int32'),predik, average='micro'))
    hm.append(hamming_loss(allytest.astype(dtype='int32'),predik))
    loss01.append(zero_one_loss(allytest.astype(dtype='int32'),predik))
    k+=1
l=1  
for his in allhistory: 
#    np.save("F:/1301154154_NANANG_NITIP/nanang/1/history"+str(l),his.history);
    np.save("history"+str(l),his.history);
    l+=1  
hma=np.mean(hm)
loss01a=np.mean(loss01)
a1=np.mean(f1ma)
a2=np.mean(f1mi)
end = time.time()
waktu=end - start
np.savetxt("hasil.txt",[hma,loss01a,a1,a2,(end-start)],fmt='%1.5f')


#
#
