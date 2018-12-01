# -*- codi ng: utf-8 -*-
"""
Created on Tue Jun 05 18:26:43 2018

@author: nanang saiful
"""
from keras.callbacks import ModelCheckpoint,EarlyStopping

from keras.preprocessing.text import Tokenizer
from keras.models import Sequential,load_model
from keras.layers import Dense
import pandas as pd
import re
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import hamming_loss,classification_report,f1_score,zero_one_loss
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pickle 
import time
for skenariomonitor in  ["acc","val_acc"]:    
    for skenariostem in ["nostem","stem"]:
        for skenariooptimezer in ["adam","sgd"]:
                start = time.time()
                dokumen=pd.ExcelFile('E:\kuliah\RISET\quranic topic classification\JST\kodingan\datasetalquran1.xlsx')
                df=dokumen.parse('Sheet1')
                
                print ("preposesing")
                #lower
                df['Terjemahan']=df['Terjemahan'].apply(lambda x: x.lower())
                #mengambil huruf saja
                df['Terjemahan']=df['Terjemahan'].apply(lambda x: re.sub('[^a-zA-z\s]','',x))
                #stemming
                ps=PorterStemmer()
                df['Terjemahan']=df['Terjemahan'].apply(lambda x:" ".join([ps.stem(y) for y in x.split(" ")] ))
                #stopword removal
                stop_word= set (stopwords.words("english"))
                stop_word.remove('did')
                df['Terjemahan']=df['Terjemahan'].apply(lambda x: " ".join([item for item in x.split(" ") if item not in stop_word]))
                
                kf=KFold(n_splits=5)
                k=0;
                indextukar=np.random.permutation(len(df))
                np.save('hasilindextukar',[indextukar,kf])
                df=df.iloc[indextukar]
                del indextukar                
                tohammingloss=[]
                hm=[]
                report=[]
                f1mi=[]
                f1ma=[]
                loss01=[]
                for train, test in kf.split(df):
                    train_data = np.array(df)[train]
                    test_data = np.array(df)[test]
                    X_train=pd.Series(np.resize(train_data[:,[3]],(train_data[:,[0]].size,)), dtype='str')
                    Y_train=train_data[:,[4,5,6,7,8,9,10,11,12,13,14]]
                    X_test=pd.Series(np.resize(test_data[:,[3]],(test_data[:,[0]].size,)), dtype='str')
                    Y_test=test_data[:,[4,5,6,7,8,9,10,11,12,13,14]]
                
                    tokenizer = Tokenizer( split=' ')
                    tokenizer.fit_on_texts(X_train)
                    
                    X_train = tokenizer.texts_to_matrix(X_train, mode='tfidf')
                    X_test=tokenizer.texts_to_matrix(X_test, mode='tfidf')
                    
                    print(X_train.shape,Y_train.shape)
                    print(X_test.shape,Y_test.shape)
                    
                    
                    print ("save tokenizer")
                    with open('tokenizer250.pickle', 'wb') as handle:
                        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    
                    
                    #mlp
                    model = Sequential()
                    model.add(Dense(50, activation='sigmoid', input_shape=(X_train.shape[1],)))
                    model.add(Dense(25, activation='sigmoid'))
                    model.add(Dense(11, activation='sigmoid'))
                    model.compile(loss = 'binary_crossentropy', optimizer='sgd',metrics = ['accuracy'])
                #    print(model.summary())
                    
                    
                    batch_size = 1
                    ckpt=ModelCheckpoint('model'+skenariooptimezer+skenariomonitor+skenariostem+str(k)+'.h5', monitor="acc",verbose=1,save_best_only=True)
                    es=EarlyStopping(monitor="acc",verbose=1,patience=2)
                    history=model.fit(X_train, Y_train, epochs = 1, batch_size=batch_size, verbose = 2,validation_data=(X_test,Y_test),callbacks=[ckpt,es])
                    
                        
                    
                    model=load_model('model'+skenariooptimezer+skenariomonitor+skenariostem+str(k)+'.h5')
                    predik=np.where(model.predict(X_test) > 0.5, 1, 0)
                    print("Hammming loss: %.4f" % (hamming_loss(Y_test.astype(dtype='int32'),predik)))
                    report.append(classification_report(Y_test.astype(dtype='int32'),predik))
                    f1ma.append(f1_score(Y_test.astype(dtype='int32'),predik, average='macro'))
                    f1mi.append(f1_score(Y_test.astype(dtype='int32'),predik, average='micro'))
                    hm.append(hamming_loss(Y_test.astype(dtype='int32'),predik))
                    loss01.append(zero_one_loss(Y_test.astype(dtype='int32'),predik))
                    k+=1
                hma=np.mean(hm)
                loss01a=np.mean(loss01)
                a1=np.mean(f1ma)
                a2=np.mean(f1mi)
                end = time.time()
                waktu=end - start
                np.savetxt("hasil"+skenariooptimezer+skenariomonitor+skenariostem+".txt",[hma,loss01a,a1,a2,(end-start)],fmt='%1.5f')
                del X_test,X_train,Y_test,Y_train,batch_size,df,report,test,test_data,train,train_data
                break
        break
    break


## mlp
#
##mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=10, alpha=1e-4,
##                    solver='sgd', verbose=10, tol=1e-4, random_state=1,
##                    learning_rate_init=.1)
##
##mlp.fit(X_train, Y_train)
##print("Training set score: %f" % mlp.score(X_train, Y_train))
##print("Test set score: %f" % mlp.score(X_test, Y_test))
#
#
