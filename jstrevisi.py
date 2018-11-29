
import nltk
nltk.download('stopwords')
nltk.download('punkt')
# -*- codi ng: utf-8 -*-
"""
Created on Tue Jun 05 18:26:43 2018

@author: nanang saiful
"""
for skenariomonitor in  ["acc","val_acc"]:    
      for skenariostem in ["stem","nostem"]:
         for skenariooptimezer in ["adam","sgd"]:
            for skenariosampling in ["under","real","smote"]: 
                from keras.preprocessing.text import Tokenizer
                from keras.models import Sequential,load_model
                from keras.layers import Dense
                
                from keras.callbacks import ModelCheckpoint,EarlyStopping
                import pandas as pd
                import re
                import numpy as np
                from sklearn.model_selection import KFold
                from sklearn.metrics import hamming_loss,classification_report,f1_score,zero_one_loss
                from nltk.corpus import stopwords
                from nltk.stem import PorterStemmer
                from imblearn.over_sampling import SMOTE#RandomOverSampler
                from imblearn.under_sampling import RandomUnderSampler
                import pickle 
                import time
                import sys
                start = time.time()
                dokumen=pd.ExcelFile('datasetalquran.xlsx')
                df=dokumen.parse('Sheet1')
                kelas={}
    
                print ("preposesing")
                #lower
                df['Terjemahan']=df['Terjemahan'].apply(lambda x: x.lower())
                #mengambil huruf saja
                df['Terjemahan']=df['Terjemahan'].apply(lambda x: re.sub('[^a-zA-z\s]','',x))
                #stemming
                if skenariostem=="stem":
                    ps=PorterStemmer()
                    df['Terjemahan']=df['Terjemahan'].apply(lambda x: " ".join([ps.stem(y) for y in x.split(" ")]))      
                #stopword removal
                stop_word= set (stopwords.words("english"))
                stop_word.remove('did')
                df['Terjemahan']=df['Terjemahan'].apply(lambda x: " ".join([item for item in x.split(" ") if item not in stop_word]))
                
                kf=KFold(n_splits=5)
                k=0;
                tohammingloss=[]
                hm=[]
                report=[]
                f1mi=[]
                f1ma=[]
                loss01=[]
                for train, test in kf.split(df):
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
                    print ("save tokenizer")
            #        with open('F:/1301154154_NANANG_NITIP/nanang/1/tokenizer'+str(k)+str(kelas)+'.pickle', 'wb') as handle:
                    with open('tokenizer'+skenariosampling+skenariostem+str(k)+'.pickle', 'wb') as handle:
                            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
                       
                    labeltrain=[]
                    allhistory=[]
                    for kelas in range(15):       
                        Y_train=train_data[:,[kelas+4]]
                        Y_test=test_data[:,[kelas+4]]
                        if skenariosampling=="smote":
                            sm = SMOTE(random_state=42)
                            X_train, Y_train = sm.fit_resample(X_train1, Y_train.ravel())
                        elif skenariosampling=="under":
                            rm=RandomUnderSampler()
                            X_train, Y_train = rm.fit_resample(X_train1, Y_train)
                        indextukar=np.random.permutation(len(X_train))
                        X_train=X_train[indextukar]
                        Y_train=Y_train[indextukar]
                        print(X_train.shape,Y_train.shape)
                        print(X_test.shape,Y_test.shape)
                        
                         
                        
                        #mlp
                        print("build model")
                        model = Sequential()
                #        model.add(Dense(75, activation='sigmoid', input_shape=(X_train.shape[1],)))
                        model.add(Dense(75, activation='sigmoid', input_shape=(X_train.shape[1],)))
                        model.add(Dense(50, activation='sigmoid'))
                        model.add(Dense(1, activation='sigmoid'))
                        model.compile(loss = 'binary_crossentropy', optimizer=skenariooptimezer,metrics = ["accuracy"])
                    #    print(model.summary())
                        
                        print("training")
                        batch_size = 1
                #        ckpt=ModelCheckpoint('F:/1301154154_NANANG_NITIP/nanang/1/model'+str(k)+str(kelas)+'.h5', monitor="val_loss",verbose=1,save_best_only=True)
                        ckpt=ModelCheckpoint('model'+skenariooptimezer+skenariomonitor+skenariosampling+skenariostem+str(k)+str(kelas)+'.h5', monitor=skenariomonitor,verbose=1,save_best_only=True)
                        es=EarlyStopping(monitor=skenariomonitor,verbose=1,patience=2)
                        history=model.fit(X_train, Y_train[:].ravel(), epochs = 1, batch_size=batch_size, verbose = 2,validation_data=(X_test,Y_test),callbacks=[ckpt,es])
                        
                        print("save model")
                
                        # save model
                        print("Saved model to disk")
                
                #        model=load_model('F:/1301154154_NANANG_NITIP/nanang/1/model'+str(k)+str(kelas)+'.h5')
                        model=load_model('model'+skenariooptimezer+skenariomonitor+skenariosampling+skenariostem+str(k)+str(kelas)+'.h5')
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
                    np.save("history"+skenariooptimezer+skenariomonitor+skenariosampling+skenariostem+str(l),his.history);
                    l+=1  
                hma=np.mean(hm)
                loss01a=np.mean(loss01)
                a1=np.mean(f1ma)
                a2=np.mean(f1mi)
                end = time.time()
                waktu=end - start
                np.savetxt("hasil"+skenariooptimezer+skenariomonitor+skenariosampling+skenariostem+".txt",[hma,loss01a,a1,a2,(end-start)],fmt='%1.5f')
                Del X_test,Y_test,X_train,X_train1,Y_train,df,train_data,test_data,allpredik,allhistory

#
#
