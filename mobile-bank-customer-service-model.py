#libraries
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
#reading file
df=pd.read_csv('banka.csv')
#data compression
df=df[['sorgu','label']]
#import stepwords
stopwords =['fakat','lakin','ancak','acaba', 'ama', 'aslında', 'az', 'bazı', 'belki', 'biri', 'birkaç', 'birşey', 'biz', 'bu', 'çok', 'çünkü', 'da', 'daha', 'de', 'defa', 'diye', 'eğer', 'en', 'gibi', 'hem', 'hep', 'hepsi', 'her', 'hiç', 'için', 'ile', 'ise', 'kez', 'ki', 'kim', 'mı', 'mu', 'mü', 'nasıl', 'ne', 'neden', 'nerde', 'nerede', 'nereye', 'niçin', 'niye', 'o', 'sanki', 'şey', 'siz', 'şu', 'tüm', 've', 'veya', 'ya', 'yani']
#get input and add to end of the data table
mesaj = input('Yapmak istediğiniz işlemi giriniz: ')
mesajdf = pd.DataFrame({'sorgu':mesaj, 'label':0}, index=[42])
df = pd.concat([df,mesajdf], ignore_index=True)
#cleaning data from stopwords
for word in stopwords:
  word = " "+ word + " "
  df['sorgu']=df['sorgu'].str.replace(word,' ')
#determining important workds for learning
cv = CountVectorizer(max_features=50)
#extraction important words from 'sorgu' and transform into an array, defining x and y
x= cv.fit_transform(df['sorgu']).toarray()
y=df['label']
#tahmin = last line of arrayed df: x    ///in other words, the message for test
tahmin=x[-1].copy()
#x,y: all data except last line         ///for train
x=x[:-1]
y=y[:-1]
#train-test-split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3, random_state=21)
#for data classification and learning
rf=RandomForestClassifier()
model = rf.fit(x_train,y_train)
score=model.score(x_test,y_test)
#testing
result = model.predict([tahmin])
print('Sonuc: ', result, 'Skor: ', score)
