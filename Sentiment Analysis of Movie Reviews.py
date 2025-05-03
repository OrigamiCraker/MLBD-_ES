# L'esercizio consiste nell'utilizzo del dataset delle recensioni di IMBd. Esso contiene il testo delle reviews, ognuna con una label che indica se è positiva o negativa.
# Prima di tutto dobbiamo uplodare il train e provare il test data a server

# Siccessivamente importiamo pandas come libreria di data analisi

import pandas as pd

train = pd.read_csv('http://ailab.uniud.it/wp-content/uploads/2019/05/Train_Movie_Data.csv')
print("First Elements of train set:\n {}".format(train.head(3)))
text_train = train['review'].values
y_train = train['sentiment'].values
print('Number of train samples: ', len(y_train))
print()
test = pd.read_csv('http://ailab.uniud.it/wp-content/uploads/2019/05/Test_Movie_Data.csv')
print("First Elements of test set:\n {}".format(test.head(3)))
text_test = test['review'].values
y_test = test['sentiment'].values
print('Number of test samples: ', len(y_test))