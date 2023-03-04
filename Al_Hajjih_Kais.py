import pandas as pd
import numpy as np
from sklearn import neural_network, metrics, datasets


#data = pd.read_csv('eighthr.data')
#data = data.replace('?', 0)
#data.to_csv('eighthr1.csv',index= False)


data = pd.read_csv('onehr1.csv')

new_data = pd.DataFrame(data).to_numpy()
new_data = new_data[:, 2:].astype('float')


#IMPARTIRE IN TRAIN SI TEST
data_train = new_data[:2000, :71]
data_test = new_data[2000:, :71]

etichete_train = new_data[:2000, 71]
etichete_test = new_data[2000:, 71]





#Crearea si Antrenarea MLP

clf = neural_network.MLPClassifier(hidden_layer_sizes=(50), learning_rate_init=0.1)
clf.fit(data_train, etichete_train)


#Testam MLP

predictii = clf.predict(data_test)


#Acuratete
acc=0
for i in range(len(etichete_test)):
    if etichete_test[i]==predictii[i]:
        acc=acc+1
print('Acuratetea1=' + str((acc/len(etichete_test))*100) + '%')



#2
clf = neural_network.MLPClassifier(hidden_layer_sizes=200, learning_rate_init=0.1)
clf.fit(data_train, etichete_train)

predictii = clf.predict(data_test)


acc=0
for i in range(len(etichete_test)):
    if etichete_test[i]==predictii[i]:
        acc=acc+1
print('Acuratetea2=' + str((acc/len(etichete_test))*100) + '%')






#3
clf = neural_network.MLPClassifier(hidden_layer_sizes=(70,50), learning_rate_init=0.1)
clf.fit(data_train, etichete_train)

predictii = clf.predict(data_test)

acc=0
for i in range(len(etichete_test)):
    if etichete_test[i]==predictii[i]:
        acc=acc+1
print('Acuratetea3=' + str((acc/len(etichete_test))*100) + '%')

#print(etichete_test)
