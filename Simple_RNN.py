
import numpy as np
import pandas as pd
from tensorflow.keras.datasets import imdb
#from tensorflow.keras.preprocessing import Sequence
from tensorflow.keras.models import  Sequential
from tensorflow.keras.layers import Embedding,SimpleRNN,Dense

##loading the imdb dataset

max_features=10000
(X_train,y_train),(X_test,y_test)=imdb.load_data(num_words=max_features)

#shape of the data
print(f'Training data shape: {X_train.shape}, Training labels shape: {y_train.shape}')
print(f'Testing data shape: {X_train.shape}, Testing labels shape: {y_test.shape}')

print(X_train[0],y_train[0])

#checking a sample review and its label
sample_review=X_train[0]
sample_label=y_train[0]
print(f"Sample review (as integers):{sample_review}")
print(f'Sample label: {sample_label}')


### MApping of words index bacl to words(for understanding)
word_index=imdb.get_word_index()
#word_index
reverse_word_index = {value: key for key, value in word_index.items()}
print(reverse_word_index)
print('-----------------------------')
decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in sample_review])
print('decoded_review')
print('-------------------------------')
from tensorflow.keras.preprocessing import sequence
max_len=500
X_train=sequence.pad_sequences(X_train,maxlen=max_len)
X_test = sequence.pad_sequences(X_test, maxlen=max_len)
X_train
print('-------------------------------')
## Train Simple RNN
model=Sequential()
model.add(Embedding(max_features,128,input_length=max_len)) ## Embedding Layers
model.add(SimpleRNN(128,activation='relu'))
model.add(Dense(1,activation="sigmoid"))
print('----------------------------------')

print(model.summary())
print('------------------------------------')
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
print('--------------------------------------')
## Create an instance of EarlyStoppping Callback
from tensorflow.keras.callbacks import EarlyStopping
earlystopping=EarlyStopping(monitor='val_loss',patience=5,restore_best_weights=True)
earlystopping
print('-------------------------------------')
## Train the model with early sstopping
history=model.fit(
    X_train,y_train,epochs=10,batch_size=32,
    validation_split=0.2,
    callbacks=[earlystopping]
)
print('-------------------------------------')
## Save model file
model.save('simple_rnn_imdb.h5')
print('-------------------------------------')

