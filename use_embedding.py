from keras.preprocessing.text import Tokenizer
import nltk
from nltk.tokenize import word_tokenize
import numpy as np
import re
import pandas as pd
from keras.utils import to_categorical
from doc3 import training_doc3

from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Input
from keras.layers import Layer
from keras.preprocessing.sequence import pad_sequences

import tensorflow_hub as hub
import tensorflow as tf

import pickle


def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

cleaned = re.sub(r'\W+', ' ', training_doc3).lower()
tokens = word_tokenize(cleaned)
train_len = 3+1
text_sequences = []
for i in range(train_len,len(tokens)):
    seq = tokens[i-train_len:i]
    text_sequences.append(seq)
sequences = {}
count = 1
for i in range(len(tokens)):
    if tokens[i] not in sequences:
        sequences[tokens[i]] = count
        count += 1
tokenizer = Tokenizer()
tokenizer.fit_on_texts(text_sequences)
sequences = tokenizer.texts_to_sequences(text_sequences) 


vocabulary_size = len(tokenizer.word_counts)+1
print(vocabulary_size)
n_sequences = np.empty([len(sequences),train_len], dtype='int32')
for i in range(len(sequences)):
    n_sequences[i] = sequences[i]

texts_list = []
for line in text_sequences:
    texts_list += line
index_list = []
for line in sequences:
    index_list += line
frame = pd.DataFrame({'text':texts_list, 'loc_index':index_list})    
frame = frame.drop_duplicates()

#Load the Encoder Model if saved locally - this can take a while
encoderModel = hub.load('module')

#uncomment this instead to use from tf_hub
#encoderModel = hub.load('https://tfhub.dev/google/universal-sentence-encoder-large/5')

print ("module loaded")
def embed(input):
    return encoderModel(input)

#organize training inputs and targets
train_inputs = n_sequences[:,:-1]
train_targets = n_sequences[:,-1]
train_targets = to_categorical(train_targets, num_classes=vocabulary_size)
seq_len = train_inputs.shape[1]
train_inputs.shape

#get embeddings dictionary
vocab = list(tokenizer.word_counts.keys())
vocab_embeddings = [np.array(embed([word])) for word in vocab]
embeddings_dict = {}
for i in range(len(vocab)):
    embeddings_dict[vocab[i]] = vocab_embeddings[i]

#get embeddings matrix - this becomes the weights of the non-trainable embedding layer
embeddings_matrix = np.zeros(shape=(len(frame) + 1,512))

for i in range(len(frame)):
    embeddings_matrix[frame.iloc[i].loc_index] = embeddings_dict[frame.iloc[i].text]


#Build the Network    
embedding_layer = Embedding(len(frame) + 1, 512, weights=[embeddings_matrix], input_length=seq_len, trainable=False)
model = Sequential()
model.add(embedding_layer)
model.layers[0].trainable = False
model.add(LSTM(50,return_sequences=True))
model.add(LSTM(50))
model.add(Dense(50,activation='relu'))
model.add(Dense(vocabulary_size, activation='softmax'))
print(model.summary())


#Compile the Network
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_inputs,train_targets,epochs=500,verbose=1)

#Save model and tokenizer
model.save("word_embedding.h5")
save_obj(tokenizer, 'tokenizer_saved')

