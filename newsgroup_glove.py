'''
Code to train a text classification model on the 20 newsgroups dataset.
The script uses a pre-trained GloVe word embeddings model.
'''

import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Sequential
from keras.models import Model

NEWSGROUP_DATA_DIR = r"data\20_newsgroup"
MAX_WORDS = 20000
MAX_SEQUENCE_LENGTH = 1000
EMBEDDING_DIM = 100

# Create embedding index using Glove
print ("Indexing GloVe word vectors...")
embeddings_index = {}
fhd = open('data/glove.6B.100d.txt', 'r', encoding='utf8')
for line in fhd:
    values = line.split()
    word = values[0]
    vector = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = vector
fhd.close()
print ("Length of GloVe embedding:", len(embeddings_index))

print ("Starting to process newsgroup data...")
texts = []      # list of text samples
labels_index = {}       # dictionary mapping label name to label id
labels = []     # list of label ids for corresponding text
for name in sorted(os.listdir(NEWSGROUP_DATA_DIR)):
    path = os.path.join(NEWSGROUP_DATA_DIR, name)
    if os.path.isdir(path):
        label_id = len(labels_index)
        labels_index[name] = label_id
        for fname in sorted(os.listdir(path)):
            if fname.isdigit():
                fpath = os.path.join(path, fname)
                if sys.version_info.major < 3:
                    fhd = open(fpath)
                else:
                    fhd = open(fpath, encoding='latin-1')
                text = fhd.read()
                head_index = text.find("\n\n")
                if head_index > 0:
                    text = text[head_index:]
                texts.append(text)
                fhd.close()
                labels.append(label_id)
print ("Number of texts found:", len(texts))

# tokenize text
tokenizer = Tokenizer(num_words=MAX_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print ("Number of unique tokens:", len(word_index))

# create 2D data matrix
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

labels = to_categorical(np.asarray(labels))

print ("Shape of data matrix:", data.shape)
print ("Shape of labels matrix:", labels.shape)

# split data into train and validation set
x_train, x_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=42)
            
print ("Starting to prepare embedding matrix ...")
#num_words = min(MAX_WORDS, len(word_index))
#embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    # if i >= MAX_WORDS:
    #     continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

embedding_layer = Embedding(
    #num_words,
    len(word_index) + 1,
    EMBEDDING_DIM,
    weights=[embedding_matrix],
    input_length=MAX_SEQUENCE_LENGTH,
    trainable=False)

sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = Conv1D(128, 5, activation='relu')(embedded_sequences)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(35)(x)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
preds = Dense(len(labels_index), activation='softmax')(x)

model = Model(sequence_input, preds)
model.compile(
    loss='categorical_crossentropy',
    optimizer='rmsprop',
    metrics=['acc'])

model.fit(
    x_train, 
    y_train, 
    validation_data=(x_val, y_val),
    epochs=2,
    batch_size=128,
    verbose=2)





