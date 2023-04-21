import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from tqdm import tqdm
from sklearn.model_selection import train_test_split

'''
function to load, tokenize, pad the quora dataset and create an embedding matrix
returns: dataset in a dataframe, embedding matrix (to be used in embedding layer later), 
        and tokenizer (will need it while doing the reverse token ID to word conversion later)
'''

def import_data(vocabulary_size, max_len):
    quoradf = pd.read_table('drive/Shareddrives/Project128/quora_duplicate_questions.tsv', index_col='id')
    quoradf = quoradf[quoradf.is_duplicate == 1].reset_index(drop=True)
    quoradf = quoradf.drop(labels=['qid1', 'qid2', 'is_duplicate'], axis=1)
    #quoradf['total'] = quoradf['question1']+quoradf['question2']
    quoradf['question2'] = "<SOS>" + quoradf['question2'] + "<EOS>"

    x_tr, x_val, y_tr, y_val = train_test_split(
        np.array(quoradf["question1"]),
        np.array(quoradf["question2"]),
        test_size=0.1,
        random_state=0,
        shuffle=True,
    )

    x_tokenizer = keras.preprocessing.text.Tokenizer()
    x_tokenizer.fit_on_texts(x_tr)

    cnt = 0
    tot_cnt = 0

    for key, value in x_tokenizer.word_counts.items():
        tot_cnt = tot_cnt + 1
        if value < thresh:
            cnt = cnt + 1
        
    print("% of rare words in vocabulary: ", (cnt / tot_cnt) * 100)

    x_tokenizer = keras.preprocessing.text.Tokenizer(num_words = tot_cnt - cnt) 
    x_tokenizer.fit_on_texts(list(x_tr))

    # Convert text sequences to integer sequences 
    x_tr_seq = x_tokenizer.texts_to_sequences(x_tr) 
    x_val_seq = x_tokenizer.texts_to_sequences(x_val)

    # Pad zero upto maximum length
    x_tr = keras.preprocessing.sequence.pad_sequences(x_tr_seq,  maxlen=max_sentence_length, padding='post')
    x_val = keras.preprocessing.sequence.pad_sequences(x_val_seq, maxlen=max_sentence_length, padding='post')

    # Size of vocabulary (+1 for padding token)
    x_voc = x_tokenizer.num_words + 1

    print("Size of vocabulary in X = {}".format(x_voc))


    # Prepare a tokenizer on testing data
    y_tokenizer = keras.preprocessing.text.Tokenizer()   
    y_tokenizer.fit_on_texts(list(y_tr))

    cnt = 0
    tot_cnt = 0

    for key, value in y_tokenizer.word_counts.items():
        tot_cnt = tot_cnt + 1
        if value < thresh:
            cnt = cnt + 1
        
    print("% of rare words in vocabulary:",(cnt / tot_cnt) * 100)

    # Prepare a tokenizer, again -- by not considering the rare words
    y_tokenizer = keras.preprocessing.text.Tokenizer(num_words=tot_cnt-cnt) 
    y_tokenizer.fit_on_texts(list(y_tr))

    # Convert text sequences to integer sequences 
    y_tr_seq = y_tokenizer.texts_to_sequences(y_tr) 
    y_val_seq = y_tokenizer.texts_to_sequences(y_val) 

    # Pad zero upto maximum length
    y_tr = keras.preprocessing.sequence.pad_sequences(y_tr_seq, maxlen=max_sentence_length, padding='post')
    y_val = keras.preprocessing.sequence.pad_sequences(y_val_seq, maxlen=max_sentence_length, padding='post')

    # Size of vocabulary (+1 for padding token)
    y_voc = y_tokenizer.num_words + 1

    print("Size of vocabulary in Y = {}".format(y_voc))

    '''
    #create embedding vector
    #text file has entries like
      #word1 <50d vector>
      #word2 <50d vector>
      #...
    embedding_vector = {}
    f = open('drive/Shareddrives/Project128/embeddings/glove.6B.50d.txt')
    for line in tqdm(f):
        value = line.split(' ')
        word = value[0]
        coef = np.array(value[1:],dtype = 'float32')
        embedding_vector[word] = coef
    
    #to make embedding matrix
    #take each word in tokenizer vocabulary
    #look up embedding_vector for its 50d vector
    embedding_matrix = np.zeros((vocab_size,50))
    for word,i in tqdm(token.word_index.items()):
        embedding_value = embedding_vector.get(word)
        if embedding_value is not None:
            embedding_matrix[i] = embedding_value
    '''
    
    return x_tr, x_val, x_voc, y_tr, y_val, y_voc, x_tokenizer, y_tokenizer