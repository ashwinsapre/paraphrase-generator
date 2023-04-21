import params
import load_data

xtrain, xval, xvoc, ytrain, yval, yvoc, x_tokenizer, y_tokenizer = import_data(vocabulary_size, max_sentence_length)

# Encoder
encoder_inputs = keras.layers.Input(shape=(max_sentence_length, ))

# Embedding layer
enc_emb = keras.layers.Embedding(xvoc, embedding_dim,
                    trainable=True)(encoder_inputs)

# Encoder LSTM 1
encoder_lstm1 = keras.layers.LSTM(latent_dim, return_sequences=True,
                     return_state=True, dropout=0.4,
                     recurrent_dropout=0.4)
(encoder_output1, state_h1, state_c1) = encoder_lstm1(enc_emb)

# Encoder LSTM 2
encoder_lstm2 = keras.layers.LSTM(latent_dim, return_sequences=True,
                     return_state=True, dropout=0.4,
                     recurrent_dropout=0.4)
(encoder_output2, state_h2, state_c2) = encoder_lstm2(encoder_output1)

# Encoder LSTM 3
encoder_lstm3 = keras.layers.LSTM(latent_dim, return_state=True,
                     return_sequences=True, dropout=0.4,
                     recurrent_dropout=0.4)
(encoder_outputs, state_h, state_c) = encoder_lstm3(encoder_output2)

# Set up the decoder, using encoder_states as the initial state
decoder_inputs = keras.layers.Input(shape=(None, ))

# Embedding layer
dec_emb_layer = keras.layers.Embedding(yvoc, embedding_dim, trainable=True)
dec_emb = dec_emb_layer(decoder_inputs)

# Decoder LSTM
decoder_lstm = keras.layers.LSTM(latent_dim, return_sequences=True,
                    return_state=True, dropout=0.4,
                    recurrent_dropout=0.2)
(decoder_outputs, decoder_fwd_state, decoder_back_state) = \
    decoder_lstm(dec_emb, initial_state=[state_h, state_c])

# Dense layer
decoder_dense = keras.layers.TimeDistributed(keras.layers.Dense(yvoc, activation='softmax'))
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model
model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.0005), loss='sparse_categorical_crossentropy')

es = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2)

history = model.fit(
    [xtrain, ytrain[:, :-1]],
    ytrain.reshape(ytrain.shape[0], ytrain.shape[1], 1)[:, 1:],
    epochs=3,
    callbacks=[es],
    batch_size=128,
    validation_data=([xval, yval[:, :-1]],
                     yval.reshape(yval.shape[0], yval.shape[1], 1)[:
                     , 1:]),
    )

reverse_target_word_index = y_tokenizer.index_word
reverse_source_word_index = x_tokenizer.index_word
target_word_index = y_tokenizer.word_index


# Inference Models

# Encode the input sequence to get the feature vector
encoder_model = keras.Model(inputs=encoder_inputs, outputs=[encoder_outputs,
                      state_h, state_c])

# Decoder setup

# Below tensors will hold the states of the previous time step
decoder_state_input_h = keras.layers.Input(shape=(latent_dim, ))
decoder_state_input_c = keras.layers.Input(shape=(latent_dim, ))
decoder_hidden_state_input = keras.layers.Input(shape=(max_sentence_length, latent_dim))

# Get the embeddings of the decoder sequence
dec_emb2 = dec_emb_layer(decoder_inputs)

# To predict the next word in the sequence, set the initial states to the states from the previous time step
(decoder_outputs2, state_h2, state_c2) = decoder_lstm(dec_emb2,
        initial_state=[decoder_state_input_h, decoder_state_input_c])

# A dense softmax layer to generate prob dist. over the target vocabulary
decoder_outputs2 = decoder_dense(decoder_outputs2)

# Final decoder model
decoder_model = keras.Model([decoder_inputs] + [decoder_hidden_state_input,
                      decoder_state_input_h, decoder_state_input_c],
                      [decoder_outputs2] + [state_h2, state_c2])

def decode_sequence(input_seq):

    # Encode the input as state vectors.
    (e_out, e_h, e_c) = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1
    target_seq = np.zeros((1, 1))

    # Populate the first word of target sequence with the start word.
    target_seq[0, 0] = target_word_index['sos']

    stop_condition = False
    decoded_sentence = ''

    while not stop_condition:
        (output_tokens, h, c) = decoder_model.predict([target_seq]
                + [e_out, e_h, e_c])

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = reverse_target_word_index[sampled_token_index]

        if sampled_token != 'sos':
            decoded_sentence += ' ' + sampled_token

        # Exit condition: either hit max length or find the stop word.
        if sampled_token == 'eos' or len(decoded_sentence.split()) \
            >= max_sentence_length - 1:
            stop_condition = True

        # Update the target sequence (of length 1)
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        # Update internal states
        (e_h, e_c) = (h, c)

    return decoded_sentence

# To convert sequence to summary
def seq2summary(input_seq):
    newString = ''
    for i in input_seq:
        if i != 0 and i != target_word_index['sos'] and i \
            != target_word_index['eos']:
            newString = newString + reverse_target_word_index[i] + ' '

    return newString


# To convert sequence to text
def seq2text(input_seq):
    newString = ''
    for i in input_seq:
        if i != 0:
            newString = newString + reverse_source_word_index[i] + ' '

    return newString

for i in range(0, 19):
    print ('Review:', seq2text(xtrain[i]))
    print ('Original sentence:', seq2summary(ytrain[i]))
    print ('Paraphrase:', decode_sequence(xtrain[i].reshape(1,
           max_sentence_length)))
    print ('\n')