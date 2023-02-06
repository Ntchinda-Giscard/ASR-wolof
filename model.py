from tensorflow.keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, LSTM, Multiply, GRU
from tensorflow.keras.layers import RepeatVector, Dense, Activation, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model, Model
import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np



class Model:
    def __init__(self, Tx) -> None:
        # gloabal variable for attention part
        self.repeator = RepeatVector(Tx)
        self.concatenator = Concatenate(axis= -1)
        self.densor1 = Dense(10, activation='tanh')
        self.densor2 = Dense(1, activation='relu')
        self.activator = Activation('softmax', name='attention_weights' )
        self.dotor = Dot(axes = 1)

        #global variable for model
        self.n_a = 32 # number of units for the pre-attention, bi-directional LSTM's hidden state 'a'
        self.n_s = 64 # number of units for the post-attention LSTM's hidden state "s"
        self.machine_vocab = 40
        self.m = 0
        self.decoder_LSTM_cell = LSTM(self.n_s, return_state = True)
        self.encoder_biLSTM_cell = Bidirectional(LSTM(self.n_a, return_sequences=True),input_shape=(self.m, Tx, self.n_a*2))
        self.output_layer = Dense(len(self.machine_vocab), activation='softmax')
    def one_step_attention(self, a, s_prev):
        """
        Performs one step of attention: Outputs a context vector computed as a dot product of the attention weights
        "alphas" and the hidden states "a" of the Bi-LSTM.
        
        Arguments:
        a -- hidden state output of the Bi-LSTM, numpy-array of shape (m, Tx, 2*n_a)
        s_prev -- previous hidden state of the (post-attention) LSTM, numpy-array of shape (m, n_s)
        
        Returns:
        context -- context vector, input of the next (post-attention) LSTM cell
        
        """
        s_prev = self.repeator(s_prev)
        concat = self.concatenator([a, s_prev])
        e = self.densor1(concat)
        energies = self.densor2(e)
        alphas = self.activator(energies)
        context = self.dotor([alphas, a])
        
        return context

    def model(self, Tx, Ty, human_vocab_size):
        """
        Arguments:
        Tx -- length of the input sequence
        Ty -- length of the output sequence
        n_a -- hidden state size of the Bi-LSTM
        n_s -- hidden state size of the post-attention LSTM
        human_vocab_size -- size of the python dictionary "human_vocab"
        machine_vocab_size -- size of the python dictionary "machine_vocab"

        Returns:
        model -- Keras model instance
        """
        X_input = Input(shape=(Tx, human_vocab_size), name='Inputs for words')
        init_hidden_state = Input(shape=(self.n_s), name='initail hidden states')
        init_cell_state = Input(shape=(self.n_s), name='initial cell states')

        hidden_state_enc = init_hidden_state
        cell_states_enc = init_cell_state

        outputs = []
        out_dec = self.encoder_biLSTM_cell(X_input)

        for i in range(Tx):
            context = self.one_step_attention(out_dec, hidden_state_enc)
            hidden_state_enc, _, cell_states_enc = self.decoder_LSTM_cell(inputs=context, initial_state=[hidden_state_enc,cell_states_enc])
            out = self.output_layer(inputs=hidden_state_enc)

            outputs.append(out)
        model = Model(inputs=[X_input, init_hidden_state, init_cell_state], outputs=outputs)
        return model