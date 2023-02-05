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
        self.repeator = RepeatVector(Tx)
        self.concatenator = Concatenate(axis= -1)
        self.densor1 = Dense(10, activation='tanh')
        self.densor2 = Dense(1, activation='relu')
        self.activator = Activation('softmax', name='attention_weights' )
        self.dotor = Dot(axes = 1)
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