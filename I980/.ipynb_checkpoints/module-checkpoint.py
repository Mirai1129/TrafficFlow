import tensorflow as tf
import numpy as np 
from keras import layers, models, Input, initializers, constraints, regularizers
tf.keras.backend.clear_session()

# Define the attention layer
class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, hidden_size, bias_init = None, weight_init = None,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.hidden_size = hidden_size
        # self.bias_init = initializers.RandomNormal(seed=0)
        self.bias_init = initializers.GlorotUniform(seed=0)
        # self.weight_init = initializers.RandomNormal(seed=0)
        self.weight_init = initializers.GlorotUniform(seed=0)
        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(AttentionLayer, self).__init__(**kwargs)
        
    def build(self, input_shape):
        
        last_dim = tf.compat.dimension_value(input_shape[-1])
        # Create a trainable weight variable for this layer
        self.W = self.add_weight(shape=(last_dim, self.hidden_size ,),
                                 initializer=self.weight_init,
                                 name='att_weight',
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        # Setting bias
        if self.bias:
            self.b = self.add_weight(
                                     shape=(self.hidden_size ,),
                                     initializer=self.bias_init,
                                     name='att_bias',
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        # Compute the attention scores
        # inputs shape: (batch_size, time_steps, hidden_size)
        e = tf.tensordot(inputs, self.W, axes=1)  # Shape: (batch_size, time_steps)
        outputs = tf.nn.bias_add(e, self.b) 
        e = tf.nn.tanh(outputs)
        # Compute attention weights
        a = tf.nn.softmax(e, axis=1)  # Shape: (batch_size, time_steps)

        # Compute the context vector as the weighted average
        context = tf.reduce_sum(inputs * a, axis=1)  # Shape: (batch_size, hidden_size)
        return context
    # def get_config(self):
        # base_config = super().get_config()
        # config = {
            # "hidden_size": self.hidden_size,
            # "bias_init": keras.saving.serialize_keras_object(self.bias_init),
            # "weight_init": keras.saving.serialize_keras_object(self.weight_init),
            # "W_regularizer": keras.saving.serialize_keras_object(self.W_regularizer),
            # "b_regularizer": keras.saving.serialize_keras_object(self.b_regularizer),
            # "W_constraint": keras.saving.serialize_keras_object(self.W_constraint),
            # "b_constraint": keras.saving.serialize_keras_object(self.b_constraint),
            # "bias": self.bias,
        # }
        # return {**base_config, **config}

        
        
        
class BiGRU_module:
    def __init__(self,
                 units = 64,
                 activation="tanh",
                 use_confined_attention=False):
        self.units = units
        self.activation = activation
        self.use_confined_attention = use_confined_attention

    def __call__(self, inputs, windows_size = None):
        
        if(self.use_confined_attention == True and windows_size==None):
            raise ValueError('Use confined attention but your windows_size is None')
        if(self.use_confined_attention == True):
            x = layers.Bidirectional(layers.GRU(self.units, return_sequences=True))(inputs)
            # x = layers.LayerNormalization()(x)
            x = layers.Bidirectional(layers.GRU(self.units, return_sequences=True))(x)
            # x = layers.LayerNormalization()(x)
            fw_outputs, bw_outputs = layers.Bidirectional(layers.GRU(self.units, return_sequences=True), merge_mode=None)(x)
            # x = layers.LayerNormalizetion()(x)
            
            
            
            fw_outputs = layers.Lambda(lambda x: x[:, windows_size:, :])(fw_outputs)
            bw_outputs = layers.Lambda(lambda x: x[:, windows_size:, :])(bw_outputs)
            
            concatenated = layers.Concatenate()([fw_outputs, bw_outputs])
            
            outputs = AttentionLayer(hidden_size = self.units * 2)(concatenated)
            
            
        else:   
            x = layers.Bidirectional(layers.GRU(self.units, return_sequences=True))(inputs)
            x = layers.Bidirectional(layers.GRU(self.units, return_sequences=True))(x)
            outputs = layers.Bidirectional(layers.GRU(self.units))(x)
            # outputs = layers.Bidirectional(layers.GRU(self.units))(inputs)
            
            
        
        
        return outputs
    
class BiLSTM_module:
    def __init__(self,
                 units = 64,
                 activation="tanh",
                 use_confined_attention=False,
                 cyclical_feature = False):
        self.units = units
        self.activation = activation
        self.weight_init = initializers.GlorotUniform(seed=0)
        self.cyclical = cyclical_feature
        self.use_confined_attention = use_confined_attention

    def __call__(self, inputs, windows_size = None):
        
        if(self.use_confined_attention == True and windows_size==None and not self.cyclical):
            raise ValueError('Use confined attention but your windows_size is None')
        if(self.use_confined_attention == True):
            
            # LSTM
            x = layers.Bidirectional(layers.LSTM(self.units, kernel_initializer=self.weight_init,  return_sequences=True))(inputs)
            # x = layers.LayerNormalization()(x)
            x = layers.Bidirectional(layers.LSTM(self.units, kernel_initializer=self.weight_init, return_sequences=True))(x)
            # x = layers.LayerNormalization()(x)
            fw_outputs, bw_outputs = layers.Bidirectional(layers.LSTM(self.units, kernel_initializer=self.weight_init, return_sequences=True), merge_mode=None)(x)
            
            if (not self.cyclical):
                fw_outputs = layers.Lambda(lambda x: x[:, windows_size:, :])(fw_outputs)
                bw_outputs = layers.Lambda(lambda x: x[:, windows_size:, :])(bw_outputs)

            
            concatenated = layers.Concatenate()([fw_outputs, bw_outputs])
            
            outputs = AttentionLayer(hidden_size = self.units * 2)(concatenated)
            
            # outputs = layers.LayerNormalization()(attention)
            
            
            
        else:   
            x = layers.Bidirectional(layers.LSTM(self.units, kernel_initializer=self.weight_init, return_sequences=True))(inputs)
            # x = layers.LayerNormalization()(x)
            x = layers.Bidirectional(layers.LSTM(self.units, kernel_initializer=self.weight_init, return_sequences=True))(x)
            # x = layers.LayerNormalization()(x)
            
            outputs = layers.Bidirectional(layers.LSTM(self.units, kernel_initializer=self.weight_init))(x)
            
            
        
        
        return outputs    
   
class GRU_module:
    def __init__(self,
                 units = 64,
                 activation="tanh",
                 use_confined_attention=False,
                 cyclical_feature = False):
        self.units = units
        self.activation = activation
        self.weight_init = initializers.GlorotUniform(seed=0)

    def __call__(self, inputs, windows_size = None):
        
        x = layers.GRU(self.units, kernel_initializer=self.weight_init, return_sequences=True)(inputs)
        # x = layers.LayerNormalization()(x)
        x = layers.GRU(self.units, kernel_initializer=self.weight_init, return_sequences=True)(x)
        # x = layers.LayerNormalization()(x)
        
        outputs = layers.GRU(self.units, kernel_initializer=self.weight_init)(x)
            
            
        
        
        return outputs    
   
class LSTM_module:
    def __init__(self,
                 units = 64,
                 activation="tanh",
                 use_confined_attention=False,
                 cyclical_feature = False):
        self.units = units
        self.activation = activation
        self.weight_init = initializers.GlorotUniform(seed=0)

    def __call__(self, inputs, windows_size = None):
        
        x = layers.LSTM(self.units, kernel_initializer=self.weight_init, return_sequences=True)(inputs)
        # x = layers.LayerNormalization()(x)
        x = layers.LSTM(self.units, kernel_initializer=self.weight_init, return_sequences=True)(x)
        # x = layers.LayerNormalization()(x)
        
        outputs = layers.LSTM(self.units, kernel_initializer=self.weight_init)(x)
            
            
        
        
        return outputs    


   
class CNN_module:
    def __init__(self, units = 64, pooling = 'max'):
        self.units = units
        self.pooling = pooling
        self.weight_init = initializers.GlorotUniform(seed=0)
        
    def __call__(self, inputs):
        
        shortcut = layers.TimeDistributed(layers.Conv1D(self.units * 2, 1, kernel_initializer=self.weight_init))(inputs)

        x = layers.TimeDistributed(layers.Conv1D(self.units, 3, padding='same', kernel_initializer=self.weight_init))(inputs)  
        x = layers.Activation("relu")(x)
        x = layers.TimeDistributed(layers.Conv1D(self.units * 2, 1, kernel_initializer=self.weight_init))(x)
        x = layers.Add()([shortcut, x])
        x = layers.Activation("relu")(x)

        shortcut = layers.TimeDistributed(layers.Conv1D(self.units * 2, 1, kernel_initializer=self.weight_init))(x)
        x = layers.TimeDistributed(layers.Conv1D(self.units, 3, padding='same', kernel_initializer=self.weight_init))(x)
        x = layers.Activation("relu")(x)
        x = layers.TimeDistributed(layers.Conv1D(self.units * 2, 1, kernel_initializer=self.weight_init))(x)
        x = layers.Add()([shortcut, x])
        x = layers.Activation("relu")(x)
        
        if self.pooling == 'max':
            outputs = layers.TimeDistributed(layers.GlobalMaxPooling1D())(x)
        elif self.pooling == 'avg':
            outputs = layers.TimeDistributed(layers.GlobalAveragePooling1D())(x)
        elif self.pooling == 'flatten':
            outputs = layers.TimeDistributed(layers.Flatten())(x)   
        
        
        return outputs    
        
def build_model(hidden_size, windows_size):
    time_input = Input(shape=(36, 1), name='time_input')


    # reshape = layers.Reshape((36, 2, 1))(time_input)


    # shortcut = layers.TimeDistributed(layers.Conv1D(hidden_size * 2, 1, padding='same', activation='relu'))(reshape)

    # conv = layers.TimeDistributed(layers.Conv1D(hidden_size, 3, padding='same'))(reshape)  
    # conv = layers.Activation("relu")(conv)
    # conv = layers.TimeDistributed(layers.Conv1D(hidden_size * 2, 3, padding='same'))(conv)
    # conv = layers.Add(name="1-1_add")([shortcut, conv])


    # shortcut = layers.TimeDistributed(layers.Conv1D(hidden_size * 2, 1, padding='same', activation='relu'))(conv)
    # conv = layers.TimeDistributed(layers.Conv1D(hidden_size, 3, padding='same'))(conv)
    # conv = layers.Activation("relu")(conv)
    # conv = layers.TimeDistributed(layers.Conv1D(hidden_size * 2, 3, padding='same'))(conv)
    # conv = layers.Add(name="1-2_add")([shortcut, conv])


    # conv = layers.Activation("relu", name="post_relu")(conv)
    # conv = layers.TimeDistributed(layers.GlobalAveragePooling1D())(conv)
    # conv = layers.TimeDistributed(layers.Flatten())(conv)

    model_LSTM = BiGRU_module(units=hidden_size)(time_input)
    # conv_attention = AttentionLayer(hidden_size = hidden_size * 4)(conv)
    # concatenated = layers.Concatenate()([conv_attention, model_LSTM])
    # x = layers.Dropout(0.2)(model_LSTM)
    x = layers.Dense(512, activation='relu')(model_LSTM)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)

    outputs = layers.Dense(1, activation='linear', name='output_layers')(x)
    model = models.Model(inputs=time_input, outputs=outputs)
    return model
    
def build_model_withcyc(hidden_size, windows_size):
    weight_init = initializers.GlorotUniform(seed=0)
    
    time_input = Input(shape=(36, 1), name='time_input')
    cyc_input = Input(shape=(7, 1), name='cyclical_input')
    cyc_input_ts = Input(shape=(7, 36), name='cyclical_input_ts')
    
    # reshape = layers.Reshape((36, 1, 1))(time_input)
    # reshape_ts = layers.Reshape((36, 7, 1))(cyc_input_ts)

    # shortcut = layers.Conv1D(hidden_size * 2, 1, padding='same')(time_input)

    # conv = layers.Conv1D(hidden_size, 3, padding='same')(time_input)  
    # conv = layers.Activation("relu")(conv)
    # conv = layers.Conv1D(hidden_size * 2, 1)(conv)
    # conv = layers.Add(name="1-1_add")([shortcut, conv])
    # conv = layers.Activation("relu")(conv)

    # shortcut = layers.Conv1D(hidden_size * 2, 1)(conv)
    # conv = layers.Conv1D(hidden_size, 3, padding='same')(conv)
    # conv = layers.Activation("relu")(conv)
    # conv = layers.Conv1D(hidden_size * 2, 1)(conv)
    # conv = layers.Add(name="1-2_add")([shortcut, conv])
    
    
    # conv = layers.TimeDistributed(layers.Conv1D(hidden_size * 2, 3, padding='same'))(reshape)
    # conv = layers.Activation("relu")(conv)
    # conv = layers.TimeDistributed(layers.Conv1D(hidden_size * 2, 3, padding='same'))(conv)

    
    # conv = layers.TimeDistributed(layers.Flatten())(conv)
    
    # conv_ts = CNN_module(units=hidden_size, pooling = 'avg')(reshape_ts)
    # conv = CNN_module(units=hidden_size, pooling = 'flatten')(reshape)
    
    # model_LSTM = BiLSTM_module(units=hidden_size, use_confined_attention=True)(time_input, windows_size)
    model_LSTM = BiLSTM_module(units=hidden_size)(time_input)
    # conv_attention = AttentionLayer(hidden_size = hidden_size * 2)(conv)
    cyclical_feature = BiLSTM_module(units=hidden_size)(cyc_input) 
    # cyclical_feature_ts = BiLSTM_module(units=hidden_size, use_confined_attention=True, cyclical_feature=True)(cyc_input_ts)
    cyclical_feature_ts = BiLSTM_module(units=hidden_size)(cyc_input_ts)
    # x = layers.Concatenate()([conv_attention, model_LSTM, cyclical_feature, cyclical_feature_ts])

    x = layers.Concatenate()([model_LSTM, cyclical_feature, cyclical_feature_ts])
    # x = layers.Concatenate()([model_LSTM, cyclical_feature])
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(512, kernel_initializer=weight_init, activation='relu')(x)   
    x = layers.Dense(256, kernel_initializer=weight_init, activation='relu')(x)
    x = layers.Dense(128, kernel_initializer=weight_init, activation='relu')(x)
    x = layers.Dense(64, kernel_initializer=weight_init, activation='relu')(x)

    outputs = layers.Dense(1, activation='linear', name='output_layers')(x)
    model = models.Model(inputs=[time_input,cyc_input, cyc_input_ts], outputs=outputs)
    # model = models.Model(inputs=[time_input,cyc_input], outputs=outputs)
    return model
    
# def build_cnn_withcyc(hidden_size, windows_size):
    # time_input = Input(shape=(36, 2), name='time_input')
    # cyc_input = Input(shape=(7, 1), name='cyclical_input')

    
    # conv = layers.Conv1D(hidden_size, 3, padding='same')(time_input)
    # conv = layers.Activation("relu")(conv)
    # conv = layers.Conv1D(hidden_size * 2, 3, padding='same')(conv)
    # conv = layers.Activation("relu")(conv)
    # conv = layers.Conv1D(hidden_size * 4, 3, padding='same')(conv)
    # conv = layers.Activation("relu", name="post_relu")(conv)
    # conv = layers.TimeDistributed(layers.GlobalAveragePooling1D())(conv)
    # conv = layers.Flatten()(conv)
    
    # model_LSTM = BiLSTM_module(units=hidden_size, use_confined_attention=True)(conv, windows_size)
    # cyclical_feature = BiLSTM_module(units=hidden_size)(cyc_input)
    # conv_attention = AttentionLayer(hidden_size = hidden_size * 4)(conv)
    # x = layers.Concatenate()([conv_attention, model_LSTM, cyclical_feature])
    # x = layers.Concatenate()([model_LSTM, cyclical_feature])
    # x = layers.Dropout(0.2)(x)
    # x = layers.Dense(512, activation='relu')(x)   
    # x = layers.Dense(256, activation='relu')(x)
    # x = layers.Dense(128, activation='relu')(x)
    # x = layers.Dense(64, activation='relu')(x)

    # outputs = layers.Dense(1, activation='linear', name='output_layers')(x)
    # model = models.Model(inputs=[time_input,cyc_input], outputs=outputs)
    # return model