from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten, Multiply
from keras.layers import Embedding
from keras.layers import Bidirectional, TimeDistributed, Permute
from keras.layers import LSTM, Conv1D, MaxPooling1D
from keras.layers import concatenate, Reshape, add, maximum, Activation, Lambda
from keras.utils import multi_gpu_model
from keras import optimizers, regularizers
from keras.optimizers import RMSprop
from keras import backend as K
from config import Config

from tools import *

class Models():
    def __init__(self, params, tasks):
        self.gpu_num = None
        self.params = params
        self.tasks = tasks
        self.index_embedding = Config.word_emb
        self.emb_len = Config.emb_len
        self.word_num = Config.word_num
        self.loss = {}
        self.input_layers = []
        self.output_layers = []
        self.single_models = {}
        self.dataset_num = len(tasks)
        self.loss_weights = []
        self.shared_embedding = Embedding(input_dim=self.word_num, output_dim=self.emb_len,
                                    weights=[self.index_embedding], name='global_embedding')

        ## Things for shared LSTM
        self.shared_forward_lstm = LSTM(units=params['lstm_output_dim'], return_sequences=True,
                                dropout=params['lstm_dropout'], recurrent_dropout=params['lstm_dropout'],
                                name='shared_forward_lstm')
        self.shared_backward_lstm = LSTM(units=params['lstm_output_dim'], return_sequences=True,
                                dropout=params['lstm_dropout'], recurrent_dropout=params['lstm_dropout'],
                                name='shared_backward_lstm', go_backwards=True)
        
        self.shared_lstm_tb_dense = Dense(params['bi_lstm_output_dim'], activation='tanh')
    

    def get_model(self, model, gpu_num=None):
        self.gpu_num = gpu_num
        if model == 'spbrnn':
            return self.build_spbrnn(self.params)
        elif model == 'aspbrnn':
            return self.build_aspbrnn(self.params)
        elif model == 'upbrnn':
            return self.build_upbrnn(self.params)
        return self.build_rnn(self.params)

    def _get_model(self):
        mtl_model = Model(inputs=self.input_layers, outputs=self.output_layers)
        if self.gpu_num is not None:
            mtl_model = multi_gpu_model(mtl_model, gpus=self.gpu_num)
        mtl_model.compile(loss=self.loss, loss_weights=self.loss_weights, metrics=['accuracy'], optimizer=self._get_op())
        self.mtl_model = mtl_model
        return mtl_model, self.single_models

    def _get_op(self):
        return RMSprop(lr=self.params['lr'], epsilon=1e-6, rho=0.9)
    
    def build_aspbrnn(self, params):
        for index in range(self.dataset_num):
            wp = lambda x:x+str(index)
            in_name = idx2inputname(index)
            max_len = self.tasks[index].max_len
            in_layer = Input(shape=(max_len,), dtype='int32', name=in_name)
            self.input_layers.append(in_layer)
            self.loss_weights.append(self.tasks[index].loss_weight)

            # task-specific embedding
            specific_embedding = Embedding(input_dim=self.word_num, output_dim=self.emb_len,
                                            weights=[self.index_embedding], name=wp('local_embedding_'))

            emb_local = specific_embedding(in_layer)
            emb_global = self.shared_embedding(in_layer)
            # concatenated embedding (None, 400)
            emb_layer = concatenate([emb_local, emb_global], axis=-1, name=wp('emb_concat_'))

            
            # For shared layer
            ###  Prepare layer
            shared_lstm_tb = TimeDistributed(self.shared_lstm_tb_dense, name=wp('shared_lstm_tb_'))
            ###  calculate
            fw_lstm_out = self.shared_forward_lstm(emb_layer)
            bw_lstm_out = self.shared_backward_lstm(emb_layer)
            brnn_concate = concatenate([fw_lstm_out, bw_lstm_out], axis=-1, name=wp('sh_brnn_concat_'))

            sh_out = shared_lstm_tb(brnn_concate)


            # For specific layer
            ### prepare layer
            _sp_fw = LSTM(units=params['lstm_output_dim'], return_sequences=True,
                                    dropout=self.tasks[index].lstm_in_dropout, recurrent_dropout=self.tasks[index].lstm_dropout,
                                    name=wp('sp_fw'))
            _sp_bw = LSTM(units=params['lstm_output_dim'], return_sequences=True,
                                    dropout=self.tasks[index].lstm_in_dropout, recurrent_dropout=self.tasks[index].lstm_dropout,
                                    name=wp('sp_bw'), go_backwards=True)
            _sp_tb = TimeDistributed(Dense(params['bi_lstm_output_dim'], activation='tanh'), name=wp('sp_fw_tb'))
            ### LSTM and Merge forward-backward LSTM
            sp_fw_out = _sp_fw(emb_layer)
            sp_bw_out = _sp_bw(emb_layer)
            sp_brnn_concate = concatenate([sp_fw_out, sp_bw_out], axis=-1, name=wp('sp_brnn_concat_'))
            sp_out = _sp_tb(sp_brnn_concate)
            ### Max pooling layer
            max_pooling = MaxPooling1D(pool_size=max_len, strides=1, padding='valid', data_format='channels_last')
            sp_final_out = max_pooling(sp_out)
            sp_final_out = Reshape((params['bi_lstm_output_dim'],),
                                name='sp_' + str(index) + '_final')(sp_final_out)
            
            
            # For attention layer
            sp_sh_concat = concatenate([sp_out, sh_out], axis=-1, name=wp('sp_sh_concat_'))
            sp_sh_tanh = TimeDistributed(Dense(params['bi_lstm_output_dim'], activation='tanh', use_bias=False), name=wp('sp_sh_tanh_'))(sp_sh_concat)
            sp_sh_tanh = TimeDistributed(Dense(1, use_bias=False))(sp_sh_tanh)
            sp_sh_tanh = Reshape((max_len, ), name=wp('sp_sh_tanh_with_h_reshape_'))(sp_sh_tanh)
            sp_sh_softmax = Activation('softmax')(sp_sh_tanh)
            sp_sh_softmax = Reshape((max_len, 1), name=wp('sp_sh_softmax_reshape_'))(sp_sh_softmax)
            sp_sh_softmax = Lambda(lambda x: K.repeat_elements(x, params['bi_lstm_output_dim'], -1), name=wp('sp_sh_softmax_repeat_'))(sp_sh_softmax)
            sh_out_with_relativity = Multiply()([sp_sh_softmax , sh_out])
            sh_out_with_relativity = Permute((2, 1))(sh_out_with_relativity)
            sh_final_out = Lambda(lambda x: K.sum(x, axis=-1), output_shape=(params['bi_lstm_output_dim'], ))(sh_out_with_relativity)

            merge_out = concatenate([sp_final_out, sh_final_out], axis=-1)
            
            
            # fc before softmax
            merge_out = Dense(units=params['final_output_dim'], activation='relu', name=wp('fc_layer_'))(merge_out)
            merge_out = Dropout(self.tasks[index].fc_dropout)(merge_out)
            # Design output layer: softmax
            out_name = idx2outputname(index)
            num_class = self.tasks[index].label_size
            self.loss[out_name] = 'categorical_crossentropy'
            out_layer = Dense(units=num_class, activation='softmax',
                            activity_regularizer=regularizers.l2(self.tasks[index].regularizers), name=out_name)(merge_out)
            self.output_layers.append(out_layer)
            ## Generate model
            curr_model = Model(inputs=in_layer, outputs=out_layer)
            if self.gpu_num is not None:
                curr_model = multi_gpu_model(curr_model, gpus=self.gpu_num)
            curr_model.compile(loss=self.loss[out_name], optimizer=self._get_op(), metrics=['accuracy'])
            self.single_models['model_'+str(index)] = curr_model

        return self._get_model()

    def build_spbrnn(self, params):
        for index in range(self.dataset_num):
            wp = lambda x:x+str(index)
            in_name = idx2inputname(index)
            max_len = self.tasks[index].max_len
            in_layer = Input(shape=(max_len,), dtype='int32', name=in_name)
            self.input_layers.append(in_layer)
            self.loss_weights.append(self.tasks[index].loss_weight)

            # task-specific embedding
            specific_embedding = Embedding(input_dim=self.word_num, output_dim=self.emb_len,
                                            weights=[self.index_embedding], name=wp('local_embedding_'))

            emb_local = specific_embedding(in_layer)
            emb_global = self.shared_embedding(in_layer)
            # concatenated embedding (None, 400)
            emb_layer = concatenate([emb_local, emb_global], axis=-1, name=wp('emb_concat_'))

            # For shared layer
            ###  Prepare layer
            shared_lstm_tb = TimeDistributed(self.shared_lstm_tb_dense, name=wp('shared_lstm_tb_'))
            ###  calculate
            fw_lstm_out = self.shared_forward_lstm(emb_layer)
            bw_lstm_out = self.shared_backward_lstm(emb_layer)
            brnn_concate = concatenate([fw_lstm_out, bw_lstm_out], axis=-1, name=wp('sh_brnn_concat_'))

            sh_out = shared_lstm_tb(brnn_concate)

            ### calculate relativity
            x2h = concatenate([emb_local, sh_out], axis=-1, name=wp('concate_sh_emb_'))
            wx2h = TimeDistributed(Dense(params['lstm_output_dim'], activation='sigmoid', use_bias=False), name=wp('relative_'))(x2h)

            sh_out_with_relativity = Multiply()([wx2h, sh_out])
            
            ## For specific layer
            ### prepare layer
            _sp_fw = LSTM(units=params['lstm_output_dim'], return_sequences=True,
                                    dropout=params['lstm_dropout'], recurrent_dropout=params['lstm_dropout'],
                                    name=wp('sp_fw'))
            _sp_bw = LSTM(units=params['lstm_output_dim'], return_sequences=True,
                                    dropout=params['lstm_dropout'], recurrent_dropout=params['lstm_dropout'],
                                    name=wp('sp_bw'), go_backwards=True)
            _sp_tb = TimeDistributed(Dense(params['lstm_output_dim'], activation='tanh'), name=wp('sp_fw_tb'))
            ### calculate
            sp_fw_out = _sp_fw(emb_layer)
            sp_bw_out = _sp_bw(emb_layer)
            sp_brnn_concate = concatenate([sp_fw_out, sp_bw_out], axis=-1, name=wp('sp_brnn_concat_'))
            sp_out = _sp_tb(sp_brnn_concate)

            sh_sp_concate = concatenate([sp_out, sh_out_with_relativity], axis=-1, name=wp('sh_sp_concate'))
            sh_sp_out = Dense(params['bi_lstm_output_dim'], name=wp('sh_sp_out'), activation='tanh')(sh_sp_concate)
            sh_sp_out = Dropout(0.5)(sh_sp_out)
            # max pooling layer
            max_pooling = MaxPooling1D(pool_size=max_len, strides=1, padding='valid', data_format='channels_last')
            merge_out = max_pooling(sh_sp_out)
            merge_out = Reshape((params['bi_lstm_output_dim'],),
                                name='reshape_' + str(index) + '_final')(merge_out)
            # fc before softmax
            merge_out = Dense(units=params['bi_lstm_output_dim'], activation='relu', name=wp('fc_layer_'))(merge_out)
            merge_out = Dropout(0.5)(merge_out)
            ## Design output layer: softmax
            out_name = idx2outputname(index)
            num_class = self.tasks[index].label_size
            self.loss[out_name] = 'categorical_crossentropy'
            out_layer = Dense(units=num_class, activation='softmax',
                            activity_regularizer=regularizers.l2(self.tasks[index].regularizers), name=out_name)(merge_out)
            self.output_layers.append(out_layer)
            ## Generate model
            curr_model = Model(inputs=in_layer, outputs=out_layer)
            if self.gpu_num is not None:
                curr_model = multi_gpu_model(curr_model, gpus=self.gpu_num)
            curr_model.compile(loss=self.loss[out_name], optimizer=self._get_op(), metrics=['accuracy'])
            self.single_models['model_'+str(index)] = curr_model

        return self._get_model()

    def build_rnn(self, params):
        lstm = LSTM(units=params['lstm_output_dim'], return_sequences=False,
                                dropout=params['lstm_dropout'], recurrent_dropout=params['lstm_dropout'],
                                name='lstm')
        for index in range(self.dataset_num):
            wp = lambda x:x+str(index)
            in_name = idx2inputname(index)
            out_name = idx2outputname(index)
            max_len = self.tasks[index].max_len
            in_layer = Input(shape=(max_len,), dtype='int32', name=in_name)
            self.input_layers.append(in_layer)
            self.loss_weights.append(self.tasks[index].loss_weight)

            # concate shared embedding and specific embedding
            specific_embedding = Embedding(input_dim=self.word_num, output_dim=self.emb_len,
                                            weights=[self.index_embedding], name=wp('local_embedding_'))
            emb_local = specific_embedding(in_layer)

            ## LSTM
            fw_lstm_out = lstm(emb_local)
            fc_layer = Dense(units=params['bi_lstm_output_dim'], activation='relu', name=wp('fc_layer_'))(fw_lstm_out)

            ## Design output layer: softmax
            num_class = self.tasks[index].label_size
            self.loss[out_name] = 'categorical_crossentropy'
            out_layer = Dense(units=num_class, activation='softmax',
                            activity_regularizer=regularizers.l2(self.tasks[index].regularizers), name=out_name)(fc_layer)
            self.output_layers.append(out_layer)

            ## Generate model
            curr_model = Model(inputs=in_layer, outputs=out_layer)
            if self.gpu_num is not None:
                curr_model = multi_gpu_model(curr_model, gpus=self.gpu_num)
            curr_model.compile(loss=self.loss[out_name], optimizer=self._get_op(), metrics=['accuracy'])
            self.single_models['model_'+str(index)] = curr_model

        return self._get_model()

    def build_upbrnn(self, params):
        for index in range(self.dataset_num):

            in_name = idx2inputname(index)
            in_layer = Input(shape=(params['max_len_list'][index],), dtype='int32', name=in_name)
            self.input_layers.append(in_layer)

            
            # design the task-specific layer
            specific_embedding = Embedding(input_dim=params['num_words'], output_dim=params['embedding_len'],
                                            weights=[self.index_embedding], name='local_embedding_'+str(index))

            emb_local = specific_embedding(in_layer)
            emb_global = self.shared_embedding(in_layer)

            emb_layer = concatenate([emb_local, emb_global], axis=2, name='emb_concat_'+str(index))

            ###  Prepare layer
            shared_forward_lstm_tb = TimeDistributed(self.share_tb_forward_dense, name=wp('shared_forward_lstm_tb_'))
            shared_backward_lstm_tb = TimeDistributed(self.share_tb_backward_dense, name=wp('shared_backward_lstm_tb'))
            ###  calculate
            fw_lstm_out = self.shared_forward_lstm(emb_layer)
            bw_lstm_out = self.shared_backward_lstm(emb_layer)

            fw_tb_out = self.shared_forward_lstm_tb(fw_lstm_out)
            bw_tb_out = self.shared_backward_lstm_tb(bw_lstm_out)

            brnn_out = concatenate([fw_tb_out, bw_tb_out], axis=2, name='brnn_concat_'+str(index))
            merge_out = Dense(params['bi_lstm_output_dim'], activation='tanh', name='combine_brnn_'+str(index))(brnn_out)
            merge_out = Dropout(0.5)(merge_out)
            # max pooling layer
            max_pooling = MaxPooling1D(pool_size=params['max_len_list'][index], strides=1, padding='valid', data_format='channels_last')
            merge_out = max_pooling(merge_out)
            merge_out = Reshape((params['bi_lstm_output_dim'],),
                                name='reshape_' + str(index) + '_final')(merge_out)
            out_name = idx2outputname(index)
            num_class = self.tasks[index].label_size
            # if num_class == 2:
            # 	loss[out_name] = 'binary_crossentropy'
            # 	out_layer = Dense(units=1, activation='sigmoid',
            # 					  activity_regularizer=params['regularizers'][index], name=out_name)(merge_out)
            # else:
            self.loss[out_name] = 'categorical_crossentropy'
            out_layer = Dense(units=num_class, activation='softmax',
                            activity_regularizer=params['regularizers'][index], name=out_name)(merge_out)
            self.output_layers.append(out_layer)

            curr_model = Model(inputs=in_layer, outputs=out_layer)
            if self.gpu_num is not None:
                curr_model = multi_gpu_model(curr_model, gpus=self.gpu_num)
            curr_model.compile(loss=self.loss[out_name], optimizer=self._get_op(), metrics=['accuracy'])
            self.single_models['model_'+str(index)] = curr_model

        return self._get_model()
