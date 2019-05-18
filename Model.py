import tensorflow as tf
from TrainingUtil import get_rnn_cell

"""
Encoder描述了Encoder的框架
"""


class Encoder(object):
    def __init__(self, mode, params, name: str = "forword_RNN_encoder"):
        self._name = name
        self._params = params
        self.__revise_rnn_cell_config_dropout_by_mode(mode)
        return

    def __call__(self, *args, **kwargs):
        return self._build(*args, **kwargs)

    def _build(self, inputs, sequence_length, **kwargs):
        """
        build函数用来构建网络，input是placeholder，sequence_length也是placeholder，它们都是外部提供的，用来提供输入！
        :param inputs:
        :param sequence_length:
        :param kwargs:
        :return:
        """
        cell = get_rnn_cell(**self._params)
        outputs, state = tf.nn.dynamic_rnn(
            cell=cell,
            inputs=inputs,
            sequence_length=sequence_length,
            dtype=tf.float32,
            **kwargs)  # dynamic_rnn需要三个关键输入，cell，inputs和sequence_length
        return outputs, state

    def __revise_rnn_cell_config_dropout_by_mode(self, mode):
        if mode != tf.contrib.learn.ModeKeys.TRAIN:
            self._params["rnn_cell_config"]["dropout_input_keep_prob"] = 1.0
            self._params["rnn_cell_config"]["dropout_output_keep_prob"] = 1.0


"""
Decoder描述了Decoder的框架
"""
