"""
Trainer提供的功能有
1、初始化训练训练状态
2、计算准确率
3、run train loop
4、run test loop
5、保存模型
"""
import tensorflow as tf
import numpy as np
from tensorflow.contrib.seq2seq import sequence_loss
from tensorflow.python.layers.core import Dense

import DataSet
from Model import Encoder
from Vocabulary import SequenceVocabulary

source_vocab_size = 30
encoding_embedding_size = 100
lr = 0.01
source_path = "data/dataSource"
target_path = 'data/dataTarget'
batch_size = 4
max_sequence_length = 10
epoch = 5
train_data_size = 800

vocabulary = SequenceVocabulary()
seqDataSet = DataSet.SeqDataSet(source_path, target_path, batch_size, vocabulary)


def _get_inputs(batchsize):
    input_data = tf.placeholder(tf.int32, [batchsize, None], name='input_ids')
    # 定义target序列最大长度（之后target_sequence_length和source_sequence_length会作为feed_dict的参数）
    source_sequence_length = tf.placeholder(tf.int32, (batchsize,), name='source_sequence_length')
    target = tf.placeholder(tf.int32, [batchsize, None], name="target_ids")
    target_sequence_length = tf.placeholder(tf.int32, (batch_size,), name="target_sequence_length")
    return input_data, source_sequence_length, target, target_sequence_length


def _get_default_params():
    return {
        "cell_class": "tensorflow.contrib.rnn.BasicLSTMCell",
        "cell_params": {
            "num_units": 32
        },
        "dropout_input_keep_prob": 1.0,
        "dropout_output_keep_prob": 1.0
    }


input_data, source_sequence_length, targets, target_sequence_length = _get_inputs(batch_size)
encoder_embed_input = tf.contrib.layers.embed_sequence(input_data, source_vocab_size, encoding_embedding_size)
encoder = Encoder(tf.contrib.learn.ModeKeys.TRAIN, _get_default_params())
outputs, final_state = encoder(encoder_embed_input, source_sequence_length)

layers = Dense(source_vocab_size)
# I think I could use outputs to decoder real output.
outputs = layers(outputs)  # b * t * source_v_size
masks = tf.sequence_mask(source_sequence_length, max_sequence_length)  # b * t
masks = tf.cast(masks, tf.float32)
cost = sequence_loss(outputs, targets, masks)
optimizer = tf.train.AdamOptimizer(lr)
gradients = optimizer.compute_gradients(cost)
capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
train_op = optimizer.apply_gradients(capped_gradients)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print("start trainning")
    for epoch_i in range(1, epoch + 1):
        for batch_i, (source_batch, target_batch, source_length, target_length) in enumerate(
                seqDataSet.train_batch_generator()):
            _, loss = sess.run([train_op, cost], feed_dict={
                input_data: source_batch,
                targets: target_batch,
                target_sequence_length: target_length,
                source_sequence_length: source_length
            })

            if batch_i % 10 == 0:
                for batch_ii, (
                valid_source_batch, valid_target_batch, valid_source_length, valid_target_length) in enumerate(
                        seqDataSet.valid_batch_generator()):
                    sample1 = valid_source_batch[0]
                    source_sentence = [vocabulary.get_token_by_index(word_idx) for word_idx in sample1]

                    validation_loss, outputs_tensor = sess.run(
                        [cost, outputs],
                        {input_data: valid_source_batch,
                         targets: valid_target_batch,
                         target_sequence_length: valid_target_length,
                         source_sequence_length: valid_source_length})
                    if batch_ii % 10 == 0:
                        target_sample1 = np.argmax(outputs_tensor[0], axis=1).tolist()
                        target_sentence = [vocabulary.get_token_by_index(int(word_idx)) for word_idx in target_sample1]
                        print("source sentense " + str(source_sentence))
                        print("target sentense " + str(target_sentence))
                        print('Epoch {:>3}/{} Batch {:>4}/{} - Training Loss: {:>6.3f}  - Validation loss: {:>6.3f}'
                              .format(epoch_i,
                                      epoch,
                                      batch_i,
                                      train_data_size // batch_size,
                                      loss,
                                      validation_loss))

# you can find that the loss is decrease but the target is not my target, it's so strange.
