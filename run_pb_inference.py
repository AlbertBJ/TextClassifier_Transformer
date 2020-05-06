import argparse
import os
import sys
import tokenization
import tensorflow as tf
import numpy as np
import time
parser = argparse.ArgumentParser(
    description='BERT model pb model case/batch test program, exit with q')

parser.add_argument('--model', type=str,
                    default='./inference/robert_tiny_clue/frozen_model.pb', help='the path for the model')
#TODO: CHECK vocab file
parser.add_argument('--vocab_file', type=str,
                    # default='../ALBERT/albert_base/vocab_chinese.txt')
                    default='/Users/huangdongxiao2/CodeRepos/SesameSt/albert_zh/inference/robert_tiny_clue/vocab.txt')
parser.add_argument('--labels', type=list, default=[
                    'happy', 'anger', 'lost', 'fear', 'sad', 'other', 'anxiety'], help='label list')
parser.add_argument('--max_seq_length', type=int, default=128,
                    help='the length of sequence for text padding')
parser.add_argument('--tensor_input_ids', type=str, default='input_ids:0',
                    help='the input_ids op_name for graph, format： <op_name>:<output_index>')
parser.add_argument('--tensor_input_mask', type=str, default='input_mask:0',
                    help='the input_mask op_name for graph, format： <op_name>:<output_index>')
parser.add_argument('--tensor_segment_ids', type=str, default='segment_ids:0',
                    help='the segment_ids op_name for graph, format： <op_name>:<output_index>')
parser.add_argument('--tensor_output', type=str, default='loss/pred_prob:0',
                    help='the output op_name for graph, format： <op_name>:<output_index>')
parser.add_argument('--MODE', type=str, default='SINGLE',
                    help='SINGLE prediction or BATCH prediction')
args_in_use = parser.parse_args()
"""
gpu settting
"""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # default: 0
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
"""
load pb model and predict
"""


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None):
        """Constructs a InputExample.
        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 is_real_example=True):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.is_real_example = is_real_example


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def convert_single_example(example, label_list, max_seq_length,
                           tokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    tokens_a = tokenizer.tokenize(example.text_a)
    tokens_b = None
    if example.text_b:
        tokens_b = tokenizer.tokenize(example.text_b)

    if tokens_b:
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0     0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    if tokens_b:
        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        is_real_example=True)
    return feature


with tf.Graph().as_default():
    output_graph_def = tf.GraphDef()
    label_list = args_in_use.labels
    label_map = {i: label for i, label in enumerate(label_list)}

    max_seq_length = args_in_use.max_seq_length
    """
   load pb model
    """
    with open(args_in_use.model, 'rb') as f:
        output_graph_def.ParseFromString(f.read())
        tf.import_graph_def(output_graph_def, name='')
    """
    enter a text and predict
    """
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        input_ids = sess.graph.get_tensor_by_name(
            args_in_use.tensor_input_ids)
        input_mask = sess.graph.get_tensor_by_name(
            args_in_use.tensor_input_mask)
        segment_ids = sess.graph.get_tensor_by_name(
            args_in_use.tensor_segment_ids)
        tokenizer = tokenization.FullTokenizer(
            vocab_file=args_in_use.vocab_file, do_lower_case=True)
        output = args_in_use.tensor_output
        if args_in_use.MODE == 'SINGLE':
            while 1:
                question = input("enter a sentence:")
                if question == 'q' or question == 'quit()':
                    break
                predict_example = InputExample('id', question, None)
                feature = convert_single_example(
                    predict_example, label_list, max_seq_length, tokenizer)
                # print(feature.input_ids)
                # print(feature.input_mask)
                # print(feature.segment_ids)
                feed_dict = {
                    input_ids: [feature.input_ids],
                    input_mask: [feature.input_mask],
                    segment_ids: [feature.segment_ids],
                }
                # works fine
                # feed_dict = {
                #     args_in_use.tensor_input_ids: [feature.input_ids],
                #     args_in_use.tensor_input_mask: [feature.input_mask],
                #     args_in_use.tensor_segment_ids: [feature.segment_ids],
                # }
                start_time = time.time()
                y_pred_cls = sess.run(output, feed_dict=feed_dict)
                print(f'elapsed time: {time.time()-start_time}s')
                max_index = np.argmax(y_pred_cls[0])
                print(" current results ", y_pred_cls)
                print(f'label: {label_map[max_index]}')
        elif args_in_use.MODE == 'BATCH':
            questions = [
                '我要投诉的',
                '我很不开心',
                '我好喜欢你'
            ]
            features = []
            for i, question in enumerate(questions):
                predict_example = InputExample(f'id{i}', question, None)
                feature = convert_single_example(
                    predict_example, label_list, max_seq_length, tokenizer)
                features.append(feature)
            feed_dict = {
                input_ids: [feature.input_ids for feature in features],
                input_mask: [feature.input_mask for feature in features],
                segment_ids: [feature.segment_ids for feature in features],
            }
            y_pred_cls = sess.run(output, feed_dict=feed_dict)
            max_idxs = np.argmax(y_pred_cls, 1)
            print(y_pred_cls)
            print(
                f'labels: {[label_map[max_index] for max_index in max_idxs]}')
        else:
            raise ValueError('unsupported mode')
