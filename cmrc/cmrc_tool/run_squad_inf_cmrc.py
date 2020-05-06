# -*- coding: utf-8 -*-
import argparse
import os
import sys
import math
import tokenization
import collections
import tensorflow as tf
import numpy as np
import time
import six
import pdb

parser = argparse.ArgumentParser(
    description='BERT model saved model case/batch test program, exit with q')

parser.add_argument('--model', type=str,
                    default='./cmrc_1588284341', help='the path for the model')
parser.add_argument('--vocab_file', type=str,
                    default='./1586959006/vocab.txt')
parser.add_argument('--max_seq_length', type=int, default=512,
                    help='the length of sequence for text padding')
parser.add_argument('--do_lower_case', type=bool, default=True,
                    help='Whether to lower case the input text. Should be True for uncased models and False for cased models.')
parser.add_argument('--max_answer_length', type=int, default=312,
                    help='The maximum length of an answer that can be generated. This is needed because the start and end predictions are not conditioned on one another.')
parser.add_argument('--n_best_size', type=int, default=20,
                    help='"The total number of n-best predictions to generate in the nbest_predictions.json output file.')
parser.add_argument('--doc_stride', type=int, default=128,
                    help='the length of document stride')
parser.add_argument('--max_query_length', type=int, default=64,
                    help='the max length of query length')
parser.add_argument('--tensor_start_positions', type=str, default='start_positions',
                    help='the start_positions feature name for saved model')
parser.add_argument('--tensor_end_positions', type=str, default='end_positions',
                    help='the end_positions feature name for saved model')
parser.add_argument('--tensor_unique_ids', type=str, default='unique_ids',
                    help='the unique_ids feature name for saved model')
parser.add_argument('--tensor_input_ids', type=str, default='input_ids',
                    help='the input_ids feature name for saved model')
parser.add_argument('--tensor_input_mask', type=str, default='input_mask',
                    help='the input_mask feature name for saved model')
parser.add_argument('--tensor_segment_ids', type=str, default='segment_ids',
                    help='the segment_ids feature name for saved model')
parser.add_argument('--tensor_input_span_mask', type=str, default='input_span_mask',
                    help='the input_span_mask feature name for saved model')
parser.add_argument('--MODE', type=str, default='SINGLE',
                    help='SINGLE prediction or BATCH prediction')
args_in_use = parser.parse_args()


class SquadExample(object):
    """A single training/test example for simple sequence classification.

       For examples without an answer, the start and end position are -1.
    """

    def __init__(self,
                 qas_id,
                 question_text,
                 doc_tokens,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None,
                 is_impossible=False):
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % (tokenization.printable_text(self.qas_id))
        s += ", question_text: %s" % (
            tokenization.printable_text(self.question_text))
        s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
        if self.start_position:
            s += ", start_position: %d" % (self.start_position)
        if self.start_position:
            s += ", end_position: %d" % (self.end_position)
        if self.start_position:
            s += ", is_impossible: %r" % (self.is_impossible)
        return s


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 unique_id,
                 example_index,
                 doc_span_index,
                 tokens,
                 token_to_orig_map,
                 token_is_max_context,
                 input_ids,
                 input_mask,
                 segment_ids,
                 input_span_mask,
                 start_position=None,
                 end_position=None):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.input_span_mask = input_span_mask
        self.start_position = start_position
        self.end_position = end_position


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""

    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + \
            0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


class ChineseFullTokenizer(object):
    """Runs end-to-end tokenziation."""

    def __init__(self, vocab_file, do_lower_case=False):
        self.vocab = tokenization.load_vocab(vocab_file)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.wordpiece_tokenizer = tokenization.WordpieceTokenizer(
            vocab=self.vocab)
        self.do_lower_case = do_lower_case

    def tokenize(self, text):
        split_tokens = []
        for token in customize_tokenizer(text, do_lower_case=self.do_lower_case):
            for sub_token in self.wordpiece_tokenizer.tokenize(token):
                split_tokens.append(sub_token)

        return split_tokens

    def convert_tokens_to_ids(self, tokens):
        return tokenization.convert_by_vocab(self.vocab, tokens)

    def convert_ids_to_tokens(self, ids):
        return tokenization.convert_by_vocab(self.inv_vocab, ids)


def convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 doc_stride, max_query_length):
    """Loads a data file into a list of `InputBatch`s."""

    unique_id = 1000000000
    tokenizer = ChineseFullTokenizer(
        vocab_file=args_in_use.vocab_file, do_lower_case=args_in_use.do_lower_case)

    features = []
    for (example_index, example) in enumerate(examples):
        query_tokens = tokenizer.tokenize(example.question_text)

        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]

        # 将原文变成token之后，token和原始文本index的对应关系
        tok_to_orig_index = []
        # 将原文变成token之后，原始文本和token的index的对应关系
        orig_to_tok_index = []
        all_doc_tokens = []
        for (i, token) in enumerate(example.doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

        tok_start_position = None
        tok_end_position = None

        # The -3 accounts for [CLS], [SEP] and [SEP]
        max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

        # We can have documents that are longer than the maximum sequence length.
        # To deal with this we do a sliding window approach, where we take chunks
        # of the up to our max length with a stride of `doc_stride`.
        _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
            "DocSpan", ["start", "length"])
        doc_spans = []
        start_offset = 0
        while start_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            # 如果length是小于max_tokens_for_doc的话，那么就会直接break
            if start_offset + length == len(all_doc_tokens):
                break
            start_offset += min(length, doc_stride)

        for (doc_span_index, doc_span) in enumerate(doc_spans):
            tokens = []
            token_to_orig_map = {}
            token_is_max_context = {}
            # segment_ids中0代表[CLS]、第一个[SEP]和query_tokens，1代表doc和第二个[SEP]
            segment_ids = []
            input_span_mask = []
            tokens.append("[CLS]")
            segment_ids.append(0)
            input_span_mask.append(1)
            for token in query_tokens:
                tokens.append(token)
                segment_ids.append(0)
                input_span_mask.append(0)
            tokens.append("[SEP]")
            segment_ids.append(0)
            input_span_mask.append(0)  # TODO:check why

            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                # len(tokens)为[CLS]+query_tokens+[SEP]的大小，应该是doc_tokens第i个token
                token_to_orig_map[len(
                    tokens)] = tok_to_orig_index[split_token_index]

                is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                       split_token_index)
                token_is_max_context[len(tokens)] = is_max_context
                tokens.append(all_doc_tokens[split_token_index])
                segment_ids.append(1)
                input_span_mask.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)
            input_span_mask.append(0)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
                input_span_mask.append(0)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(input_span_mask) == max_seq_length

            start_position = None
            end_position = None

            features.append(
                InputFeatures(
                    unique_id=unique_id,
                    example_index=example_index,
                    doc_span_index=doc_span_index,
                    tokens=tokens,  # 可能会引入UNK
                    token_to_orig_map=token_to_orig_map,
                    token_is_max_context=token_is_max_context,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    input_span_mask=input_span_mask,
                    start_position=start_position,
                    end_position=end_position))
            unique_id += 1

    return features


def customize_tokenizer(text, do_lower_case=False):
    tokenizer = tokenization.BasicTokenizer(do_lower_case=do_lower_case)
    temp_x = ""
    text = tokenization.convert_to_unicode(text)
    for c in text:
        if tokenizer._is_chinese_char(ord(c)) or tokenization._is_punctuation(c) or tokenization._is_whitespace(c) or tokenization._is_control(c):
            temp_x += " " + c + " "
        else:
            temp_x += c
    if do_lower_case:
        temp_x = temp_x.lower()
    return temp_x.split()


def read_squad_examples(input_file):
    """Read a SQuAD json file into a list of SquadExample."""
#   with tf.gfile.Open(input_file, "r") as reader:
#     input_data = json.load(reader)["data"]

    #
    examples = []
    for entry in input_data:
        for paragraph in entry["paragraphs"]:
            paragraph_text = paragraph["context"]
            raw_doc_tokens = customize_tokenizer(
                paragraph_text, do_lower_case=args_in_use.do_lower_case)
            doc_tokens = []
            char_to_word_offset = []
            prev_is_whitespace = True

            k = 0
            temp_word = ""
            for c in paragraph_text:
                # c is whitespace
                if tokenization._is_whitespace(c) or not c.split():
                    char_to_word_offset.append(k-1)
                    continue
                else:
                    temp_word += c
                    char_to_word_offset.append(k)
                if args_in_use.do_lower_case:
                    temp_word = temp_word.lower()
                if temp_word == raw_doc_tokens[k]:
                    doc_tokens.append(temp_word)
                    temp_word = ""
                    k += 1
            if k != len(raw_doc_tokens):
                print(paragraph)
                print(doc_tokens)
                print(raw_doc_tokens)
            assert k == len(raw_doc_tokens)

            for qa in paragraph["qas"]:
                qas_id = qa["id"]
                question_text = qa["question"]
                start_position = None
                end_position = None
                orig_answer_text = None

                example = SquadExample(
                    qas_id=qas_id,
                    question_text=question_text,
                    doc_tokens=doc_tokens,
                    orig_answer_text=orig_answer_text,
                    start_position=start_position,
                    end_position=end_position)
                examples.append(example)
    tf.logging.info("**********read_squad_examples complete!**********")

    return examples


def get_final_text(pred_text, orig_text, do_lower_case):
    """Project the tokenized prediction back to the original text."""

    # When we created the data, we kept track of the alignment between original
    # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
    # now `orig_text` contains the span of our original text corresponding to the
    # span that we predicted.
    #
    # However, `orig_text` may contain extra characters that we don't want in
    # our prediction.
    #
    # For example, let's say:
    #   pred_text = steve smith
    #   orig_text = Steve Smith's
    #
    # We don't want to return `orig_text` because it contains the extra "'s".
    #
    # We don't want to return `pred_text` because it's already been normalized
    # (the SQuAD eval script also does punctuation stripping/lower casing but
    # our tokenizer does additional normalization like stripping accent
    # characters).
    #
    # What we really want to return is "Steve Smith".
    #
    # Therefore, we have to apply a semi-complicated alignment heruistic between
    # `pred_text` and `orig_text` to get a character-to-charcter alignment. This
    # can fail in certain cases in which case we just return `orig_text`.

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    # We first tokenize `orig_text`, strip whitespace from the result
    # and `pred_text`, and check if they are the same length. If they are
    # NOT the same length, the heuristic has failed. If they are the same
    # length, we assume the characters are one-to-one aligned.
    tokenizer = tokenization.BasicTokenizer(do_lower_case=do_lower_case)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        tf.logging.info(
                "Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        if args_in_use.verbose_logging:
            tf.logging.info("Length not equal after stripping spaces: '%s' vs '%s'",
                            orig_ns_text, tok_ns_text)
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in six.iteritems(tok_ns_to_s_map):
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        if args_in_use.verbose_logging:
            tf.logging.info("Couldn't map start position")
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        if args_in_use.verbose_logging:
            tf.logging.info("Couldn't map end position")
        return orig_text

    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text


def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(
        enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def get_predictions(all_examples, all_features, all_results, n_best_size,
                    max_answer_length, do_lower_case):

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction",
        ["feature_index", "start_index", "end_index", "start_logit", "end_logit"])

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]

        prelim_predictions = []
        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]

            start_indexes = _get_best_indexes(result.start_logits, n_best_size)
            end_indexes = _get_best_indexes(result.end_logits, n_best_size)
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=result.start_logits[start_index],
                            end_logit=result.end_logits[end_index]))

        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_logit + x.end_logit),
            reverse=True)

        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", ["text", "start_logit", "end_logit", "start_index", "end_index"])

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]

            tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
            orig_doc_start = feature.token_to_orig_map[pred.start_index]
            orig_doc_end = feature.token_to_orig_map[pred.end_index]
            orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
            tok_text = " ".join(tok_tokens)

            # De-tokenize WordPieces that have been split off.
            tok_text = tok_text.replace(" ##", "")
            tok_text = tok_text.replace("##", "")

            # Clean whitespace
            tok_text = tok_text.strip()
            tok_text = " ".join(tok_text.split())
            orig_text = " ".join(orig_tokens)

            final_text = get_final_text(tok_text, orig_text, do_lower_case)
            final_text = final_text.replace(' ', '')
            if final_text in seen_predictions:
                continue

            seen_predictions[final_text] = True
            nbest.append(
                _NbestPrediction(
                    text=final_text,
                    start_logit=pred.start_logit,
                    end_logit=pred.end_logit,
                    start_index=pred.start_index,
                    end_index=pred.end_index))

        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(
                _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

        assert len(nbest) >= 1

        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_logit + entry.end_logit)
            if not best_non_null_entry:
                if entry.text:
                    best_non_null_entry = entry

        probs = _compute_softmax(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            output["start_index"] = entry.start_index
            output["end_index"] = entry.end_index
            nbest_json.append(output)

        assert len(nbest_json) >= 1

        all_predictions[example.qas_id] = nbest_json[0]["text"]
        all_nbest_json[example.qas_id] = nbest_json
    return all_predictions, all_nbest_json


RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "start_logits", "end_logits"])


if __name__ == '__main__':
    predict_fn = tf.contrib.predictor.from_saved_model(args_in_use.model)
    max_seq_length = args_in_use.max_seq_length
    tokenizer = tokenization.FullTokenizer(vocab_file=args_in_use.vocab_file,
                                           do_lower_case=True)
    if args_in_use.MODE == 'SINGLE':
        # while True:
            # paragraph = input('(PRESS q to quit)请输入段落\n> ')
            # question = input('(PRESS q to quit)请输入问题\n> ')
        # if question == 'q':
        #     break

        """
            DEMO EXAMPLES
        """
        # paragraph = "《战国无双3》（）是由光荣和ω-force开发的战国无双系列      的正统第三续作。本作以三大故事为主轴，分别是以武田信玄等人为主的《关东三国      志》，织田信长等人为主的《战国三杰》，石田三成等人为主的《关原的年轻武者》      ，丰富游戏内的剧情。此部份专门介绍角色，欲知武器情报、奥义字或擅长攻击类型      等，请至战国无双系列1.由于乡里大辅先生因故去世，不得不寻找其他声优接手。从      猛将传 and Z开始。2.战国无双 编年史的原创男女主角亦有专属声优。此模式是任 >      天堂游戏谜之村雨城改编的新增模式。本作中共有20张战场地图（不含村雨城），后      来发行的猛将传再新增3张战场地图。但游戏内战役数量繁多，部分地图会有兼用的>      状况，战役虚实则是以光荣发行的2本「战国无双3 人物真书」内容为主，以下是相>      关介绍。（注：前方加☆者为猛将传新增关卡及地图。）合并本篇和猛将传的内容"
        # question = "《战国无双3》是由哪两个公司合作开发的？"
        # paragraph = "弗赖永广场（Freyung）是奥地利首都维也纳的一个三角形广场，位于内城（第一区）。这个广场最初位于古罗马堡垒Vindabona城墙以外，在12世纪，奥地利公爵亨利二世邀请爱尔兰僧侣来此修建了“苏格兰修道院”（Schottenkloster），因为当时爱尔兰被称为“新苏格兰”。修道院周围的广场也称为苏格兰广场。“弗赖永”这个名称源于古代德语词汇“frey”，意为自由。这是因为修道院拥有不受公爵管理的特权，还能保护逃亡者。1773年，其侧又新建小修道院，因其形状被称为鞋盒房子（Schubladkastenhaus）。弗赖永广场成为重要的市场，各种各样的街头艺术家以表演为生。其中一种表演是“维也纳小丑”（Wiener Hanswurst）。由于霍夫堡皇宫距此不远，在17世纪和18世纪，许多奥地利贵族在广场上和附近的Herrengasse兴建他们的城市住所。1856年，拆除了弗赖永广场和毗邻的Am Hof广场之间的房屋，以加宽街道。19世纪后期，银行和其他金融机构也迁来此处设立总部。"
        # paragraph = "“弗赖永”这个名称源于古代德语词汇“frey”，意为自由。这是因为修道院拥有不受公爵管理的特权，还能保护逃亡者。1773年，其侧又新建小修道院，因其形状被称为鞋盒房子（Schubladkastenhaus）。弗赖永广场成为重要的市场，各种各样的街头艺术家以表演为生。其中一种表演是“维也纳小丑”（Wiener Hanswurst）。由于霍夫堡皇宫距此不远，在17世纪和18世纪，许多奥地利贵族在广场上和附近的Herrengasse兴建他们的城市住所。1856年，拆除了弗赖永广场和毗邻的Am Hof广场之间的房屋，以加宽街道。19世纪后期，银行和其他金融机构也迁来此处设立总部。"
        # question = "为什么修道院的名称取自意为自由的古代德语词汇“frey”？"
        paragraph = "历时五年四次审议的《电子商务法》已于今年1月1日起实施。《电子商务法》的颁布实施为我国规范当前电子商务市场秩序、维护公平竞争环境、保障参与主体权益、促进电子商务健康快速发展奠定了法律基础。从总体上，应该看到《电子商务法》是一部以促进电子商务发展为立法目标之一的法律，是一部权益法，也是一部促进法。《电子商务法》专门设立了“电子商务促进”章节，明确了国家发展电子商务的重点方向。其中，农村电商和电商扶贫成为促进的重点."
        question = "电子商务法的目的"
        paragraph2 ="阳光板大部分使用的是聚碳酸酯（PC）原料生产，利用空挤压工艺在耐候性脆弱的PC板材上空挤压UV树脂，质量好一点的板面均匀分布有高浓度的UV层，阻挡紫外线的穿过，防止板材黄变，延长板材寿命使产品使用寿命达到10年以上。并且产品具有长期持续透明性的特点。（有单面和双面UV防护）。用途：住宅/商厦采光天幕，工厂厂房 仓库采光顶，体育场馆采光顶，广告牌，通道/停车棚，游泳池/温室覆盖，室内隔断。另本司有隔热保温的PC板材做温棚 遮阳棚 都不错2832217048@qq.com"
        question2 = "阳光板雨棚能用几年"
        paragraph3 = "藏蓝色，兼于蓝色和黑色之间，既有蓝色的沉静安宁，也有黑色的神秘成熟，既有黑色的收敛效果，又不乏蓝色的洁净长久，虽然不会大热流行，却是可以长久的信任，当藏蓝色与其他颜色相遇，你便会懂得它内在的涵养。藏蓝色+橙色单纯的藏蓝色会给人很严肃的气氛，橙色的点缀让藏蓝色也充满时尚活力。藏蓝色+白色白色是藏蓝色的最佳搭档，两者搭档最容易显得很干净，藏蓝色和白色营造的洗练感，让通勤装永远都不会过时，展现出都市女性的利落感。藏蓝色+粉色藏蓝色和粉色组合散发出成熟优雅的女人味，让粉色显出别样娇嫩。藏蓝色+米色藏蓝色和米色的搭配散发出浓郁的知性气质，稚气的设计细节更显年轻。藏蓝色+红色藏蓝色和红色的搭配更加的沉稳，也更具存在感，如果是面积差不多的服装来搭配，可以用红色的小物点缀来巧妙的平衡。藏蓝色+松石绿藏蓝色搭配柔和的松石绿色给人上品好品质的感觉，用凉鞋和项链来点缀更加具有层次感。藏蓝色+黄色明亮的黄色热情洋溢的融化了藏蓝色的冰冷静谧，细节感的设计更加具有轻松休闲的气氛。藏蓝色+金色推荐单品：藏蓝色"
        question3 = "藏蓝色配什么颜色好看"
        # question = "电子商务法的生效日期" # bad case
        # question = "电子商务法生效时间" # 一般的 case
        # question = "电子商务法实施时间" # good case
        input_data = [{
            "paragraphs": [
                {
                    "context": paragraph,
                    "qas": [
                        {
                            "question": question,
                            "id": "RANDOM_QUESTION_ID"
                        }
                    ]
                },
                {
                    "context": paragraph2,
                    "qas": [
                        {
                            "question": question2,
                            "id": "RANDOM_QUESTION_ID2"
                        }
                    ]
                },
                {
                    "context": paragraph3,
                    "qas": [
                        {
                            "question": question3,
                            "id": "RANDOM_QUESTION_ID3"
                        }
                    ]
                },
            ]
        }]
        predict_examples = read_squad_examples(input_data)

        features = convert_examples_to_features(
            examples=predict_examples,
            tokenizer=tokenizer,
            max_seq_length=args_in_use.max_seq_length,
            doc_stride=args_in_use.doc_stride,
            max_query_length=args_in_use.max_query_length)

        start_time = time.time()
        results = predict_fn({
            args_in_use.tensor_unique_ids: [feature.unique_id for feature in features],
            args_in_use.tensor_input_ids: [feature.input_ids for feature in features],
            args_in_use.tensor_input_mask: [feature.input_mask for feature in features],
            args_in_use.tensor_segment_ids: [feature.segment_ids for feature in features],
            args_in_use.tensor_input_span_mask: [feature.input_span_mask for feature in features],
        })
        print(f'elapsed time: {time.time()-start_time}s')
        print(np.shape(results['end_logits']))
        unique_ids = results['unique_ids']
        start_logits_list = results['start_logits']
        end_logits_list = results['end_logits']
        all_results = []
        for unique_id, start_logits, end_logits in zip(unique_ids, start_logits_list, end_logits_list):
            # unique_id = int(result["unique_ids"])
            # start_logits = [float(x) for x in result["start_logits"].flat]
            # end_logits = [float(x) for x in result["end_logits"].flat]
            _raw_result = RawResult(
                unique_id=unique_id,
                start_logits=start_logits,
                end_logits=end_logits)
            all_results.append(_raw_result)
        all_predictions, all_nbest_json = get_predictions(predict_examples, features, all_results,
                                                          args_in_use.n_best_size, args_in_use.max_answer_length,
                                                          args_in_use.do_lower_case)
        print(all_predictions)
        # print(all_nbest_json)
        # 分数对不上

    elif args_in_use.MODE == 'BATCH':
        pass
        # TO BE IMPLEMENTED
    else:
        raise ValueError('unsupported mode')
