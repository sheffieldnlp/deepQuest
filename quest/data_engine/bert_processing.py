#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# bert_processing.py
#
# Copyright (C) 2019 Frederic Blain (feedoo) <f.blain@sheffield.ac.uk>
#
# Licensed under the "THE BEER-WARE LICENSE" (Revision 42):
# Fred (feedoo) Blain wrote this file. As long as you retain this notice you
# can do whatever you want with this stuff. If we meet some day, and you think
# this stuff is worth it, you can buy me a tomato juice or coffee in return
#

import os
import codecs
import plac

import tensorflow as tf
import tensorflow_hub as hub

import bert_tokenization as tokenization


"""
This script processes input data to be used with the BERT layer

Steps:
    1. load data from file. Access to data to be given either as args, or
    deepquest config file
    2. create BERT tokenizer from Tenforflow HUB
    3. process the data
    4. save the data

The output should be three files (_token, _mask, _segids).
"""


# By default, we use the multilingual cased BERT model
bert_hub_module_handle = 'https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1'


def create_bert_vocab(dataset, id_name):
    """
    This function 'overwrites' the 'build_vocabulary()' function from
    keras_wrapper/dataset.py, in order to consider the vocab as defined by BERT
    """
    vocab_dict = get_bert_vocab(bert_hub_module_handle)

    inv_vocab_dict = {v: k for k, v in vocab_dict.items()}

    dataset.vocabulary[id_name]['words2idx'] = vocab_dict
    dataset.vocabulary[id_name]['idx2words'] = inv_vocab_dict

    dataset.vocabulary_len[id_name] = len(vocab_dict.keys())
    # if id_name not in dataset.vocabulary:
    #     inv_vocab_dict = {v: k for k, v in vocab_dict.items()}
    #
    #     dataset.vocabulary[id_name]['words2idx'] = vocab_dict
    #     dataset.vocabulary[id_name]['idx2words'] = inv_vocab_dict
    #
    #     dataset.vocabulary_len[id_name] = len(vocab_dict.keys())
    #
    # else:
    #     print("/!\ updating old keys...")
    #     old_keys = dataset.vocabulary[id_name]['words2idx'].keys()
    #     new_keys = vocab_dict.keys()
    #     added = 0
    #     for key in new_keys:
    #         if key not in old_keys:
    #             dataset.vocabulary[id_name]['words2idx'][key] = dataset.vocabulary_len[id_name]
    #             dataset.vocabulary_len[id_name] += 1
    #             added += 1
    #
    #     inv_dictionary = {v: k for k, v in dataset.vocabulary[id_name]['words2idx'].items()}
    #     dataset.vocabulary[id_name]['idx2words'] = inv_dictionary
    #
    #     if not dataset.silence:
    #         print('Appending ' + str(added) + ' words to dictionary with id "' + id_name + '".')
    #         logging.info('\tThe new total is ' + str(dataset.vocabulary_len[id_name]) + '.')


def get_bert_vocab(bert_hub_module_handle=bert_hub_module_handle):
  """
  Build and return a dict with BERT vocab file from the Hub module.
  """
  with tf.Graph().as_default():
    bert_module = hub.Module(bert_hub_module_handle)
    tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
    with tf.Session() as sess:
      vocab_file = sess.run([tokenization_info["vocab_file"]])
  # voca_dict is not empty because of the hardcoded:
  # X_out[i, j + offset_j] = vocab.get(w, vocab['<unk>'])
  # from LoadText() in keras_wrapper/dataset.py
  # therefore in 'words2idx', both '<unk>' and '[PAD]' point to the same indice (i.e. 0)
  # while in 'idx2words', '0' points to '[PAD]' (normally
  vocab_dict = {'<unk>': 0}
  with codecs.open(vocab_file[0], 'r', 'utf-8') as fh:
      idx = 0
      for line in fh:
        tok = line.rstrip()
        vocab_dict[tok] = idx
        idx += 1
  return vocab_dict


def create_tokenizer_from_hub_module(bert_hub_module_handle):
  """
  Get the vocab file and casing info from the Hub module,
  and return a tokenizer
  source: https://github.com/google-research/bert/blob/master/run_classifier_with_tfhub.py
  """
  with tf.Graph().as_default():
    bert_module = hub.Module(bert_hub_module_handle)
    tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
    with tf.Session() as sess:
      vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"],
                                            tokenization_info["do_lower_case"]])
  return tokenization.FullTokenizer(
      vocab_file=vocab_file, do_lower_case=do_lower_case)

def convert_sentence_to_features(sentence, tokenizer, max_seq_len):
    tokens = ['[CLS]']
    tokens.extend(tokenizer.tokenize(sentence))
    if len(tokens) > max_seq_len-1:
        tokens = tokens[:max_seq_len-1]
    tokens.append('[SEP]')

    segment_ids = [0] * len(tokens)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)

    #Zero Mask till seq_length
    zero_mask = [0] * (max_seq_len-len(tokens))
    input_ids.extend(zero_mask)
    input_mask.extend(zero_mask)
    segment_ids.extend(zero_mask)

    return tokens, input_ids, input_mask, segment_ids


def convert_sentences_to_features(sentences, tokenizer, max_seq_len=70):
    all_input_ids = []
    all_input_mask = []
    all_segment_ids = []
    all_tokens = []

    for sentence in sentences:
        tokens, input_ids, input_mask, segment_ids = convert_sentence_to_features(sentence, tokenizer, max_seq_len)
        all_tokens.append(tokens)
        all_input_ids.append(input_ids)
        all_input_mask.append(input_mask)
        all_segment_ids.append(segment_ids)

    return all_tokens, all_input_ids, all_input_mask, all_segment_ids

def preprocessTextBert(config):
    '''
    def setInput(self, path_list, set_name, type='raw-image', id='image', repeat_set=1, required=True,
    '''
    """
    def preprocessText(self, annotations_list, id, set_name, tokenization, build_vocabulary, max_text_len,
                       max_words, offset, fill, min_occ, pad_on_batch, words_so_far, bpe_codes=None, separator='@@', use_extra_words=True, use_pad=False):
        Preprocess 'text' data type: Builds vocabulary (if necessary) and preprocesses the sentences.
        Also sets Dataset parameters.

        :param annotations_list: Path to the sentences to process.
        :param id: Dataset id of the data.
        :param set_name: Name of the current set ('train', 'val', 'test')
        :param tokenization: Tokenization to perform.
        :param build_vocabulary: Whether we should build a vocabulary for this text or not.
        :param max_text_len: Maximum length of the text. If max_text_len == 0, we treat the full sentence as a class.
        :param max_words: Maximum number of words to include in the dictionary.
        :param offset: Text shifting.
        :param fill: Whether we path with zeros at the beginning or at the end of the sentences.
        :param min_occ: Minimum occurrences of each word to be included in the dictionary.
        :param pad_on_batch: Whether we get sentences with length of the maximum length of the minibatch or sentences with a fixed (max_text_length) length.
        :param words_so_far: Experimental feature. Should be ignored.
        :param bpe_codes: Codes used for applying BPE encoding.
        :param separator: BPE encoding separator.

        :return: Preprocessed sentences.
        """

    # we load the configuration options
    import imp
    conf = imp.load_source("load_parameters", config)
    params = conf.load_parameters()

    cache_dir = params.get('TFHUB_CACHE_DIR', None)
    if not cache_dir:
        import tempfile
        cache_dir = os.path.join(tempfile.gettempdir(), "tfhub_modules")
    # print("using {} as cache_dir".format(cache_dir))
    os.environ['TFHUB_CACHE_DIR'] = cache_dir

    # we import/create the BERT tokenizer
    tokenizer = create_tokenizer_from_hub_module(bert_hub_module_handle)

    # we process data for each dataset
    for split in ['train', 'val', 'test']:
        for lang in [params['SRC_LAN'], params['TRG_LAN']]:
            sentences = []
            cur_file = "{}/{}{}".format(
                    params['DATA_ROOT_PATH'],
                    params['TEXT_FILES'][split],
                    lang
                    )
            with codecs.open(cur_file, 'r', 'utf-8') as list_:
                for line in list_:
                    sentences.append(line.rstrip('\n'))

            tok_sentences, sentences_ids, sentences_mask, segment_ids = convert_sentences_to_features(
                    sentences,
                    tokenizer,
                    params['MAX_INPUT_TEXT_LEN']
                    )

            with codecs.open("{}.ids".format(cur_file), 'w', encoding='utf-8') as fsentids, \
                codecs.open("{}.mask".format(cur_file), 'w', encoding='utf-8') as fmask, \
                codecs.open("{}.segids".format(cur_file), 'w', encoding='utf-8') as fsegids:
                    # for sids, smask, segids in zip(sentences_ids, sentences_mask, segment_ids):
                    for sids, smask, segids in zip(tok_sentences, sentences_mask, segment_ids):
                        # fsentids.write(u"{}\n".format(' '.join(list(map(str, sids)))))
                        fsentids.write(u"{}\n".format(' '.join(sids)))
                        fmask.write(u"{}\n".format(' '.join(list(map(str, smask)))))
                        fsegids.write(u"{}\n".format(' '.join(list(map(str, segids)))))

if __name__ == '__main__':
    import plac
    plac.call(preprocessTextBert)
