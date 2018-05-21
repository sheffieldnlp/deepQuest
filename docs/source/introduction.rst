==================================
Introduction to Quality Estimation
==================================

Quality Estimation (QE) (`Specia et al. 2009`_) targets the prediction of MT quality without any human intervention. QE results can be particularly useful during the costly Post-Edition (PE) process, the process of manually correcting MT output to achieve a publishable quality. QE indicates if an MT unit (word, phrase, sentence, paragraph and document) is *worth post-editing*. For instance, sentence-level QE scores help rank sentences that are worth post-editing, while word-level QE aims to spot words that need to be changed during PE. Document-level QE, on the other hand, scores or ranks documents according to their quality for fully automated MT usage scenarios where no post-editing can be performed, e.g. MT for gisting of News articles online.

The QE task is usually cast as a supervised regression or classification task with a rather small amount of manually annotated or/and post-edited data. Those data can be labeled using automatic metrics and the post-edited references. Whereas at the document, paragraph or sentence levels QE predicts automatic scores (e.g., BLEU, TER, etc.), at the word and phrase levels predictions are often binary: *OK* or *BAD*.

.. _Specia et al. 2009: http://clg.wlv.ac.uk/papers/Specia_EAMT2009.pdf

******************************
Traditional Quality Estimation
******************************

For the document, paragraph and sentence levels, QE models are usually trained using various regression algorithms (e.g., Support Vector Machines (SVMs), Multilayer Perceptron). For the word and phrase levels, algorithms such as Conditional Random Fields (CRFs) or Random Forests are used. 

QE features are traditionally characterized as *black-box* (system-independent) or *glass-box* (system-dependent, extracted from the translation process): e.g., at the word level we can distinguish the POS and the lemma of a word as system-independent features, and the system posterior probability of producing a certain word in a certain position as a glass-box feature. 

*************************
Neural Quality Estimation
*************************
Recently, neural methods have been successfully exploited to improve QE performance.
Those methods mostly rely on the encoder-decoder architecture (`Sutskever et al. 2014`_, `Bahdanau et al. 2015`_) for sequence-to-sequence prediction problems. This approach has become very popular in many applications where inputs and outputs are sequential, as natural language data. 

In an encoder-decoder approach, an input sequence is encoded into an internal representation (roughly, features learned automatically), and then an output sequence is generated from this representation. Current best practices implement encoder-decoder approaches using RNNs, which handle inputs in a sequence, while taking previous computations into account.
 
We implement two different architectures, both based on the RNN encoder-decoder approach: 

1. **POSTECH-inspired** architecture:

This architecture is a two-stage end-to-end stacked neural QE model that combines inspired by the architecture of (`Kim et al. 2017`_):

- a *Predictor*, an encoder-decoder RNN model to predict words based on their important context representations. To be more precise, it uses a modification of the standard NMT encoder-decoder architecture, which at each timestep predicts the next word :math:`e_i` taking into account not only the previously generated word :math:`e_{i-1}`, but also the following word :math:`e_{i+1}` (MT is given for the QE task). This Predictor architecture is pre-trained separately using a significant amount of parallel data; 

- an *Estimator* which is a bidirectional RNN model to produce quality estimates for words, phrases and sentences based on representations from the Predictor, the so-called *QE feature vectors* (QEFVs). These QEFVs are extracted by decomposing the Predictor softmax layer and contain weights assigned by the Predictor to the words of actual MT we seek to evaluate. 

Following (`Kim et al. 2017`_), we also reimplement a stacked architecture for multi-task learning (MTL). The MTL consists in alternating between word prediction and sentence/word/phrase-level QE objectives with one objective at a time.  

2. our **BiRNN** architecture:

It uses only two bi-directional RNNs (bi-RNN) as encoders to learn the representation of the (source, MT) sentence pair. We train source and  MT bi-RNNs independently (see Figure below). The two representations are then combined via concatenation:

.. image:: images/sent.jpg
 :height: 350px
 
For word-level QE
However, sentence-level QE scores are not simple aggregations of word-level representations: they reflect some importance of words within a sentence. Thus, a certain weighting should be applied to those representations. Such weighting is provided by the attention mechanism. 
We apply the following attention function computing a normalized weight for each hidden state :math:`h_{j}` of the RNN:

.. math:: \alpha_j = \frac{\exp(W_ah_j^\top)}{\sum_{k=1}^{J}\exp(W_ah_k^\top)}
 :label: attention
 
The resulting sentence vector is thus a weighted sum of word vectors:

.. math:: v = \sum_{j=1}^{J}\alpha_j h_{j}

This vector is then used to produce quality scores.
 
.. _Sutskever et al. 2014: https://arxiv.org/abs/1409.3215
.. _Bahdanau et al. 2015: https://arxiv.org/abs/1409.0473
.. _Kim et al. 2017: https://dl.acm.org/citation.cfm?id=3109480


*********************************
Document-Level Quality Estimation
*********************************

Our document-level framework is also a bi-RNN (an encoder, see Figure below).

.. image:: images/doc.jpg
 :height: 350px

RNNs have been successfully used for document representation.  

The document-level quality predictor takes as its input a set of sentence-level representations. The last hidden state of the decoder can be taken as the summary of an entire sequence. However, some document-level QE scores are not a simple aggregations of sentence-level QE scores. So such cases, we provide the architecture with the attention mechanism :eq:`attention` to learn weights to different representations (different sentences). Finally, the last hidden state / weighted sum of the sentence representations are used directly to make classification decisions.
 