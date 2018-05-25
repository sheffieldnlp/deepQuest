#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import division

import os
import subprocess
import threading
import random

from math import exp
from operator import mul
from collections import defaultdict

from scorer import Scorer
from reference import Reference

# Assumes tercom.7.25 and tercom.7.25.jar is in the ../ter/ directory.  Change as needed.
TER_JAR = '../ter/tercom.7.25'


class SentenceTerScorer(Scorer):
    """
    Scores SmoothedBleuReference objects.
    """

    def __init__(self, argument_string):
        """
        Initialises metric-specific parameters.
        """
        Scorer.__init__(self, argument_string='')
        self._reference = None
        # use n-gram order of 4 by default
        self.additional_flags = argument_string

    def set_reference(self, reference_tokens):
        """
        Sets the reference against hypotheses are scored.
        """

        if hasattr(self._reference, 'extension'):
            self._reference.lock.acquire()
            clean_p = subprocess.Popen(self._reference.clean_cmd, shell=True)
            clean_p.communicate()
            self._reference.lock.release()
        self._reference = SentenceTerReference(
            reference_tokens,
            additional_flags=self.additional_flags
        )

class SentenceTerReference(Reference):
    """
    Smoothed sentence-level BLEU as as proposed by Lin and Och (2004).
    Implemented as described in (Chen and Cherry, 2014).
    """

    def __init__(self, reference_tokens, additional_flags=''):
        """
        Computes the TER of a sentence.
        :param reference_tokens: the reference translation that hypotheses shall be
                         scored against. Must be an iterable of tokens (any
                 /tmp/3420971.ref        type).
        :param additional_flags: additional TERCOM flags.
        """
        self.d = dict(os.environ.copy())
        self.d['LANG'] = 'C'
        self.extension = str(random.randint(0, 10000000))
        self.hyp_filename = "/tmp/" + self.extension + ".hyp"
        self.ref_filename = "/tmp/" + self.extension + ".ref"
        self.ter_cmd = "bash " + TER_JAR + " -r " + self.ref_filename + " -h " + self.hyp_filename \
                       + additional_flags + "| grep TER | awk '{print $3}'"
        self.clean_cmd = "rm -f " + self.ref_filename + " " + self.hyp_filename
        # Used to guarantee thread safety
        self.lock = threading.Lock()
        Reference.__init__(self, reference_tokens)
        self._gts_ter = ' '.join(reference_tokens) + '\t(sentence%d)\n' % 0
        with open(self.ref_filename, 'w') as f:
                f.write(self._gts_ter)

        # preprocess reference


    def score(self, hypothesis_tokens):
        """
        Scores @param hypothesis against this reference.

        @return the smoothed sentence-level TER score: 1.0 is best, 0.0 worst.
        """
        self.lock.acquire()
        res_ter = ' '.join(hypothesis_tokens) + '\t(sentence%d)\n' % 0
        with open(self.hyp_filename, 'w') as f:
            f.write(res_ter)

        self.ter_p = subprocess.Popen(self.ter_cmd, cwd=os.path.dirname(os.path.abspath(__file__)),
                                      stdout=subprocess.PIPE, shell=True, env=self.d)
        score = self.ter_p.stdout.read()
        self.lock.release()
        self.ter_p.kill()
        return float(score)


