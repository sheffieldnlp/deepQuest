import sys
import numpy as np
import random
import os

init = [line.rstrip('\n') for line in open(sys.argv[1])]
in_path = os.path.dirname(os.path.realpath(sys.argv[1]))

np.random.seed(0)
data = np.arange(len(init))
np.random.shuffle(data)
nb_test = 3000
nb_dev = 3000

dev_test = data[:nb_test+nb_dev]
dev = dev_test[:nb_dev]
test = dev_test[nb_dev:]
train = data[nb_test+nb_dev:]

with open(in_path + '/train', 'w') as ftr, open(in_path + '/dev', 'w') as fdev, open(in_path + '/test', 'w') as ftest: 
    ftr.writelines(["%s\n" % init[i]  for i in train])
    fdev.writelines(["%s\n" % init[i]  for i in dev])
    ftest.writelines(["%s\n" % init[i]  for i in test])
