############
Installation
############

DeepQuest is written in Python and we highly recommend that you use a Conda_ in order to keep under control your working environment, without interfering with your system-wide configuration, neither former installation of dependencies.

Assuming you are working in a dedicated Python environment, get DeepQuest as follows::

    git clone git@github.com:sheffieldnlp/deepQuest.git
    conda install theano
    cd deepQuest
    pip install -r requirements.txt

Computational Requirements
**************************

DeepQuest is GPU compatible and we highly recommend to train your models on GPU with a minimum of 8Go memory available.
Of course, if you don't have access to such resources, you can also use DeepQuest on a CPU server.
This will considerably extend the training time, especially for complex architecture, such as the POSTECH model (`Kim et al., 2017`_), on large datasets, but architectures such as BiRNN should work fine and take about 12 hours to be trained (while ~20min on GPU).



.. ==============================================================================
.. _Conda: https://conda.io/docs/user-guide/tasks/manage-environments.html
.. _Keras: https://github.com/MarcBS/keras
.. _Multimodal Keras Wrapper: https://github.com/lvapeab/multimodal_keras_wrapper
.. _pip: https://en.wikipedia.org/wiki/Pip_(package_manager)
.. _`NMT-Keras`: https://nmt-keras.readthedocs.io/en/latest/requirements.html
.. _`Kim et al., 2017`: http://www.statmt.org/wmt17/pdf/WMT63.pdf
