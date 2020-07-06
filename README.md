# deepQuest-mod -- Framework for neural-based Multimodal Quality Estimation

deepQuest-mod is a modified version of [deepQuest][1], a state-of-the-art neural-based Quality Estimation framework developed by the [University of Sheffield][2]. deepQuest-mod handles external sources of information, namely visual features, in order to improve its performance.

deepQuest-mod is part of the tools designed and used in my MSc Individual Project for Imperial College London.

This repository presents the updated version of deepQuest-mod. The original version submitted as part of my MSc Individual Project is available on https://github.com/shuokabe/MSc_Individual_Project-deepQuest-mod.


## System requirements

This framework runs successfully on **Python 2** with a **Theano** backend.

The TensorFlow version that is used is 1.13.1.

This framework has been tested on Microsoft Azure Virtual Machine.


## Documentation about the original deepQuest

Information about the original deepQuest framework is available on https://github.com/sheffieldnlp/deepQuest and https://sheffieldnlp.github.io/deepQuest.


## Documentation

deepQuest-mod can be used as deepQuest, since its general structure has been kept. New models were however added to incorporate visual features corresponding to the data.

Two QE levels are supported by deepQuest-mod for visual features: sentence-level biRNN model (EncSentVis) and document-level biRNN model (EncDocVis). Both require on top of the usual deepQuest input files, a file containing the visual features (dense vector), where each line represents one document.


### Quick launch files

deepQuest-mod contains two shell scripts facilitating the launching of the models: launch.sh (for document-level QE) and sent-launch.sh (for sentence-level QE).
Both copy the data files and configuration files from the root folder (one folder before deepQuest) to the folder required by deepQuest.

For example, the following command will prepare the files for a document-level QE biRNN model (EncDoc) where the task name is 'doc-level' and the name of the data is 'doc-qe-data-vis':
```
./launch.sh --task doc-level --data doc-qe-data-vis
```

In order to launch the same model but with visual features, the command will be:
```
./launch.sh --task doc-level --data doc-qe-data-vis --vis true
```

Here, the ```vis``` argument will indicate if the desired model is a Multimodal QE model with visual features or not.


### Configuration files

Since new models have been introduced, new configuration files are needed. In a similar way, deepQuest-mod also contains new shell files launching the training and testing process of Multimodal Quality Estimation.
Those new files are located in the main root folder where these have explicit names of the model such as ```config-docBiRNN-vis.py``` and ```train-test-docQEBiRNN-vis.sh``` for document-level biRNN with visual features.

The original deepQuest's command to launch the model is still usable. However, this framework also adds two new arguments:
- ```visual```: ending of the file containing the visual features
- ```vis_method```: the merging strategy of the visual features; accepted arguments are: 'concatenation', 'add-embedding' and 'time-step'

Hence, this would be a possible command to launch the document-level biRNN model with the concatenation merging strategy:
```
./train-test-docQEBiRNN-vis.sh --task doc-level --source src --target mt --visual vis --score mqm --docsize 50 --activation linear --vis_method concatenation --seed 124 --device cuda0 > log-docQEbRNN-vis.txt 2>&1 &
```


[1]: http://aclweb.org/anthology/C18-1266
[2]: https://www.sheffield.ac.uk
