# Spoken Language Recognition on Open-Source Datasets

This was our capstone project at SMU. Link to the paper can be found here: https://scholar.smu.edu/datasciencereview/vol3/iss2/3/

Abstract:  
The field of speaker and language recognition is constantly being researched and developed, but much of this research is done on private or expensive datasets, making the field more inaccessible than many other areas of machine learning. In addition, many papers make performance claims without comparing their models to other recent research. With the recent development of public multilingual speech corpora such as Mozilla's Common Voice as well as several single-language corpora, we now have the resources to attempt to address both of these problems. We construct an eight-language dataset from Common Voice and a Google Bengali corpus as well as a five-language holdout test set from Audio Lingua. We then compare one filterbank-based model and two waveform-based models found in recent literature, all based on convolutional neural networks. We find that our filterbank-based model achieves the strongest results, with a 90.5% test accuracy on our eight-language test set and a 74.8% test accuracy on our five-language Audio Lingua test set. We conclude that some models originally trained on private datasets are also applicable to our public datasets and make suggestions on how this performance can be improved further.

Code is not intended to be run as-is, but is provided as a reference for similar projects. Some code requires folders to be created in advance or files to be moved.

Datasets used:  
[Common Voice](https://voice.mozilla.org/en/datasets)  
[Google Bengali](http://www.openslr.org/53/)  
Audio Lingua - scraped using `audio_lingua_downloader.py` script in Code folder

`dnn_models.py` and model code in `model_sincnet.py` taken from the [SincNet paper repository](https://github.com/mravanelli/SincNet/)  
`xvec` module in `model_xvec.py` provided by Daniel Garcia-Romero

Recommended citation:  
Arendale, Brady; Zarandioon, Samira; Goodwin, Ryan; and Reynolds, Douglas (2020) "Spoken Language Recognition on Open-Source Datasets," SMU Data Science Review: Vol. 3 : No. 2 , Article 3.