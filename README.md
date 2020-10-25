# Machine Learning - Project 1

The code of the project of team 'answer42' for [EPFL Machine Learning Higgs](https://www.aicrowd.com/challenges/epfl-machine-learning-higgs) challenge for year 2020.

## Software requirements

* Python version: TO BE ADDED

_Disclaimer_: The requirements aren't strict, but are recommended as all the code was tested using them

## Tested models

| Model            | Peak accuracy (%) |
|:-----------------|:-----------------:|
| Ensemble model   | 84                |

### Author's notes

Pretty late into experimenting with big number of augmented features we noticed that some of the implemented functions (notably cross validation [mainly because of stratification part] and pairwise mutliplication) are not memory optimized for work with huge amount of data and therefore require a lot of memory. This effect is especially noticeable in Google Colab, which was used for testing different setups and models, as sessions would often crash due to the lack of RAM. We don't fully understand the effect this may have on local execution of the code on PC, but expect it to be rather slow if there is not enough RAM. The  files provided to recreate the final submission are expected to require anywhere between 12 and 16 GB of memory.

## Authors

* Andrei Atanov
* Valentina Šumovskaya
* Miloš Vujasinović
