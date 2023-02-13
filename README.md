# Infomin Representation Learning

<p align="center"><img width="60%" src="materials/thumbnail.png" /></p>

--------------------------------------------------------------------------------
This repository provides a PyTorch implementation of "Scalable Infomin Learning" [Paper](https://openreview.net/pdf?id=Ojakr9ofova), NeurIPS 2022.



## Introduction
Do we really need to estimate mutual information in order to minimise it? No!


## Materials

* [Paper](https://openreview.net/pdf?id=Ojakr9ofova)
* [Slides](materials/slides.pdf)
* [Poster](materials/poster.png)



## Prerequisite


### Libraries

* Python 3.5+
* Pytorch 1.12.1
* Torchvision 0.13.1
* Numpy, scipy, matplotlib



### Data
Please run the following script to download the PIE dataset (contributed by [https://github.com/bluer555/CR-GAN](https://github.com/bluer555/CR-GAN))
```
bash scripts/download_pie.sh
```



## MI estimators/proxies

at /mi

* Nonparametric methods: Pearson correlation, distance correlation
* Neural estimators: neural total correlation, neural Renyi correlation
* Variational estimator: CLUB
* Slice method 


## Applications

at /tasks

* Fairness
* Disentangled representation learning
* Domain adaptation
