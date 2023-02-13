# Infomin Representation Learning

<p align="center"><img width="60%" src="materials/front.png" /></p>

--------------------------------------------------------------------------------
This repository provides a PyTorch implementation of "Scalable Infomin Learning" [Paper](https://openreview.net/pdf?id=Ojakr9ofova), NeurIPS 2022.



## Introduction
We consider learning representation with the following objective:
$$\min L(f(X), Y) - \beta \cdot I(f(X); T)$$
where $I$ is the mutual information. This objective is ubiquitous in fairness, disentangledment, domain adaptation.

Traditionally we need to (re-)estimate $I$ first before every update to $f$, which is usually done by training a neural net. This leads to an annoying minmax problem similar to GAN. We show that minmax learning is indeed not needed.


## Materials

* [Paper](https://openreview.net/pdf?id=Ojakr9ofova)
* [Slides](materials/slides.pdf)
* [Poster](materials/poster.png)



## Prerequisite


### 1. Libraries

* Python 3.5+
* Pytorch 1.12.1
* Torchvision 0.13.1
* Numpy, scipy, matplotlib
We recommend to use conda to update library dependence. 




### 2. Data
Please run the following script to download the PIE dataset (contributed by [https://github.com/bluer555/CR-GAN](https://github.com/bluer555/CR-GAN))
```
bash scripts/download_pie.sh
```



## MI estimators/proxies

at /mi

* Pearson Correlation
* Distance Correlation
* Neural Total Correlation
* Neural Renyi Correlation
* CLUB
* Slice method 




## Applications

at /tasks

* Fairness
* Disentangled representation learning
* Domain adaptation
