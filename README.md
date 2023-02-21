# Infomin Representation Learning

<p align="center"><img width="60%" src="materials/front.png" /></p>

--------------------------------------------------------------------------------
This repository provides a PyTorch implementation of the paper ["Scalable Infomin Learning"](https://openreview.net/pdf?id=Ojakr9ofova), NeurIPS 2022.






## Introduction
We consider learning representation with the following objective:
$$\min L(f(X), Y) - \beta \cdot I(f(X); T)$$
where $L$ is some loss (e.g. BCE) and $I$ is the mutual information. This objective is ubiquitous in fairness, invariance, disentangled representation learning, domain adaptation, etc. In the figure above, $Y$ is the digit and $T$ is the color.  

We show that to minimise $I(f(X); T)$ above, we really need not to estimate it, which is challenging. Rather, we can simply consider a random 'slice' of $I(f(X); T)$ in each mini-batch during learning, which is much easier to estimate.

<!---
As byproduct, this project also implements a set of methods for assessing $Z \perp T$ for vectors
--->

<!---
To optimise this objective, traditionally we need to (re-)estimate $I$ before every update to $f$. However estimating $I$ is challenging. We show to minimise $I$ we indeed need not to estimate it: just consider random `slices' of it is enough.
--->


See also the followiing materials: [Poster](materials/poster.png), [Slides](materials/slides.pdf), [Demo](demo.ipynb). The demo is a minimalist jupyter notebook for trying.




## Prerequisite


### 1. Libraries

* Python 3.5+
* Pytorch 1.12.1
* Torchvision 0.13.1
* Numpy, scipy, matplotlib

We strongly recommend to use conda to manage/update library dependence:
```
conda install pytorch torchvision matplotlib
```




### 2. Data
Please run the following script to download the PIE dataset (contributed by [https://github.com/bluer555/CR-GAN](https://github.com/bluer555/CR-GAN))
```
bash scripts/download_pie.sh
```
For fairness experiments, the data is in the /data folder.



## MI estimators/independence tests

at /mi

* Pearson Correlation
* Distance Correlation
* Neural Total Correlation
* Neural Renyi Correlation
* CLUB
* Sliced mutual information




## Applications

at /tasks

* Fairness
* Disentangled representation learning
* Domain adaptation

## Results



<p align="center"><img width="95%" src="materials/result_DA.png" /></p>


<p align="center"><img width="95%" src="materials/result_disentanglement.png" /></p>


