# CS147 Final Project: CUDA Parallelized KNN


> Group Members: [Linda Ghunaim](https://github.com/lindgh), [Glider Mapalad](https://github.com/gmapa002), [Sabaipon Phimmala](https://github.com/bphimmala),
>
> Team: LGS

#### [Demo Recording](https://youtu.be/AReen-qWeDs) | [Presentation Slides](https://docs.google.com/presentation/d/17SKTet6EpyyGicJcMiO0oEIWMOr8oVImZsZNAaius_I/edit?usp=sharing)    



## Project Overview

LGS used CUDA to parallelize a K-Nearest neighbor (KNN) classifier that predicts the ‘star-rating’ of Amazon reviews for digital music products. The classifier used TF-IDF feature vectors extracted from the review text. Our project optimizes and improves Glider’s portion of a project from CS105. Below are links to the original project and dataset we used.

- [Original Project](https://colab.research.google.com/drive/19okzU6i1WmpZDTi0z8zsshTYogO_CSpa?usp=sharing)

- [Dataset](https://www.google.com/url?q=https%3A%2F%2Fhuggingface.co%2Fdatasets%2FMcAuley-Lab%2FAmazon-Reviews-2023)

- [Stratified Dataset](https://docs.google.com/spreadsheets/d/13_LRyNGp4SQ74kcQfe5plDogiOQ_Be6hhIiEPIni6g8/edit?usp=sharing)

## Documentation

We ran our program on the Bender server provided by BCOE Systems. After signing in, the user must `git clone` our repository. Afterwards, they must run this command to get into the Apptainer: 

```sh
apptainer shell --nv /singularity/cs217/cs217.2024-12-12.sif
```


Next, import the following dependencies by running: 

```sh
python
import numba
import pandas
``` 


Exit out of the Python specific container. Now the project is runnable through the Apptainer terminal by running:

```sh
python3 main.py
```
