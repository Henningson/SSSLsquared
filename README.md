![LGDV](images/lgdv_small.png) ![Phoniatric Division](images/Uniklinikum-Erlangen.svg)

![Python3](https://img.shields.io/badge/python-3.5%20%7C%203.6%20%7C%203.7-blue)
![GPL-3.0](https://img.shields.io/github/license/Henningson/vocaloid)


# SSSL^2
This repository accompanies the paper <a href="https://henningson.github.io/Vocal3D/assets/Paper.pdf">Joint Segmentation and Sub-Pixel Localization in Structured Light Laryngoscopy</a>.
This is a joint work of the <a href="https://www.lgdv.tf.fau.de/">Chair of Visual Computing</a> of the Friedrich-Alexander University of Erlangen-Nuremberg and the <a href="https://www.hno-klinik.uk-erlangen.de/phoniatrie/">Phoniatric Division</a> of the University Hospital Erlangen. 

# Why SSSL^2?
SSSL^2 := SSSLSSSL stands for **S**emantic **S**egmentation and **S**ub-Pixel Accurate **L**ocalization in **S**ingle-**S**hot **S**tructured **LL**ight Laryngoscopy.
Our approach estimates a semantic segmentation of the Vocal Folds and Glottal Gap while simultaneously predicting sub-pixel accurate laserpoint positions in an efficient manner.
It is used on a per frame basis in structured light laryngoscopy.


## Dataset
The updated Dataset was integrated into the original repository and can be found <a href="https://github.com/Henningson/HLEDataset.git">here on GitHub</a>!  

## Prerequisites
Make sure that you have a Python version >=3.5 installed.
A CUDA capable GPU is recommended, but not necessary.
However, note that inference times are most definitely higher, and training a network from scratch may take ages.

## Installation
TODO.

## Visualizing Results
We supply a Viewer that you can use to visualize the Networks.
You can find it in ```Visualization/QTViewer.py```.

## Evaluating a Trained Network
TODO.

## Training a Network from Scratch
TODO.

## Things to note
TODO.

## Limitations
TODO.

## Citation
Please cite this paper, if this work helps you with your research:
```
@InProceedings{SegAndLocalize,
  author="Henningson, Jann-Ole and Semmler, Marion and D{\"o}llinger, Michael and Stamminger, Marc",
  title="Joint Segmentation and Sub-Pixel Localization in Structured Light Laryngoscopy",
  booktitle="Medical Image Computing and Computer Assisted Intervention -- MICCAI 2023",
  year="2023",
  pages="?",
  isbn="?"
}
```
A PDF of the Paper is included in the `assets/` Folder of this repository.
However, you can also find it <a href="https://google.com/">here</a>.
