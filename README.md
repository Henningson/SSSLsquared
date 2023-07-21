![LGDV](assets/lgdv_small.png) ![Phoniatric Division](assets/Uniklinikum-Erlangen.svg)  
![Python3](https://img.shields.io/badge/python-3.5%20%7C%203.6%20%7C%203.7-blue)
![GPL-3.0](https://img.shields.io/github/license/Henningson/vocaloid)


# SSSL^2
This repository accompanies the paper <a href="https://henningson.github.io/Vocal3D/assets/Paper.pdf">Joint Segmentation and Sub-Pixel Localization in Structured Light Laryngoscopy</a>.
This is a joint work of the <a href="https://www.lgdv.tf.fau.de/">Chair of Visual Computing</a> of the Friedrich-Alexander University of Erlangen-Nuremberg and the <a href="https://www.hno-klinik.uk-erlangen.de/phoniatrie/">Phoniatric Division</a> of the University Hospital Erlangen. 

## Why SSSL^2?
SSSL^2 := SSSLSSSL stands for **S**emantic **S**egmentation and **S**ub-Pixel Accurate **L**ocalization in **S**ingle-**S**hot **S**tructured **L**ight Laryngoscopy.
Our approach estimates a semantic segmentation of the Vocal Folds and Glottal Gap while simultaneously predicting sub-pixel accurate laserpoint positions in an efficient manner.
It is used on a per frame basis (single-shot) in an active reconstruction setting (structured light) in laryngeal endoscopy (laryngoscopy).

## Dataset
The updated Dataset was integrated into the original repository and can be found <a href="https://github.com/Henningson/HLEDataset.git">here on GitHub</a>!  

## Quantitative Evaluation and Comparison
|            | Precision :arrow_up:   | F1-Score :arrow_up:    | IoU :arrow_up:        | DICE :arrow_up:       | Inf. Speed (ms) :arrow_down: | FPS :arrow_up: |
|------------|-------------|-------------|-------------|-------------|-----------------|-----|
| Baseline   | 0.64        | 0.69        |             |             |                 |     |
| U-LSTM[8]  | 0.70 ± 0.41 | 0.58 ± 0.32 | 0.52 ± 0.18 | 0.77 ± 0.08 | 65.57 ± 0.31    | 15  |
| U-Net[18]  | **0.92** ± 0.08 | **0.88** ± 0.04 | **0.68** ± 0.08 | **0.88** ± 0.02 | 4.54 ± 0.03     | 220 |
| Sharan[21] | 0.17 ± 0.19 | 0.16 ± 0.17 |             |             | 5.97 ± 0.25     | 168 |
| 2.5D U-Net | 0.90 ± 0.08 | 0.81 ± 0.05 | 0.65 ± 0.06 | 0.87 ± 0.02 | **1.08** ± 0.01     | **926** |

## Prerequisites
Make sure that you have a Python version >=3.5 installed.
A CUDA capable GPU is recommended, but not necessary.
However, note that inference times are most definitely higher, and training a network from is not recommended.

## Installation
We supply a environment.yaml inside this folder.
You can use this to easily setup a conda environment.

## Pretrained Models
TODO.

## Visualizing Results
![Inference Viewer](assets/InferenceViewer.png)
We supply a Viewer that you can use to visualize the predictions of the trained networks.
You can use it via ```inference.py```, with  
`A`: Show Previous Frame  
`D`: Show Next Frame  
`W`: Toggle Predicted Keypoints (Green)  
`S`: Toggle Ground-Truth Keypoints (Blue)  
`Scroll Mousewheel`: Zoom In and Out  
`Click Mousewheel`: Drag View.  


## Evaluating a Trained Network
Can be done using ```evaluate.py```.

## Training a Network from Scratch
TODO.

## Things to note
We are currently in the process of heavily refactoring this code. The most recent version can be found in the **refactor** branch.

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
A PDF of the Paper will be included in the `assets/` Folder of this repository.
However, you can also find it <a href="https://google.com/">here</a> (at a later point in time).