# Saliency in VR: How do people explore virtual environments?
Project web-site with more infos videos, and dataset download: [link](https://vsitzmann.github.io/vr-saliency/).
Link to our group website with other VR, AR, and computational imaging projects: [link](http://www.computationalimaging.org/)

This repository contains example code to work with the dataset proposed in the following paper:

*Sitzmann, Vincent, et al. "Saliency in VR: How do people explore virtual environments?." IEEE Transactions on Visualization and Computer Graphics (2017).

If you use this code or dataset in you research, please consider citing our paper with the following Bibtex code:

```
@article{Sitzmann_TVCG_VR-saliency, 
    author = {Vincent Sitzmann and Ana Serrano and Amy Pavel and Maneesh Agrawala and Diego Gutierrez and Belen Masia and Gordon Wetzstein}, 
    title = {How do people explore virtual environments?}, 
    journal = {IEEE Transactions on Visualization and Computer Graphics}, 
    year = {2017}
} 
```
A pre-print is available at this [link](http://ieeexplore.ieee.org/document/8269807/).

## Abstract

Understanding how people explore immersive virtual environments is crucial for many applications, such as designing virtual reality (VR) content, developing new compression algorithms, or learning computational models of saliency or visual attention. Whereas a body of recent work has focused on modeling saliency in desktop viewing conditions, VR is very different from these conditions in that viewing behavior is governed by stereoscopic vision and by the complex interaction of head orientation, gaze, and other kinematic constraints. To further our understanding of viewing behavior and saliency in VR, we capture and analyze gaze and head orientation data of 169 users exploring stereoscopic, static omni-directional panoramas, for a total of 1980 head and gaze trajectories for three different viewing conditions. We provide a thorough analysis of our data, which leads to several important insights, such as the existence of a particular fixation bias, which we then use to adapt existing saliency predictors to immersive VR conditions. In addition, we explore other applications of our data and analysis, including automatic alignment of VR video cuts, panorama thumbnails, panorama video synopsis, and saliency-based compression. 

## Using the dataset

The dataset can be downloaded [here](https://drive.google.com/file/d/1BHtigR_egB6E-N4irZA9wSAH_902_PBh/view?usp=sharing). It contains the raw head and eye tracking files as .csv files as well as pre-processed .pck files (the preprocessing code will be made available soon).

To explore the dataset with the sample code, simply unpack it into the root directory of the cloned github repo - it should create a directory "data". The code in the jupyter notebook "src/Analysis.ipynb" should then run just fine. 
