# Singh_2024_NatComm
This repo contains the cell shape analysis script referenced in the publication from Singh et al. published 2024 in Nature Communications. (Link will be included as soon as the paper is published)

# How to use
## Extract cell shapes
In order to use the analysis notebook you first have to extract the cell shapes from your image data. We used [cellpose](https://github.com/MouseLand/cellpose) to do so. If you are not familiar with cellpose please check their [website](https://www.cellpose.org/) and [paper](https://www.nature.com/articles/s41592-022-01663-4).

## Installation
Download this repo and create a conda environement using the .yml-file included in this repo:
```
conda env create -n cellShapeAnalysis_env --file cellShapeAnalysis_env.yml
```
## Run Analysis
Run the analysis notebook from JupyterNotebook or JupyterLab. Make sure that you have selected the just created env as kernel.

For analysing your data, just copy your image data (.tif-files) and your cellpose shape data (.npy-files) to the "Data" folder, update the filepaths in the notebook and run the analysis.

