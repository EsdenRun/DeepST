# STMAP: identify spatial domains in spatial transcriptomics


![Workflow](./Fig/workflow.png)

## Introduction
STMAP first uses H&E staining to extract tissue morphology information through a pre-trained deep learning model, and normalizes each spot’s gene expression according to the similarity of adjacent spots. STMAP further learns a spatial adjacency matrix on spatial location for the construction of graph convolutional network. STMAP utilizes a denoising autoencoder network and a variational graph autoencoder to generate the final latent embeddings. The output of STMAP can be applied to identify spatial domains, batch effect correction and downstream analysis.

## Package: `STMAP`

We created the python package called `STMAP` that uses `scanpy` to streamline the integration of spatial transcriptomics datasets and
evaluate the results. STMAP is implemented in the open-source python using [`PyTorch`](https://pytorch.org/) and [`PyG`](https://github.com/pyg-team/pytorch_geometric) libraries.

### Installation
The STMAP python package is in the folder STMAP. You can install it from the root of this repository using

```bash
pip install .
```
Alternatively, you can also install the package directly from GitHub via

```bash
pip install git+https://github.com/JiangBioLab/STMAP.git
```

### Installing additional packages

<details>
  <summary> 1. Install PyTorch package </summary>
  
  + Installation via [Anaconda](https://anaconda.org/pyg/pyg).
```bash
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
```
+ Installation via [Pip Wheels](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html#installation-via-pip-wheels)
```bash
pip3 install torch torchvision torchaudio
```
</details>

<details>
  <summary> 2. Install PyG package </summary>
           
+ Installation via [Anaconda](https://anaconda.org/pyg/pyg).

You can now install PyG via Anaconda for all major OS/PyTorch/CUDA combinations 🤗 Given that you have [PyTorch >= 1.8.0](https://pytorch.org/get-started/locally/) installed, simply run:
```bash
conda install pyg -c pyg -c conda-forge
```
+ Installation via [Pip Wheels](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html#installation-via-pip-wheels)

We have outsourced a lot of functionality of PyG to other packages, which needs to be installed in advance. These packages come with their own CPU and GPU kernel implementations based on the PyTorch C++/CUDA extension interface. We provide pip wheels for these packages for all major OS/PyTorch/CUDA combinations:
```bash
conda install pyg -c pyg -c conda-forge
```
1). Ensure that at least PyTorch 1.4.0 is installed:
```bash
python -c "import torch; print(torch.__version__)"
>>> 1.9.0
```
2). Find the CUDA version PyTorch was installed with:
```bash
python -c "import torch; print(torch.version.cuda)"
>>> 11.1
```
3). Install the relevant packages:
```bash
pip install torch-scatter -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
pip install torch-geometric

#### where ${CUDA} and ${TORCH} should be replaced by the specific CUDA version (cpu, cu92, cu101, cu102, cu110, cu111) and PyTorch version (1.4.0, 1.5.0, 1.6.0, 1.7.0, 1.7.1,  1.8.0, 1.8.1, 1.9.0, 1.9.1), respectively. For example, for PyTorch 1.9.0/1.9.1 and CUDA 11.1, type:
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.9.0+cu111.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-1.9.0+cu111.html
pip install torch-geometric

#### For PyTorch 1.8.0/1.8.1 and CUDA 10.2, type:
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.8.0+cu102.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-1.8.0+cu102.html
pip install torch-geometric
```
4). Install additional packages (optional):
To add additional functionality to PyG, such as k-NN and radius graph generation or SplineConv support, run
```bash
pip install torch-cluster -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
pip install torch-spline-conv -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
```
</details>



## Usage
We recommend checking out the [tutorial notebook](./notebooks/scachepy_tutorial.ipynb). In essence, you can:
```python
import os
from STMAP import run
data_path = './data/'
data_name = '151673' ###### if multiple datasets, just set data_name to list. data_name = ['151673','151674',['151675']]
save_path = './Results/'

H_mo = run(data_path=data_path, 
           data_name=data_name, 
           save_path=save_path, 
           pre_epochs=500, 
           epochs=800,
           concat_pca_dim=40,
           linear_encoder_hidden=[100,20],
           linear_decoder_hidden=[30],
           conv_hidden=[32,8],
           eval_cluster_n=7,
           save=False, 
           k=11,
           distType='euclidean',
           pca_n_comps=50,
           neighbour_k=3,
           Conv_type='GCNConv',)
adata, stmap_feat = H_mo.fit()
H_mo.plot_clustering(adata, color='STMAP',img_key=None)
...
```

## Compared tools
Tools that are compared include: 
* [BayesSpace](https://github.com/edward130603/BayesSpace)
* [stLearn](https://github.com/BiomedicalMachineLearning/stLearn)
* [SpaGCN](https://github.com/jianhuupenn/SpaGCN)
* [Seurat](https://satijalab.org/seurat/)
* [SEDR](https://github.com/JinmiaoChenLab/SEDR/)

### Download data
|      Platform      |       Tissue     |    SampleID   |
|:----------------:|:----------------:|:------------:|
| [10x Visium](https://support.10xgenomics.com) | Human dorsolateral pre-frontal cortex (DLPFC) | [151507,](https://spatial-dlpfc.s3.us-east-2.amazonaws.com/h5/151507_filtered_feature_bc_matrix.h5) [151508,](https://spatial-dlpfc.s3.us-east-2.amazonaws.com/h5/151508_filtered_feature_bc_matrix.h5) [151509,](https://spatial-dlpfc.s3.us-east-2.amazonaws.com/h5/151509_filtered_feature_bc_matrix.h5) [151510,](https://spatial-dlpfc.s3.us-east-2.amazonaws.com/h5/151510_filtered_feature_bc_matrix.h5) [151669,](https://spatial-dlpfc.s3.us-east-2.amazonaws.com/h5/151669_filtered_feature_bc_matrix.h5) [151670,](https://spatial-dlpfc.s3.us-east-2.amazonaws.com/h5/151570_filtered_feature_bc_matrix.h5) [151671,](https://spatial-dlpfc.s3.us-east-2.amazonaws.com/h5/151671_filtered_feature_bc_matrix.h5) [151672,](https://spatial-dlpfc.s3.us-east-2.amazonaws.com/h5/151672_filtered_feature_bc_matrix.h5) [151673,](https://spatial-dlpfc.s3.us-east-2.amazonaws.com/h5/151673_filtered_feature_bc_matrix.h5) [151674,](https://spatial-dlpfc.s3.us-east-2.amazonaws.com/h5/151674_filtered_feature_bc_matrix.h5) [151675,](https://spatial-dlpfc.s3.us-east-2.amazonaws.com/h5/151675_filtered_feature_bc_matrix.h5) [151676](https://spatial-dlpfc.s3.us-east-2.amazonaws.com/h5/151676_filtered_feature_bc_matrix.h5)
| [10x Visium](https://support.10xgenomics.com) | Mouse brain section| [Coronal,](https://www.10xgenomics.com/resources/datasets/mouse-kidney-section-coronal-1-standard-1-1-0) [Sagittal-Anterior,](https://www.10xgenomics.com/resources/datasets/mouse-brain-serial-section-1-sagittal-anterior-1-standard-1-1-0) [Sagittal-Posterior](https://www.10xgenomics.com/resources/datasets/mouse-brain-serial-section-1-sagittal-posterior-1-standard-1-1-0)
| [10x Visium](https://support.10xgenomics.com) | Human breast cancer| [Invasive Ductal Carcinoma breast,](https://www.10xgenomics.com/resources/datasets/human-breast-cancer-block-a-section-1-1-standard-1-1-0) [Ductal Carcinoma In Situ & Invasive Carcinoma](https://www.10xgenomics.com/resources/datasets/human-breast-cancer-ductal-carcinoma-in-situ-invasive-carcinoma-ffpe-1-standard-1-3-0) 
| [Stereo-Seq](https://www.biorxiv.org/content/10.1101/2021.01.17.427004v2) | Mouse olfactory bulb| [Olfactory bulb](https://github.com/BGIResearch/stereopy) 
| [Slide-seq](https://www.biorxiv.org/content/10.1101/2021.10.10.463829v1) |  Mouse hippocampus| [Coronal](https://www.spatialomics.org/SpatialDB/download/slideseq_30923225.tar.gz) 
| [MERFISH](https://www.pnas.org/content/116/39/19490) |  Mouse brain slice| [Hypothalamic preoptic region](https://www.spatialomics.org/SpatialDB/download/merfish_30385464.tar.gz) 


### Please cite:

**Identifying spatial domains in spatial transcriptomics with STMAP.**
QH Jiang,  bioRxiv 2021.10.18.111161; doi: https://doi.org/10.1101/2020.05.22.111161_

