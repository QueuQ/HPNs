# Hierarchical Prototype Networks for Continual Graph Representation Learning (HPNs)

Implementations of Hierarchical Prototype Networks for Continual Graph Representation Learning. [paper](https://doi.org/10.1109/TPAMI.2022.3186909)


<div align="center">
    <img src="resources/pipeline.jpg">
</div>


 ```
@article{Zhang23TPAMI,
  title        = {Hierarchical Prototype Networks for Continual Graph Representation Learning},
  author       = {Xikun Zhang and Dongjin Song and Dacheng Tao},
  journal      = {{IEEE} Trans. Pattern Anal. Mach. Intell.},
  volume       = {45},
  number       = {4},
  pages        = {4622--4636},
  year         = {2023},
  url          = {https://doi.org/10.1109/TPAMI.2022.3186909},
  doi          = {10.1109/TPAMI.2022.3186909},
}
 ```

## Dependencies
Our final test was done with the following configurations. However, early versions of these packages may also work.

* Numpy == 1.19.1
* PyTorch == 1.7.1
* ogb == 1.3.1
 
## Data preprocessing
All datasets are uploaded except OGB-Arxiv and OGB-Products which are too large for github. For these two datasets, the preprocessing codes below provide the entire process for downloading and preprocessing (splitting datasets into a sequence of tasks).
On these large datasets, to avoid wasting time on preprocessing each time running the programs, the code will also store the preprocessed data. 
If you wish to preprocess both OGB-Arxiv and OGB-Products together, please run the following command.
``` shell
bash OGB_preprocess.sh
```
Or if you want to preprocess one of them, please run the following commands.
``` shell
python OGB_preprocess.py --data_name ogbn-arxiv
```
``` shell
python OGB_preprocess.py --data_name ogbn-products
```
## Run experiments

### run experiments on Cora
``` shell
bash run_cora.sh
```
### run experiments on Citeseer
``` shell
bash run_citeseer.sh
```
### run experiments on Actor
``` shell
bash run_actor.sh
```
### run experiments on OGB-Arxiv
``` shell
bash run_arxiv.sh
```
### run experiments on OGB-Products
``` shell
bash run_products.sh
```


