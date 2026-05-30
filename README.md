# Partial-GeCo

This repository contains the code of the paper
"An Integer Linear Programming Approach to Geometrically Consistent Partial-Partial Shape Matching", V.Ehm, P. Roetzer, F. Bernard, D. Cremers. 3DV 2026.
It provides code to solve the partial-to-partial shape matching problem with geometric consistency using an ILP.

## ⚙️ Installation
Add a conda environment and install required packages

```
conda create -n partial-geco python=3.8
conda activate partial-geco
pip install -r requirements.txt
```

To compute surface cycles and generate the product space, we use the implementation of GeCo [1]. To use that run 

```
pip install git+https://github.com/paul0noah/GeCo
```

Additionally install the Gurobi optimizer (www.gurobi.com) for solving the optimization problem.

## 🧑‍💻️ Usage
Run the ``main.py`` to see an example shape being matched.

## Datasets
CP2P24 can be dowloaded from [huggingface](https://huggingface.co/datasets/xieyizheng/CP2P24) and PARTIALSMAL can be downladed from this [GC-PPSM](https://github.com/vikiehm/gc-ppsm) [2].

## Feature Computation
We use features and overlap predicions from EchoMatch [3]. You can find the repository [here](https://github.com/vikiehm/echo-match/). Save the features and overlap predictions in ``example_data/features``.

## 🎓 Attribution
```
@inproceedings{ehminteger,
  title={An Integer Linear Programming Approach to Geometrically Consistent Partial-Partial Shape Matching},
  author={Ehm, Viktoria and Roetzer, Paul and Bernard, Florian and Cremers, Daniel},
  booktitle={Thirteenth International Conference on 3D Vision}
}
```

## References 
[1] Paul Roetzer, Florian Bernard. Fast Globally Optimal and Geometrically Consistent 3D Shape Matching. ICCV, 2025.

[2] Viktoria Ehm, Maolin Gao, Paul Roetzer, Marvin Eisenberger, Daniel Cremers, Florian Bernard. Partial-to-Partial Shape Matching with Geometric Consistency. CVPR, 2024.

[3] Yizheng Xie, Viktoria Ehm, Paul Roetzer, Nafie El Amrani, Maolin Gao, Florian Bernard, Daniel Cremers. CVPR 2025.
