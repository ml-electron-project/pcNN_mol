# pcNN_mol
PySCF version of the physically constrained NN-based XC functional.
https://arxiv.org/abs/2111.15593

## Requirements
Python3  
numpy  
PySCF  
pytorch   
JAX  
(For pytorch and JAX, CUDA version is not necessary.)

## Usage
Run dft_pcnn.py and you will be asked to input the place of coodinate file (absolute path), spin number, and charge number.　　　　
　　　
   
(Example)
```
input the target .xyz path:
./h2o.txt
input the number of unpaired electrons:
0
input the charge number:
0
```
(Final output)
converged SCF energy = -76.3462003834499

## Citing this XC functinoal
@misc{nagai2021machinelearningbased,
      title={Machine-Learning-Based Exchange-Correlation Functional with Physical Asymptotic Constraints}, 
      author={Ryo Nagai and Ryosuke Akashi and Osamu Sugino},
      year={2021},
      eprint={2111.15593},
      archivePrefix={arXiv},
      primaryClass={cond-mat.mtrl-sci}
}


Copyright 2021 Ryo Nagai
