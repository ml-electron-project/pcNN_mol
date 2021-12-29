# pcNN_mol
PySCF version of the physically constrained NN-based XC functional.
https://arxiv.org/abs/2111.15593

## Requirements
Python3  
numpy  
PySCF == 2.0.1
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


## License
Copyright 2021 Ryo Nagai

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
