# deepMOF
This repository has pretrained Neural Network Potentential for IRMOF-n (n=1, 4, 6, 7, 10)
 
##### Requirements:
- python 3
- ASE
- schnetpack
- PyTorch (>=0.4.1)

Note: We recommend using a GPU for training the neural networks.

# Installation

```
git clone https://github.com/otayfuroglu/deepMOF.git
cd deepMOF
```
## Install requirements
```
pip install -r requirements.txt
```
# How to use

## Quick test example

The Quick test example scripts allows to load model and geometry optimization of IRMOF-1.
The can be started using:

```
python deepmof_quicktest.py
```
  
You can choose another MOF in IRMOF series and perform molecular dynamic simulations, geometry optimization and others calculations at DFT level.


