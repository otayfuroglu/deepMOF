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

```
## Install requirements
```
pip install -r requirements.txt
```
# How to use

## Quick test example

The quick test example scripts allows to load model and geometry optimization of IRMOF-1.
The can be started using:

```
cd /path/to/deepMOF
python deepmof_quicktest.py
```
  
You can choose another MOF structures in IRMOF series and perform molecular dynamic simulations, geometry optimization and others calculations. The deepMOF model will provide with DFT-level calculations in other molecular dynamics software environments compatible whichi is the schnetpack.

## Documentation
For the full reference, visit our related paper in PCCP, https://pubs.rsc.org/en/content/articlelanding/2022/cp/d1cp05973d

If you are using deepMOF models in your research, please cite:
Tayfuroglu O, Kocak A, Zorlu Y, A neural network potential for the IRMOF series and its application for thermal and mechanical behaviors, Phys. Chem. Chem. Phys., 2022, Advance Article, https://doi.org/10.1039/D1CP05973D


## References
For the full Documentation schnetpack visit https://schnetpack.readthedocs.io/ 
and reach to details of the  schnet paper follow;
K.T. Schütt, P. Kessel, M. Gastegger, K. Nicoli, A. Tkatchenko, K.-R. Müller.
SchNetPack: A Deep Learning Toolbox For Atomistic Systems.
J. Chem. Theory Comput.
[10.1021/acs.jctc.8b00908](http://dx.doi.org/10.1021/acs.jctc.8b00908)


