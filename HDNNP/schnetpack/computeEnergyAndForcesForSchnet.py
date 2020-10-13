#
import openmm_py
import torch
import sys
import numpy as np
from ase import Atoms
from ase.io import write

def warn(*argv):
    # write to stderr
    print(*argv, file=sys.stderr, flush=True)

class EnergyComputer(openmm_py.PyCall):
    """ This class implements the PyCall C++ interface.
        It's computeEnergyAndForces() method will be used as a callback from C++
        to compute energies and forces.
    """
    def __init__(self, chemicalSymbols, aseCalculator):
        warn("PY__INIT EnergyComputer")
        super().__init__()

        self.device  = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        #self.device  = "cpu"
        self.aseCalculator   = aseCalculator
        self.chemicalSymbols = chemicalSymbols


    def computeEnergyAndForces(self, positions, includeForces=True, includeEnergy=True):
        """ positions: atomic postions in [nM]
                       numparitcal * 3 PyCall.FloatVector with
                       x1, y1, z1, x2, ... zn coordinates passed to us from openMM
            includeForces: boolean, if True force computation is requested.
            includeEnergy: boolean, if True energy computation is requested.

            return: PyCall.NNPResult with energy [kJ/mol] and forces [kJ/mol/nm]
        """

        pos = np.array(positions, dtype=np.float32)
        warn("py computeEnergyAndForces {} positions[nm] {}"
             .format(type(positions),pos))

        # correction for openmm positions to ase atoms positions
        pos = pos.reshape((int(pos.shape[0]/3), 3))
        pos = pos * 10.0

        # compute energy
        atoms = Atoms(self.chemicalSymbols, pos)

        atoms.set_calculator(self.aseCalculator)
        pred = torch.tensor(atoms.get_potential_energy(), device=self.device, dtype=torch.float)
        if includeForces:
            # use PyTorch autograd to compute:
            #     force = - derivative of enrgy wrt. coordinates
            forces = torch.tensor(atoms.get_forces()).cpu().numpy()
            forces = forces.ravel()
        else:
            forces = np.zeros(len(positions))
        # Return result in type openmm_py.NNPResult
        # this is a C++ struct with two fields: energy [kJ/Mol/nM]
        #    energy [kJ/Mol]
        #    force [kJ/Mol/nM]
        res = openmm_py.NNPResult();
        res.energy = pred.cpu().item()
        res.force = openmm_py.FloatVector(forces.tolist());
        warn("Py Energy {} Forces {}".format(pred,forces))

        return res;
