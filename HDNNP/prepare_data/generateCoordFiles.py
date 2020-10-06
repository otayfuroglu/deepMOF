from ase import Atoms
from ase.io import write, read
from ase.md import Langevin
import ase.units as units
from rdkit import Chem
from rdkit.Chem import AllChem
# from asap3 import EMT
from ase.calculators.dftb import Dftb
import argparse
import os

# parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--conformations', type=int, default=2048)
parser.add_argument('--temperature', type=int, default=100)
parser = parser.parse_args()

calc = Dftb(Hamiltonian_='DFTB',  # this line is included by default
            Hamiltonian_SCC='Yes',
            Hamiltonian_SCCTolerance=1e-8,
            Hamiltonian_MaxAngularMomentum_='',
            Hamiltonian_MaxAngularMomentum_H='s',
            Hamiltonian_MaxAngularMomentum_C="p",
            Hamiltonian_MaxAngularMomentum_O='p',
            Hamiltonian_MaxAngularMomentum_Zn="p",
           )

def genertCoord(fregDIR, fregName):
    #m = Chem.MolFromMolFile("%s/%s.mol" % (fregDIR, fregName))
    #AllChem.EmbedMolecule(m, useRandomCoords=True)
    #AllChem.UFFOptimizeMolecule(m)
    #pos = m.GetConformer().GetPositions()
    #natoms = m.GetNumAtoms()
    #species = [m.GetAtomWithIdx(j).GetSymbol() for j in range(natoms)]

    #atoms = Atoms(species, positions=pos)
    atoms = read("%s/%s.xyz" %(fregDIR, fregName))
    atoms.calc = calc
    md = Langevin(atoms, 1 * units.fs, temperature=parser.temperature * units.kB,
                  friction=0.01)
    if not os.path.exists("%s/nonEquGeometries" %fregDIR):
        os.mkdir("%s/nonEquGeometries" %fregDIR)

    for i in range(parser.conformations):
        print(i)
        if os.path.exists("%s/nonEquGeometries/%s_"%(fregDIR, fregName)+"{0:0>5}".format(i)+".xyz"):
            continue

        md.run(1)
        species = atoms.get_chemical_symbols()
        positions = atoms.get_positions()
        traj = Atoms(species, positions=positions)
        write("%s/nonEquGeometries/%s_"%(fregDIR, fregName)+"{0:0>5}".format(i)+".xyz", traj)

def main():
    fregDIR = "mof_fragments/xyzFiles"

    for i in range(1, 6):
        genertCoord(fregDIR, "mof5_f"+str(i))

if __name__ == "__main__":
    main()
