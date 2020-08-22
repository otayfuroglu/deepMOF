from ase.io import read
from ase.db import connect

from schnetpack import AtomsData
import os
from dftd4 import D4_model
from ase.calculators.orca import ORCA
#from gpaw import GPAW, PW

import numpy as np
import pandas as pd
import multiprocessing

mol_path ="/home/modellab/workspace/omer/deepMOF/HDNNP/prepare_data/non_equ_geom_xyz_files"
db_path = "/home/modellab/workspace/omer/deepMOF/HDNNP/prepare_data"
db_name = "non_equ_geom_energy_forces_withORCA.db"
csv_name = "calculated_files.csv"
#os.chdir(mol_path)
file_names = os.listdir(mol_path)
#n_cpu = multiprocessing.cpu_count()
n_cpu = 4
n_proc =  int(multiprocessing.cpu_count() / n_cpu)

fragments = []
for i in range(2, 6):
    fragments.append([item for item in file_names if "mof5_f%s"%i in item])

def orca_calculator(label, n_cpu, initial_gbw=['','']):
    return ORCA(label=label,
                   maxiter=2000,
                   charge=0, mult=1,
                   orcasimpleinput='SP PBE D4 DEF2-SVP DEF2/J RIJDX MINIPRINT NOPRINTMOS NOPOP' + ' ' + initial_gbw[0],
                   orcablocks='%scf Convergence sloppy \n maxiter 300 end \n %pal nprocs ' + str(n_cpu) + ' end' + initial_gbw[1]
                   )


class CaculateData():
    def __init__(self):
        self.i = 0
        self.new_db = AtomsData("%s/%s" %(db_path, db_name),
                           available_properties=[
                               "energy",
                               "forces"
                           ])
    def _add_calculated_file(self, df_calculated_files, file_base):
        df_calculated_files_new = pd.DataFrame([file_base], columns=["FileNames"])
        df_calculated_files_new.to_csv("%s/%s" %(db_path, csv_name), mode='a', header=False, index=None)

    def _caculate_data(self, file_name):
        file_base = file_name.replace(".xyz", "")
        #db_calculated = connect("%s/%s" %(db_path, db_name)).select()
        #calculated_files = [row["name"] for row in db_calculated]
        df_calculated_files = pd.read_csv("%s/%s" %(db_path, csv_name), index_col=None)
        calculated_files = df_calculated_files["FileNames"].to_list()
        if file_base in calculated_files:
            print("The %s file have already calculated" %file_base)
            return None

        # file base will be add to calculted csv file
        self._add_calculated_file(df_calculated_files, file_base)
        print(file_base)
        mol = read("%s/%s.xyz" %(mol_path, file_base))

        label = "orca_%s" %file_base
        #try:
        if self.i == 0:
            mol.set_calculator(orca_calculator(label, n_cpu))
        else:
            initial_gbw = ['MORead',  '\n%moinp "initial.gbw"']
            mol.set_calculator(orca_calculator(label, n_cpu, initial_gbw))
        energy = mol.get_potential_energy()
        #print(energy)
        forces = mol.get_forces()
        energy = np.array([energy], dtype=np.float32)
        forces = np.array([forces], dtype=np.float32)
        if self.i == 0:
            os.system("mv %s.gbw initial.gbw" %label)
        os.system("rm %s*" %label)
        self.i += 1
        self.new_db.add_system(mol, file_base, energy=energy, forces=forces)
        os.system("rm %s*" %label)
       # except:
       #     print("Error for %s" %file_base)
       #     self.i += 1
       #     os.system("rm %s*" %label)
        print(self.i)

    def caculate_data(self, n_proc):
        #atoms_list = []
        #property_list = []
        for file_names in fragments:
            with multiprocessing.Pool(n_proc) as pool:
                pool.map(self._caculate_data, file_names)

    def print_data(self):
        self.new_db = AtomsData("%s/%s" %(db_path, db_name))
        print('Number of reference calculations:', len(self.new_db))
        print('Available properties:')
        for p in self.new_db.available_properties:
            print(p)
        print
        i = 1
        example = self.new_db[i]
        print('Properties of molecule with id %s:' % i)
        for k, v in example.items():
            print('-', k, ':', v.shape)


calculate = CaculateData()
calculate.caculate_data(n_proc)
