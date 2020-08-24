# 
from ase.db import connect
from ase.io import read
from schnetpack import AtomsData
import os
import numpy as np
import pandas as pd
#db_1 = AtomsData('./non_equ_geom_energy_forces_withORCA_0.db')
#db_2 = AtomsData('./non_equ_geom_energy_forces_withORCA.db')


new_db = AtomsData("./non_equ_geom_energy_coh_energy_forces_withORCA_v4.db",
                   available_properties=[
                       "total_E",
                       "cohesive_E_perAtom",
                       "forces",
                   ])

example = new_db[1]
print('Properties of molecule with id 0:')

for k, v in example.items():
        print('-', k, ':', v.shape)

print(new_db[0]["cohesive_E_perAtom"])

def main():
    new_db = AtomsData("./non_equ_geom_energy_forces_withORCA.db",
                       available_properties=[
                           "energy",
                       ])
    mol_path ="/home/modellab/workspace/omer/deepMOF/HDNNP/prepare_data/non_equ_geom_xyz_files"
    #os.chdir(mol_path)
    file_names = os.listdir(mol_path)
    for i, file_name in enumerate(file_names):
        atoms = read("%s/%s" %(mol_path, file_name))
        name = file_name.replace(".xyz", "")
        energy = np.array([0.0], dtype=np.float32)
        #print(available_properties_dict)
        #new_db.add_systems([atoms], [available_properties_dict])
        new_db.add_system(atoms, name, energy=energy)
        if i == 0:
            break
#main()
#db = AtomsData("./test.db")
#for p in db.available_properties:
#    print(p)
#
#print(db[0]["energy"])

def _add_calculated_file(df_calculated_files, file_base):
    df_calculated_files_new = pd.DataFrame([file_base], columns=["FileNames"])
    df_calculated_files_new.to_csv("./calculated_files.csv", mode='a', header=False, index=None)

def get_calculated_files():
    db = connect("./non_equ_geom_energy_forces_withORCA_v4.db")
    db = db.select()
    #
    df = pd.DataFrame()
    data_list =  np.array([[row["name"], row["energy"]] for row in db])
    df["FileNames"] = data_list[:, 0]
    #df["Energies"] = data_list[:, 1]
    df.to_csv("calculated_files.csv", index=None)
    #df = pd.read_csv("./calculated_files.csv", index_col=None)
    _add_calculated_file(df, "NEW")
    #
    #new_db = AtomsData("./non_equ_geom_energy_forces_withORCA.db",
    #                   available_properties=[
    #                       "energy", "forces",
    #                   ])
    #print(len(new_db))

#get_calculated_files()
