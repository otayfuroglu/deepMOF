from ase.io import read
from ase.db import connect
from ase.calculators.orca import ORCA
from ase import Atoms

import torch
from schnetpack.datasets import AtomsData
import schnetpack as spk

from schnetpack import AtomsData
import os
from dftd4 import D4_model
#from gpaw import GPAW, PW

import numpy as np
import pandas as pd
import multiprocessing

n_cpu = 4

def orca_calculator(multiplicity, label, n_cpu):
    return ORCA(label="ase_orca_%s"%label,
              maxiter=2000,
              charge=0, mult=multiplicity,
                   orcasimpleinput='SP PBE D4 DEF2-SVP DEF2/J RIJDX MINIPRINT NOPRINTMOS NOPOP',
                   orcablocks='%scf Convergence sloppy \n maxiter 300 end \n %pal nprocs ' + str(n_cpu) + " end"
              )

def calc_SP(atoms, calc_tor):
    atoms.set_calculator(calc_tor)
    return atoms.get_potential_energy()

def calc_free_atom_energy():
    multiplicity_atoms = {"H": 2, "C": 3, "O": 3, "Zn": 1}
    free_atoms_energies = {}
    for chemical_symbol, multiplicity in multiplicity_atoms.items():
        free_atoms_energies[chemical_symbol] = calc_SP(Atoms(chemical_symbol), orca_calculator(multiplicity_atoms[chemical_symbol], chemical_symbol, n_cpu))
    os.system("rm ase_orca_*")
    return free_atoms_energies

def calc_cohesive_E(atoms, total_E, free_atoms_energies):
    chemical_symbols = atoms.get_chemical_symbols()
    chemical_symbols_numbers = {i:chemical_symbols.count(i) for i in chemical_symbols}
    free_energies_all_atoms = 0.0
    for chemical_symbol, number_of_atoms in chemical_symbols_numbers.items():
        free_energies_all_atoms += number_of_atoms * free_atoms_energies[chemical_symbol]
    return (total_E - free_energies_all_atoms) / len(atoms)

def caculate_data():
    new_db = AtomsData("./non_equ_geom_energy_coh_energy_forces_withORCA_v4.db",
                       available_properties=[
                           "total_E",
                           "cohesive_E_perAtom",
                           "forces"
                       ])
    free_HCOZn_energies = calc_free_atom_energy()

    property_list = []
    atoms_list = []
    name_list = []

    db1 = connect("./non_equ_geom_energy_forces_withORCA_v4_1.db").select()
    db2 = connect("./non_equ_geom_energy_forces_withORCA_v4_2.db").select()
    for db in [db1, db2]:
        for i, row in enumerate(db):
            if i % 100 == 0:
                print(i)
            file_base = row["name"]
            mol = row.toatoms()
            total_E = row["energy"]
            total_E = np.array([total_E], dtype=np.float32)

            forces = row["forces"]
            forces = np.array(forces, dtype=np.float32)

            cohesive_E_perAtom = calc_cohesive_E(mol, total_E, free_HCOZn_energies)
            cohesive_E_perAtom = np.array(cohesive_E_perAtom, dtype=np.float32)

            atoms_list.append(mol)
            name_list.append(file_base)
            property_list.append({"total_E": total_E,
                                 "forces": forces,
                                 "cohesive_E_perAtom": cohesive_E_perAtom,
                                })

    new_db.add_systems(atoms_list, name_list, property_list)

def get_data_from_csv():
    new_db = AtomsData("./non_equ_geom_energy_coh_energy_forces_withORCA_v4_1.db",
                       available_properties=[
                           "total_E",
                           "cohesive_E_perAtom",
                           "forces"
                       ])
    df = pd.read_csv("./non_equ_geom_energy_coh_energy_forces_withORCA_v4.csv")
    db = AtomsData("./non_equ_geom_energy_forces_withORCA_v4.db")
    for i, row in df.iterrows():

        #db_row = db[i]

        file_base = row["name"]
        #if db_row["name"] != file_base:
        #    print("HATA")

        mol = db.get_atoms(i)
        total_E = row["total_E"]
        total_E = np.array([total_E], dtype=np.float32)

        forces = db[i]["forces"]
        forces = np.array(forces, dtype=np.float32)
        cohesive_E_perAtom = row["cohesive_E_perAtom"]
        cohesive_E_perAtom = np.array([cohesive_E_perAtom], dtype=np.float32)

        new_db.add_system(mol, file_base, total_E=total_E, cohesive_E_perAtom=cohesive_E_perAtom, forces=forces)
        #if i == 10:
        #    break

def ase_db_to_csv():
    db = AtomsData("./non_equ_geom_energy_coh_energy_forces_withORCA_v4.db")
    #
    df = pd.DataFrame()
    df["name"] = [db.get_name(i) for i in range(len(db))]

    for label in ["total_E", "cohesive_E_perAtom"]:
        df[label] = [float(db[i][label]) for i in range(len(db))]
    df.to_csv("./non_equ_geom_energy_coh_energy_forces_withORCA_v4.csv")

def get_fragment_data(db, properties, fragment_keyword):
    property_list = []
    atoms_list = []
    name_list = []
    for i in range(len(db)):
        if i % 100 == 0:
            print(i)
            file_base = db.get_name(i)
        #properties = ["total_E", "forces", "cohesive_E_perAtom"]
        if fragment_keyword in file_base:
            property_values = []
            for propert in properties:

                mol = db.get_atoms(i)
                target_propert = db[i][propert]
                target_propert = np.array(target_propert, dtype=np.float32)
                property_values.append(target_propert)

            #combine two lists into a dictionary 
            property_dict = dict(zip(properties, property_values))
            atoms_list.append(mol)
            name_list.append(file_base)
            property_list.append(property_dict)

    return atoms_list, name_list, property_list


def run_get_fragment_data():

    db = AtomsData("../prepare_data/non_equ_geom_energy_forces_withORCA_new.db")
    new_db_path = "../prepare_data/non_equ_geom_energy_forces_withORCA_new_f1.db"
    if os.path.exists(new_db_path):
        os.remove(new_db_path)

    atoms, properties = db.get_properties(0)
    properties = [propert for propert in properties.keys() if "_" not in propert]
    new_db = AtomsData(new_db_path,
                       available_properties=properties
                      )

    for fragment_keyword in ["mof5_new_f1",]:# "mof5_new_f2", "mof5_new_f3"]:
        atoms_list, name_list, property_list = get_fragment_data(db, properties, fragment_keyword)
        new_db.add_systems(atoms_list, name_list, property_list)

def statics_data(dataset, data_dir):
    print(data_dir)
    atoms, properties = dataset.get_properties(0)
    properties = list(properties.keys())
    forcetut=data_dir
    os.mkdir(data_dir)
    train, val, test = spk.train_test_split(
                data=dataset,
                num_train=4400,
                num_val=5,
                split_file=os.path.join(forcetut, "split.npz"),
            )
    train_loader = spk.AtomsLoader(train, batch_size=12, shuffle=True, num_workers=8)
    val_loader = spk.AtomsLoader(val, batch_size=4)

    means, stddevs = train_loader.get_statistics(
        properties[0],
        divide_by_atoms=True,
    )

    print('Mean atomization energy / atom:      {:12.4f} [kcal/mol]'.format(means[properties[0]][0]))
    print('Std. dev. atomization energy / atom: {:12.4f} [kcal/mol]'.format(stddevs[properties[0]][0]))

#run_get_fragment_data()

#caculate_data()
#get_data_from_csv()
#ase_db_to_csv()

dataset = AtomsData("non_equ_geom_energy_forces_withORCA_new_f1.db",
                    #available_properties=properties,
                    #load_only=properties,
                    collect_triples=True)
print(len(dataset))
statics_data(dataset, "mof5_new_f1")
#
#new_db = AtomsData("./non_equ_geom_energy_coh_energy_forces_withORCA_v4.db")
#print(len(new_db))
##new_db = connect("./non_equ_geom_energy_coh_energy_forces_withORCA_v4.db").select()
#for i in range(len(new_db)):
#    print(new_db[i]["total_E"])
#    break
#print(new_db.get_atoms())
