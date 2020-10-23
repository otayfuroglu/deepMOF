#! /truba/home/otayfuroglu/miniconda3/bin/python

import os
import schnetpack as spk
from schnetpack.md import System
from schnetpack.md import MaxwellBoltzmannInit
from schnetpack.md.integrators import VelocityVerlet
from schnetpack.md.calculators import SchnetPackCalculator
from schnetpack import Properties
from schnetpack.md import Simulator
from schnetpack.md.simulation_hooks import thermostats
from schnetpack.md.simulation_hooks import logging_hooks
from schnetpack.md.integrators import RingPolymer

from ase.io import read
import torch
import argparse
import ntpath

def main(molecule_path, md_type, n_steps):
    
    # Gnerate a directory of not present
    
    # Get the parent directory of SchNetPack
    BASE_DIR = "/truba_scratch/otayfuroglu/deepMOF/HDNNP/schnetpack"
    MODEL_DIR = BASE_DIR + "/" + "models/rayTune_Behler_v3"
    WORKS_DIR = BASE_DIR + "/" + "md_" + md_type

    if md_type == "rpmd":
        n_replicas = 12 
    else:
        n_replicas = 1
   
    print("# of replicas --> ", n_replicas)

    if not os.path.exists(WORKS_DIR):
        os.mkdir(WORKS_DIR)

    properties = ["energy", "forces"]  # properties used for training

    _, name = ntpath.split(molecule_path) 
    name = name[:-4]
    
    # Load model and structure
    model_path = os.path.join(MODEL_DIR, "best_model")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    md_model = torch.load(model_path, map_location=device).to(device)
    
    md_calculator = SchnetPackCalculator(
    md_model,
    required_properties=properties,
    force_handle=properties[1],
    position_conversion="A",
    force_conversion="kcal/mol/A",
    )

    # Number of beads in RPMD simulation
    system_temperature = 100
    bath_temperature = 100
    time_constant = 100
    buffer_size = 100

   # Set up the system, load structures, initialize
    system = System(n_replicas, device=device)
    system.load_molecules_from_xyz(molecule_path)

    # Initialize momenta
    initializer = MaxwellBoltzmannInit(
        system_temperature,
        remove_translation=True,
        remove_rotation=True)

    initializer.initialize_system(system)

    # Here, a smaller timestep is required for numerical stability
    
    if md_type == "rpmd":
        time_step = 0.2 # fs

        # Initialize the integrator, RPMD also requires a polymer temperature which determines the coupling of beads.
        # Here, we set it to the system temperature
        print("set of the RingPlymer Integrator")
        integrator = RingPolymer(
            n_replicas,
            time_step,
            system_temperature,
            device=device
        )

        # Initialize the thermostat
        pile = thermostats.PILELocalThermostat(bath_temperature, time_constant)
    else:

        print("set of the VelocityVerlet Integrator")
        time_step = 0.5 # fs
        integrator = VelocityVerlet(time_step)
        langevin = thermostats.LangevinThermostat(bath_temperature, time_constant)

    # Logging
    log_file = os.path.join(WORKS_DIR, '%s.hdf5' %name)
    data_streams = [
        logging_hooks.MoleculeStream(),
        logging_hooks.PropertyStream(),
    ]
    file_logger = logging_hooks.FileLogger(
        log_file,
        buffer_size,
        data_streams=data_streams
    )

    # Checkpoints
    chk_file = os.path.join(WORKS_DIR, '%s.chk' %name)
    checkpoint = logging_hooks.Checkpoint(chk_file, every_n_steps=100)

    # Assemble the hooks:
    if md_type == "rpmd":
        simulation_hooks = [
            pile,
            file_logger,
            checkpoint
        ]
    else:
        simulation_hooks = [
            langevin,
            file_logger,
            checkpoint
        ]

    # Assemble the simulator
    simulator = Simulator(system, integrator, md_calculator, simulator_hooks=simulation_hooks)

    n_steps = n_steps
    simulator.simulate(n_steps)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Give something ...")
    parser.add_argument("-mol", "--mol_path", type=str, required=True, help="give full molecule path ")
    parser.add_argument("-mdtype", "--md_type", type=str, required=True, help="give MD type classical or Ring Polymer (rpmd) MD")
    parser.add_argument("-n", "--n_steps", type=int, required=True, help="give number of stepes")

    args = parser.parse_args()
    main(molecule_path=args.mol_path, md_type=args.md_type, n_steps=args.n_steps)
