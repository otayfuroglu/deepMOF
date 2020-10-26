#! /truba/home/otayfuroglu/miniconda3/bin/python
import os
import logging
from torch.optim import Adam
import schnetpack as spk
import schnetpack.atomistic.model

from schnetpack.train import Trainer, CSVHook, ReduceLROnPlateauHook
from schnetpack.train.metrics import MeanAbsoluteError, RootMeanSquaredError
from schnetpack.train import build_mse_loss
from schnetpack.datasets import *

import torch

import get_atomrefs

from functools import partial
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

from datetime import datetime

import warnings
warnings.filterwarnings("ignore")

# basic settings
BASE_DIR = "/truba_scratch/otayfuroglu/deepMOF/HDNNP"

trainingName = "hdnnWeighted_l3n50_rho01_batch2"
MODEL_DIR = os.path.join(os.getcwd(), trainingName)
DATA_DIR = "%s/prepare_data/nonEquGeometriesEnergyForcesWithORCAFromMD.db" %BASE_DIR

# logging to logFile
logFile = "%s.log" %trainingName
if os.path.exists(logFile):
    os.remove(logFile)
logging.basicConfig(filename=logFile,
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=os.environ.get("LOGLEVEL", "INFO"))

if os.path.exists(MODEL_DIR):
    logging.info("Warning: model will be restored from checkpiont in the %s directory! Are you sure?" %MODEL_DIR)
else:
    os.makedirs(MODEL_DIR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.info("Job started %s" %(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
logging.info("device type --> %s" %device)
n_gpus = torch.cuda.device_count()
logging.info("Number of cuda devices --> %s" %n_gpus,)

# data preparation
logging.info("get dataset")
dataset = AtomsData(DATA_DIR,
                    #available_properties=properties,
                    #load_only=properties,
                    collect_triples=True)

_, properties = dataset.get_properties(0)
properties = [item for item in properties.keys() if "_" != item[0]]
#del properties[1]
logging.info("available properties --> %s" %properties)

n_sample = len(dataset)
logging.info("Number of sample: %s" %n_sample)  
#properties = ["energy", "forces"]  # properties used for training
#batch_size = 2


def run_train(config,  checkpoint_dir=None):


    train, val, test = spk.train_test_split(
        data=dataset,
        num_train=int(n_sample * 0.9),
        num_val=int(n_sample * 0.1),
        split_file=os.path.join(MODEL_DIR, "split.npz"),
    )
    train_loader = spk.AtomsLoader(train, batch_size=config["batch_size"], shuffle=True, num_workers=20)
    val_loader = spk.AtomsLoader(val, batch_size=config["batch_size"], num_workers=20)

    # get statistics
    atomrefs = get_atomrefs.atomrefs_energy0(properties[0])
    #per_atom = dict(energy=True, forces=False)
    #per_atom = dict(energy=True)
    per_atom = {properties[0]: True}
    means, stddevs = train_loader.get_statistics(
        [properties[0]],
        single_atom_ref=atomrefs,
        divide_by_atoms=per_atom,
        #divide_by_atoms=True,
    )

    # model build
    logging.info("build model")
    representation = spk.representation.BehlerSFBlock(n_radial=22,
                                                      n_angular=5,
                                                      zetas={1},
                                                      cutoff_radius=config["cutoff_radius"],
                                                      elements=frozenset((1, 6, 8, 30)),
                                                      centered=False,
                                                      crossterms=False,
                                                      mode="weighted", # "weighted" for wACSF "Behler" for ACSF
                                                     )

    output_modules = [
        schnetpack.atomistic.ElementalAtomwise(
            n_in=representation.n_symfuncs,
            n_out=1,
            n_layers=config["n_layers"],
            aggregation_mode="sum",
            n_hidden=config["n_hidden"],
            elements=frozenset((1, 6, 8, 30)),
            property=properties[0],
            derivative="forces",
            mean=means[properties[0]],
            stddev=stddevs[properties[0]],
            atomref=atomrefs[properties[0]],
            negative_dr=True,
        )
    ]

    model = schnetpack.atomistic.model.AtomisticModel(representation, output_modules)
    # for multi GPU
    if n_gpus > 1:
        model = torch.nn.DataParallel(model)
        
    # build optimizer
    optimizer = Adam(params=model.parameters(), lr=config["lr"])

    # hooks
    logging.info("build trainer")
    #metrics = [MeanAbsoluteError(p, p) for p in properties]
    metrics = [RootMeanSquaredError(p, p) for p in properties]
    hooks = [CSVHook(log_path=MODEL_DIR, metrics=metrics),
             ReduceLROnPlateauHook(
                 optimizer,
                 patience=20,
                 factor=0.5,
                 min_lr=1e-6,
                 #window_length=1,
                 stop_after_min=False)]

    # trainer
    rho = config["rho"]
    loss = build_mse_loss(properties, loss_tradeoff=[rho, 1 - rho])# for ["energy", "force"]
    #loss = build_mse_loss(properties, loss_tradeoff=[0.1]) # for ["energy"]
    trainer = Trainer(
        MODEL_DIR,
        train_type="train",
        model=model,
        hooks=hooks,
        loss_fn=loss,
        optimizer=optimizer,
        train_loader=train_loader,
        validation_loader=val_loader,
    )

    # run training
    logging.info("training")
    trainer.train(device=device, n_epochs=100)

run_train({'n_layers': 3, 'n_hidden': 50, 'lr': 0.001, 'batch_size': 2, "cutoff_radius": 6.0, "rho": 0.1})
logging.info("training was done")
