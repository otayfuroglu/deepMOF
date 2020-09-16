import os
import logging
from torch.optim import Adam
import schnetpack as spk
import schnetpack.atomistic.model

from schnetpack.train import Trainer, CSVHook, ReduceLROnPlateauHook
from schnetpack.train.metrics import MeanAbsoluteError
from schnetpack.train import build_mse_loss
from schnetpack.datasets import *

import torch

import get_atomrefs

from functools import partial
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

import warnings
warnings.filterwarnings("ignore")

print("Number of cuda devices -->", torch.cuda.device_count())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device type -->", device)

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))

# basic settings
BASE_DIR ="/home/modellab/workspace/omer/deepMOF/HDNNP"
model_dir = "%s/schnetpack/mof5_model_hdnnp_forces_SVPD_test4" %BASE_DIR # directory that will be created for storing model
data_dir = "%s/prepare_data/non_equ_geom_energy_forces_withORCA_SVPD.db" %BASE_DIR

if os.path.exists(model_dir):
    print("Warning: model will be restored from checkpiont! Are you sure?")
else:
    os.makedirs(model_dir)

# data preparation
logging.info("get dataset")
dataset = AtomsData(data_dir,
                    #available_properties=properties,
                    #load_only=properties,
                    collect_triples=True)

_, properties = dataset.get_properties(0)
properties = [item for item in properties.keys() if "_" != item[0]]
#del properties[1]
#print("available properties -->", properties)

n_sample = len(dataset) / 20
#print("Number of sample: ", n_sample)
#properties = ["energy", "forces"]  # properties used for training
#batch_size = 2


def run_train(config):


    train, val, test = spk.train_test_split(
        data=dataset,
        num_train=int(n_sample * 0.9),
        num_val=int(n_sample * 0.1),
        split_file=os.path.join(model_dir, "split.npz"),
    )
    train_loader = spk.AtomsLoader(train, batch_size=config["batch_size"], shuffle=True, num_workers=4)
    val_loader = spk.AtomsLoader(val, batch_size=config["batch_size"], num_workers=4)

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
                                                      cutoff_radius=6.0,
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
    #model = spk.AtomisticModel(representation=representation, output_modules=output_modules)

    # build optimizer
    optimizer = Adam(params=model.parameters(), lr=config["lr"])

    # hooks
    logging.info("build trainer")
    metrics = [MeanAbsoluteError(p, p) for p in properties]
    hooks = [CSVHook(log_path=model_dir, metrics=metrics),
             ReduceLROnPlateauHook(
                 optimizer,
                 patience=20,
                 factor=0.5,
                 min_lr=1e-6,
                 #window_length=1,
                 stop_after_min=False)]

    # trainer
    loss = build_mse_loss(properties, loss_tradeoff=[0.001, 0.99])# for ["energy", "force"]
    #loss = build_mse_loss(properties, loss_tradeoff=[0.1]) # for ["energy"]
    trainer = Trainer(
        model_dir,
        model=model,
        hooks=hooks,
        loss_fn=loss,
        optimizer=optimizer,
        train_loader=train_loader,
        validation_loader=val_loader,
    )

    # run training
    logging.info("training")
    trainer.train(device=device, n_epochs=10)

run_train({'n_layers': 2, 'n_hidden': 50, 'lr': 0.001, 'batch_size': 1})

def ray_tune(num_samples=10, max_num_epochs=10, gpus_per_trial=2):
    config = {
        #"l1": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        #"l2": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        "n_layers": tune.grid_search([2]),#, 3, 4, 5]),
        "n_hidden": tune.grid_search([50]),#, 100, 150, 200]),
        "lr": tune.grid_search([1e-4, 1e-3]),
        "batch_size": tune.grid_search([1, 2]),#, 8, 16])
    }
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
    reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        parameter_columns=["lr", "batch_size", "n_layers", "n_hidden"],
        #metric_columns=["loss", "accuracy", "training_iteration"])
        metric_columns=["loss", "training_iteration"])
    result = tune.run(
        partial(run_train),
        resources_per_trial={"cpu": 4, "gpu": gpus_per_trial},
        config=config,
        local_dir=model_dir,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        checkpoint_at_end=True,
    )

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    #print("Best trial final validation accuracy: {}".format(
    #    best_trial.last_result["accuracy"]))


#ray_tune(num_samples=4, max_num_epochs=10, gpus_per_trial=0.5)

