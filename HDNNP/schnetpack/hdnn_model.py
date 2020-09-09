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
print("Number of cuda devices -->", torch.cuda.device_count())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device type -->", device)

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))

# basic settings
model_dir = "./mof5_model_hdnnp_forces_mof5_single"  # directory that will be created for storing model

if os.path.exists(model_dir):
    print("Warning: model will be restored from checkpiont! Are you sure?")
else:
    os.makedirs(model_dir)

#properties = ["energy", "forces"]  # properties used for training
batch_size = 4

# data preparation
logging.info("get dataset")
dataset = AtomsData("../prepare_data/non_equ_geom_energy_forces_withORCA_single.db",
                    #available_properties=properties,
                    #load_only=properties,
                    collect_triples=True)

_, properties = dataset.get_properties(0)
properties = [item for item in properties.keys() if not "_" in item]
print("availavle properties -->", properties)

n_sample = len(dataset) / 6
print("Number of sample: ", n_sample)

def main():
    train, val, test = spk.train_test_split(
        data=dataset,
        num_train=int(n_sample * 0.9),
        num_val=int(n_sample * 0.1),
        split_file=os.path.join(model_dir, "split.npz"),
    )
    train_loader = spk.AtomsLoader(train, batch_size=batch_size, shuffle=True, num_workers=batch_size)
    val_loader = spk.AtomsLoader(val, batch_size=batch_size, num_workers=batch_size)

    # get statistics
    atomrefs = dataset.get_atomref(properties[0])
    #per_atom = dict(energy=True, forces=False)
    #per_atom = dict(energy=True)
    per_atom = {properties[0]: True}
    means, stddevs = train_loader.get_statistics(
        [properties[0]],
        single_atom_ref=atomrefs,
        #divide_by_atoms=per_atom,
        divide_by_atoms=True,
    )

    # model build
    logging.info("build model")
    representation = spk.representation.BehlerSFBlock(n_radial=5,
                                                      n_angular=3,
                                                      zetas={1},
                                                      cutoff_radius=6.0,
                                                      elements=frozenset((1, 6, 8, 30)),
                                                      centered=False,
                                                      crossterms=False,
                                                      mode="Behler", # "weighted" for wACSF "Behler" for ACSF
                                                     )

    output_modules = [
        schnetpack.atomistic.ElementalAtomwise(
            n_in=representation.n_symfuncs,
            n_out=1,
            n_layers=3,
            aggregation_mode="sum",
            n_hidden=50,
            elements=frozenset((1, 6, 8, 30)),
            property=properties[0],
            derivative="forces",
            mean=means[properties[0]],
            stddev=stddevs[properties[0]],
            negative_dr=True,
        )
    ]

    model = schnetpack.atomistic.model.AtomisticModel(representation, output_modules)
    #model = spk.AtomisticModel(representation=representation, output_modules=output_modules)

    # build optimizer
    optimizer = Adam(params=model.parameters(), lr=1e-3)

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
    trainer.train(device=device, n_epochs=200)

main()
