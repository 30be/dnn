### Imports

import json
from functools import partial
import os
import time
import numpy as np
import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from datetime import datetime
import matplotlib.patches as patches
from svetlanna.units import ureg
from svetlanna import SimulationParameters, simulation_parameters
from svetlanna.parameters import ConstrainedParameter
from svetlanna.transforms import ToWavefront
from svetlanna.elements import (
    FreeSpace,
    RectangularAperture,
    DiffractiveLayer,
)
from svetlanna.setup import LinearOpticalSetup
from svetlanna.detector import Detector, DetectorProcessorClf
from src.wf_datasets import DatasetOfWavefronts
from src.clf_loops import onn_train_clf, onn_validate_clf
import itertools

C_CONST = 299_792_458  # [m / s]
NUMBER_OF_CLASSES = 10
MNIST_DATA_FOLDER = "./data"  # folder to store data
if not os.path.exists(MNIST_DATA_FOLDER):
    os.makedirs(MNIST_DATA_FOLDER)


def get_transforms(detector_size, x_layer_nodes, y_layer_nodes, modulation_type):
    resize_y = int(detector_size[0] / 2)
    resize_x = int(detector_size[1] / 2)  # shape for transforms.Resize

    # paddings along OY
    pad_top = int((y_layer_nodes - resize_y) / 2)
    pad_bottom = y_layer_nodes - pad_top - resize_y
    # paddings along OX
    pad_left = int((x_layer_nodes - resize_x) / 2)
    pad_right = x_layer_nodes - pad_left - resize_x  # params for transforms.Pad

    # compose all transforms!
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(
                size=(resize_y, resize_x),
                interpolation=InterpolationMode.NEAREST,
            ),
            transforms.Pad(
                padding=(
                    pad_left,  # left padding
                    pad_top,  # top padding
                    pad_right,  # right padding
                    pad_bottom,  # bottom padding
                ),
                fill=0,
            ),  # padding to match sizes!
            ToWavefront(modulation_type=modulation_type),  # <- select modulation type!!!
        ]
    )


def get_setup(
    num_layers, simulation_parameters, freespace_method, phase_values, apertures, aperture_size, free_space_distance
):
    free_space = lambda: [FreeSpace(simulation_parameters, free_space_distance, freespace_method)]
    aperture = lambda: [RectangularAperture(simulation_parameters, *aperture_size)] if apertures else []
    phase_layer = lambda phase: DiffractiveLayer(
        simulation_parameters,
        ConstrainedParameter(torch.full(simulation_parameters.axes_size(("H", "W")), phase), 0, 2 * np.pi),
    )

    layer = lambda i: aperture() + [phase_layer(phase_values[i].item())] + free_space()
    layers = list(itertools.chain(*map(layer, range(num_layers))))
    return LinearOpticalSetup(free_space() + layers + aperture() + [Detector(simulation_parameters, "intensity")])


import matplotlib
from matplotlib import pyplot as plt

# matplotlib.use("TkAgg")
# plt.style.use("dark_background")


def get_detector_mask(segment_size_in_neurons: int, segment_dist: int, mesh_size, order_of_digits, sim_params):
    if not len(order_of_digits) == NUMBER_OF_CLASSES:
        print("Wrong ordering list!")

    t = torch.ones(mesh_size) * -1

    x_grid, y_grid = sim_params.meshgrid(x_axis="W", y_axis="H")
    reg = 2 * torch.pi / 10
    for i in range(10):
        ang = torch.atan2(y_grid, x_grid) + torch.pi
        t[(reg * (i + 1) > ang) & (ang > reg * i)] = order_of_digits[i]
    # plt.imshow(t, cmap="rainbow")
    # plt.colorbar()
    # plt.show()
    return t

    upper = [(-1, 1), (0, 1), (1, 1)]
    lower = [(-1, -1), (0, -1), (1, -1)]
    middle = [(-1.5, 0), (-0.5, 0), (0.5, 0), (1.5, 0)]
    for i, (x, y) in enumerate(lower + middle + upper):
        if i not in order_of_digits:
            print("Wrong ordering list!")
        x_start = int(
            segment_size_in_neurons
            + x * segment_dist
            + mesh_size[1] // 2
            - segment_size_in_neurons // 2
            - segment_dist // 2
        )
        y_start = int(
            segment_size_in_neurons
            + y * segment_dist
            + mesh_size[0] // 2
            - segment_size_in_neurons // 2
            - segment_dist // 2
        )
        t[
            y_start : y_start + int(segment_size_in_neurons),
            x_start : x_start + int(segment_size_in_neurons),
        ] = order_of_digits[i]
        print(f"Adding segment for class {order_of_digits[i]} which is {i}th at ({x_start}, {y_start})")
    return t


def save(
    VARIABLES,
    DETECTOR_MASK,
    loss_func_name,
    optical_setup_to_train,
    train_epochs_losses,
    val_epochs_losses,
    train_epochs_acc,
    val_epochs_acc,
):
    today_date = datetime.today().strftime("%d-%m-%Y_%H-%M")  # date for a results folder name

    RESULTS_FOLDER = f"results/exp_radial_{today_date}"

    if not os.path.exists(RESULTS_FOLDER):
        os.makedirs(RESULTS_FOLDER)
    with open(f"{RESULTS_FOLDER}/conditions.json", "w", encoding="utf8") as json_file:
        json.dump(VARIABLES, json_file, ensure_ascii=True)

    torch.save(DETECTOR_MASK, f"{RESULTS_FOLDER}/detector_mask.pt")

    all_lasses_header = ",".join(
        [
            f"{loss_func_name.split()[0]}_train",
            f"{loss_func_name.split()[0]}_val",
            "accuracy_train",
            "accuracy_val",
        ]
    )
    print("Train epoch losses", train_epochs_losses)
    all_losses_array = np.array([train_epochs_losses, val_epochs_losses, train_epochs_acc, val_epochs_acc]).T

    # filepath to save losses
    losses_filepath = f"{RESULTS_FOLDER}/training_curves.csv"
    # saving losses
    np.savetxt(
        losses_filepath,
        all_losses_array,
        delimiter=",",
        header=all_lasses_header,
        comments="",
    )

    # filepath to save the model
    model_filepath = f"{RESULTS_FOLDER}/optical_net.pth"
    # saving model
    torch.save(optical_setup_to_train.net.state_dict(), model_filepath)


# TODO: try to remove all parameters here
def get_mnist(sim_params: SimulationParameters, train_val_split_seed, train_bs, val_bs, modulation_type):
    mesh = sim_params.axes_size(("H", "W"))  # TODO: Check x-y sizes here
    image_transform_for_ds = get_transforms(mesh, mesh[1], mesh[0], modulation_type)

    mnist = partial(torchvision.datasets.MNIST, MNIST_DATA_FOLDER, download=True)
    mnist_wf = partial(DatasetOfWavefronts, transformations=image_transform_for_ds, sim_params=sim_params)
    mnist_wf_train_ds = mnist_wf(init_ds=mnist(train=True))
    # mnist_wf_test_ds = mnist_wf(init_ds=mnist(train=False))
    train_wf_ds, val_wf_ds = torch.utils.data.random_split(
        dataset=mnist_wf_train_ds,
        lengths=[55000, 5000],  # sizes from the article
        generator=torch.Generator().manual_seed(train_val_split_seed),  # for reproducibility
    )

    train_wf_loader = torch.utils.data.DataLoader(train_wf_ds, train_bs, shuffle=True)
    val_wf_loader = torch.utils.data.DataLoader(val_wf_ds, val_bs)
    return train_wf_loader, val_wf_loader


def actually_train(
    optical_setup_to_train,
    num_epochs,
    train_wf_loader,
    val_wf_loader,
    loss_func_clf,
    LR,
    DETECTOR_PROCESSOR,
):
    # Link optimizer to a recreated net!
    optimizer_clf = torch.optim.Adam(
        params=optical_setup_to_train.net.parameters(),
        lr=LR,
    )
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_func_clf = nn.CrossEntropyLoss()

    train_epochs_losses = []
    val_epochs_losses = []  # to store losses of each epoch

    train_epochs_acc = []
    val_epochs_acc = []  # to store accuracies

    for epoch in range(num_epochs):
        print()
        print(f"Epoch #{epoch + 1} of {num_epochs}: ", end="")
        print()
        show_progress = True

        # TRAIN
        start_train_time = time.time()  # start time of the epoch (train)
        train_losses, _, train_accuracy = onn_train_clf(
            optical_setup_to_train.net,  # optical network composed in 3.
            train_wf_loader,  # dataloader of training set
            DETECTOR_PROCESSOR,  # detector processor
            loss_func_clf,
            optimizer_clf,
            device=str(DEVICE),
            show_process=show_progress,
        )  # train the model
        mean_train_loss = np.mean(train_losses)

        print("Training results")
        print(f"\tLoss : {mean_train_loss:.6f}")
        print(f"\tAccuracy : {(train_accuracy * 100):>0.1f} %")
        print(f"\t------------   {time.time() - start_train_time:.2f} s")

        # VALIDATION
        start_val_time = time.time()  # start time of the epoch (validation)
        val_losses, _, val_accuracy = onn_validate_clf(
            optical_setup_to_train.net,  # optical network composed in 3.
            val_wf_loader,  # dataloader of validation set
            DETECTOR_PROCESSOR,  # detector processor
            loss_func_clf,
            device=str(DEVICE),
            show_process=show_progress,
        )  # evaluate the model
        mean_val_loss = np.mean(val_losses)

        print("Validation results")
        print(f"\tLoss : {mean_val_loss:.6f}")
        print(f"\tAccuracy : {(val_accuracy * 100):>0.1f} %")
        print(f"\t------------   {time.time() - start_val_time:.2f} s")

        if scheduler := None:
            scheduler.step(mean_val_loss)

        # save losses
        train_epochs_losses.append(mean_train_loss)
        val_epochs_losses.append(mean_val_loss)
        # seve accuracies
        train_epochs_acc.append(train_accuracy)
        val_epochs_acc.append(val_accuracy)
    return (train_epochs_losses, val_epochs_losses, train_epochs_acc, val_epochs_acc)


def train(
    wavelength=C_CONST / (400 * ureg.GHz),
    neuron_size=0.53,
    mesh_size=(200, 200),
    use_apertures=False,
    modulation_type="amp",  # using ONLY amplitude to encode each picture in a Wavefront!
    num_of_diff_layers=3,
    num_epochs=10,
    torch_seed=int(time.time()),
    free_space_distance=40,
    free_space_method="AS",  # we use an angular spectrum method
    detector_segment_size=6.4,
    zones_order=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    train_bs=20,  # a batch size for training set
    val_bs=8,
    train_val_split_seed=int(time.time()),
    LR=1e-3,
):
    neuron_size *= wavelength
    free_space_distance *= wavelength
    detector_segment_size *= wavelength  # in neurons (int)
    detector_segment_size_m = detector_segment_size // neuron_size  # in [m]
    x_layer_nodes = mesh_size[1]
    y_layer_nodes = mesh_size[0]
    x_layer_size_m = x_layer_nodes * neuron_size  # [m]
    y_layer_size_m = y_layer_nodes * neuron_size
    detector_size = (0, 0) if use_apertures else mesh_size  # ???
    variables = {
        "wavelength": wavelength,  # working wavelength, in [m]
        "neuron_size": neuron_size,  # size of a pixel for DiffractiveLayers, in [m]
        "mesh_size": mesh_size,  # full size of a layer = numerical mesh
        "use_apertures": use_apertures,  # if we need to add apertures before each DiffractieLayer
        "aperture_size": detector_size,  # size of each aperture = a detector square for classes zones
        "detector_segment_size": detector_segment_size_m,  # size of each square class zone on a detector, in [m]
        "segments_order": zones_order,
        "train_val_seed": train_val_split_seed,
        "torch_seed": torch_seed,
        "free_space_distance": free_space_distance,  # constant free space distance for a network, in [m]
        "free_space_method": free_space_method,
        "train_batch_size": train_bs,  # batch sizes for training
        "modulation_type": modulation_type,
        "val_batch_size": val_bs,
        "adam_lr": LR,  # learning rate for Adam optimizer
        "number_of_epochs": num_epochs,  # number of epochs to train
        "num_diff_layers": num_of_diff_layers,
    }
    print(variables)
    sim_params = SimulationParameters(
        axes={
            "W": torch.linspace(-x_layer_size_m / 2, x_layer_size_m / 2, x_layer_nodes),
            "H": torch.linspace(-y_layer_size_m / 2, y_layer_size_m / 2, y_layer_nodes),
            "wavelength": wavelength,
        }
    )

    init_phases = torch.full((num_of_diff_layers,), np.pi)

    detector_mask = get_detector_mask(
        detector_segment_size_m, detector_segment_size_m * 2, mesh_size, zones_order, sim_params
    )

    loss_func_name = "CE loss"
    # Recreate a system to restart training!
    optical_setup_to_train = get_setup(
        num_of_diff_layers, sim_params, free_space_method, init_phases, use_apertures, (100, 100), free_space_distance
    )
    losses = actually_train(
        optical_setup_to_train,
        num_epochs,
        *get_mnist(sim_params, train_val_split_seed, train_bs, val_bs, modulation_type),
        loss_func_name,
        LR,
        DetectorProcessorClf(NUMBER_OF_CLASSES, sim_params, detector_mask),
    )
    save(variables, detector_mask, loss_func_name, optical_setup_to_train, *losses)


# train(num_epochs=0)  # test
for i in [1, 2, 3, 5, 6, 7]:
    print(f"Training {i}th diff layer with the radial pattern.")
    train(num_of_diff_layers=i)
