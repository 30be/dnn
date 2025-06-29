### Imports

import json
import os
import time
import numpy as np
import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from datetime import datetime
from svetlanna.units import ureg
from svetlanna import SimulationParameters
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

C_CONST = 299_792_458  # [m / s]
NUMBER_OF_CLASSES = 10
MNIST_DATA_FOLDER = "./data"  # folder to store data
if not os.path.exists(MNIST_DATA_FOLDER):
    os.makedirs(MNIST_DATA_FOLDER)


def get_transforms(detector_size, x_layer_nodes, y_layer_nodes, modulation_type):
    resize_y = int(detector_size[0] / 2)
    resize_x = int(detector_size[1] / 2)

    pad_top = int((y_layer_nodes - resize_y) / 2)
    pad_bottom = y_layer_nodes - pad_top - resize_y
    pad_left = int((x_layer_nodes - resize_x) / 2)
    pad_right = x_layer_nodes - pad_left - resize_x

    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(
                size=(resize_y, resize_x),
                interpolation=InterpolationMode.NEAREST,
            ),
            transforms.Pad(
                padding=(
                    pad_left,
                    pad_top,
                    pad_right,
                    pad_bottom,
                ),
                fill=0,
            ),
            ToWavefront(modulation_type=modulation_type),
        ]
    )


def get_setup(
    num_layers, simulation_parameters, freespace_method, phase_values, apertures, aperture_size, free_space_distance
):
    elements = [FreeSpace(simulation_parameters, free_space_distance, freespace_method)]
    for i in range(num_layers):
        if apertures:
            elements.append(RectangularAperture(simulation_parameters, *aperture_size))
        elements.append(
            DiffractiveLayer(
                simulation_parameters,
                ConstrainedParameter(torch.full(simulation_parameters.axes_size(("H", "W")), phase_values[i].item()), 0, 2 * np.pi),
            )
        )
        elements.append(FreeSpace(simulation_parameters, free_space_distance, freespace_method))
    if apertures:
        elements.append(RectangularAperture(simulation_parameters, *aperture_size))
    elements.append(Detector(simulation_parameters, "intensity"))
    return LinearOpticalSetup(elements)


def get_detector_mask(mesh_size, order_of_digits, sim_params):
    if not len(order_of_digits) == NUMBER_OF_CLASSES:
        print("Wrong ordering list!")

    t = torch.ones(mesh_size) * -1

    x_grid, y_grid = sim_params.meshgrid(x_axis="W", y_axis="H")
    reg = 2 * torch.pi / 10
    for i in range(10):
        ang = torch.atan2(y_grid, x_grid) + torch.pi
        t[(reg * (i + 1) > ang) & (ang > reg * i)] = order_of_digits[i]
    return t


def save(
    VARIABLES,
    DETECTOR_MASK,
    loss_func_name_base,
    optical_setup_to_train,
    train_epochs_losses,
    val_epochs_losses,
    train_epochs_acc,
    val_epochs_acc,
):
    today_date = datetime.today().strftime("%d-%m-%Y_%H-%M")
    RESULTS_FOLDER = f"results/exp_radial_{today_date}"

    if not os.path.exists(RESULTS_FOLDER):
        os.makedirs(RESULTS_FOLDER)
    with open(f"{RESULTS_FOLDER}/conditions.json", "w", encoding="utf8") as json_file:
        json.dump(VARIABLES, json_file, ensure_ascii=True)

    torch.save(DETECTOR_MASK, f"{RESULTS_FOLDER}/detector_mask.pt")

    all_lasses_header = ",".join(
        [
            f"{loss_func_name_base}_train",
            f"{loss_func_name_base}_val",
            "accuracy_train",
            "accuracy_val",
        ]
    )
    all_losses_array = np.array([train_epochs_losses, val_epochs_losses, train_epochs_acc, val_epochs_acc]).T

    losses_filepath = f"{RESULTS_FOLDER}/training_curves.csv"
    np.savetxt(
        losses_filepath,
        all_losses_array,
        delimiter=",",
        header=all_lasses_header,
        comments="",
    )

    model_filepath = f"{RESULTS_FOLDER}/optical_net.pth"
    torch.save(optical_setup_to_train.net.state_dict(), model_filepath)


def get_mnist(sim_params: SimulationParameters, train_val_split_seed, train_bs, val_bs, modulation_type):
    mesh = sim_params.axes_size(("H", "W"))
    image_transform_for_ds = get_transforms(mesh, mesh[1], mesh[0], modulation_type)

    mnist_train_ds = torchvision.datasets.MNIST(MNIST_DATA_FOLDER, train=True, download=True)
    mnist_wf_train_ds = DatasetOfWavefronts(init_ds=mnist_train_ds, transformations=image_transform_for_ds, sim_params=sim_params)

    train_wf_ds, val_wf_ds = torch.utils.data.random_split(
        dataset=mnist_wf_train_ds,
        lengths=[55000, 5000],
        generator=torch.Generator().manual_seed(train_val_split_seed),
    )

    train_wf_loader = torch.utils.data.DataLoader(train_wf_ds, train_bs, shuffle=True)
    val_wf_loader = torch.utils.data.DataLoader(val_wf_ds, val_bs)
    return train_wf_loader, val_wf_loader


def actually_train(
    optical_setup_to_train,
    num_epochs,
    train_wf_loader,
    val_wf_loader,
    LR,
    DETECTOR_PROCESSOR,
):
    optimizer_clf = torch.optim.Adam(
        params=optical_setup_to_train.net.parameters(),
        lr=LR,
    )
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_func_clf = nn.CrossEntropyLoss()

    train_epochs_losses = []
    val_epochs_losses = []
    train_epochs_acc = []
    val_epochs_acc = []

    for epoch in range(num_epochs):
        print(f"\nEpoch #{epoch + 1} of {num_epochs}: ")
        show_progress = True

        start_train_time = time.time()
        train_losses, _, train_accuracy = onn_train_clf(
            optical_setup_to_train.net,
            train_wf_loader,
            DETECTOR_PROCESSOR,
            loss_func_clf,
            optimizer_clf,
            device=str(DEVICE),
            show_process=show_progress,
        )
        mean_train_loss = np.mean(train_losses)

        print("Training results")
        print(f"\tLoss : {mean_train_loss:.6f}")
        print(f"\tAccuracy : {(train_accuracy * 100):>0.1f} %")
        print(f"\t------------   {time.time() - start_train_time:.2f} s")

        start_val_time = time.time()
        val_losses, _, val_accuracy = onn_validate_clf(
            optical_setup_to_train.net,
            val_wf_loader,
            DETECTOR_PROCESSOR,
            loss_func_clf,
            device=str(DEVICE),
            show_process=show_progress,
        )
        mean_val_loss = np.mean(val_losses)

        print("Validation results")
        print(f"\tLoss : {mean_val_loss:.6f}")
        print(f"\tAccuracy : {(val_accuracy * 100):>0.1f} %")
        print(f"\t------------   {time.time() - start_val_time:.2f} s")

        train_epochs_losses.append(mean_train_loss)
        val_epochs_losses.append(mean_val_loss)
        train_epochs_acc.append(train_accuracy)
        val_epochs_acc.append(val_accuracy)
    return (train_epochs_losses, val_epochs_losses, train_epochs_acc, val_epochs_acc)


def train(
    wavelength=C_CONST / (400 * ureg.GHz),
    neuron_size=0.53,
    mesh_size=(200, 200),
    use_apertures=False,
    modulation_type="amp",
    num_of_diff_layers=3,
    num_epochs=10,
    torch_seed=int(time.time()),
    free_space_distance=40,
    free_space_method="AS",
    detector_segment_size=6.4,
    zones_order=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    train_bs=20,
    val_bs=8,
    train_val_split_seed=int(time.time()),
    LR=1e-3,
):
    neuron_size_m = neuron_size * wavelength
    free_space_distance_m = free_space_distance * wavelength
    detector_segment_size_m = (detector_segment_size * wavelength) // neuron_size_m

    x_layer_nodes, y_layer_nodes = mesh_size[1], mesh_size[0]
    x_layer_size_m = x_layer_nodes * neuron_size_m
    y_layer_size_m = y_layer_nodes * neuron_size_m
    
    aperture_size = (100, 100) # This was hardcoded in the original get_setup call, so keeping it.

    variables = {
        "wavelength": wavelength,
        "neuron_size": neuron_size_m,
        "mesh_size": mesh_size,
        "use_apertures": use_apertures,
        "aperture_size": aperture_size,
        "detector_segment_size": detector_segment_size_m,
        "segments_order": zones_order,
        "train_val_seed": train_val_split_seed,
        "torch_seed": torch_seed,
        "free_space_distance": free_space_distance_m,
        "free_space_method": free_space_method,
        "train_batch_size": train_bs,
        "modulation_type": modulation_type,
        "val_batch_size": val_bs,
        "adam_lr": LR,
        "number_of_epochs": num_epochs,
        "num_diff_layers": num_of_diff_layers,
    }

    sim_params = SimulationParameters(
        axes={
            "W": torch.linspace(-x_layer_size_m / 2, x_layer_size_m / 2, x_layer_nodes),
            "H": torch.linspace(-y_layer_size_m / 2, y_layer_size_m / 2, y_layer_nodes),
            "wavelength": wavelength,
        }
    )

    init_phases = torch.full((num_of_diff_layers,), np.pi)

    detector_mask = get_detector_mask(mesh_size, zones_order, sim_params)

    optical_setup_to_train = get_setup(
        num_of_diff_layers, sim_params, free_space_method, init_phases, use_apertures, aperture_size, free_space_distance_m
    )
    losses = actually_train(
        optical_setup_to_train,
        num_epochs,
        *get_mnist(sim_params, train_val_split_seed, train_bs, val_bs, modulation_type),
        LR,
        DetectorProcessorClf(NUMBER_OF_CLASSES, sim_params, detector_mask),
    )
    save(variables, detector_mask, "CE loss", optical_setup_to_train, *losses)


for i in [1, 2, 3, 5, 6, 7]:
    print(f"Training {i}th diff layer with the radial pattern.")
    train(num_of_diff_layers=i)
