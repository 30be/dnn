import json
import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
import matplotlib.patches as patches
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
from src.clf_loops import onn_validate_clf
from pathlib import Path
import matplotlib
from matplotlib import pyplot as plt
import time
import matplotlib.animation as animation
import argparse
import itertools
import tqdm

RESULTS_FOLDER = Path("results")
NUMBER_OF_CLASSES = 10
MNIST_DATA_FOLDER = Path("./data")
MNIST_DATA_FOLDER.mkdir(exist_ok=True)

# Constants from train.py for consistency
C_CONST = 299_792_458  # [m / s]
FREESPACE_METHOD = "AS"
MODULATION_TYPE = "amp"
ZONES_HIGHLIGHT_COLOR = "w"
ZONES_LW = 0.5
DEVICE = "cpu"

def setup_matplotlib():
    matplotlib.use("TkAgg")
    plt.style.use("dark_background")
    plt.rcParams["axes.titlesize"] = 20
    plt.rcParams["axes.labelsize"] = 16
    plt.rcParams["xtick.labelsize"] = 15
    plt.rcParams["ytick.labelsize"] = 15
    plt.rcParams["legend.fontsize"] = 15

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

def get_mnist_test_dataset(sim_params: SimulationParameters, modulation_type):
    mesh = sim_params.axes_size(("H", "W"))
    image_transform_for_ds = get_transforms(mesh, mesh[1], mesh[0], modulation_type)
    mnist_test_ds = torchvision.datasets.MNIST(MNIST_DATA_FOLDER, train=False, download=True)
    return DatasetOfWavefronts(init_ds=mnist_test_ds, transformations=image_transform_for_ds, sim_params=sim_params)

def get_sim_params(v):
    x_layer_size = v["mesh_size"][1] * v["neuron_size"]
    y_layer_size = v["mesh_size"][0] * v["neuron_size"]

    return SimulationParameters(
        axes={
            "W": torch.linspace(-x_layer_size / 2, x_layer_size / 2, v["mesh_size"][1]),
            "H": torch.linspace(-y_layer_size / 2, y_layer_size / 2, v["mesh_size"][0]),
            "wavelength": v["wavelength"],
        }
    )

def load_experiment_data(exp_name):
    exp_path = RESULTS_FOLDER / exp_name
    with open(exp_path / "conditions.json") as f:
        v = json.load(f)
    
    # Handle old naming conventions and provide defaults
    v["num_diff_layers"] = v.get("num_diff_layers") or \
                            v.get("num_of_diff_layers") or \
                            v.get("number_of_diff_layers") or \
                            3 # Default value if not found

    v["modulation_type"] = v.get("modulation_type", MODULATION_TYPE) # Default value if not found
    
    v["freespace_method"] = FREESPACE_METHOD # Ensure consistency

    sim_params = get_sim_params(v)
    init_phases = torch.ones(v["num_diff_layers"]) * np.pi
    optical_setup = get_setup(
        v["num_diff_layers"],
        sim_params,
        v["freespace_method"],
        init_phases,
        v["use_apertures"],
        v["aperture_size"],
        v["free_space_distance"],
    )
    optical_setup.net.load_state_dict(torch.load(exp_path / "optical_net.pth"))
    detector_mask = torch.load(exp_path / "detector_mask.pt")
    detector_processor = DetectorProcessorClf(
        simulation_parameters=sim_params,
        num_classes=NUMBER_OF_CLASSES,
        segmented_detector=detector_mask,
    )
    return v, sim_params, optical_setup, detector_mask, detector_processor

def const_diff_layers(num_layers, extra_condition="exp"):
    for file in os.listdir(RESULTS_FOLDER):
        filename = os.fsdecode(file)
        if os.path.isdir(os.path.join(RESULTS_FOLDER, filename)):
            results_folder = RESULTS_FOLDER / filename
            conditions_file = results_folder / "conditions.json"
            if conditions_file.exists():
                with open(conditions_file) as json_file:
                    loaded_var = json.load(json_file)
                if extra_condition not in filename:
                    continue
                
                num_diff_layers = loaded_var.get("num_diff_layers") or \
                                  loaded_var.get("num_of_diff_layers") or \
                                  loaded_var.get("number_of_diff_layers")
                
                if num_diff_layers == num_layers:
                    yield filename

def plot_training_curves(exp_name):
    losses = np.genfromtxt(RESULTS_FOLDER / exp_name / "training_curves.csv", delimiter=",")
    print(f"Plotting training accuracy curves for {exp_name}")
    _, _, train_epochs_acc, val_epochs_acc = losses[1:, :].T
    number_of_epochs = len(train_epochs_acc)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, number_of_epochs + 1), train_epochs_acc, label="Тренировочная точность")
    plt.plot(range(1, number_of_epochs + 1), val_epochs_acc, linestyle="dashed", label="Валидационная точность")
    plt.ylabel("Точность, % от верных ответов")
    plt.xlabel("Эпоха обучения")
    plt.legend()
    plt.title(f"Training Curves for {exp_name}")
    plt.show()

def plot_difflayers(exp_name):
    v, _, optical_setup, _, _ = load_experiment_data(exp_name)
    diff_layers = [layer for layer in optical_setup.net if isinstance(layer, DiffractiveLayer)]
    n_cols = len(diff_layers)
    n_rows = 1

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3.2))

    for ind_diff_layer, layer in enumerate(diff_layers):
        ax_this = axs[ind_diff_layer % n_cols] if n_rows == 1 else axs[ind_diff_layer // n_cols][ind_diff_layer % n_cols]
        ax_this.set_title(f"{ind_diff_layer + 1}. DiffractiveLayer")
        im = ax_this.imshow(layer.mask.detach(), "gist_stern")
        fig.colorbar(im, ax=ax_this)
    plt.suptitle(f"Diffractive Layers for {exp_name}")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_difflayers_gif(exp_name):
    v, _, optical_setup, _, _ = load_experiment_data(exp_name)
    diff_layers = [layer for layer in optical_setup.net if isinstance(layer, DiffractiveLayer)]

    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(diff_layers[0].mask.detach().cpu().numpy(), cmap="gist_stern")
    fig.colorbar(im, ax=ax)
    ax.axis("off")
    plt.title(f"Diffractive Layers GIF for {exp_name}")

    def update(frame):
        layer = diff_layers[frame]
        mask_data = layer.mask.detach().cpu().numpy()
        im.set_data(mask_data)
        im.set_clim(vmin=mask_data.min(), vmax=mask_data.max())
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=len(diff_layers), interval=500, blit=True)
    dest = RESULTS_FOLDER / exp_name / "difflayers.gif"
    print(f"Saving the gif as {dest}")
    ani.save(dest, writer="pillow")
    plt.show()

def plot_propagation(exp_name):
    v, sim_params, optical_setup, detector_mask, detector_processor = load_experiment_data(exp_name)
    mnist_wf_test_ds = get_mnist_test_dataset(sim_params, v["modulation_type"])
    wavefront_example = mnist_wf_test_ds[0][0] # Get the first test image

    optical_setup.net.eval()
    with torch.no_grad():
        detector_output = optical_setup.net(torch.unsqueeze(wavefront_example, 0))

    plt.figure(figsize=(8, 8))
    plt.imshow(detector_output.squeeze().detach().cpu().numpy(), cmap="hot")
    for zone in get_zones_patches(detector_mask):
        plt.gca().add_patch(zone)
    plt.title(f"Detector Intensity for {exp_name}")
    plt.show()

def make_confusion_matrix(exp_name):
    print(f"Making confusion_matrix for {exp_name}")
    v, sim_params, optical_setup, detector_mask, detector_processor = load_experiment_data(exp_name)
    mnist_wf_test_ds = get_mnist_test_dataset(sim_params, v["modulation_type"])
    
    targets_test_lst = []
    preds_test_lst = []

    optical_setup.net.eval()
    for wavefront_this, target_this in tqdm.tqdm(mnist_wf_test_ds):
        batch_wavefronts = torch.unsqueeze(wavefront_this, 0)
        batch_labels = torch.unsqueeze(torch.tensor(target_this), 0)

        with torch.no_grad():
            detector_output = optical_setup.net(batch_wavefronts)
            batch_probas = detector_processor.batch_forward(detector_output)

            for ind_in_batch in range(batch_labels.size()[0]):
                targets_test_lst.append(batch_labels[ind_in_batch].item())
                preds_test_lst.append(batch_probas[ind_in_batch].argmax().item())
    
    confusion_matrix = torch.zeros(size=(NUMBER_OF_CLASSES, NUMBER_OF_CLASSES), dtype=torch.int32)
    for i in range(len(targets_test_lst)):
        confusion_matrix[targets_test_lst[i], preds_test_lst[i]] += 1

    torch.save(confusion_matrix, RESULTS_FOLDER / exp_name / "confusion_matrix.pt")
    print(f"Confusion matrix saved to {RESULTS_FOLDER / exp_name / 'confusion_matrix.pt'}")

def plot_confusion_matrix_from_file(exp_name):
    confusion_matrix_path = RESULTS_FOLDER / exp_name / "confusion_matrix.pt"
    if not confusion_matrix_path.exists():
        make_confusion_matrix(exp_name) # Generate if not exists
    
    confusion_matrix = torch.load(confusion_matrix_path)
    plt.figure(figsize=(8, 8))
    plt.imshow(confusion_matrix, cmap="nipy_spectral", vmax=2500)

    for i in range(NUMBER_OF_CLASSES):
        for j in range(NUMBER_OF_CLASSES):
            val = confusion_matrix[j, i].item()
            plt.text(i, j, str(val), va="center", ha="center", fontsize=12, color="white")

    plt.ylabel("Желаемое значение")
    plt.xlabel("Полученное значение")
    plt.xticks(range(NUMBER_OF_CLASSES))
    plt.yticks(range(NUMBER_OF_CLASSES))
    plt.title(f"Confusion Matrix for {exp_name}")
    plt.show()

def get_accuracy(exp_name):
    confusion_matrix_path = RESULTS_FOLDER / exp_name / "confusion_matrix.pt"
    if not confusion_matrix_path.exists():
        make_confusion_matrix(exp_name)
    confusion_matrix = torch.load(confusion_matrix_path)
    return confusion_matrix.diag().sum() / confusion_matrix.sum()

def plot_accuracy(filt, title="Model Accuracy"):
    colormap = plt.get_cmap("rainbow")
    gradient_values = np.linspace(0, 1, NUMBER_OF_CLASSES)
    colors = [colormap(value) for value in gradient_values]
    print(f"Plotting accuracy for {filt}")
    
    accuracies = []
    diff_layers_present = []
    
    plt.figure(figsize=(10, 6))
    
    for diff_layer in range(1, 8): # Assuming max 7 layers based on train.py loop
        exps = list(const_diff_layers(diff_layer, filt))
        if not exps:
            continue
        
        avg_acc = 0
        for exp in exps:
            acc = get_accuracy(exp)
            plt.plot(diff_layer, acc, ".", color="gray", alpha=0.6)
            avg_acc += acc
        
        diff_layers_present.append(diff_layer)
        accuracies.append(avg_acc / len(exps))
        
    plt.plot(diff_layers_present, accuracies, label="Средняя точность", color="blue", linewidth=2)

    plt.xlabel("Количество дифракционных слоев")
    plt.ylabel("Точность модели")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 1)
    plt.show()

def get_zones_patches(detector_mask, delta=0.5):
    patches_list = []
    for ind_class in range(NUMBER_OF_CLASSES):
        idx_y, idx_x = (detector_mask == ind_class).nonzero(as_tuple=True)
        if idx_x.numel() > 0 and idx_y.numel() > 0: # Ensure there are points for the class
            min_x, max_x = idx_x.min().item(), idx_x.max().item()
            min_y, max_y = idx_y.min().item(), idx_y.max().item()
            patches_list.append(patches.Rectangle(
                (min_x - delta, min_y - delta),
                max_x - min_x + 1 + 2 * delta,
                max_y - min_y + 1 + 2 * delta,
                linewidth=ZONES_LW,
                edgecolor=ZONES_HIGHLIGHT_COLOR,
                facecolor="none",
            ))
    return patches_list

def get_center(tensor, digit):
    y_coords, x_coords = (tensor == digit).nonzero(as_tuple=True)
    if x_coords.numel() == 0 or y_coords.numel() == 0:
        return (0,0) # Return a default if no points found
    return (x_coords.float().mean().item(), y_coords.float().mean().item())

def draw_zones(filt, title="Detector Zones"):
    exps = list(const_diff_layers(3, filt)) # Assuming 3 layers for zone visualization
    if not exps:
        print(f"No experiments found for filter '{filt}' with 3 diffractive layers.")
        return
    
    exp = exps[0] # Take the first one
    print(f"Plotting zones for {filt}, as taken from {exp}")
    mask = torch.load(RESULTS_FOLDER / exp / "detector_mask.pt")
    
    plt.figure(figsize=(8, 8))
    plt.imshow(mask, cmap="rainbow")
    for i in range(NUMBER_OF_CLASSES):
        center_x, center_y = get_center(mask, i)
        plt.text(center_x, center_y, str(i), va="center", ha="center", fontsize=15, color="black")
    plt.title(title)
    plt.show()

def plot_records():
    records = [
        ("расположение 0", 85),
        ("расположение 1", 84),
        ("расположение 2", 85),
        ("полосы", 83),
        ("радиальная форма", 73),
        ("mengu, 2020", 97),
    ]

    labels = [record[0] for record in records]
    values = [record[1] for record in records]

    colors = ["#7f00ff"] * len(labels)
    for i, label in enumerate(labels):
        if label == "mengu, 2020":
            colors[i] = "#9cfaa3"

    plt.figure(figsize=(10, 6))
    plt.bar(labels, values, color=colors)
    plt.ylim(10, 100)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Точность")
    plt.title("Сравнение точности")
    plt.tight_layout()
    plt.show()

def main():
    setup_matplotlib()

    parser = argparse.ArgumentParser(description="Optical Neural Network Evaluation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Plot training curves
    parser_train_curves = subparsers.add_parser("train_curves", help="Plot training and validation accuracy curves for an experiment.")
    parser_train_curves.add_argument("exp_name", type=str, help="Name of the experiment folder (e.g., 'exp_20-06-2025_14-48').")

    # Plot diffractive layers
    parser_difflayers = subparsers.add_parser("difflayers", help="Plot diffractive layers for an experiment.")
    parser_difflayers.add_argument("exp_name", type=str, help="Name of the experiment folder.")

    # Plot diffractive layers GIF
    parser_difflayers_gif = subparsers.add_parser("difflayers_gif", help="Create and save a GIF of diffractive layers for an experiment.")
    parser_difflayers_gif.add_argument("exp_name", type=str, help="Name of the experiment folder.")

    # Plot propagation
    parser_propagation = subparsers.add_parser("propagation", help="Plot detector intensity for the first test image of an experiment.")
    parser_propagation.add_argument("exp_name", type=str, help="Name of the experiment folder.")

    # Make confusion matrix
    parser_make_confusion = subparsers.add_parser("make_confusion", help="Generate and save the confusion matrix for an experiment.")
    parser_make_confusion.add_argument("exp_name", type=str, help="Name of the experiment folder.")

    # Plot confusion matrix from file
    parser_plot_confusion = subparsers.add_parser("plot_confusion", help="Plot the confusion matrix for an experiment.")
    parser_plot_confusion.add_argument("exp_name", type=str, help="Name of the experiment folder.")

    # Plot overall accuracy
    parser_accuracy = subparsers.add_parser("accuracy", help="Plot overall model accuracy filtered by experiment type.")
    parser_accuracy.add_argument("filter", type=str, help="Filter string for experiment names (e.g., 'exp_radial', 'exp_zones').")
    parser_accuracy.add_argument("--title", type=str, default="Model Accuracy", help="Title for the plot.")

    # Draw detector zones
    parser_draw_zones = subparsers.add_parser("draw_zones", help="Draw detector zones for a given experiment filter.")
    parser_draw_zones.add_argument("filter", type=str, help="Filter string for experiment names (e.g., 'exp_radial', 'exp_zones').")
    parser_draw_zones.add_argument("--title", type=str, default="Detector Zones", help="Title for the plot.")

    # Plot predefined records
    parser_records = subparsers.add_parser("records", help="Plot predefined accuracy records.")

    args = parser.parse_args()

    if args.command == "train_curves":
        plot_training_curves(args.exp_name)
    elif args.command == "difflayers":
        plot_difflayers(args.exp_name)
    elif args.command == "difflayers_gif":
        plot_difflayers_gif(args.exp_name)
    elif args.command == "propagation":
        plot_propagation(args.exp_name)
    elif args.command == "make_confusion":
        make_confusion_matrix(args.exp_name)
    elif args.command == "plot_confusion":
        plot_confusion_matrix_from_file(args.exp_name)
    elif args.command == "accuracy":
        plot_accuracy(args.filter, args.title)
    elif args.command == "draw_zones":
        draw_zones(args.filter, args.title)
    elif args.command == "records":
        plot_records()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
