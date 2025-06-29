### Imports
import json
from functools import partial
import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
import matplotlib.patches as patches
from svetlanna import SimulationParameters, wavefront
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
import itertools
from pathlib import Path
import matplotlib
import tqdm
from matplotlib import pyplot as plt
import time
import matplotlib.animation as animation

RESULTS_FOLDER = Path("results")


def exp_to_hist(exp, groups=[(4, 7, 9), (4, 5, 8)]):
    m = torch.load(exp / "confusion_matrix.pt")
    hist = {}
    for group_ind, group in enumerate(groups):
        for i, j in itertools.combinations(group, 2):
            hist[f"{group_ind}: {i}-{j}"] = m[i][j] + m[j][i]
    return hist


def plot_confusion(hist, name):
    plt.figure(figsize=(10, 6))
    plt.bar(hist.keys(), hist.values())
    plt.xlabel("Index Pairs")
    plt.ylabel("Average Value")
    plt.title(name)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def average_on(param: str, value):
    hist = {}
    amount = 0
    for exp_path in RESULTS_FOLDER.iterdir():
        conditions_path = exp_path / "conditions.json"
        if not conditions_path.exists():
            continue

        with open(conditions_path) as f:
            metadata = json.load(f)

        if metadata.get(param) == value:
            for key, val in exp_to_hist(exp_path).items():
                hist[key] = hist.get(key, 0) + val
            amount += 1

    for key in hist:
        hist[key] /= amount
    plot_confusion(hist, f"Average Histogram for '{param}' = '{value}' (averaged over {amount} experiments)")


DIR_RESULTS = "results"
MAX_PHASE = 2 * np.pi  # TODO: Try to remove it
FREESPACE_METHOD = "AS"  # We use an angular spectrum method
MODULATION_TYPE = "amp"  # We will not touch initial phase
NUMBER_OF_CLASSES = 10
ZONES_HIGHLIGHT_COLOR = "w"
ZONES_LW = 0.5
DEVICE = "cpu"
C_CONST = 299_792_458  # [m / s]
NUMBER_OF_CLASSES = 10
MNIST_DATA_FOLDER = Path("./data")  # folder to store data

MNIST_DATA_FOLDER.mkdir(exist_ok=True)


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


def get_zones_patches(detector_mask, delta=0.5):
    def zone_patch(ind_class):
        idx_y, idx_x = (detector_mask == ind_class).nonzero(as_tuple=True)
        return patches.Rectangle(
            (idx_x[0] - delta, idx_y[0] - delta),
            idx_x[-1] - idx_x[0] + 2 * delta,
            idx_y[-1] - idx_y[0] + 2 * delta,
            linewidth=ZONES_LW,
            edgecolor=ZONES_HIGHLIGHT_COLOR,
            facecolor="none",
        )

    return [zone_patch(i) for i in range(NUMBER_OF_CLASSES)]


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


# TODO: try to remove all parameters here
def get_mnist(sim_params: SimulationParameters, train_val_split_seed, train_bs, val_bs, modulation_type):
    mesh = sim_params.axes_size(("H", "W"))  # TODO: Check x-y sizes here
    image_transform_for_ds = get_transforms(mesh, mesh[1], mesh[0], modulation_type)

    mnist = partial(torchvision.datasets.MNIST, MNIST_DATA_FOLDER, download=True)
    mnist_wf = partial(DatasetOfWavefronts, transformations=image_transform_for_ds, sim_params=sim_params)
    mnist_wf_train_ds = mnist_wf(init_ds=mnist(train=True))
    mnist_wf_test_ds = mnist_wf(init_ds=mnist(train=False))
    train_wf_ds, val_wf_ds = torch.utils.data.random_split(
        dataset=mnist_wf_train_ds,
        lengths=[55000, 5000],  # sizes from the article
        generator=torch.Generator().manual_seed(train_val_split_seed),  # for reproducibility
    )

    train_wf_loader = torch.utils.data.DataLoader(train_wf_ds, train_bs, shuffle=True)
    val_wf_loader = torch.utils.data.DataLoader(val_wf_ds, val_bs)
    return train_wf_loader, val_wf_loader, mnist_wf_test_ds


RESULTS_FOLDER = Path("results")


def plot_train_accuracy(exp):
    losses = np.genfromtxt(RESULTS_FOLDER / exp / "training_curves.csv", delimiter=",")
    print("Plotting training accuracy curves for", exp)
    _, _, train_epochs_acc, val_epochs_acc = losses[1:, :].T
    number_of_epochs = len(train_epochs_acc)

    plt.plot(range(1, number_of_epochs + 1), train_epochs_acc, label="Тренировочная точность")
    plt.plot(range(1, number_of_epochs + 1), val_epochs_acc, linestyle="dashed", label="Валидационная точность")
    plt.ylabel("Точность, % от верных ответов")
    plt.xlabel("Эпоха обучения")
    plt.legend()
    plt.show()


def plot_train_accuracy2(exps):
    def get_epochs(exp):
        return len(np.genfromtxt(RESULTS_FOLDER / exp / "training_curves.csv", delimiter=",")[1:, :].T[0])

    plot_train_accuracy(max(list(exps), key=get_epochs))


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


def plot_difflayers(exp):
    with open(RESULTS_FOLDER / exp / "conditions.json") as f:
        v = json.load(f)
    print(v)
    # TODO: Abstract all of that away
    v["freespace_method"] = "AS"

    if "num_of_diff_layers" in v.keys():
        v["num_diff_layers"] = v["num_of_diff_layers"]
    elif "number_of_diff_layers" in v.keys():
        v["num_diff_layers"] = v["number_of_diff_layers"]

    init_phases = torch.ones(v["num_diff_layers"]) * np.pi
    sim_params = get_sim_params(v)
    optical_setup = get_setup(
        v["num_diff_layers"],
        sim_params,
        v["freespace_method"],
        init_phases,
        None,
        v["aperture_size"],
        v["free_space_distance"],
    )
    optical_setup.net.load_state_dict(torch.load(RESULTS_FOLDER / exp / "optical_net.pth"))
    diff_layers = [layer for layer in optical_setup.net if isinstance(layer, DiffractiveLayer)]
    n_cols = len(diff_layers)
    n_rows = 1

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3.2))

    for ind_diff_layer, layer in enumerate(diff_layers):
        if n_rows > 1:
            ax_this = axs[ind_diff_layer // n_cols][ind_diff_layer % n_cols]
        else:
            ax_this = axs[ind_diff_layer % n_cols]

        ax_this.set_title(f"{ind_diff_layer + 1}. DiffractiveLayer")
        im = ax_this.imshow(layer.mask.detach(), "gist_stern")
        fig.colorbar(im)
    plt.show()


def plot_difflayers_gif(exp):
    with open(RESULTS_FOLDER / exp / "conditions.json") as f:
        v = json.load(f)
    v["freespace_method"] = "AS"

    if "num_of_diff_layers" in v.keys():
        v["num_diff_layers"] = v["num_of_diff_layers"]
    elif "number_of_diff_layers" in v.keys():
        v["num_diff_layers"] = v["number_of_diff_layers"]

    init_phases = torch.ones(v["num_diff_layers"]) * np.pi
    sim_params = get_sim_params(v)
    optical_setup = get_setup(
        v["num_diff_layers"],
        sim_params,
        v["freespace_method"],
        init_phases,
        None,
        v["aperture_size"],
        v["free_space_distance"],
    )
    optical_setup.net.load_state_dict(torch.load(RESULTS_FOLDER / exp / "optical_net.pth"))
    diff_layers = [layer for layer in optical_setup.net if isinstance(layer, DiffractiveLayer)]

    fig, ax = plt.subplots(figsize=(4, 4))

    im = ax.imshow(diff_layers[0].mask.detach().cpu().numpy(), cmap="gist_stern")
    fig.colorbar(im, ax=ax)
    ax.axis("off")

    def update(frame):
        layer = diff_layers[frame]
        mask_data = layer.mask.detach().cpu().numpy()
        im.set_data(mask_data)
        im.set_clim(vmin=mask_data.min(), vmax=mask_data.max())
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=len(diff_layers), interval=500, blit=True)
    dest = RESULTS_FOLDER / exp / "difflayers.gif"
    print("saving the gif as ", dest)
    ani.save(dest, writer="pillow")
    plt.show()


def plot_wavefront(test_wavefront, ind_test):
    cmap = "hot"

    plt.title(f"intensity (id={ind_test})")
    plt.imshow(test_wavefront.intensity, cmap)


def plot_propagation(exp):
    plt.title("Интенсивность детектора")
    with open(RESULTS_FOLDER / exp / "conditions.json") as f:
        v = json.load(f)
    print(v)
    v["freespace_method"] = "AS"

    if "num_of_diff_layers" in v.keys():
        v["num_diff_layers"] = v["num_of_diff_layers"]
    elif "number_of_diff_layers" in v.keys():
        v["num_diff_layers"] = v["number_of_diff_layers"]

    init_phases = torch.ones(v["num_diff_layers"]) * np.pi
    sim_params = get_sim_params(v)
    optical_setup_loaded = get_setup(
        v["num_diff_layers"],
        sim_params,
        v["freespace_method"],
        init_phases,
        None,
        v["aperture_size"],
        v["free_space_distance"],
    )
    # TODO: What is that?
    optical_setup_loaded.net.load_state_dict(torch.load(RESULTS_FOLDER / exp / "optical_net.pth"))
    detector_segment_size_m = v["detector_segment_size"] * v["neuron_size"]
    detector_mask = torch.load(RESULTS_FOLDER / exp / "detector_mask.pt")
    detector_processor = DetectorProcessorClf(  # TODO: What does Clf mean
        simulation_parameters=sim_params,
        num_classes=NUMBER_OF_CLASSES,
        segmented_detector=detector_mask,
    )
    selected_detector_mask = detector_processor.segmented_detector.clone().detach()  # ???
    _, _, mnist_wf_test_ds = get_mnist(
        get_sim_params(v),
        train_val_split_seed=v["train_val_seed"],
        train_bs=v["train_batch_size"],
        val_bs=v["val_batch_size"],
        modulation_type="AS",
    )
    wavefront = mnist_wf_test_ds[0][0]
    optical_setup_loaded.net.eval()

    batch_wavefronts = torch.unsqueeze(wavefront, 0)

    with torch.no_grad():
        detector_output = optical_setup_loaded.net(batch_wavefronts)
    plt.imshow(wavefront.detach(), cmap="hot")
    for zone in get_zones_patches(detector_mask):
        plt.axes().add_patch(zone)
    plt.show()


def make_confusion(exp):
    print(f"Making confusion_matrix for {exp}")
    targets_test_lst = []
    preds_test_lst = []  # lists of targets and model predictioons
    with open(RESULTS_FOLDER / exp / "conditions.json") as f:
        v = json.load(f)
    print(v)
    v["freespace_method"] = "AS"

    if "num_of_diff_layers" in v.keys():
        v["num_diff_layers"] = v["num_of_diff_layers"]
    elif "number_of_diff_layers" in v.keys():
        v["num_diff_layers"] = v["number_of_diff_layers"]

    init_phases = torch.ones(v["num_diff_layers"]) * np.pi
    sim_params = get_sim_params(v)
    optical_setup_loaded = get_setup(
        v["num_diff_layers"],
        sim_params,
        v["freespace_method"],
        init_phases,
        None,
        v["aperture_size"],
        v["free_space_distance"],
    )
    # TODO: What is that?
    optical_setup_loaded.net.load_state_dict(torch.load(RESULTS_FOLDER / exp / "optical_net.pth"))
    detector_segment_size_m = v["detector_segment_size"] * v["neuron_size"]
    detector_mask = torch.load(RESULTS_FOLDER / exp / "detector_mask.pt")
    detector_processor = DetectorProcessorClf(  # TODO: What does Clf mean
        simulation_parameters=sim_params,
        num_classes=NUMBER_OF_CLASSES,
        segmented_detector=detector_mask,
    )
    selected_detector_mask = detector_processor.segmented_detector.clone().detach()  # ???
    _, _, mnist_wf_test_ds = get_mnist(
        get_sim_params(v),
        train_val_split_seed=v["train_val_seed"],
        train_bs=v["train_batch_size"],
        val_bs=v["val_batch_size"],
        modulation_type="AS",
    )
    for ind, (wavefront_this, target_this) in enumerate(tqdm.tqdm(mnist_wf_test_ds)):
        optical_setup_loaded.net.eval()

        batch_wavefronts = torch.unsqueeze(wavefront_this, 0)
        batch_labels = torch.unsqueeze(torch.tensor(target_this), 0)  # to use forwards for batches

        with torch.no_grad():
            detector_output = optical_setup_loaded.net(batch_wavefronts)
            # process a detector image
            batch_probas = detector_processor.batch_forward(detector_output)

            for ind_in_batch in range(batch_labels.size()[0]):
                label_this = batch_labels[ind_in_batch].item()  # true label

                targets_test_lst.append(label_this)
                preds_test_lst.append(batch_probas[ind_in_batch].argmax().item())
    confusion_matrix = torch.zeros(
        size=(10, 10),  # TODO: What is the size of the matrix?
        dtype=torch.int32,
    )

    for ind in range(len(mnist_wf_test_ds)):
        confusion_matrix[targets_test_lst[ind], preds_test_lst[ind]] += 1

    with open(RESULTS_FOLDER / exp / "confusion_matrix.pt", "wb") as f:
        torch.save(confusion_matrix, f)


def plot_confusion_matrix(confusion_matrix):
    # plt.imshow(confusion_matrix, vmax=2000)  # , cmap="Blues")
    plt.imshow(confusion_matrix, cmap="nipy_spectral", vmax=2500)  # , cmap="Blues")
    # plt.colorbar()
    # plt.imshow(confusion_matrix, cmap="gist_stern", vmax=500)  # , cmap="Blues")

    for i in range(NUMBER_OF_CLASSES):
        for j in range(NUMBER_OF_CLASSES):
            val = confusion_matrix[j, i].item()
            plt.text(i, j, str(val), va="center", ha="center", fontsize=12, color="white")

    plt.ylabel("Желаемое значение")
    plt.xlabel("Полученное значение")
    plt.xticks(range(10))
    plt.yticks(range(10))
    plt.show()


def const_diff_layers(num_layers, extra_condition="exp"):
    for file in os.listdir(DIR_RESULTS):
        filename = os.fsdecode(file)
        if os.path.isdir(os.path.join(DIR_RESULTS, filename)):  ## OS DECODE?
            results_folder = f"{DIR_RESULTS}/{filename}"
            conditions_file = f"{results_folder}/conditions.json"
            if os.path.exists(conditions_file):
                with open(conditions_file) as json_file:
                    loaded_var = json.load(json_file)
                if extra_condition not in filename:
                    continue
                if "num_diff_layers" in loaded_var.keys() and loaded_var["num_diff_layers"] == num_layers:
                    yield filename
                if "number_of_diff_layers" in loaded_var.keys() and loaded_var["number_of_diff_layers"] == num_layers:
                    yield filename
                if "num_of_diff_layers" in loaded_var.keys() and loaded_var["num_of_diff_layers"] == num_layers:
                    yield filename


def plot_filtered(ax, filt, title, prec=False):
    print(f"Plotting '{title}' as {filt}-filtered {'precision' if prec else 'recall'}")

    def digit_acc(digit, exp):
        try:
            m = torch.load(RESULTS_FOLDER / exp / "confusion_matrix.pt")
        except:
            return None
        if prec:
            if sum(m[i][digit] for i in range(len(m[9]))) == 0:
                return 0
            return m[digit][digit] / sum(m[i][digit] for i in range(len(m[9])))
        else:
            if sum(m[digit]) == 0:
                return 0
            return m[digit][digit] / sum(m[digit])

    colormap = plt.get_cmap("rainbow")
    gradient_values = np.linspace(0, 1, 10)
    colors = [colormap(value) for value in gradient_values]

    def plot_digit(i):
        avg_accs = []
        rng = []
        for diff_layer in range(1, 6):
            avg_acc = 0
            filtered_exps = list(const_diff_layers(diff_layer, filt))
            with_matrix = 0
            for exp in filtered_exps:
                acc = digit_acc(i, exp)
                if acc is not None:
                    avg_acc += acc
                    with_matrix += 1
            if with_matrix != 0:
                ax.plot([diff_layer], [avg_acc / with_matrix], ".", color=colors[i])
                avg_accs.append(avg_acc / with_matrix)
                rng.append(diff_layer)
        print(f"Digit {i}'s accuracies: {avg_accs}")
        ax.plot(rng, avg_accs, color=colors[i], label=i)
        return avg_accs

    plt.xticks(range(1, 6))
    # TODO: Fix no-experiment issue
    digit_lists = []
    for i in range(10):
        digit_lists.append(plot_digit(i))

    plt.legend(loc="upper left")
    # Add grid
    ax.grid(True)

    # Set y-axis scale from 0 to 1 (for accuracy)
    print(filt)
    if filt == "exp_radial":
        ax.set_ylim(0, 1)
    elif prec:
        ax.set_ylim(0.5, 1)
    else:
        ax.set_ylim(0.2, 1)
    ax.set_ylabel("Точность" if prec else "Полнота")
    ax.set_xlabel("Количество дифракционных слоев")
    return digit_lists


def plot_recall(filt, title):
    fig, axes = plt.subplots(1, 1, figsize=(14, 6))
    dl1 = plot_filtered(axes, filt, "Recall for processor setup 1", prec=False)  # [[layers] * digits]
    plt.title(title)
    plt.show()
    return dl1


def plot_precision(filt, title):
    fig, axes = plt.subplots(1, 1, figsize=(14, 6))
    dl1 = plot_filtered(axes, filt, "Precision for processor setup 1", prec=True)  # [[layers] * digits]
    plt.title(title)
    plt.show()
    return dl1


def plot_precision_for_digits(dl1, dl2):
    for i, (digit1_list, digit2_list) in enumerate(zip(dl1, dl2)):
        data = [digit1_list[i] - digit2_list[i] for i in range(len(digit2_list))]
        plt.plot([1, 2, 3, 5], data, label=i)
        plt.xlabel("Количество дифракционных слоев")
        plt.ylabel("Разность в точности для цифр")
        plt.legend()
    plt.plot()


def plot_confusion_matrices(filt, name, diff_layers):
    for layer in diff_layers:
        if exps := list(const_diff_layers(layer, filt)):
            if not (RESULTS_FOLDER / exps[0] / "confusion_matrix.pt").exists():
                make_confusion(exps[0])
            confusion_matrix = torch.load(RESULTS_FOLDER / exps[0] / "confusion_matrix.pt")
            title = f"{name}(n={layer})"
            plt.title(title)
            print(f"Plotting '{title}' matrix")
            plot_confusion_matrix(confusion_matrix)
            time.sleep(0.1)
        else:
            print(f"No experiments filtered as {filt} for {layer} difflayers")


def get_accuracy(exp):
    if not (RESULTS_FOLDER / exp / "confusion_matrix.pt").exists():
        make_confusion(exp)
    confusion_matrix = torch.load(RESULTS_FOLDER / exp / "confusion_matrix.pt")
    return confusion_matrix.diag().sum() / confusion_matrix.sum()


def plot_accuracy(filt, title):
    colormap = plt.get_cmap("rainbow")
    gradient_values = np.linspace(0, 1, 10)
    colors = [colormap(value) for value in gradient_values]
    print(f"Plotting accuracy for {filt}")
    accuracies = []
    diff_layers = []
    once_flag = True  # HACK: i dont know how to label once
    for diff_layer in range(1, 6):
        avg_acc = 0
        accuracies_found = 0
        exps = list(const_diff_layers(diff_layer, filt))
        print(f"Found {len(exps)} experiments filtered as {filt} for {diff_layer} difflayers, namely {exps}")
        for exp in exps:
            avg_acc += (acc := get_accuracy(exp))
            if once_flag:
                plt.plot(diff_layer, acc, ".", label="Эксперименты", color="gray")
                once_flag = False
            else:
                plt.plot(diff_layer, acc, ".", color="gray")
            accuracies_found += 1
        if accuracies_found:
            diff_layers.append(diff_layer)
            accuracies.append(avg_acc / accuracies_found)
    plt.plot(diff_layers, accuracies, label="Средняя точность")

    plt.xlabel("Количество дифракционных слоев")
    plt.ylabel("Точность модели")
    plt.title(title)
    plt.legend()
    plt.show()


def get_center(tensor, digit):
    y_coords, x_coords = (tensor == digit).nonzero(
        as_tuple=True
    )  # TODO: What is the result of .nonzero??? 1-d array? the fuck. What is as_tuple?

    # TODO: What will happen, if I will do .mean on an array of integers?
    return (x_coords.float().mean(), y_coords.float().mean())


FONT_SIZE = 15


def draw_zones(filt, title):
    exp = list(const_diff_layers(3, filt))[0]
    print(f"Plotting zones for {filt}, as taken from {exp}")
    mask = torch.load(RESULTS_FOLDER / exp / "detector_mask.pt")
    plt.imshow(mask, cmap="rainbow")
    for i in range(10):
        plt.text(*get_center(mask, i), str(i), va="center", ha="center", fontsize=FONT_SIZE, color="black")
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

    # Separate labels and values for plotting
    labels = [record[0] for record in records]
    values = [record[1] for record in records]

    # Define colors for the bars
    colors = ["#7f00ff"] * len(labels)
    for i, label in enumerate(labels):
        if label == "mengu, 2020":
            colors[i] = "#9cfaa3"

    # Create the bar chart
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, values, color=colors)

    # Set the x-axis range from 10 to 100
    plt.ylim(10, 100)

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=30, ha="right")

    # Add labels and title
    plt.ylabel("Точность")
    plt.title("Сравнение точности ")

    # Display the plot
    plt.tight_layout()  # Adjust layout to prevent labels from being cut off
    plt.show()


def plot_all_for(filt, filt_name):
    # draw_zones(filt, "Расположение детекторов")  # f"{filt_name}: зоны детектора")
    # plot_confusion_matrices(filt, f"{filt_name}: матрица ошибок", [2, 3, 5])
    # plot_accuracy(filt, f"{filt_name}: точность")
    # plot_recall(filt, f"{filt_name}: полнота(recall)/цифра")
    time.sleep(0.1)
    return plot_precision(filt, f"{filt_name}: точность/цифра")
    return []


def find_outlier(filt, diff_layer, threshold=0.2):
    for exp in const_diff_layers(diff_layer, filt):
        accuracy = get_accuracy(exp)
        if accuracy < threshold:
            print(f"Outlier found: {exp} with accuracy {accuracy:.4f}")
            return exp
    return None


def main():
    print("Finished importing, starting")
    matplotlib.use("TkAgg")
    plt.style.use("dark_background")
    plt.rcParams["axes.titlesize"] = 20  # Title font size
    plt.rcParams["axes.labelsize"] = 16  # Axis label font size
    plt.rcParams["xtick.labelsize"] = 15  # X-tick label font size
    plt.rcParams["ytick.labelsize"] = 15  # Y-tick label font size
    plt.rcParams["legend.fontsize"] = 15  # Legend font size

    if TRANSPARENT := 0:
        plt.rcParams["figure.facecolor"] = (0.0, 0.0, 0.0, 0.0)
        plt.rcParams["axes.facecolor"] = (0.0, 0.0, 0.0, 0.0)
        plt.rcParams["savefig.facecolor"] = (0.0, 0.0, 0.0, 0.0)

    if DISABLE_PLOTS := 0:
        plt.show = lambda: None  # Disable plotting at all
        print("Warning: Plotting is disabled.")

    if PLOT_ALL := 0:
        digits_1 = plot_all_for("exp_2", "Изначальное расположение цифр")
        plot_all_for("exp_zones", "Расположение цифр 1")
        digits_2 = plot_all_for("exp_triplets", "Расположение цифр 2")
        plot_all_for("exp_stripes", "Форма детектора 1")
        plot_all_for("exp_radial", "Форма детектора 2")
        plt.title("Разность точностей для цифр для зон 1 и 2")
        plot_precision_for_digits(digits_1, digits_2)
    # # draw_zones("exp_2", "")
    plot_records()
    # plt.figure(figsize=(7, 7))
    # plt.title("Два дифракционных слоя")
    # plot_train_accuracy2(const_diff_layers(2, "exp_2"))
    # plt.figure(figsize=(7, 7))
    # plt.title("Три дифракционных слоя")
    # plot_train_accuracy2(const_diff_layers(3, "exp_2"))
    # plt.figure(figsize=(7, 7))
    # plt.title("Пять дифракционных слоев")
    # plot_train_accuracy2(const_diff_layers(5, "exp_2"))
    # const_diff_layers()
    # plot_accuracy("exp_2", "Значения точности")
    # plot_confusion_matrices("exp_2", "Матрица ошибок", [2, 3, 5])

    # plot_confusion_matrices("exp_zones", "Матрица ошибок", [3])
    # plot_confusion_matrices("exp_triplets", "Матрица ошибок", [3])
    # plot_confusion_matrices("exp_stripes", "Матрица ошибок", [3])
    # plot_confusion_matrices("exp_radial", "Матрица ошибок", [5])
    # plot_all_for("exp_zones", "Расположение цифр 1")
    # plot_loss_curves("exp_23-06-2025_12-42")
    # plot_difflayers_gif(list(const_diff_layers(3, "exp_2"))[1])
    # plot_propagation(next(const_diff_layers(3, "exp_2")))
    # plot_loss_curves(next(const_diff_layers(3, "exp_2")))
    # plot_loss_curves("exp_23-06-2025_12-42")
    # plot_loss_curves("exp_23-06-2025_12-42")
    # Your data


if __name__ == "__main__":
    main()

# TODO: draw_final(exp, id)
# confusion_matrix with adequate color pattern
# Validation and train vs accuracy
# Plot 4 accuracies on one plot
