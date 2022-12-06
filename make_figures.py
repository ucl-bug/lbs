import os
from test import TRAIN_IDS

import numpy as np
from fire import Fire
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from scipy.io import loadmat


def show_example(
    sos_map,
    pred_field,
    true_field,
):
    # Find the maximum value of the field
    max_val = np.max(np.abs(true_field))
    vmax = 2  # max_val / 2.

    # Prepare figure
    fig, axs = plt.subplots(1, 3, figsize=(8, 2.2), dpi=300)

    raster1 = axs[0].imshow(np.real(true_field), vmin=-vmax, vmax=vmax, cmap="seismic")
    axs[0].axis("off")
    axs[0].set_title("Reference")
    fig.colorbar(raster1, ax=axs[0])

    ax = fig.add_axes([0.10, 0.65, 0.25, 0.25])
    raster2 = ax.imshow(sos_map, vmin=1, vmax=2, cmap="inferno")
    ax.axis("off")

    raster3 = axs[1].imshow(np.real(pred_field), vmin=-vmax, vmax=vmax, cmap="seismic")
    axs[1].axis("off")
    axs[1].set_title("Prediction")
    fig.colorbar(raster3, ax=axs[1])

    # Normalized error map
    error_field = np.abs(true_field - pred_field)
    error_field = 100 * error_field / max_val
    raster4 = axs[2].imshow(error_field, cmap="inferno")
    axs[2].axis("off")
    axs[2].set_title("Difference %")
    cbar = fig.colorbar(raster4, ax=axs[2])
    # cbar.set_ticks(np.log10([0.1, 0.01, 0.001, 0.0001]))
    # cbar.set_ticklabels(["10%", "1%", "0.1%", "0.01%"])
    plt.tight_layout()


def load_data(name, is_born=False):
    path = "results/" + TRAIN_IDS[name] + ".mat"
    data = loadmat(path)

    # Add scaling factor used in loss
    # TODO: Remove
    if not is_born:
        data["prediction"] = data["prediction"] / 10.0
    return data


def get_single_sample(data, example):
    sos = data["sound_speed"][example, ..., 0]
    pred = data["prediction"][example, ..., 0]
    true_field = data["true_field"][example, ..., 0]
    return sos, pred, true_field


def make_example_figure(data, example):
    sos, pred, true_field = get_single_sample(data, example)
    show_example(sos, pred, true_field)


def compute_example_error(pred, true, loss_kind):
    error = np.abs(pred - true)

    # Normalize by maximum value
    error = error / np.max(np.abs(true))

    if loss_kind == "l_infty":
        return 100 * np.amax(error)
    else:
        raise ValueError(f"Unknown loss kind {loss_kind}")


def errors_for_model(name, loss_kind):
    is_born = "born" in name
    data = load_data(name, is_born)

    # Remove channels
    data["prediction"] = data["prediction"][..., 0]
    data["true_field"] = data["true_field"][..., 0]

    # Compute loss for each example
    errors = [
        compute_example_error(data["prediction"][i], data["true_field"][i], loss_kind)
        for i in range(data["prediction"].shape[0])
    ]
    return errors


def make_iterations_error_figure(loss_kind):
    plt.figure(figsize=(8, 3))
    bno_models = {
        "6_stages": 2,
        "base": 6,
        "24_stages": 10,
    }
    cbs_models = {
        "born_series_6": 2,
        "born_series_12": 6,
        "born_series_24": 10,
    }

    linear_models = {
        "linear": 6,
    }

    linear_2_channels_models = {"2_channels": 7}

    for name, num_stages in bno_models.items():
        print(name)
        errors = errors_for_model(name, loss_kind)
        plt.boxplot(
            errors,
            positions=[num_stages - 0.5],
            widths=0.75,
            medianprops=dict(color="black"),
            whiskerprops=dict(color="black"),
            capprops=dict(color="black"),
            flierprops=dict(color="black", marker="x"),
        )

    # Linear model
    for name, num_stages in linear_models.items():
        print(name)
        errors = errors_for_model(name, loss_kind)
        plt.boxplot(
            errors,
            positions=[num_stages + 1.5],
            widths=0.75,
            boxprops=dict(color="green"),
            medianprops=dict(color="green"),
            whiskerprops=dict(color="green"),
            capprops=dict(color="green"),
            flierprops=dict(color="green", marker="x"),
        )

    # Linear model with 2 channels
    for name, num_stages in linear_2_channels_models.items():
        print(name)
        errors = errors_for_model(name, loss_kind)
        plt.boxplot(
            errors,
            positions=[num_stages + 1.5],
            widths=0.75,
            boxprops=dict(color="orange"),
            medianprops=dict(color="orange"),
            whiskerprops=dict(color="orange"),
            capprops=dict(color="orange"),
            flierprops=dict(color="orange", marker="x"),
        )

    # Repeat for CBS, but using red color
    for name, num_stages in cbs_models.items():
        print(name)
        errors = errors_for_model(name, loss_kind)
        plt.boxplot(
            errors,
            positions=[num_stages + 0.5],
            widths=0.75,
            boxprops=dict(color="red"),
            medianprops=dict(color="red"),
            whiskerprops=dict(color="red"),
            capprops=dict(color="red"),
            flierprops=dict(color="red", marker="x"),
        )

    # Add title
    titles = {"l_infty": "Maximum error %"}
    plt.ylabel(titles[loss_kind])

    # Enlarge x-axis
    plt.xlim(0, 12)
    plt.xticks([0, 2, 6, 10], [2, 6, 12, 24])
    plt.xlabel("Iterations")

    # Make legend
    legend_elements = []
    legend_elements.append(Patch(edgecolor="black", label="BNO", facecolor="white"))
    legend_elements.append(
        Patch(edgecolor="red", label="Born Series", facecolor="white")
    )
    legend_elements.append(
        Patch(edgecolor="green", label="Linear BNO", facecolor="white")
    )
    legend_elements.append(
        Patch(edgecolor="orange", label="Linear BNO 2 channels", facecolor="white")
    )
    plt.legend(handles=legend_elements, fontsize=8)

    plt.grid(axis="y")


def show_iterations(example):
    bno_models = ["6_stages", "base", "24_stages"]
    cbs_models = ["born_series_6", "born_series_12", "born_series_24"]
    vmax = 2

    fix, ax = plt.subplots(2, 3, figsize=(9, 6))
    for i, name in enumerate(bno_models):
        data = load_data(name, False)
        _, pred, _ = get_single_sample(data, example)
        ax[0, i].imshow(pred.real, vmin=-vmax, vmax=vmax, cmap="seismic")
        ax[0, i].axis("off")

    for i, name in enumerate(cbs_models):
        data = load_data(name, True)
        _, pred, _ = get_single_sample(data, example)
        ax[1, i].imshow(pred.real, vmin=-vmax, vmax=vmax, cmap="seismic")
        ax[1, i].axis("off")

    plt.tight_layout()


def main(
    results: str = "test",
    figure: str = "example",
    loss_kind: str = "l_infty",
    example: int = 0,
    save_fig: bool = True,
):
    # Make figure
    if figure == "example":
        # Load data
        is_born = "born" in results
        data = load_data(results, is_born)
        make_example_figure(data, example)
    elif figure == "iterations_error":
        make_iterations_error_figure(loss_kind)
    elif figure == "show_iterations":
        show_iterations(example)
    else:
        raise ValueError(f"Unknown figure {figure}")

    # Save figure
    if save_fig:
        if not os.path.exists("figures"):
            os.makedirs("figures")
        plt.savefig(f"figures/{results}_{figure}_{example}.eps", bbox_inches="tight")
        # Save as png
        plt.savefig(
            f"figures/{results}_{figure}_{example}.png", bbox_inches="tight", dpi=300
        )


def make_all_figures():
    pass


if __name__ == "__main__":
    Fire(main)
