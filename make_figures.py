import os
from test import TRAIN_IDS

import numpy as np
from fire import Fire
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from scipy.io import loadmat, savemat


def show_example(
    sos_map,
    pred_field,
    true_field,
):
    # Find the maximum value of the field
    max_val = np.max(np.abs(true_field))
    vmax = 2  # max_val / 2.

    # Prepare figure
    fig, axs = plt.subplots(1, 3, figsize=(8, 2.0), dpi=300)

    raster1 = axs[0].imshow(np.real(true_field), vmin=-vmax, vmax=vmax, cmap="seismic")
    axs[0].axis("off")
    axs[0].set_title("Reference")
    fig.colorbar(raster1, ax=axs[0])

    ax = fig.add_axes([0.152, 0.61, 0.25, 0.25])
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
    # plt.tight_layout()


def load_data(name, is_born=False):

    # Add scaling factor used in loss
    # TODO: Remove
    if not is_born:
        path = "results/" + TRAIN_IDS[name] + ".mat"
        data = loadmat(path)
        data["prediction"] = data["prediction"] / 10.0  #
    else:
        path = "results/" + name + ".mat"
        data = loadmat(path)
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
    plt.figure(figsize=(5, 3))
    bno_models = {
        "6_stages": 2,
        "base": 5.5,
        "24_stages": 10,
    }
    cbs_models = {
        "born_series_6": 2,
        "born_series_12": 5.5,
        "born_series_24": 10,
    }

    linear_2_channels_models = {"2_channels": 5.5}

    for name, num_stages in bno_models.items():
        print(name)
        errors = errors_for_model(name, loss_kind)
        plt.boxplot(
            errors,
            patch_artist=True,
            positions=[num_stages - 0.5],
            widths=0.75,
            boxprops=dict(facecolor="white", color="black"),
            medianprops=dict(color="black"),
            whiskerprops=dict(color="black"),
            capprops=dict(color="black"),
            flierprops=dict(color="black", marker="."),
        )

    # Linear model with 2 channels
    for name, num_stages in linear_2_channels_models.items():
        print(name)
        errors = errors_for_model(name, loss_kind)
        plt.boxplot(
            errors,
            patch_artist=True,
            positions=[num_stages + 1.5],
            widths=0.75,
            boxprops=dict(facecolor="white", color="orange"),
            medianprops=dict(color="orange"),
            whiskerprops=dict(color="orange"),
            capprops=dict(color="orange"),
            flierprops=dict(color="orange", marker="."),
        )

    # Repeat for CBS, but using red color
    for name, num_stages in cbs_models.items():
        print(name)
        errors = errors_for_model(name, loss_kind)
        plt.boxplot(
            errors,
            patch_artist=True,
            positions=[num_stages + 0.5],
            widths=0.75,
            boxprops=dict(facecolor="white", color="red"),
            medianprops=dict(color="red"),
            whiskerprops=dict(color="red"),
            capprops=dict(color="red"),
            flierprops=dict(color="red", marker="."),
        )

    # Add title
    titles = {"l_infty": "Maximum error %"}
    plt.ylabel(titles[loss_kind])

    # Enlarge x-axis
    # plt.xlim(0, 12)
    plt.xticks([2, 6, 10], [6, 12, 24])
    plt.xlabel("Iterations")

    # Make legend
    legend_elements = []
    legend_elements.append(Patch(edgecolor="black", label="LBS", facecolor="white"))
    legend_elements.append(Patch(edgecolor="red", label="LBS", facecolor="white"))
    legend_elements.append(
        Patch(edgecolor="orange", label="Linear LBS 2 ch.", facecolor="white")
    )
    plt.legend(handles=legend_elements, fontsize=8)

    plt.grid(axis="y", which="both")
    # plt.yscale("log")


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


def make_cbs_errors(recompute_data):
    # This function loads the CBS results for every number of iterations and builds up a
    # 2D array with axis [iteration, example] containing the error value
    # It then saves the array to a file
    # If the file already exists, it loads the data from the file if recompute_data is False
    matfile = "results/error_data_for_pareto.mat"
    if os.path.exists(matfile) and not recompute_data:
        data = loadmat(matfile)
        return data["errors"]

    # Generate data
    # Load all results starting with "born_series_", the remaining of the
    # name is the number of iterations
    results = [f for f in os.listdir("results") if f.startswith("born_series_")]
    results = sorted(results, key=lambda x: int(x.split("_")[2][:-4]))
    errors = []
    for result in results:
        print(result)
        # remove the ".mat" extension
        result = result[:-4]
        error_value = errors_for_model(result, "l_infty")
        errors.append(error_value)
    errors = np.asarray(errors)
    savemat(matfile, {"errors": errors})
    return errors


def show_pareto(results, recompute_data=False):
    plt.figure(figsize=(7, 4))

    # Load error for cbs
    cbs_errors = make_cbs_errors(recompute_data)

    # Compute error for bno

    # For each example, find the number of cbs iterations that has the error
    # closest to the bno error, but not larger TODO
    runs = ["6_stages", "base", "24_stages"]
    num_iterations = [6, 12, 24]
    colors = ["black", "darkred", "darkgreen"]
    light_colors = ["#999999", "#ff9999", "#99ff99"]

    x_plot = []
    y_plot = []
    for run, num_iters, col, light_col in zip(
        runs, num_iterations, colors, light_colors
    ):
        print(run)
        bno_errors = errors_for_model(run, "l_infty")

        num_cbs_iterations = []
        for i in range(len(bno_errors)):
            error = bno_errors[i]
            idx = np.argmin(np.abs(cbs_errors[:, i] - error))
            num_cbs_iterations.append(idx)

        # Sort the bno_errors and num_cbs_iterations by bno_errors
        bno_errors = np.asarray(bno_errors)
        idx = np.argsort(bno_errors)
        bno_errors = bno_errors[idx]
        num_cbs_iterations = np.array(num_cbs_iterations)[idx]

        # Transform the num_cbs_iterations to the speed_up_factor
        num_cbs_iterations = num_cbs_iterations / num_iters

        # Plot a point with errorbars for both x and y
        x = np.median(bno_errors)
        y = np.median(num_cbs_iterations)
        xerr_left = np.percentile(bno_errors, 5)
        xerr_right = np.percentile(bno_errors, 95)
        xerr = np.array([[x - xerr_left], [xerr_right - x]])
        yerr_left = np.percentile(num_cbs_iterations, 5)
        yerr_right = np.percentile(num_cbs_iterations, 95)
        yerr = np.array([[y - yerr_left], [yerr_right - y]])
        plt.scatter(bno_errors, num_cbs_iterations, marker=".", color=light_col)
        plt.errorbar(
            x,
            y,
            xerr=xerr,
            yerr=yerr,
            fmt="o",
            label=f"{num_iters} stages",
            color=col,
            capsize=5,
        )

        # Print the mean speed up factor
        print(f"Mean speed up factor: {np.mean(num_cbs_iterations)} for {run}")

        x_plot.append(x)
        y_plot.append(y)
    plt.ylabel("Speed up factor")
    plt.xlabel("Maximum error %")
    plt.xscale("log")

    # Plot the line that connects the points
    plt.plot(x_plot, y_plot, color="black", linestyle="--")

    # Don't use the scientific notation for y ticks
    plt.xticks(
        [1, 1.2, 1.5, 2, 3, 4, 5, 6, 10, 15, 20],
        [1, 1.2, 1.5, 2, 3, 4, 5, 6, 10, 15, 20],
    )
    plt.legend()


def main(
    results: str = "test",
    figure: str = "example",
    loss_kind: str = "l_infty",
    example: int = 0,
    save_fig: bool = True,
    recompute_data: bool = False,
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
    elif figure == "show_pareto":
        show_pareto(results, recompute_data)
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
