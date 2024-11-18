import h5py
import matplotlib.pyplot as plt
import numpy as np


def plot_delta(delta, matsubara_energy, labels=None, plot_params=None, ax=None):
    """
    Plots the real and imaginary parts of delta as a function of the imaginary part of Matsubara energy.

    Parameters:
        delta (np.ndarray): Array containing the delta data with shape (n_impurities, n_matsubara).
        matsubara_energy (np.ndarray): Array of Matsubara energies (complex) from which the imaginary part will be used.
        labels (list of str, optional): List of labels for each impurity curve. Defaults to "Impurity i" if not provided.
        plot_params (dict, optional): Dictionary of plot parameters to customize the plot appearance.
        ax (matplotlib.axes.Axes, optional): Axis to plot on. Creates a new figure if None.
    """
    n_impurities = delta.shape[0]
    if labels is None:
        labels = [f"Impurity {i}" for i in range(n_impurities)]
    if len(labels) != n_impurities:
        raise ValueError(
            "Length of labels must match the number of impurity curves in delta."
        )

    matsubara_imag = matsubara_energy.imag
    if plot_params is None:
        plot_params = {}

    # Create figure and axis if not provided
    if ax is None:
        fig, axs = plt.subplots(
            2, 1, figsize=plot_params.get("figsize", (10, 8)), sharex=True
        )
        fig.suptitle(plot_params.get("title", "Delta Plot"))
    else:
        axs = [ax, ax]

    # Plot real part
    for i in range(n_impurities):
        axs[0].plot(matsubara_imag, delta[i].real, label=labels[i])
    axs[0].set_ylabel(plot_params.get("ylabel_real", "Re(Δ)"))
    axs[0].legend()
    axs[0].grid(plot_params.get("grid", True))
    axs[0].set_xlim(plot_params.get("xlim"))
    axs[0].set_ylim(plot_params.get("ylim_real"))

    # Plot imaginary part
    for i in range(n_impurities):
        axs[1].plot(matsubara_imag, delta[i].imag, label=labels[i])
    axs[1].set_ylabel(plot_params.get("ylabel_imag", "Im(Δ)"))
    axs[1].set_xlabel(plot_params.get("xlabel", "Imaginary part of Matsubara energy"))
    axs[1].grid(plot_params.get("grid", True))
    axs[1].set_xlim(plot_params.get("xlim"))
    axs[1].set_ylim(plot_params.get("ylim_imag"))

    if ax is None:
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()


def plot_gfloc_spectral_function(
    gfloc, energy_grid, labels=None, plot_params=None, ax=None
):
    """
    Plots -(1/π) * Im(gfloc[i]) for each impurity as a function of energy.

    Parameters:
        gfloc (np.ndarray): Array containing gfloc data with shape (n_impurities, n_energies).
        energy_grid (np.ndarray): Array of energy values corresponding to gfloc.
        labels (list of str, optional): List of labels for each impurity. Defaults to "Impurity i" if not provided.
        plot_params (dict, optional): Dictionary of plot parameters to customize the plot appearance.
        ax (matplotlib.axes.Axes, optional): Axis to plot on. Creates a new figure if None.
    """
    n_impurities = gfloc.shape[0]
    if labels is None:
        labels = [f"Impurity {i}" for i in range(n_impurities)]
    if len(labels) != n_impurities:
        raise ValueError(
            "Length of labels must match the number of impurities in gfloc."
        )

    spectral_function = -(1 / np.pi) * np.imag(gfloc)
    if plot_params is None:
        plot_params = {}

    # Create figure and axis if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=plot_params.get("figsize", (10, 6)))

    for i in range(n_impurities):
        ax.plot(energy_grid, spectral_function[i], label=labels[i])

    ax.set_xlabel(plot_params.get("xlabel", "Energy"))
    ax.set_ylabel(plot_params.get("ylabel", r"$-\frac{1}{\pi} \mathrm{Im}(G_{loc})$"))
    ax.set_title(plot_params.get("title", "Spectral Function for Each Impurity"))
    ax.set_xlim(plot_params.get("xlim"))
    ax.set_ylim(plot_params.get("ylim"))
    ax.legend(loc=plot_params.get("legend_loc", "upper right"), fontsize="small")
    ax.grid(plot_params.get("grid", True))

    if ax is None:
        plt.tight_layout()
        plt.show()


def plot_trace_sigma(sigmas, energy_grid, plot_params=None, ax=None):
    """
    Plots the real and imaginary parts of the trace of sigmas as a function of energy.

    Parameters:
        sigmas (np.ndarray): Array containing sigma data with shape (n_energies, n_impurities, n_impurities).
        energy_grid (np.ndarray): Array of complex energies corresponding to the first dimension of sigmas.
        plot_params (dict, optional): Dictionary of plot parameters to customize the plot appearance.
        ax (matplotlib.axes.Axes, optional): Axis to plot on. Creates a new figure if None.
    """
    trace_sigma = np.trace(sigmas, axis1=1, axis2=2)
    if plot_params is None:
        plot_params = {}

    # Create figure and axis if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=plot_params.get("figsize", (10, 6)))

    ax.plot(energy_grid.real, trace_sigma.real, label="Re(Tr(Σ))", color="blue")
    ax.plot(energy_grid.real, trace_sigma.imag, label="Im(Tr(Σ))", color="red")

    ax.set_xlabel(plot_params.get("xlabel", "Energy (Real part of z_ret)"))
    ax.set_ylabel(plot_params.get("ylabel", "Tr(Σ)"))
    ax.set_title(
        plot_params.get(
            "title", "Real and Imaginary Parts of Tr(Σ) as a Function of Energy"
        )
    )
    ax.legend(loc=plot_params.get("legend_loc", "upper right"))
    ax.set_xlim(plot_params.get("xlim"))
    ax.set_ylim(plot_params.get("ylim"))
    ax.grid(plot_params.get("grid", True))

    if ax is None:
        plt.tight_layout()
        plt.show()


def plot_charge_per_impurity(
    datasets, labels=None, dataset_labels=None, plot_params=None, ax=None
):
    """
    Plots charge per impurity for multiple datasets as a grouped bar plot for comparison.

    Parameters:
        datasets (list of np.ndarray): List of arrays, each containing charge per impurity data.
        labels (list of str, optional): List of labels for each impurity.
        dataset_labels (list of str, optional): List of labels for each dataset.
        plot_params (dict, optional): Dictionary of plot parameters to customize the plot appearance.
        ax (matplotlib.axes.Axes, optional): Axis to plot on. Creates a new figure if None.
    """
    n_impurities = datasets[0].shape[0]
    n_datasets = len(datasets)
    if labels is None:
        labels = [f"Impurity {i}" for i in range(n_impurities)]
    if len(labels) != n_impurities:
        raise ValueError(
            "Length of labels must match the number of impurities in each dataset."
        )
    if dataset_labels is None:
        dataset_labels = [f"Dataset {i}" for i in range(n_datasets)]
    if len(dataset_labels) != n_datasets:
        raise ValueError("Number of dataset labels must match the number of datasets.")

    if plot_params is None:
        plot_params = {}

    # Create figure and axis if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=plot_params.get("figsize", (10, 6)))

    x = np.arange(n_impurities)
    width = 0.8 / n_datasets
    colors = plt.cm.tab10.colors
    colors = colors * (n_datasets // len(colors) + 1)

    for i, data in enumerate(datasets):
        ax.bar(
            x + (i - n_datasets / 2) * width,
            data,
            width,
            label=dataset_labels[i],
            color=colors[i],
        )

    ax.set_xlabel(plot_params.get("xlabel", "Impurity"))
    ax.set_ylabel(plot_params.get("ylabel", "Charge"))
    ax.set_title(plot_params.get("title", "Comparison of Charge per Impurity"))
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend(loc=plot_params.get("legend_loc", "upper right"))
    ax.set_ylim(plot_params.get("ylim"))
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    if ax is None:
        plt.tight_layout()
        plt.show()


def plot_bath_energies(h5_file_path, labels=None, plot_params=None, ax=None):
    """
    Plots the bath energies (ek) for each impurity across iterations.

    Parameters:
        h5_file_path (str): Path to the HDF5 file containing bath parameters.
        labels (list of str, optional): List of labels for each impurity.
        plot_params (dict, optional): Dictionary of plot parameters to customize the plot appearance.
        ax (matplotlib.axes.Axes, optional): Axis to plot on. Creates a new figure if None.
    """
    with h5py.File(h5_file_path, "r") as f:
        iterations = sorted(
            f["all_bath_parameters"].keys(), key=lambda x: int(x.split("_")[-1])
        )
        n_impurities = len(f[f"all_bath_parameters/{iterations[0]}"].keys()) // 2

        if labels is None:
            labels = [f"Impurity {i}" for i in range(n_impurities)]

        ek_values = {i: [] for i in range(n_impurities)}
        iter_nums = []

        for iter_key in iterations:
            iter_num = int(iter_key.split("_")[-1])
            iter_nums.append(iter_num)
            for i in range(n_impurities):
                ek_values[i].append(f[f"all_bath_parameters/{iter_key}/ek_{i}"][:])

    if plot_params is None:
        plot_params = {}

    if ax is None:
        fig, ax = plt.subplots(figsize=plot_params.get("figsize", (10, 6)))

    for i in range(n_impurities):
        ek_values_array = np.array(ek_values[i])
        for j in range(ek_values_array.shape[1]):
            ax.plot(iter_nums, ek_values_array[:, j], label=f"{labels[i]} - ek_{j}")

    ax.set_xlabel(plot_params.get("xlabel", "Iteration"))
    ax.set_ylabel(plot_params.get("ylabel", "Bath Energy (ek)"))
    ax.set_title(
        plot_params.get("title", "Bath Energies (ek) per Impurity Across Iterations")
    )
    ax.set_xlim(plot_params.get("xlim", None))
    ax.set_ylim(plot_params.get("ylim", None))

    legend_columns = plot_params.get("legend_columns", 5)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=legend_columns,
        fontsize="small",
    )
    ax.grid(plot_params.get("grid", True))

    if ax is None:
        plt.show()


def plot_bath_couplings(h5_file_path, labels=None, plot_params=None, ax=None):
    """
    Plots the bath couplings (vk) for each impurity across iterations.

    Parameters:
        h5_file_path (str): Path to the HDF5 file containing bath parameters.
        labels (list of str, optional): List of labels for each impurity.
        plot_params (dict, optional): Dictionary of plot parameters to customize the plot appearance.
        ax (matplotlib.axes.Axes, optional): Axis to plot on. Creates a new figure if None.
    """
    with h5py.File(h5_file_path, "r") as f:
        iterations = sorted(
            f["all_bath_parameters"].keys(), key=lambda x: int(x.split("_")[-1])
        )
        n_impurities = len(f[f"all_bath_parameters/{iterations[0]}"].keys()) // 2

        if labels is None:
            labels = [f"Impurity {i}" for i in range(n_impurities)]

        vk_values = {i: [] for i in range(n_impurities)}
        iter_nums = []

        for iter_key in iterations:
            iter_num = int(iter_key.split("_")[-1])
            iter_nums.append(iter_num)
            for i in range(n_impurities):
                vk_values[i].append(f[f"all_bath_parameters/{iter_key}/vk_{i}"][:])

    if plot_params is None:
        plot_params = {}

    if ax is None:
        fig, ax = plt.subplots(figsize=plot_params.get("figsize", (10, 6)))

    for i in range(n_impurities):
        vk_values_array = np.array(vk_values[i])
        for j in range(vk_values_array.shape[1]):
            ax.plot(iter_nums, vk_values_array[:, j], label=f"{labels[i]} - vk_{j}")

    ax.set_xlabel(plot_params.get("xlabel", "Iteration"))
    ax.set_ylabel(plot_params.get("ylabel", "Bath Coupling (vk)"))
    ax.set_title(
        plot_params.get("title", "Bath Couplings (vk) per Impurity Across Iterations")
    )
    ax.set_xlim(plot_params.get("xlim", None))
    ax.set_ylim(plot_params.get("ylim", None))

    legend_columns = plot_params.get("legend_columns", 5)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=legend_columns,
        fontsize="small",
    )
    ax.grid(plot_params.get("grid", True))

    if ax is None:
        plt.show()


def plot_transmission(
    energies, dft_transmission=None, dmft_transmission=None, plot_params=None
):
    """
    Plots both DFT and DMFT transmission functions on the same plot for comparison.

    Parameters:
        energies (np.ndarray): Array of energy values (real part).
        dft_transmission (np.ndarray, optional): Transmission data for DFT.
        dmft_transmission (np.ndarray, optional): Transmission data for DMFT.
        plot_params (dict, optional): Dictionary of plot parameters to customize the plot appearance.
    """
    if plot_params is None:
        plot_params = {}

    fig, ax = plt.subplots(figsize=plot_params.get("figsize", (10, 6)))

    # Check if yscale is set to "log"
    use_log_scale = plot_params.get("yscale") == "log"
    # Prepare DFT transmission for log scale if applicable
    if dft_transmission is not None:
        if use_log_scale:
            positive_indices_dft = dft_transmission > 0
            if not np.any(positive_indices_dft):
                print(
                    "Warning: DFT transmission has no positive values, unable to apply log scale."
                )
                return
            if not np.all(positive_indices_dft):
                print(
                    "Warning: Some DFT transmission values are non-positive, excluding them for log scale."
                )
            # Filter energies and transmission for positive values only
            ax.plot(
                energies[positive_indices_dft].real,
                dft_transmission[positive_indices_dft],
                label="DFT Transmission",
                color="blue",
            )
        else:
            ax.plot(
                energies.real, dft_transmission, label="DFT Transmission", color="blue"
            )

    # Prepare DMFT transmission for log scale if applicable
    if dmft_transmission is not None:
        if use_log_scale:
            positive_indices_dmft = dmft_transmission > 0
            if not np.any(positive_indices_dmft):
                print(
                    "Warning: DMFT transmission has no positive values, unable to apply log scale."
                )
                return
            if not np.all(positive_indices_dmft):
                print(
                    "Warning: Some DMFT transmission values are non-positive, excluding them for log scale."
                )
            # Filter energies and transmission for positive values only
            ax.plot(
                energies[positive_indices_dmft].real,
                dmft_transmission[positive_indices_dmft],
                label="DMFT Transmission",
                color="red",
            )
        else:
            ax.plot(
                energies.real, dmft_transmission, label="DMFT Transmission", color="red"
            )

    # Customize plot labels, title, and other parameters
    ax.set_xlabel(plot_params.get("xlabel", "Energy (Real part of z_ret)"))
    ax.set_ylabel(plot_params.get("ylabel", "Transmission"))
    ax.set_title(plot_params.get("title", "Transmission Comparison"))
    ax.legend(loc=plot_params.get("legend_loc", "upper right"))

    # Apply log scale if specified
    if use_log_scale:
        ax.set_yscale("log")

    ax.set_xlim(plot_params.get("xlim"))
    ax.set_ylim(plot_params.get("ylim"))
    ax.grid(plot_params.get("grid", True))

    plt.tight_layout()
    plt.show()
