from brian2.units import *
import numpy as np

from src.network_single_imprint import (
    NetworkSingleImprint,
)
from src.network_multiple_contexts_multiple_assemblies import (
    NetworkMultipleContextsMultipleAssemblies,
)
from src.network_ff_inhibition import run_sim
import matplotlib.pyplot as plt

plt.style.use("../plots_style.txt")


parameter_dict = {}
parameters_for_run = {
    "runtime_imprint": 40 * second,  # 40
    "runtime_baseline": 2.5 * second,
    "normalize": True,
    "seed": 11,
    "run_association": False,
    "monitor_dt_weights": 100 * ms,
}


def Fig_S3(only_load_results=False):
    (
        fig,
        axes_normalization_weights_over_time,
        axes_weight_matrix_norm_no_norm,
        ax_input_spikes_1,
        ax_input_spikes_2,
        ax_recurrent_spikes,
        ax_recurrent_spikes,
        axes_ff_plasticity,
        ff_inhibition_axes,
        rec_inhibition_axes,
    ) = create_figure_layout_Fig_S3()

    networks = {}

    show_range = [
        0,
        parameters_for_run["runtime_imprint"] / msecond
        + 2 * parameters_for_run["runtime_baseline"] / msecond,
    ]

    for normalization in ["normalization", "no normalization"]:
        if normalization == "no normalization":
            parameters_for_run["normalize"] = False

        net = NetworkSingleImprint(
            parameter_file_name="parameters",
            parameters_for_run=parameters_for_run,
            save_file_name="data_Fig_S3_single",
            parameter_dict=parameter_dict,
            only_load_results=False,
        )

        net.only_load_results = only_load_results
        net.run(report_style="text")
        networks[normalization] = net

    # show spikes
    networks["normalization"].show_spike_rasters(
        show_plot=False,
        axes=np.array([ax_input_spikes_1, ax_input_spikes_2, ax_recurrent_spikes]),
        show_range=show_range,
    )

    for ii in range(2):
        ax = axes_weight_matrix_norm_no_norm[ii]
        show_weight_matrix_sorted(
            networks["normalization"],
            ax=ax,
            sort_by_firing_rate_or_strongest_avg_weight=(ii == 0),
        )
        ax.set_title("Normalized: sorted by firing rate")
        if ii == 1:
            ax.set_title("Normalized: sorted by strongest avg weight")

    feedforward_plasticity(
        networks["normalization"],
        axes=axes_ff_plasticity,
        show_range=show_range,
        max_pre_id=30,
    )

    for normalization in ["normalization", "no normalization"]:
        plot_norm_vs_no_norm(
            networks[normalization],
            axes_normalization_weights_over_time,
            random_samples=0,
            show_range=show_range,
            label=normalization,
            set_ylim=normalization == "no normalization",
        )

    for ii in range(2):
        ax = axes_weight_matrix_norm_no_norm[ii + 2]
        show_weight_matrix_sorted(
            networks["no normalization"],
            ax=ax,
            sort_by_firing_rate_or_strongest_avg_weight=(ii == 0),
        )
        ax.set_title("Not Normalized: sorted by firing rate")
        if ii == 1:
            ax.set_title("Not Normalized: sorted by strongest avg weight")

    show_ltp_ltd_threshold_and_n_potentiated_dendrites(
        axes=ff_inhibition_axes,
        n_active_start=15,
        n_active_end=41,
        n_seeds=64,
        prevent_plasticity=True,
        only_load_results=True,
    )

    reccurrent_inhibition_compare_on_vs_off(
        n_seeds=500,
        axes=rec_inhibition_axes,
    )

    fig.tight_layout()
    fig.savefig("../results/figures/Fig_S3.pdf", dpi=800)


def create_figure_layout_Fig_S3():
    fig = plt.figure(figsize=(48, 48))
    gs = fig.add_gridspec(16, 16, hspace=1, wspace=1)

    ax_input_spikes_1 = fig.add_subplot(gs[:1, 4:8])
    ax_input_spikes_2 = fig.add_subplot(gs[1:2, 4:8])
    ax_recurrent_spikes = fig.add_subplot(gs[2:3, 4:8])
    ax_weights_norm_firing_rate = fig.add_subplot(gs[1:3, 8:10])
    ax_weights_norm_avg_weights = fig.add_subplot(gs[1:3, 10:12])

    ax_norm_group_1 = fig.add_subplot(gs[4:5, 4:8])
    ax_norm_group_2 = fig.add_subplot(gs[5:6, 4:8])
    ax_norm_group_3 = fig.add_subplot(gs[6:7, 4:8])
    ax_norm_group_4 = fig.add_subplot(gs[7:8, 4:8])

    ax_weights_no_norm_firing_rate = fig.add_subplot(gs[5:7, 8:10])
    ax_weights_no_norm_avg_weights = fig.add_subplot(gs[5:7, 10:12])

    ax_ff_over_time = fig.add_subplot(gs[4:6, 0:4])
    ax_ff_matrix_before = fig.add_subplot(gs[6:8, 0:2])
    ax_ff_matrix_after = fig.add_subplot(gs[6:8, 2:4])

    ax_ff_inhibtion_threshold = fig.add_subplot(gs[8:10, 4:6])
    ax_ff_inhibtion_n_potentiated = fig.add_subplot(gs[10:12, 4:6])

    # Main scatter axis
    main_ax_rates = fig.add_subplot(gs[9:12, 6:9])
    x_hist_rates = fig.add_subplot(gs[8:9, 6:9], sharex=main_ax_rates)
    y_hist_rates = fig.add_subplot(gs[9:12, 9:10], sharey=main_ax_rates)

    return (
        fig,
        (ax_norm_group_1, ax_norm_group_2, ax_norm_group_3, ax_norm_group_4),
        (
            ax_weights_norm_firing_rate,
            ax_weights_norm_avg_weights,
            ax_weights_no_norm_firing_rate,
            ax_weights_no_norm_avg_weights,
        ),
        ax_input_spikes_1,
        ax_input_spikes_2,
        ax_recurrent_spikes,
        ax_recurrent_spikes,
        (ax_ff_over_time, ax_ff_matrix_before, ax_ff_matrix_after),
        (ax_ff_inhibtion_threshold, ax_ff_inhibtion_n_potentiated),
        (main_ax_rates, x_hist_rates, y_hist_rates),
    )


def feedforward_plasticity(
    net,
    axes=None,
    show_plot=False,
    random_samples=0,
    show_range=[2000, 32000],
    max_pre_id=None,
):
    if axes is None:
        fig, axes = plt.subplots(1, 3)

    ax1, ax2, ax3 = axes

    _, assembly_neuron_ids = net.sort_neurons_by_firing_rate(shuffle_rest=False)

    pot_pot = list(net.potenitially_potentiated_dendrites)

    active_synapses = []
    non_active_synapses = []
    for neuron_id in assembly_neuron_ids:
        dendrite_id = net.area.params["n_dend_each"] * neuron_id

        all_weight_ids = np.where((net.area.input_synapses[0].j)[:] == dendrite_id)[0]
        all_source_ids = net.area.input_synapses[0].i[all_weight_ids]
        all_weights_at_that_dendrite = net.save_dict[
            f"weight_w_ff_1_{pot_pot.index(dendrite_id)}_pot_pot"
        ]

        for source, weight in zip(all_source_ids, all_weights_at_that_dendrite):
            if source < net.area.params["assembly_size"]:
                active_synapses.append(weight)
            else:
                if len(non_active_synapses) < len(
                    active_synapses
                ):  # this ensures the same number in both groups
                    non_active_synapses.append(weight)
    plot_time = net.save_dict["voltage_weights_t"]

    for weight, label, color in zip(
        [active_synapses, non_active_synapses],
        [
            f"high pre activity ({net.area.params['assembly_firing_rate']})",
            f"low pre activity ({net.area.params['ff_bck']})",
        ],
        ["#1f78b4", "#33a02c"],
    ):
        if random_samples == 0:
            ax1.plot(
                plot_time,
                np.mean(weight, axis=0)[:],
                label=label,
            )

        else:
            np.random.seed(5)
            samples = np.random.choice(len(weight), size=random_samples, replace=False)

            for sp in samples:
                ax1.plot(plot_time, weight[sp][:], color=color, alpha=0.4, label=label)
                label = None

    ax1.set(xlabel="Time in ms", ylabel="avg. weight ff", xlim=show_range)
    ax1.legend()

    weights_ff = np.vstack(
        [net.save_dict["weights_ff_1"], net.save_dict["weights_ff_2"]]
    )
    weights_ff_pre = np.copy(weights_ff)
    weights_ff_pre[weights_ff != 0] = net.area.params["ff_w"]

    sorted_neuron_ids, _ = net.sort_neurons_by_firing_rate(shuffle_rest=False)
    sorted_neuron_ids = sorted_neuron_ids[::-1]
    sorted_dendrite_ids = []

    for nn in range(400):
        sorted_dendrite_ids += [
            ii + net.parameters["n_dend_each"] * sorted_neuron_ids[nn]
            for ii in range(net.parameters["n_dend_each"])
        ]

    y_max = 180

    if max_pre_id is None:
        max_pre_id = net.parameters["n_somas"] * 2

    for ww, ax in zip([weights_ff_pre, weights_ff], [ax2, ax3]):
        weights = ww[
            np.ix_([ii for ii in range(max_pre_id)], sorted_dendrite_ids[:y_max])
        ]
        _ = ax.imshow(
            weights.T,
            cmap="Greys",
            origin="lower",
            extent=[-0.5, max_pre_id - 0.5, -0.5, y_max - 0.5],
            vmin=0,
            vmax=net.parameters["w_max_ff"],
        )
        yticks = [ii for ii in range(y_max) if ii % net.parameters["n_dend_each"] == 0]
        ax.set(
            aspect=max_pre_id / y_max,
            xlabel="presynaptic soma\n(sorted by firing rate)",
            ylabel="postsynaptic dendrite\n(sorted by firing rate)",
            yticks=yticks,
            yticklabels=[ii if ii % 18 == 0 else None for ii in yticks],
        )
    if show_plot:
        plt.show()


def show_weight_matrix_sorted(
    net,
    ax,
    sort_by_firing_rate_or_strongest_avg_weight=True,
    take_first_x_neurons=400,
    y_max=180,
):
    weights_loaded_recurrent = net.save_dict["weights"]
    # sorted_neuron_ids, _ = net.sort_neurons_by_firing_rate(shuffle_rest=False)

    if sort_by_firing_rate_or_strongest_avg_weight == False:
        weight_sums_input = np.sum(weights_loaded_recurrent[:, ::6], 0)
        sorted_neuron_ids_by_inputs = np.argsort(weight_sums_input)[::-1]
        # now we select the first X neurons that receive the strongest inputs
        threshold_id = np.argmax(
            weight_sums_input[sorted_neuron_ids_by_inputs]
            <= np.mean((weight_sums_input[sorted_neuron_ids_by_inputs])[:20]) / 3.0
        )
        selected_ids = sorted_neuron_ids_by_inputs[:threshold_id]

        sorted_neuron_ids_by_outputs = list(
            np.argsort(np.sum(weights_loaded_recurrent, 1))
        )[::-1]

        sorted_neuron_ids = list(selected_ids) + [
            n_id for n_id in sorted_neuron_ids_by_outputs if n_id not in selected_ids
        ]
    else:
        sorted_neuron_ids, _ = net.sort_neurons_by_firing_rate(shuffle_rest=False)
        sorted_neuron_ids = sorted_neuron_ids[::-1]  # we only look at one context

    sorted_dendrite_ids = []

    for nn in range(take_first_x_neurons):
        sorted_dendrite_ids += [
            ii + net.parameters["n_dend_each"] * sorted_neuron_ids[nn]
            for ii in range(net.parameters["n_dend_each"])
        ]

    weights = weights_loaded_recurrent[:, :]
    weights = weights[np.ix_(sorted_neuron_ids, sorted_dendrite_ids[:y_max])]
    _ = ax.imshow(
        weights.T,
        cmap="Greys",
        origin="lower",
        extent=[-0.5, net.parameters["n_somas"] - 0.5, -0.5, y_max - 0.5],
    )
    yticks = [ii for ii in range(y_max) if ii % net.parameters["n_dend_each"] == 0]
    ax.set(
        aspect=2,
        xlabel="presynaptic soma\n(sorted)",
        ylabel="postsynaptic dendrite\n(sorted)",
        yticks=yticks,
        yticklabels=[ii if ii % 18 == 0 else None for ii in yticks],
    )


def plot_norm_vs_no_norm(
    net, ax, label, random_samples=0, show_range=[2000, 12000], set_ylim=False
):
    pot_pot = net.potenitially_potentiated_dendrites

    cat_1 = []  # gated and at least four inputs from within assembly
    cat_2 = []  # gated and at least four dendrites from outside assembly
    cat_3 = []  # gated and three dendrites from within assembly
    cat_4 = []  # gated and three dendrites from outside assembly

    _, assembly_neuron_ids = net.sort_neurons_by_firing_rate(shuffle_rest=False)

    for ii in range(len(pot_pot)):
        n_strong_inputs = net.area.counts_gated[0][pot_pot[ii]]

        # now we select all the synapses that target this dendrite
        all_weight_ids = np.where((net.area.synapses_E.j)[:] == pot_pot[ii])[0]
        all_weights_at_that_dendrite = net.save_dict[f"weight_w_{ii}_pot_pot"]

        for wii, weight_id in enumerate(all_weight_ids):
            source_neuron = net.area.synapses_E.i[weight_id]
            target_neuron = net.area.synapses_E.j[weight_id] // 6
            if source_neuron in assembly_neuron_ids:
                if n_strong_inputs >= 4 and target_neuron in assembly_neuron_ids:
                    cat_1.append(all_weights_at_that_dendrite[wii, :])
                if n_strong_inputs == 3 and target_neuron not in assembly_neuron_ids:
                    cat_3.append(all_weights_at_that_dendrite[wii, :])

            else:
                if n_strong_inputs >= 4 and target_neuron in assembly_neuron_ids:
                    cat_2.append(all_weights_at_that_dendrite[wii, :])
                if n_strong_inputs == 3 and target_neuron not in assembly_neuron_ids:
                    cat_4.append(all_weights_at_that_dendrite[wii, :])

    names = ["1", "2", "3", "4"]
    colors = ["#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]

    plot_time = net.save_dict["voltage_weights_t"]

    for ii, (cat, name, color) in enumerate(
        zip([cat_1, cat_2, cat_3, cat_4], names, colors)
    ):
        # randomly sample x synapses
        ls = "-"
        if label == "no normalization":
            ls = "--"

        if random_samples == 0:
            ax[ii].plot(
                plot_time,
                np.mean(cat, axis=0)[:],
                color=color,
                label=name + " " + label,
                ls=ls,
            )

        else:
            np.random.seed(5)
            samples = np.random.choice(len(cat), size=random_samples, replace=False)

            new_label = name + " " + label
            for sp in samples:
                ax.plot(plot_time, cat[sp, :], color=color, alpha=0.4, label=new_label)
                label = None

        ax[ii].axhline(y=0.1, color="k", alpha=0.2, label="starting weight")
        ax[ii].legend()

        if set_ylim:
            ax[ii].set_ylim([0, ax[ii].get_ylim()[1]])
        ax[ii].set_xlim(show_range)


def show_ltp_ltd_threshold_and_n_potentiated_dendrites(
    axes=None,
    n_active_start=15,
    n_active_end=41,
    n_seeds=16,
    prevent_plasticity=True,
    only_load_results=True,
):
    n_max_inputs = 12
    n_different_inptus = n_active_end - n_active_start

    n_ltp_in_total = np.zeros((n_seeds, 2, n_different_inptus)).astype(float) * float(
        "nan"
    )
    fraction_of = np.zeros((n_seeds, n_max_inputs, 2, n_different_inptus)).astype(
        float
    ) * float("nan")
    ltp_ltd_threshold = np.zeros((n_seeds, 2, n_different_inptus)).astype(
        float
    ) * float("nan")

    for seed_id, seed in enumerate(range(n_seeds)):
        for ii in range(n_different_inptus):
            n_active_inputs = ii + n_active_start
            for jj in range(2):
                net = run_sim(
                    seed=seed,
                    n_active_inputs=n_active_inputs,
                    include_adaptive_feedforward_inhibition=(jj == 1),
                    return_results=True,
                    prevent_plasticity=prevent_plasticity,
                    only_load_results=only_load_results,
                    save_file_name="data_Fig_S3_ff_inhibition",
                )

                weights = net.save_dict["silent_synapses_weight"]
                # use average for all silent synapses (there are 10)
                weights = np.mean(
                    weights.reshape(10, weights.shape[0] // 10, weights.shape[1]),
                    axis=0,
                )
                # only look at gated dendrites
                weights = weights[::6, :]
                counts_gated = net.save_dict["counts_gated"].astype(int)

                for n_inputs in range(n_max_inputs):
                    ltp = 0
                    ltd = 0
                    for nn, count in enumerate(counts_gated[0]):
                        if count == n_inputs:
                            if (
                                weights[nn, -1]
                                > net.parameters_for_run[
                                    "silent_synapse_starting_weight"
                                ]
                            ):
                                ltp += 1
                            else:
                                ltd += 1

                    if ltp == 0 and ltd == 0:
                        fraction_of[seed_id, n_inputs, jj, ii] = float("nan")
                    else:
                        fraction_of[seed_id, n_inputs, jj, ii] = ltp / (ltp + ltd)

                all_fractions = fraction_of[seed_id, :, jj, ii]

                first_larger_0 = np.argmax(all_fractions > 0)
                ltp_ltd_threshold[seed_id, jj, ii] = first_larger_0 + (
                    1 - all_fractions[first_larger_0]
                )

                n_ltp_in_total[seed_id, jj, ii] = np.sum(
                    weights[:, -1]
                    > net.parameters_for_run["silent_synapse_starting_weight"]
                )

    if axes is None:
        fig, (ax1, ax2) = plt.subplots(1, 2)
    else:
        ax1, ax2 = axes
    X = [n_active_start + ii for ii in range(n_different_inptus)]

    n_ltp_in_total = np.nanmean(n_ltp_in_total, axis=0)
    ltp_ltd_threshold = np.nanmean(ltp_ltd_threshold, axis=0)

    label = "without ff inhibition"

    for jj in range(2):
        if jj == 1:
            label = "with ff inhibition"
        ax1.plot(X, ltp_ltd_threshold[jj], label=label)
        ax2.plot(X, n_ltp_in_total[jj], label=label)

    ax1.set(xlabel="n active presynaptic inputs", ylabel="LTP/LTD Threshold")
    ax2.set(xlabel="n active presynaptic inputs", ylabel="Potentiated Dendrites")

    ax1.legend()
    ax2.legend()
    if axes is None:
        plt.show()


def load_recurrent_inhibition_comparison(
    n_seeds=500,
):
    estimated_size_from_rates_and_weights = np.zeros((n_seeds, 2)) * float("NaN")
    estimated_size_from_weights = np.zeros((n_seeds, 2)) * float("NaN")

    for seed_id, seed in enumerate(range(n_seeds)):
        print(f"START WITH {seed}")
        save_file_name = "data_Fig_S3_recurrent_inhibition_many"
        assembly_ids = [(0, -1)]
        contexts = [0]

        parameter_dict = {}

        parameters_for_run = {
            "runtime_imprint": 32 * second,  # 12
            "runtime_baseline": 1.5 * second,
            "seed": seed,
            "all_assembly_ids": assembly_ids,
            "all_context_ids": contexts,
            "debug_mode": False,
            "save_weights": True,
            "monitor_dt": 500 * ms,
            "save_most_active_neuron_weights": True,  ###
            "no_recall": True,
        }

        for jj in range(2):
            if jj == 1:
                parameter_dict["rec_inhib_rate"] = 0 * Hz

            net = NetworkMultipleContextsMultipleAssemblies(
                parameter_file_name="parameters",
                parameters_for_run=parameters_for_run,
                save_file_name=save_file_name,
                parameter_dict=parameter_dict,
                only_load_results=True,
            )
            net.run(report_style="text", report_period=30 * second)

            if not net.save_dict:
                continue

            _, assembly_sizes_by_weights = net.sort_neurons_by_weights(show_plot=False)
            _, assembly_neuron_ids, _ = net.sort_neurons_by_firing_rate()
            # not a good naming convention, this method acutally uses to sort by weight and rate

            estimated_size_from_rates_and_weights[seed_id, jj] = len(
                assembly_neuron_ids
            )
            estimated_size_from_weights[seed_id, jj] = assembly_sizes_by_weights[0]

        for jj in range(2):
            if np.isnan(estimated_size_from_rates_and_weights[seed_id, jj]):
                estimated_size_from_rates_and_weights[seed_id, (jj + 1) % 2] = float(
                    "nan"
                )
                estimated_size_from_weights[seed_id, (jj + 1) % 2] = float("nan")

    return (estimated_size_from_rates_and_weights, estimated_size_from_weights)


def reccurrent_inhibition_compare_on_vs_off(
    n_seeds=500,
    axes=None,
    min_val=10,
    max_val=120,
):
    estimated_size_from_rates, _ = load_recurrent_inhibition_comparison(
        n_seeds=n_seeds,
    )

    show_corner_plot(
        y=estimated_size_from_rates[:, 0],
        x=estimated_size_from_rates[:, 1],
        ylabel="rec on (sorted by rates)",
        xlabel="rec off (sorted by rates)",
        axes=axes[:3],
        min_val=min_val,
        max_val=max_val,
    )


def show_corner_plot(x, y, xlabel="x", ylabel="y", axes=None, min_val=10, max_val=120):
    # Create figure and grid

    if axes is None:
        fig = plt.figure(figsize=(8, 8))
        grid = plt.GridSpec(4, 4, hspace=0.2, wspace=0.2)

        # Main scatter axis
        main_ax = fig.add_subplot(grid[1:4, 0:3])
        x_hist = fig.add_subplot(grid[0, 0:3], sharex=main_ax)
        y_hist = fig.add_subplot(grid[1:4, 3], sharey=main_ax)
    else:
        main_ax, x_hist, y_hist = axes

    min_val = 0
    max_val = np.nanmax([100, np.nanmax(x), np.nanmax(y)])

    main_ax.scatter(x, y, alpha=0.5)
    main_ax.set(xlim=[min_val, max_val], ylim=[min_val, max_val])

    # Histogram for the x axis
    x_hist.hist(x, bins=np.arange(max_val) - 0.5, alpha=0.7, density=True)
    plt.setp(x_hist.get_xticklabels(), visible=False)

    # Histogram for the y axis
    y_hist.hist(
        y,
        bins=np.arange(max_val) - 0.5,
        orientation="horizontal",
        alpha=0.7,
        density=True,
    )
    plt.setp(y_hist.get_yticklabels(), visible=False)

    # Hide spines for histograms
    x_hist.spines["top"].set_visible(False)
    x_hist.spines["right"].set_visible(False)
    y_hist.spines["top"].set_visible(False)
    y_hist.spines["right"].set_visible(False)

    # Labels and titles
    main_ax.set_xlabel(xlabel)
    main_ax.set_ylabel(ylabel)


if __name__ == "__main__":
    Fig_S3(only_load_results=True)
