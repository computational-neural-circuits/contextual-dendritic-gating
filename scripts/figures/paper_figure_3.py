from brian2.units import *
import numpy as np

from src.network_multiple_contexts_over_time_with_association import (
    NetworMultipleContextsOverTimeWithAssociation,
)
from src.network_recall import (
    NetworkRecall,
)
from src.network_single_imprint import (
    NetworkSingleImprint,
)
from src.network_multiple_contexts_multiple_assemblies import (
    NetworkMultipleContextsMultipleAssemblies,
)
from src.get_activity_metrics import get_activity_metrics_from_assembly_neurons
from src.utils import get_firing_rate_for_single_neuron, get_assembly_neuron_ids_by_weight_and_rate


import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colorbar import ColorbarBase
import math
from multiprocessing import Pool
from collections import Counter

plt.style.use("../../plots_style.txt")

parameter_dict = {}

parameters_for_run = {
    "runtime_imprint": 40 * second,  # 40
    "runtime_baseline": 2.5 * second,
    "normalize": True,
    "seed": 11,
    "run_association": False,
    "monitor_dt_weights": 100 * ms,
}


def paper_figure_3(only_load_results=True):
    (
        fig,
        axes_normalization_weights_over_time,
        axes_weight_matrix_norm_no_norm,
        ax_weights_norm_soma_weights,
        ax_input_spikes_1,
        ax_input_spikes_2,
        ax_recurrent_spikes,
        ax_recurrent_spikes,
        axes_for_large_imprint,
        ax_for_theoretical_imprint_limit,
        ax_recall_example_firing_rate,
        ax_recall_example_n_active,
        ax_recall_example_colorbar,
        ax_recall_example_firing_rate_over_time,
        ax_recall_example_n_active_over_time,
        ax_recall_projection_fr,
        ax_recall_projection_na,
        ax_recall_association_fr,
        ax_recall_association_na,
    ) = create_figure_layout_paper_fig_3()

    ########################
    # Single FF imprit (start)
    #######################

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
            save_file_name="run_single_imprint",
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
        ax.set_title("Normalized: sorted by firing rate and weights")
        if ii == 1:
            ax.set_title("Normalized: sorted by strongest avg weight")

    show_weight_matrix_sorted(
        net=networks["normalization"],
        ax=ax_weights_norm_soma_weights,
        sort_by_firing_rate_or_strongest_avg_weight=True,
        show_neuron_matrix=True,
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

    ########################
    # larg imprint with recall (start)
    ######################

    all_seeds_for_large_imprint = run_large_imprint_with_recall_on_server(only_get_seeds=True)

    all_assembly_ids_for_areas = [[(0, ii, -1)] for ii in range(20)]
    parameters_for_run_large_imprint = {
        "runtime_imprint": 30 * second,  # 40
        "runtime_baseline": 1 * second,
        "seed": 0,
        "all_assembly_ids_for_areas": all_assembly_ids_for_areas,
        "area_names": ["A"],
        "all_context_ids_for_areas": [[(0, 0)] for _ in all_assembly_ids_for_areas],
        "save_network_after_each_imprint": True,
    }
    net = NetworkRecall(
        parameter_file_name="parameters",
        parameters_for_run=parameters_for_run_large_imprint,
        save_file_name="recall_results_large_imprint_paper_figure_3",
        parameter_dict=parameter_dict,
        only_load_results=False,
    )
    net.only_load_results = only_load_results

    # [ 24, 612]
    all_n_neurons_in_assemblies = np.zeros((len(all_seeds_for_large_imprint), 21))

    all_origial_assembly_sizes = []
    all_firing_rates_after_imprint = []
    for seed_id, seed in enumerate(all_seeds_for_large_imprint):
        net.parameters_for_run["seed"] = seed
        (
            recall_firing,
            recall_active,
            assembly_sizes_for_all_imprints,
            original_assembly_ids_for_all_imprints,
            firing_rate_of_assemblies_after_imprint,
        ) = run_large_imprint_with_recall(
            net=net,
        )

        all_origial_assembly_sizes += [len(xx) for xx in original_assembly_ids_for_all_imprints]
        all_firing_rates_after_imprint += firing_rate_of_assemblies_after_imprint

        label = None
        if seed_id == 0:
            net.show_spike_rasters(axes=axes_for_large_imprint)
            label = "different networks"

            net.show_weight_matrix(
                show_plot=False,
                matlab_export_name=f"../../results/figures/figure_weights/paper_figure_3_for_example_imprint_D",
            )

        show_recall_results_for_large_imprint(
            seed=seed_id,
            axes=[
                ax_recall_example_firing_rate,
                ax_recall_example_n_active,
                ax_recall_example_colorbar,
            ],
            recall_firing=recall_firing,
            recall_active=recall_active,
        )

        y_values = np.cumsum(assembly_sizes_for_all_imprints[-1, :])
        ax_for_theoretical_imprint_limit.plot(
            [ii + 1 for ii in range(len(y_values))],
            y_values,
            color="#0570b0",
            alpha=0.2,
            label=label,
        )
        y_values = assembly_sizes_for_all_imprints[:, 0]
        print(y_values)
        ax_recall_example_firing_rate_over_time.plot(
            [ii + 1 for ii in range(len(y_values))],
            y_values,
            color="#0570b0",
            alpha=0.2,
            label=label,
        )
        ax_recall_example_firing_rate_over_time.set(
            xlabel="# of imprint", ylabel="Assembly size of first imprint"
        )

    show_imprint_results_for_large_imprint(
        axes=[
            ax_recall_example_firing_rate,
            ax_recall_example_n_active,
        ],
        imprint_firing_rates=all_firing_rates_after_imprint,
        imprint_sizes=all_origial_assembly_sizes,
        seed=0,
    )

    theoretical_imprint_limit(
        ax=ax_for_theoretical_imprint_limit,
        n_runs=1000,
        show_plot=False,
    )
    ax_for_theoretical_imprint_limit.legend()

    ########################
    # larg imprint with recall (end)
    ######################

    ########################
    # Recall (start)
    ######################

    all_network_seeds, all_network_seeds_association = run_recall_for_multiple_instances_on_server(
        get_seeds=True
    )

    run_recall_for_multiple_instances(
        axes=ax_recall_projection_fr,
        change_firing_rate=True,
        run_association=False,
        only_load_results=True,
        all_network_seeds=all_network_seeds,
    )
    run_recall_for_multiple_instances(
        axes=ax_recall_projection_na,
        change_firing_rate=False,
        run_association=False,
        only_load_results=True,
        all_network_seeds=all_network_seeds,
    )
    run_recall_for_multiple_instances(
        axes=ax_recall_association_fr,
        change_firing_rate=True,
        run_association=True,
        only_load_results=True,
        all_network_seeds=all_network_seeds_association,
    )
    run_recall_for_multiple_instances(
        axes=ax_recall_association_na,
        change_firing_rate=False,
        run_association=True,
        only_load_results=True,
        all_network_seeds=all_network_seeds_association,
    )

    fig.savefig("../../results/figures/paper_fig_3.pdf", dpi=800)


def run_large_imprint_with_recall(
    net,
    all_context_ids_for_areas_recall=[[(0, 0)], [(0, 1)]],
    recall_area_id=0,
):
    net.run_imprint(
        report_style="text",
        report_period=900 * second,
        restore_beginning=False,
    )

    all_assembly_ids_for_areas = net.parameters_for_run["all_assembly_ids_for_areas"]
    n_imprints = len(all_assembly_ids_for_areas)
    # first we need to find all the neuron assemblies at the end of each the imprint
    assembly_sizes_for_all_imprints = np.zeros((n_imprints, n_imprints)) * float("nan")
    original_assembly_ids_for_all_imprints = []
    firing_rate_of_assemblies_after_imprint = []
    for ii in range(n_imprints):
        network_filename = net.save_dict["filename_for_stored_network"].decode("utf-8") + f"_{ii}"
        net.network.restore(filename=net.get_path_to_stored_networks(file_name=network_filename))
        _, all_assembly_ids, _, all_rates_of_imprint = net.sort_neurons_by_firing_rate(
            area_name="A", return_rates_for_imprint=ii
        )
        # it actually uses rates and weights
        for jj in range(n_imprints):
            if jj > ii:
                break
            assembly_sizes_for_all_imprints[ii, jj] = len(all_assembly_ids[jj])

        print("^^^", assembly_sizes_for_all_imprints[ii])
        original_assembly_ids_for_all_imprints.append(all_assembly_ids[ii])
        firing_rate_of_assemblies_after_imprint.append(
            np.mean(all_rates_of_imprint[all_assembly_ids[ii]])
        )
        # net.show_spike_rasters(
        #     show_plot=True,
        #     highlight_neuron_ids=[[0, original_assembly_ids_for_all_imprints[0]]],
        #     show_vertical_lines_at=[1000, 31000],
        # )

    rtm = net.parameters_for_run["runtime_imprint"] / msecond
    bsl = net.parameters_for_run["runtime_baseline"] / msecond
    runtime_recall = 2 * second
    rtm_rec = runtime_recall / msecond

    recall_firing = np.zeros((n_imprints, 1 + len(all_context_ids_for_areas_recall))) * float("nan")
    recall_active = np.zeros_like(recall_firing) * float("nan")
    for context_id, context_ids_for_areas in enumerate(all_context_ids_for_areas_recall):
        for imprint_id, assembly_ids_for_areas in enumerate(all_assembly_ids_for_areas):
            net.parameters_for_run.update(
                {
                    "all_assembly_ids_for_areas_recall": [assembly_ids_for_areas],
                    "all_context_ids_for_areas_recall": [context_ids_for_areas],
                    "runtime_baseline_recall": 0.1 * second,
                    "runtime_recall": runtime_recall,
                    "run_recall_after_imprint": True,
                    "recall_after_imprint_id": 19,
                }
            )
            net.run_recall(report_style="text")

            if not net.save_dict:
                continue

            start_time = bsl + len(all_assembly_ids_for_areas) * (bsl + rtm)
            end_time = start_time + rtm_rec
            (
                avg_firing_rate_recall,
                n_active_neurons_recall,
                avg_firing_rate_recall_bck,
                n_active_neurons_recall_bck,
            ) = get_activity_metrics_from_assembly_neurons(
                active_threshold=4,
                net=net,
                area=net.all_areas[recall_area_id],
                selected_ids=original_assembly_ids_for_all_imprints[imprint_id],
                start_time=start_time,
                end_time=end_time,
                select_randomly_for_background=True,
            )

            recall_firing[imprint_id, 1 + context_id] = avg_firing_rate_recall
            recall_active[imprint_id, 1 + context_id] = n_active_neurons_recall

            if context_id == 0:
                recall_firing[imprint_id, 0] = avg_firing_rate_recall_bck
                recall_active[imprint_id, 0] = n_active_neurons_recall_bck

    return (
        recall_firing,
        recall_active,
        assembly_sizes_for_all_imprints,
        original_assembly_ids_for_all_imprints,
        firing_rate_of_assemblies_after_imprint,
    )

    return n_neurons_in_assemblies


def run_large_imprint_with_recall_on_server(max_cores=80, only_get_seeds=False):
    all_network_seeds = [
        24,
        612,
        2062,
        485,
        932,
        52,
        995,
        625,
        3523,
        673,
        733,
        7387,
        34,
        78,
        31,
        789,
        321,
        89,
        32,
        63,
    ]

    if only_get_seeds:
        return all_network_seeds

    axes = None
    # iterate over seed
    only_load_results = False
    show_plot = False
    show_recall_for_assembly_id_over_time = None
    area_names = ["A"]
    all_context_ids_for_areas_recall = [[(0, 0)], [(0, 1)]]
    recall_area_id = 0
    show_spike_rasters = False
    new_parameter_dict = None
    only_get_neurons_in_assemblies = True

    params = []
    for seed in all_network_seeds:
        params.append(
            (
                axes,
                seed,
                only_load_results,
                show_plot,
                show_recall_for_assembly_id_over_time,
                area_names,
                all_context_ids_for_areas_recall,
                recall_area_id,
                show_spike_rasters,
                new_parameter_dict,
                only_get_neurons_in_assemblies,
            )
        )

    n_cores = np.min([len(params), max_cores])
    with Pool(n_cores) as pool:
        _ = pool.starmap(
            run_large_imprint_with_recall,
            params,
        )

    only_get_neurons_in_assemblies = False
    params = []
    for seed in all_network_seeds:
        for show_recall_for_assembly_id_over_time in [None, 0, 1, 2, 3, 4, 5]:
            for mm, context_ids_for_areas in enumerate([[(0, 0)], [(0, 1)]]):
                if mm == 1 and show_recall_for_assembly_id_over_time is not None:
                    continue
                all_context_ids_for_areas_recall = [context_ids_for_areas]

                params.append(
                    (
                        axes,
                        seed,
                        only_load_results,
                        show_plot,
                        show_recall_for_assembly_id_over_time,
                        area_names,
                        all_context_ids_for_areas_recall,
                        recall_area_id,
                        show_spike_rasters,
                        new_parameter_dict,
                        only_get_neurons_in_assemblies,
                    )
                )

    n_cores = np.min([len(params), max_cores])
    with Pool(n_cores) as pool:
        _ = pool.starmap(
            run_large_imprint_with_recall,
            params,
        )


def run_recall_for_multiple_instances_on_server(max_cores=200, get_seeds=False):
    all_network_seeds = [452, 213, 394, 839, 320, 100, 78, 912, 444, 102]
    all_network_seeds_association = [573, 812, 552, 602, 5992, 103, 942, 111, 325, 832]

    if get_seeds:
        return all_network_seeds, all_network_seeds_association

    axes = None
    change_firing_rate = None
    only_load_results = False
    show_results = False
    # iterate over run_association
    show_plot = False
    # iterate over specific_seed
    only_run_imprint = True

    params = []

    for run_association in [True, False]:
        all_specific_seeds = all_network_seeds
        if run_association:
            all_specific_seeds = all_network_seeds_association
        for specific_seed in all_specific_seeds:
            params.append(
                (
                    axes,
                    change_firing_rate,
                    only_load_results,
                    show_results,
                    run_association,
                    show_plot,
                    specific_seed,
                    only_run_imprint,
                )
            )

    n_cores = np.min([len(params), max_cores])
    with Pool(n_cores) as pool:
        _ = pool.starmap(
            run_recall_for_multiple_instances,
            params,
        )

    only_run_imprint = False
    params = []
    for change_firing_rate in [True, False]:
        for run_association in [True, False]:
            all_specific_seeds = all_network_seeds
            if run_association:
                all_specific_seeds = all_network_seeds_association
            for specific_seed in all_specific_seeds:
                params.append(
                    (
                        axes,
                        change_firing_rate,
                        only_load_results,
                        show_results,
                        run_association,
                        show_plot,
                        specific_seed,
                        only_run_imprint,
                    )
                )

    n_cores = np.min([len(params), max_cores])
    with Pool(n_cores) as pool:
        _ = pool.starmap(
            run_recall_for_multiple_instances,
            params,
        )


def feedforward_plasticity(
    net, axes=None, show_plot=False, random_samples=0, show_range=[2000, 32000], max_pre_id=None
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

            this_label = label
            for sp in samples:
                ax1.plot(plot_time, weight[sp][:], color=color, alpha=0.4, label=label)
                label = None

    ax1.set(xlabel="Time in ms", ylabel="avg. weight ff", xlim=show_range)
    ax1.legend()

    weights_ff_1 = net.save_dict["weights_ff_1"]
    weights_ff_2 = net.save_dict["weights_ff_2"]
    weights_ff = np.vstack([net.save_dict["weights_ff_1"], net.save_dict["weights_ff_2"]])
    weights_ff_pre = np.copy(weights_ff)
    weights_ff_pre[weights_ff != 0] = net.area.params["ff_w"]

    sorted_neuron_ids, _ = net.sort_neurons_by_firing_rate(shuffle_rest=False)
    sorted_neuron_ids = sorted_neuron_ids[0][::-1]
    sorted_dendrite_ids = []

    # resorted_weights = np.zeros_like(weights_loaded_recurrent)

    for nn in range(400):
        # for mm in range(net.parameters["n_dend_each"]):
        sorted_dendrite_ids += [
            ii + net.parameters["n_dend_each"] * sorted_neuron_ids[nn]
            for ii in range(net.parameters["n_dend_each"])
        ]

    y_max = 180

    if max_pre_id is None:
        max_pre_id = net.parameters["n_somas"] * 2

    for ww, ax in zip([weights_ff_pre, weights_ff], [ax2, ax3]):
        weights = ww[np.ix_([ii for ii in range(max_pre_id)], sorted_dendrite_ids[:y_max])]
        im = ax.imshow(
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


def plot_norm_vs_no_norm(net, ax, label, random_samples=0, show_range=[2000, 12000], set_ylim=False):
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

    for ii, (cat, name, color) in enumerate(zip([cat_1, cat_2, cat_3, cat_4], names, colors)):
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


def show_weight_matrix_sorted(
    net,
    ax,
    sort_by_firing_rate_or_strongest_avg_weight=True,
    take_first_x_neurons=400,
    y_max=180,
    show_neuron_matrix=False,
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

        sorted_neuron_ids_by_outputs = list(np.argsort(np.sum(weights_loaded_recurrent, 1)))[::-1]

        sorted_neuron_ids = list(selected_ids) + [
            n_id for n_id in sorted_neuron_ids_by_outputs if n_id not in selected_ids
        ]
    else:
        sorted_neuron_ids, _ = net.sort_neurons_by_firing_rate(shuffle_rest=False)
        sorted_neuron_ids = sorted_neuron_ids  # we only look at one context

    print("#####", sorted_neuron_ids)
    sorted_dendrite_ids = []

    for nn in range(take_first_x_neurons):
        # for mm in range(net.parameters["n_dend_each"]):
        sorted_dendrite_ids += [
            ii + net.parameters["n_dend_each"] * sorted_neuron_ids[nn]
            for ii in range(net.parameters["n_dend_each"])
        ]

    weights = weights_loaded_recurrent[:, :]
    weights = weights[np.ix_(sorted_neuron_ids, sorted_dendrite_ids[:y_max])]
    ylabel = "postsynaptic dendrite\n(sorted)"

    if show_neuron_matrix:
        weights = weights_loaded_recurrent[:, ::6]
        weights = weights[np.ix_(sorted_neuron_ids, sorted_neuron_ids)]
        y_max = len(sorted_neuron_ids)
        ylabel = "postsynaptic soma\n(sorted)"

    im = ax.imshow(
        weights.T,
        cmap="Greys",
        origin="lower",
        extent=[-0.5, net.parameters["n_somas"] - 0.5, -0.5, y_max - 0.5],
    )
    # ax[1].set_aspect(2)
    yticks = [ii for ii in range(y_max) if ii % net.parameters["n_dend_each"] == 0]
    ax.set(
        aspect="equal",
        xlabel="presynaptic soma\n(sorted)",
        ylabel=ylabel,
        yticks=yticks,
        yticklabels=[ii if ii % 18 == 0 else None for ii in yticks],
    )


def show_recall_results_for_large_imprint(axes, recall_firing, recall_active, seed=0):
    cmap = plt.cm.winter  # define the colormap
    max_val = recall_firing.shape[0]
    bounds = np.arange(0, max_val + 1)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    # colors = [colormap(ii / recall_firing.shape[0]) for ii in range(recall_firing.shape[0])]

    np.random.seed(seed)
    for ii in range(recall_firing.shape[1]):
        for jj in range(recall_firing.shape[0]):
            xx = ii + 0.2 * (0.5 - np.random.rand())
            if ii == 1:
                xx += 0.4
            axes[0].scatter(xx, recall_firing[jj, ii], color=cmap(norm(jj)), s=20)
    axes[0].set(ylabel="avg firing rate of assembly neurons")

    for ii in range(recall_firing.shape[1]):
        for jj in range(recall_active.shape[0]):
            xx = ii + 0.2 * (0.5 - np.random.rand())
            if ii == 1:
                ii + 0.5
            axes[1].scatter(xx, recall_active[jj, ii], color=cmap(norm(jj)), s=20)
    axes[1].set(ylabel="n active")

    cb = ColorbarBase(
        axes[2],
        cmap=cmap,
        norm=norm,
        spacing="proportional",
        ticks=bounds,
        boundaries=bounds,
        format="%1i",
    )
    axes[2].set_title("Position of learning")

    # ax_recall_example.errorbar(
    #     [ii for ii in range(3)],
    #     np.nanmean(assembly_firing_rate, axis=0),
    #     yerr=np.nanstd(assembly_firing_rate, axis=0),
    # )
    for ax in axes:
        xticks = [0, 0.6, 1.4, 2]
        xticklabels = [
            "Background",
            "Assembly\nimprint",
            "Assembly\ncontext 1",
            "Assembly\ncontext 2",
        ]
        ax.set(
            xticks=xticks,
            xticklabels=xticklabels,
        )


def show_imprint_results_for_large_imprint(axes, imprint_firing_rates, imprint_sizes, seed=0):
    np.random.seed(seed)
    for rate in imprint_firing_rates:
        xx = 0.6 + 0.2 * (0.5 - np.random.rand())  # we shift to 0.6 because above we shifted to 1.4
        axes[0].scatter(xx, rate, color="k", s=20)
    axes[0].scatter(0.6, np.mean(imprint_firing_rates), color="r")

    for size in imprint_sizes:
        xx = 0.6 + 0.2 * (0.5 - np.random.rand())
        axes[1].scatter(xx, size, color="k", s=20)
    axes[1].scatter(0.6, np.mean(imprint_sizes), color="r")


def create_figure_layout_paper_fig_3():
    fig = plt.figure(figsize=(53, 48))
    gs = fig.add_gridspec(18, 17, hspace=1, wspace=1)

    ax_input_spikes_1 = fig.add_subplot(gs[:1, 4:8])
    ax_input_spikes_2 = fig.add_subplot(gs[1:2, 4:8])
    ax_recurrent_spikes = fig.add_subplot(gs[2:3, 4:8])
    ax_weights_norm_firing_rate = fig.add_subplot(gs[1:3, 8:10])
    ax_weights_norm_avg_weights = fig.add_subplot(gs[1:3, 10:12])
    ax_weights_norm_soma_weights = fig.add_subplot(gs[1:3, 12:14])

    ax_norm_group_1 = fig.add_subplot(gs[4:5, 4:8])
    ax_norm_group_2 = fig.add_subplot(gs[5:6, 4:8])
    ax_norm_group_3 = fig.add_subplot(gs[6:7, 4:8])
    ax_norm_group_4 = fig.add_subplot(gs[7:8, 4:8])

    axes_for_large_imprint = [
        [
            fig.add_subplot(gs[8:12, :4]),
            fig.add_subplot(gs[12:16, :4]),
            fig.add_subplot(gs[8:12, 4:8]),
        ]
    ]
    ax_for_theoretical_imprint_limit = fig.add_subplot(gs[8:12, 8:12])
    ax_recall_example_firing_rate = fig.add_subplot(gs[8:10, 12:14])
    ax_recall_example_n_active = fig.add_subplot(gs[8:10, 14:16])
    ax_recall_example_colorbar = fig.add_subplot(gs[8:10, 16:17])

    ax_recall_example_firing_rate_over_time = fig.add_subplot(gs[10:12, 12:14])
    ax_recall_example_n_active_over_time = fig.add_subplot(gs[10:12, 14:16])

    ax_recall_projection_fr = [fig.add_subplot(gs[13:15, 5:7]), fig.add_subplot(gs[13:15, 7:9])]
    ax_recall_projection_na = [fig.add_subplot(gs[15:17, 5:7]), fig.add_subplot(gs[15:17, 7:9])]

    ax_recall_association_fr = [fig.add_subplot(gs[13:15, 10:12]), fig.add_subplot(gs[13:15, 12:14])]
    ax_recall_association_na = [fig.add_subplot(gs[15:17, 10:12]), fig.add_subplot(gs[15:17, 12:14])]

    return (
        fig,
        (ax_norm_group_1, ax_norm_group_2, ax_norm_group_3, ax_norm_group_4),
        (
            ax_weights_norm_firing_rate,
            ax_weights_norm_avg_weights,
            None,
            None,
        ),
        ax_weights_norm_soma_weights,
        ax_input_spikes_1,
        ax_input_spikes_2,
        ax_recurrent_spikes,
        ax_recurrent_spikes,
        axes_for_large_imprint,
        ax_for_theoretical_imprint_limit,
        ax_recall_example_firing_rate,
        ax_recall_example_n_active,
        ax_recall_example_colorbar,
        ax_recall_example_firing_rate_over_time,
        ax_recall_example_n_active_over_time,
        ax_recall_projection_fr,
        ax_recall_projection_na,
        ax_recall_association_fr,
        ax_recall_association_na,
    )


def show_weight_dynamics_of_assemblies_over_time(net, ax, show_plot=False):
    all_assembly_weights = []

    unique_contexts = list(np.sort(np.unique(net.parameters_for_run["all_context_ids"])))
    # unique_assembly_ids = list(np.sort(np.unique(self.parameters_for_run["all_assembly_ids"])))

    # all_weights = [[[] for ii in unique_assembly_ids] for jj in unique_contexts]

    colors = [
        ["#08519c", "#4292c6", "#9ecae1"],
        ["#006d2c", "#41ab5d", "#a1d99b"],
        ["#d94801", "#fd8d3c", "#fdd0a2"],
    ]

    n_assembly = [-1 for ii in unique_contexts]

    for context_id, assembly_ids in zip(
        net.parameters_for_run["all_context_ids"], net.parameters_for_run["all_assembly_ids"]
    ):
        assembly_neuron_ids = net.get_assembly_neuron_ids(context_id, assembly_ids)

        aa = unique_contexts.index(context_id)
        # bb = unique_assembly_ids.index(assembly_id)

        n_assembly[aa] += 1
        all_weights = []

        all_post_dend_ids = []

        for ii, post_dend_id in enumerate(net.rec_list):
            post_neuron_id = post_dend_id // net.parameters["n_dend_each"]

            if post_neuron_id in assembly_neuron_ids:
                if post_dend_id % net.parameters["n_dend_each"] == context_id:
                    # now we go through all weights and see if they come from within assembly neurons

                    synapse_ids = np.where((net.area.synapses_E.j)[:] == post_dend_id)[0]

                    synapse_sources = net.area.synapses_E.i[synapse_ids]

                    synapses_from_assembly = [
                        mm
                        for mm, source in enumerate(synapse_sources)
                        if source in assembly_neuron_ids
                    ]

                    print(
                        "#",
                        len(synapses_from_assembly),
                        len(assembly_neuron_ids),
                        post_dend_id,
                        post_neuron_id,
                    )

                    weights = net.save_dict["pot_pot_recurrent_w"][
                        ii
                        * (net.parameters["n_somas"] - 1) : (ii + 1)
                        * (net.parameters["n_somas"] - 1),
                        :,
                    ]  # n_somas - 1, since we have no self connection

                    weights_from_within = weights[synapses_from_assembly, :]

                    # all_weights[aa][bb].append(weights_from_within)
                    all_weights.append(weights_from_within)

        try:
            # all_weights[aa][bb] = np.vstack(all_weights[aa][bb])
            # these_weights = all_weights[aa][bb]
            all_weights = np.vstack(all_weights)
            these_weights = all_weights
            ax.plot(
                net.save_dict["weights_t"],
                np.mean(these_weights, axis=0),
                label=f"ctxt:{context_id} | assembly:{assembly_ids[0]} - {assembly_ids[1]}",
                color=colors[aa][n_assembly[aa]],
            )

            # ax.plot(
            #     net.save_dict["weights_t"],
            #     (these_weights.T)[:, :40],
            #     color=colors[aa][bb],
            #     lw=0.8,
            #     alpha=0.4,
            # )
        except ValueError:
            pass

    ax.legend()
    ax.set(xlabel="Time in ms", ylabel="average weight within assembly")

    if show_plot:
        plt.show()


def theoretical_imprint_limit(ax=None, n_runs=1000, show_plot=False):
    if ax is None:
        fig, ax = plt.subplots()

    all_ids = [ii for ii in range(400)]

    all_numbers = np.zeros((n_runs, 20))

    estimated_size_from_rates_and_weights, _ = load_recurrent_inhibition_comparison(
        n_seeds=500,
        run_association=False,
    )

    all_empirical_assembly_sizes = estimated_size_from_rates_and_weights[:, 0].astype(
        int
    )  # results with inhibition 'on'

    for run in range(n_runs):
        all_used_ids = []

        all_assembly_ids = [[] for _ in range(20)]
        for imprint_id in range(20):
            # vary the assembly size
            assembly_size = np.random.choice(all_empirical_assembly_sizes, 1)

            assembly_ids = list(np.random.choice(all_ids, assembly_size, replace=False))
            # draw_ids = []
            all_assembly_ids[imprint_id] = assembly_ids
            for jj in range(imprint_id):
                for a_id in assembly_ids:
                    if a_id in all_assembly_ids[jj]:
                        all_assembly_ids[jj].remove(a_id)

        all_numbers[run, :] = np.cumsum([len(ai) for ai in all_assembly_ids])

    x = [ii + 1 for ii in range(all_numbers.shape[1])]
    ax.errorbar(x, np.mean(all_numbers, axis=0), yerr=np.std(all_numbers, axis=0), label="Theory")

    ax.set(
        xlabel="# number of imprint",
        ylabel="# cumulative assembly sizes after all imprints",
        ylim=[0, 400],
    )

    if show_plot:
        plt.show()


def load_recurrent_inhibition_comparison(
    n_seeds=500,
    run_association=False,
):
    estimated_size_from_rates_and_weights = np.zeros((n_seeds, 2)) * float("NaN")
    estimated_size_from_weights = np.zeros((n_seeds, 2)) * float("NaN")

    for seed_id, seed in enumerate(range(n_seeds)):
        print(f"START WITH {seed}")
        save_file_name = "recurrent_inhibition_multicore_run"
        assembly_ids = [(0, -1)]
        if run_association:
            assembly_ids = [(0, 0)]
            save_file_name += "_association"
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

            estimated_size_from_rates_and_weights[seed_id, jj] = len(assembly_neuron_ids)
            estimated_size_from_weights[seed_id, jj] = assembly_sizes_by_weights[0]

        for jj in range(2):
            if np.isnan(estimated_size_from_rates_and_weights[seed_id, jj]):
                estimated_size_from_rates_and_weights[seed_id, (jj + 1) % 2] = float("nan")
                estimated_size_from_weights[seed_id, (jj + 1) % 2] = float("nan")

    return (estimated_size_from_rates_and_weights, estimated_size_from_weights)


def run_recall_for_multiple_instances(
    axes=None,
    change_firing_rate=False,
    only_load_results=False,
    show_results=True,
    run_association=False,
    show_plot=False,
    specific_seed=None,
    only_run_imprint=False,
    all_network_seeds=None,
):
    parameters_for_run = {
        "runtime_imprint": 30 * second,  # 40
        "runtime_baseline": 2 * second,
        "all_assembly_ids_for_areas": [[(0, 0, -1)]],
        "area_names": ["A"],
        "seed": 0,
        "all_context_ids_for_areas": [[(0, 0)]],
        "save_network_after_each_imprint": True,
    }

    if all_network_seeds is None:
        if not run_association:
            all_network_seeds = [452, 213]  # , 394, 839, 320, 100, 78, 912, 444, 102]
            # $$ already ran  : 452, 213
            # $$ running on server: , 394, 839, 320, 100, 78, 912, 444, 102
        else:
            all_network_seeds = [573, 812]  # , 552, 602, 5992, 103, 942, 111, 325, 832]
            # $$ already ran : 573, 812
            # $$ running on server: 552, 602, 5992, 103, 942, 111, 325, 832

    all_recall_inputs = [[(0, 0, -1)]]
    all_recall_contexts = [[(0, 0)], [(0, 1)]]
    all_recall_seeds = [0, 1]
    all_recall_sizes = [ii for ii in range(21)]

    runtime_recall = 2 * second

    if run_association:
        parameters_for_run["all_assembly_ids_for_areas"] = [[(0, 0, 0)]]
        all_recall_inputs = [[(0, 0, 0)]]

    net = NetworkRecall(
        parameter_file_name="parameters",
        parameters_for_run=parameters_for_run,
        save_file_name="recall_results_figure_4",
        parameter_dict=parameter_dict,
    )

    avg_firing_rates = np.zeros(
        (
            len(all_network_seeds),
            len(all_recall_inputs),
            len(all_recall_seeds) - 1 * int(10 in all_recall_seeds),
            len(all_recall_sizes),
            len(all_recall_contexts),
            2,
        )
    ) * float("nan")
    n_active_neurons = np.zeros_like(avg_firing_rates) * float("nan")

    avg_firing_rates_end_of_imprint = np.zeros((len(all_network_seeds), 2)) * float("nan")
    n_active_neurons_end_of_imprint = np.zeros_like(avg_firing_rates_end_of_imprint) * float("nan")

    for network_seed_id, network_seed in enumerate(all_network_seeds):
        if specific_seed is not None:
            if network_seed != specific_seed:
                continue

        parameters_for_run["seed"] = network_seed
        net.parameters_for_run["seed"] = network_seed

        net.run_imprint(report_style="text", report_period=1000 * second)

        network_filename = net.save_dict["filename_for_stored_network"].decode("utf-8") + "_0"
        net.network.restore(filename=net.get_path_to_stored_networks(file_name=network_filename))

        if only_run_imprint:
            return

        sorted_neuron_ids = []
        selected_ids = []

        bsl = net.parameters_for_run["runtime_baseline"] / msecond
        rtm = net.parameters_for_run["runtime_imprint"] / msecond
        rtm_recall = runtime_recall / msecond

        for area_id, area in enumerate(net.all_areas):
            a, _, b = net.sort_neurons_by_firing_rate(area=area)
            sorted_neuron_ids.append(a)
            selected_ids.append(b)

            active_threshold = 4
            (
                avg_fr_after_imprint,
                n_act_n_after_imprint,
                avg_firing_rate_not_in_assembly_after_imprint,
                n_active_neurons_not_in_assembly_after_imprint,
            ) = get_activity_metrics_from_assembly_neurons(
                active_threshold=active_threshold,
                net=net,
                area=area,
                sorted_neuron_ids=sorted_neuron_ids[area_id],
                selected_ids=selected_ids[area_id],
                start_time=bsl + rtm - rtm_recall,
                end_time=bsl + rtm,
            )

            avg_firing_rates_end_of_imprint[network_seed_id, 0] = avg_fr_after_imprint
            avg_firing_rates_end_of_imprint[
                network_seed_id, 1
            ] = avg_firing_rate_not_in_assembly_after_imprint
            n_active_neurons_end_of_imprint[network_seed_id, 0] = n_act_n_after_imprint
            n_active_neurons_end_of_imprint[
                network_seed_id, 1
            ] = n_active_neurons_not_in_assembly_after_imprint

            print(len(a), len(b))

        for seed_id, recall_seed in enumerate(all_recall_seeds):
            for input_id, recall_input in enumerate(all_recall_inputs):
                for context_id, recall_context in enumerate(all_recall_contexts):
                    x_values = []
                    for active_id, recall_size in enumerate(all_recall_sizes):
                        print("######:", seed_id, input_id, active_id)
                        run_recall_after_imprint = True

                        parameters_for_run.update(
                            {
                                "all_assembly_ids_for_areas_recall": [recall_input],
                                "all_context_ids_for_areas_recall": [recall_context],
                                "runtime_baseline_recall": 0.1 * second,
                                "runtime_recall": runtime_recall,
                                "run_recall_after_imprint": run_recall_after_imprint,
                                "recall_after_imprint_id": 0,
                                "assembly_neuron_selection_seed_recall": recall_seed,
                            }
                        )

                        if change_firing_rate:
                            parameters_for_run["assembly_firing_rate_recall"] = (
                                net.parameters["assembly_firing_rate"] * recall_size
                            ) / net.parameters["assembly_size"]
                        else:
                            parameters_for_run["assembly_size_recall"] = recall_size

                        if change_firing_rate:
                            x_values.append(parameters_for_run["assembly_firing_rate_recall"])
                        else:
                            x_values.append(parameters_for_run["assembly_size_recall"])
                            if run_association:
                                x_values[-1] *= 2

                        net.parameters_for_run.update(parameters_for_run)

                        # net.run_imprint(report_style="text")
                        print(net.save_dict)
                        net.run_recall(report_style="text")

                        # print(rmoe)
                        # if remove_n_neurons_from_first_assembly > 6:
                        #     net.show_spike_rasters(
                        #         show_plot=True, highlight_neuron_ids=[[0, neurons_to_silence]]
                        #     )

                        if not net.save_dict or not show_results:
                            continue

                        start_recall = bsl + (rtm + bsl) * (
                            1 + net.parameters_for_run["recall_after_imprint_id"]
                        )
                        end_recall = (
                            start_recall + net.parameters_for_run["runtime_recall"] / msecond
                        )

                        for area_id, area in enumerate(net.all_areas):
                            (
                                avg_fr,
                                n_act_n,
                                avg_firing_rate_not_in_assembly,
                                n_active_neurons_not_in_assembly,
                            ) = get_activity_metrics_from_assembly_neurons(
                                active_threshold=active_threshold,
                                net=net,
                                area=area,
                                sorted_neuron_ids=sorted_neuron_ids[area_id],
                                selected_ids=selected_ids[area_id],
                                start_time=start_recall,
                                end_time=end_recall,
                            )

                            if recall_seed == 10:
                                # I need to save this result differently
                                # its the result from recalling before training

                                continue

                            avg_firing_rates[
                                network_seed_id, input_id, seed_id, active_id, context_id, 0
                            ] = avg_fr
                            avg_firing_rates[
                                network_seed_id, input_id, seed_id, active_id, context_id, 1
                            ] = avg_firing_rate_not_in_assembly

                            n_active_neurons[
                                network_seed_id, input_id, seed_id, active_id, context_id, 0
                            ] = n_act_n
                            n_active_neurons[
                                network_seed_id, input_id, seed_id, active_id, context_id, 1
                            ] = n_active_neurons_not_in_assembly

    if change_firing_rate:
        xlabel = "starting neurons firing rate in Hz"
    else:
        xlabel = "n of starting neurons active"

    if show_results:
        # (network_seeds, inputs, seeds, activated,contexts,  in/out)

        in_out = 0

        normalize_results = True
        if normalize_results:
            a, b = avg_firing_rates_end_of_imprint.shape
            reshaped = avg_firing_rates_end_of_imprint.reshape(a, 1, 1, 1, 1, b)

            # Perform the division
            avg_firing_rates /= reshaped

            reshaped = n_active_neurons_end_of_imprint.reshape(a, 1, 1, 1, 1, b)

            # Perform the division
            n_active_neurons /= reshaped

        if axes is None:
            fig, (ax1, ax2) = plt.subplots(2)
        else:
            (ax1, ax2) = axes

        color_for_correct_context = "#ec7014"
        color_for_incorrect_context = "#6baed6"

        colors = [color_for_correct_context, color_for_incorrect_context]
        # for kk in range(2):

        data_fr = np.nanmean(np.nanmean(avg_firing_rates[:, :, :, :, :, in_out], 1), 1)
        data_na = np.nanmean(np.nanmean(n_active_neurons[:, :, :, :, :, in_out], 1), 1)
        for nn, network_seed in enumerate(all_network_seeds):
            for cc, contex_values in enumerate(data_fr[nn, :, :].T):
                ax1.plot(x_values, contex_values, color=colors[cc], alpha=0.3)
                if nn == 0:
                    ax1.plot(
                        x_values,
                        np.nanmean(data_fr, axis=0)[:, cc],
                        color=colors[cc],
                        label=f"context {cc}",
                    )

            for cc, contex_values in enumerate(data_na[nn, :, :].T):
                ax2.plot(x_values, contex_values, color=colors[cc], alpha=0.3)
                if nn == 0:
                    ax2.plot(
                        x_values,
                        np.nanmean(data_na, axis=0)[:, cc],
                        color=colors[cc],
                        label=f"context {cc}",
                    )

        title = "projection"
        if run_association:
            title = "association"
        ax1.set(
            xlabel=xlabel,
            ylabel=f"avg firing of assembly neurons",
        )
        ax2.set(
            xlabel=xlabel,
            ylabel=f"active assembly neurons",
        )
        ax1.legend()

        if show_plot:
            plt.show()


if __name__ == "__main__":
    paper_figure_3(only_load_results=True)
