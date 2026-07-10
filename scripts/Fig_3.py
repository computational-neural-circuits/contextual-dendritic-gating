from brian2.units import *
import numpy as np

from src.network_recall import (
    NetworkRecall,
)
from src.network_single_imprint import (
    NetworkSingleImprint,
)
from src.get_activity_metrics import get_activity_metrics_from_assembly_neurons
from src.utils import get_path_to_save_file_name

from Fig_S3 import load_recurrent_inhibition_comparison

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colorbar import ColorbarBase
from multiprocessing import Pool

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

FIGURE_NAME = "Fig_3"


def Fig_3(only_load_results=True, axes_for_currents_and_dendrites=None):
    (
        fig,
        axes_normalization_weights_over_time,
        ax_weights_norm_firing_rate,
        ax_weights_norm_soma_weights,
        ax_input_spikes_1,
        ax_recurrent_spikes,
        ax_weights_large_imprint,
        ax_for_theoretical_imprint_limit,
        ax_recall_example_firing_rate,
        ax_recall_example_n_active,
        ax_recall_example_colorbar,
        ax_recall_projection_fr,
        ax_recall_projection_na,
    ) = create_figure_layout_paper_fig_3()

    ########################
    # Single FF imprit (start)
    #######################

    normalization_clocks = None

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
            save_file_name="data_Fig_3_single_imprint",
            parameter_dict=parameter_dict,
            only_load_results=False,
            figure_name=FIGURE_NAME,
        )

        net.only_load_results = only_load_results
        net.run(report_style="text")
        networks[normalization] = net

    if axes_for_currents_and_dendrites is not None:
        get_dendritic_current_mean_and_std(
            networks["normalization"],
            use_non_gated=True,
            axes_for_currents_and_dendrites=axes_for_currents_and_dendrites,
        )

    normalization_clocks = [networks["normalization"].area.normalization.clock]
    # show spikes
    networks["normalization"].show_spike_rasters(
        show_plot=False,
        axes=np.array([ax_input_spikes_1, None, ax_recurrent_spikes]),
        show_range=show_range,
    )

    show_weight_matrix_sorted(
        networks["normalization"],
        ax=ax_weights_norm_firing_rate,
        sort_by_firing_rate_or_strongest_avg_weight=True,
    )
    ax_weights_norm_firing_rate.set_title(
        "Normalized: sorted by firing rate and weights"
    )

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

    if axes_for_currents_and_dendrites is not None:
        ## we want to quantify the currents in comparison of feedforward vs recurrent.
        ## therefore we need the neurons that become part of the assembly and track the
        ## currents in the gated dendrites. We want to compare before and after

        _, assembly_neuron_ids = networks["normalization"].sort_neurons_by_firing_rate(
            shuffle_rest=False
        )
        parameters_for_run["normalize"] = True
        parameters_for_run["monitor_currents"] = True
        print("Assembly ids: ", assembly_neuron_ids, type(assembly_neuron_ids))
        parameters_for_run["assembly_ids_for_monitoring_currents"] = [
            int(ii) for ii in assembly_neuron_ids
        ]
        parameters_for_run["context_id_for_monitoring_currents"] = 0
        net = NetworkSingleImprint(
            parameter_file_name="parameters",
            parameters_for_run=parameters_for_run,
            save_file_name="data_Fig_S2_single_imprint_with_currents_recorded",
            parameter_dict=parameter_dict,
            only_load_results=False,
        )
        net.run(report_style="text")

        analyze_dendritic_currents(
            net_original=networks["normalization"],
            net_with_currents=net,
            axes=axes_for_currents_and_dendrites,
        )

        return normalization_clocks
    ########################
    # larg imprint with recall (start)
    ######################

    all_seeds_for_large_imprint = run_large_imprint_with_recall_on_server(
        only_get_seeds=True
    )

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
        save_file_name="data_Fig_3_large_imprint",
        parameter_dict=parameter_dict,
        only_load_results=False,
        normalization_clocks=normalization_clocks,
        figure_name=FIGURE_NAME,
    )
    net.only_load_results = only_load_results

    # [ 24, 612]
    all_origial_assembly_sizes = []
    all_firing_rates_after_imprint = []

    recall_export_keys = ["bck", "same_ctxt", "diff_ctxt"]
    recall_export = [{}, {}]
    for ii in range(2):
        for key in recall_export_keys:
            recall_export[ii][key] = []

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

        all_origial_assembly_sizes += [
            len(xx) for xx in original_assembly_ids_for_all_imprints
        ]
        all_firing_rates_after_imprint += firing_rate_of_assemblies_after_imprint

        for pp, values in enumerate([recall_firing, recall_active]):
            for imprint_id, entries in enumerate(values):
                for key_id, key in enumerate(recall_export_keys):
                    recall_export[pp][key].append([seed, imprint_id, entries[key_id]])

        label = None
        if seed_id == 0:
            net.show_weight_matrix(
                show_plot=False,
                matlab_export_name=get_path_to_save_file_name("Fig_3", "weights_3C"),
                axes_for_weights=(None, ax_weights_large_imprint),
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

    for ii, save_name in enumerate(["avg_fr", "n_active"]):
        for key in recall_export_keys:
            values = np.array(recall_export[ii][key])
            path = get_path_to_save_file_name("Fig_3", "F_" + save_name + "_" + key)
            np.savetxt(path, values)

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

    all_network_seeds, all_network_seeds_association = (
        run_recall_for_multiple_instances_on_server(get_seeds=True)
    )

    run_recall_for_multiple_instances(
        axes=ax_recall_projection_fr,
        change_firing_rate=True,
        run_association=False,
        only_load_results=True,
        all_network_seeds=all_network_seeds,
        normalization_clocks=normalization_clocks,
    )
    run_recall_for_multiple_instances(
        axes=ax_recall_projection_na,
        change_firing_rate=False,
        run_association=False,
        only_load_results=True,
        all_network_seeds=all_network_seeds,
        normalization_clocks=normalization_clocks,
    )

    fig.savefig("../results/figures/Fig_3.pdf", dpi=800)


def analyze_dendritic_currents(net_original, net_with_currents, axes=None):

    for key in [
        "spikes_somas_t",
        "spikes_somas_i",
        "spikes_inputs_t_1",
        "spikes_inputs_i_1",
        "spikes_inputs_t_2",
        "spikes_inputs_i_2",
    ]:
        assert np.array_equal(
            net_original.save_dict[key], net_with_currents.save_dict[key]
        )

    print("Both runs are identical")

    recorded_time = net_original.save_dict["voltage_dends_t"]
    # in ms

    NMDA_inputs = [[], []]
    V = [[], []]
    AMPA_inputs = [[], []]
    for ii in range(2):
        for nn in range(3):

            key = f"iTotNMDA{nn+1}_{ii}"
            # need to change this later to f"iTotNMDA1_{ii}"
            NMDA_inputs[ii].append(net_with_currents.save_dict[key])
            key = f"iDendAMPA{nn+1}_{ii}"
            AMPA_inputs[ii].append(net_with_currents.save_dict[key])
        key = f"V_{ii}"  # need to change to f"V_{ii]}"
        V[ii].append(net_with_currents.save_dict[key])

    NMDA_inputs = np.array(NMDA_inputs)
    V = np.array(V)
    AMPA_inputs = np.array(AMPA_inputs)

    print(NMDA_inputs.shape)
    print(AMPA_inputs.shape)
    print(V.shape)
    print(recorded_time.shape)

    if axes is None:
        fig, axes = plt.subplots(2)
        ax1, ax2 = axes.flatten()
    else:
        _, _, ax1, ax2 = axes

    runtime_baseline = net_with_currents.parameters_for_run["runtime_baseline"]
    runtime_imprint = net_with_currents.parameters_for_run["runtime_imprint"]

    eval_time = 2 * second

    start_id_baseline = 0
    end_id_baseline = np.argmax(recorded_time > eval_time / ms)

    start_id_ff_before = np.argmax(recorded_time > runtime_baseline / ms)
    end_id_ff_before = np.argmax(recorded_time > (runtime_baseline + eval_time) / ms)

    start_id_ff_after = np.argmax(
        recorded_time > (runtime_baseline + runtime_imprint - eval_time) / ms
    )
    end_id_ff_after = len(recorded_time)

    print(start_id_baseline, end_id_baseline)
    print(start_id_ff_before, end_id_ff_before)
    print(start_id_ff_after, end_id_ff_after)

    np.random.seed(0)
    x_noise = 0.5 * (0.5 - np.random.rand(NMDA_inputs.shape[2]))

    colors = ["#377eb8", "#e41a1c"]
    labels = ["assembly neurons", "non-assembly neurons"]

    for ii, name in enumerate([["assembly neurons"], ["non-assembly neurons"]]):

        NMDA = NMDA_inputs[ii]
        AMPA = AMPA_inputs[ii]

        NMDA_recurrent = NMDA[0]
        NMDA_ff = NMDA[1] + NMDA[2]

        AMPA_recurrent = AMPA[0]
        AMPA_ff = AMPA[1] + AMPA[2]

        for counter, (start_id, end_id) in enumerate(
            zip(
                [start_id_baseline, start_id_ff_before, start_id_ff_after],
                [end_id_baseline, end_id_ff_before, end_id_ff_after],
            )
        ):

            for shift, values in enumerate(
                [NMDA_recurrent, NMDA_ff, AMPA_recurrent, AMPA_ff]
            ):
                y = np.mean(values[:, start_id:end_id], axis=1)
                x = counter * 10 + ii + shift * 2
                label = None
                if counter == 0 and shift == 0:
                    label = labels[ii]
                ax1.scatter(x + x_noise, y, color=colors[ii], alpha=0.5, label=label)
                ax1.scatter(x, np.mean(y), color=colors[ii], marker="P")

    ax1.set(
        xticks=[0.5, 2.5, 4.5, 6.5, 10.5, 12.5, 14.5, 16.5, 20.5, 22.5, 24.5, 26.5],
        xticklabels=["NMDA rec", "NMDA ff", "AMPA rec", "AMPA ff"] * 3,
        ylabel="Dendric input currents",
    )
    ax1.legend()

    # get all weights for the dendrites
    _, assembly_neuron_ids = net_original.sort_neurons_by_firing_rate(
        shuffle_rest=False
    )

    print(
        net_original.save_dict["weights"].shape,
        net_original.save_dict["weights_ff_1"].shape,
        net_original.save_dict["weights_ff_2"].shape,
    )

    weights_rec = net_original.save_dict["weights"]
    weights_ff = net_original.save_dict["weights_ff_1"]

    syn_rec = net_original.area.synapses_E
    synapses_from_assembly_neurons = []
    for neuron_id in assembly_neuron_ids:
        synapse_ids_pre = np.where((syn_rec.i)[:] == neuron_id)[0]

        synapses_from_assembly_neurons += list(synapse_ids_pre)

    np.random.seed(12)

    recurrent_within = []
    recurrent_from_outside = []
    ff_source = []
    ff_non_source = []

    for ii, neuron_id in enumerate(assembly_neuron_ids):

        dummy_ids = [kk for kk in assembly_neuron_ids if kk != neuron_id]

        label = None
        if ii == 0:
            label = "Recurrent weights (assembly neurons)"

        weights = weights_rec[dummy_ids, neuron_id * 6]
        recurrent_within += list(weights.flatten())

        ax2.scatter(
            [ii + (np.random.rand() - 0.5) * 0.5 for _ in dummy_ids],
            weights,
            color="r",
            alpha=0.5,
            label=label,
        )
        ax2.scatter(
            [len(assembly_neuron_ids) + (np.random.rand() - 0.5) * 0.5],
            np.mean(weights_rec[dummy_ids, neuron_id * 6]),
            color="r",
        )

        if ii == 0:
            label = f"Recurrent weights ({len(assembly_neuron_ids)} random non-assembly neurons)"

        non_assembly = ([mm for mm in range(400) if mm not in assembly_neuron_ids])[
            : len(assembly_neuron_ids)
        ]
        weights = weights_rec[non_assembly, neuron_id * 6]
        recurrent_from_outside += list(weights.flatten())

        ax2.scatter(
            [ii + (np.random.rand() - 0.5) * 0.5 for _ in non_assembly],
            weights,
            color="b",
            alpha=0.5,
            label=label,
        )
        ax2.scatter(
            [len(assembly_neuron_ids) + (np.random.rand() - 0.5) * 0.5],
            np.mean(weights_rec[non_assembly, neuron_id * 6]),
            color="b",
        )

        if ii == 0:
            label = "FF weight from active units"

        weights = weights_ff[[mm for mm in range(20)], neuron_id * 6]
        ff_source += list(weights.flatten())
        ax2.scatter(
            [ii + (np.random.rand() - 0.5) * 0.5 for _ in range(20)],
            weights,
            color="k",
            alpha=0.5,
            label=label,
        )
        ax2.scatter(
            [len(assembly_neuron_ids) + (np.random.rand() - 0.5) * 0.5],
            np.mean(weights_ff[[mm for mm in range(20)], neuron_id * 6]),
            color="k",
            alpha=0.5,
        )

        if ii == 0:
            label = "FF weight from 20 random non-active units"

        non_active = [mm for mm in range(20, 40)]
        weights = weights_ff[non_active, neuron_id * 6]
        ff_non_source += list(weights.flatten())
        ax2.scatter(
            [ii + (np.random.rand() - 0.5) * 0.5 for _ in range(20)],
            weights,
            color="g",
            alpha=0.5,
            label=label,
        )
        ax2.scatter(
            [len(assembly_neuron_ids) + (np.random.rand() - 0.5) * 0.5],
            np.mean(weights_ff[non_active, neuron_id * 6]),
            color="g",
            alpha=0.5,
        )

    ax2.legend()
    ax2.set(
        xlabel="Weights from assembly neurons or acrive pre-units to gated assembly dendrite\n(Last one is avg of weights for each dendrite)",
        ylabel="Weight",
    )


def run_large_imprint_with_recall(
    net,
    all_context_ids_for_areas_recall=[[(0, 0)], [(0, 1)]],
    recall_area_id=0,
    recall_after_imprint_id=None,
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
        if recall_after_imprint_id is not None:
            if ii != recall_after_imprint_id:
                original_assembly_ids_for_all_imprints.append([])
                firing_rate_of_assemblies_after_imprint.append([])
                continue

        network_filename = (
            net.save_dict["filename_for_stored_network"].decode("utf-8") + f"_{ii}"
        )
        filename = net.get_path_to_stored_networks(file_name=network_filename)
        net.network.restore(filename=filename)
        _, all_assembly_ids, _, all_rates_of_imprint = net.sort_neurons_by_firing_rate(
            area_name="A", return_rates_for_imprint=ii
        )
        # it actually uses rates and weights
        for jj in range(n_imprints):
            if jj > ii:
                break
            assembly_sizes_for_all_imprints[ii, jj] = len(all_assembly_ids[jj])

        original_assembly_ids_for_all_imprints.append(all_assembly_ids[ii])
        firing_rate_of_assemblies_after_imprint.append(
            np.mean(all_rates_of_imprint[all_assembly_ids[ii]])
        )

    rtm = net.parameters_for_run["runtime_imprint"] / msecond
    bsl = net.parameters_for_run["runtime_baseline"] / msecond
    runtime_recall = 2 * second
    rtm_rec = runtime_recall / msecond

    all_recall_sizes = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

    recall_firing = np.zeros(
        (n_imprints, 1 + len(all_context_ids_for_areas_recall), len(all_recall_sizes))
    ) * float("nan")
    recall_active = np.zeros_like(recall_firing) * float("nan")

    for context_id, context_ids_for_areas in enumerate(
        all_context_ids_for_areas_recall
    ):
        for imprint_id, assembly_ids_for_areas in enumerate(all_assembly_ids_for_areas):

            for size_id, assembly_size_recall in enumerate(all_recall_sizes):

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

                if recall_after_imprint_id is not None:
                    if imprint_id != recall_after_imprint_id or context_id == 1:
                        continue
                    net.parameters_for_run["recall_after_imprint_id"] = (
                        recall_after_imprint_id
                    )
                    net.parameters_for_run["assembly_size_recall"] = (
                        assembly_size_recall
                    )

                else:
                    if assembly_size_recall != 20:
                        continue

                if assembly_size_recall == 20:
                    if "assembly_size_recall" in net.parameters_for_run:
                        del net.parameters_for_run["assembly_size_recall"]

                net.run_recall(report_style="text")

                if not net.save_dict:
                    continue

                factor = len(all_assembly_ids_for_areas)
                if recall_after_imprint_id is not None:
                    factor = recall_after_imprint_id + 1

                start_time = bsl + factor * (bsl + rtm)
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

                recall_firing[imprint_id, 1 + context_id, size_id] = (
                    avg_firing_rate_recall
                )
                recall_active[imprint_id, 1 + context_id, size_id] = (
                    n_active_neurons_recall
                )

                if context_id == 0:
                    recall_firing[imprint_id, 0, size_id] = avg_firing_rate_recall_bck
                    recall_active[imprint_id, 0, size_id] = n_active_neurons_recall_bck

    if recall_after_imprint_id is None:

        return (
            recall_firing[:, :, -1],
            recall_active[:, :, -1],
            assembly_sizes_for_all_imprints,
            original_assembly_ids_for_all_imprints,
            firing_rate_of_assemblies_after_imprint,
        )
    else:

        return (
            recall_firing,
            recall_active,
            assembly_sizes_for_all_imprints,
            original_assembly_ids_for_all_imprints,
            firing_rate_of_assemblies_after_imprint,
        )


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
        weight_sums_input = np.sum(
            weights_loaded_recurrent[:, :: net.parameters["n_dend_each"]], 0
        )
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
        sorted_neuron_ids = sorted_neuron_ids  # we only look at one context

    sorted_dendrite_ids = []

    for nn in range(take_first_x_neurons):
        sorted_dendrite_ids += [
            ii + net.parameters["n_dend_each"] * sorted_neuron_ids[nn]
            for ii in range(net.parameters["n_dend_each"])
        ]

    weights = weights_loaded_recurrent[:, :]
    weights = weights[np.ix_(sorted_neuron_ids, sorted_dendrite_ids[:y_max])]
    ylabel = "postsynaptic dendrite\n(sorted)"

    if show_neuron_matrix:
        weights = weights_loaded_recurrent[:, :: net.parameters["n_dend_each"]]
        weights = weights[np.ix_(sorted_neuron_ids, sorted_neuron_ids)]
        y_max = len(sorted_neuron_ids)
        ylabel = "postsynaptic soma\n(sorted)"

    _ = ax.imshow(
        weights.T,
        cmap="Greys",
        origin="lower",
        extent=[-0.5, net.parameters["n_somas"] - 0.5, -0.5, y_max - 0.5],
    )
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
                xx += 0.4
            axes[1].scatter(xx, recall_active[jj, ii], color=cmap(norm(jj)), s=20)
    axes[1].set(ylabel="n active")

    _ = ColorbarBase(
        axes[2],
        cmap=cmap,
        norm=norm,
        spacing="proportional",
        ticks=bounds,
        boundaries=bounds,
        format="%1i",
    )
    axes[2].set_title("Position of learning")

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


def show_imprint_results_for_large_imprint(
    axes, imprint_firing_rates, imprint_sizes, seed=0
):
    np.random.seed(seed)
    for rate in imprint_firing_rates:
        xx = 0.6 + 0.2 * (
            0.5 - np.random.rand()
        )  # we shift to 0.6 because above we shifted to 1.4
        axes[0].scatter(xx, rate, color="k", s=20)
    axes[0].scatter(0.6, np.mean(imprint_firing_rates), color="r")

    for size in imprint_sizes:
        xx = 0.6 + 0.2 * (0.5 - np.random.rand())
        axes[1].scatter(xx, size, color="k", s=20)
    axes[1].scatter(0.6, np.mean(imprint_sizes), color="r")


def create_figure_layout_paper_fig_3():
    fig = plt.figure(figsize=(45, 20))
    gs = fig.add_gridspec(8, 25, hspace=1, wspace=1)

    ax_input_spikes_1 = fig.add_subplot(gs[:1, 5:11])
    ax_recurrent_spikes = fig.add_subplot(gs[1:2, 5:11])
    ax_weights_norm_firing_rate = fig.add_subplot(gs[:2, 12:16])
    ax_weights_norm_soma_weights = fig.add_subplot(gs[:2, 17:21])

    ax_norm_group_1 = fig.add_subplot(gs[:1, :2])
    ax_norm_group_2 = fig.add_subplot(gs[:1, 2:4])
    ax_norm_group_3 = fig.add_subplot(gs[1:2, :2])
    ax_norm_group_4 = fig.add_subplot(gs[1:2, 2:4])
    ax_weights_large_imprint = fig.add_subplot(gs[3:5, 5:9])

    ax_for_theoretical_imprint_limit = fig.add_subplot(gs[3:5, 10:14])
    ax_recall_example_firing_rate = fig.add_subplot(gs[3:5, 15:19])
    ax_recall_example_n_active = fig.add_subplot(gs[3:5, 20:24])
    ax_recall_example_colorbar = fig.add_subplot(gs[3:5, 24:25])

    ax_recall_projection_fr = [
        fig.add_subplot(gs[6:8, 17:21]),
        fig.add_subplot(gs[6:8, 12:16]),
    ]
    ax_recall_projection_na = [
        fig.add_subplot(gs[6:8, 6:10]),
        fig.add_subplot(gs[6:8, 1:5]),
    ]

    return (
        fig,
        (ax_norm_group_1, ax_norm_group_2, ax_norm_group_3, ax_norm_group_4),
        ax_weights_norm_firing_rate,
        ax_weights_norm_soma_weights,
        ax_input_spikes_1,
        ax_recurrent_spikes,
        ax_weights_large_imprint,
        ax_for_theoretical_imprint_limit,
        ax_recall_example_firing_rate,
        ax_recall_example_n_active,
        ax_recall_example_colorbar,
        ax_recall_projection_fr,
        ax_recall_projection_na,
    )


def theoretical_imprint_limit(ax=None, n_runs=1000, show_plot=False):
    if ax is None:
        fig, ax = plt.subplots()

    all_ids = [ii for ii in range(400)]

    all_numbers = np.zeros((n_runs, 20))

    estimated_size_from_rates_and_weights, _ = load_recurrent_inhibition_comparison(
        n_seeds=500,
    )

    all_empirical_assembly_sizes = estimated_size_from_rates_and_weights[:, 0].astype(
        int
    )  # results with inhibition 'on'

    for run in range(n_runs):
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
    ax.errorbar(
        x,
        np.mean(all_numbers, axis=0),
        yerr=np.std(all_numbers, axis=0),
        label="Theory",
    )

    ax.set(
        xlabel="# number of imprint",
        ylabel="# cumulative assembly sizes after all imprints",
        ylim=[0, 400],
    )

    if show_plot:
        plt.show()


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
    normalization_clocks=None,
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
        save_file_name="data_Fig_3_recall",
        parameter_dict=parameter_dict,
        normalization_clocks=normalization_clocks,
        figure_name=FIGURE_NAME,
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

    avg_firing_rates_end_of_imprint = np.zeros((len(all_network_seeds), 2)) * float(
        "nan"
    )
    n_active_neurons_end_of_imprint = np.zeros_like(
        avg_firing_rates_end_of_imprint
    ) * float("nan")

    for network_seed_id, network_seed in enumerate(all_network_seeds):
        if specific_seed is not None:
            if network_seed != specific_seed:
                continue

        parameters_for_run["seed"] = network_seed
        net.parameters_for_run["seed"] = network_seed

        net.run_imprint(report_style="text", report_period=1000 * second)

        network_filename = (
            net.save_dict["filename_for_stored_network"].decode("utf-8") + "_0"
        )
        net.network.restore(
            filename=net.get_path_to_stored_networks(file_name=network_filename)
        )

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
            avg_firing_rates_end_of_imprint[network_seed_id, 1] = (
                avg_firing_rate_not_in_assembly_after_imprint
            )
            n_active_neurons_end_of_imprint[network_seed_id, 0] = n_act_n_after_imprint
            n_active_neurons_end_of_imprint[network_seed_id, 1] = (
                n_active_neurons_not_in_assembly_after_imprint
            )

        for seed_id, recall_seed in enumerate(all_recall_seeds):
            for input_id, recall_input in enumerate(all_recall_inputs):
                for context_id, recall_context in enumerate(all_recall_contexts):
                    x_values = []
                    for active_id, recall_size in enumerate(all_recall_sizes):
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
                            x_values.append(
                                parameters_for_run["assembly_firing_rate_recall"]
                            )
                        else:
                            x_values.append(parameters_for_run["assembly_size_recall"])
                            if run_association:
                                x_values[-1] *= 2

                        net.parameters_for_run.update(parameters_for_run)

                        net.run_recall(report_style="text")

                        if not net.save_dict or not show_results:
                            continue

                        start_recall = bsl + (rtm + bsl) * (
                            1 + net.parameters_for_run["recall_after_imprint_id"]
                        )
                        end_recall = (
                            start_recall
                            + net.parameters_for_run["runtime_recall"] / msecond
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
                                network_seed_id,
                                input_id,
                                seed_id,
                                active_id,
                                context_id,
                                0,
                            ] = avg_fr
                            avg_firing_rates[
                                network_seed_id,
                                input_id,
                                seed_id,
                                active_id,
                                context_id,
                                1,
                            ] = avg_firing_rate_not_in_assembly

                            n_active_neurons[
                                network_seed_id,
                                input_id,
                                seed_id,
                                active_id,
                                context_id,
                                0,
                            ] = n_act_n
                            n_active_neurons[
                                network_seed_id,
                                input_id,
                                seed_id,
                                active_id,
                                context_id,
                                1,
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
            ax1, ax2 = axes

        color_for_correct_context = "#ec7014"
        color_for_incorrect_context = "#6baed6"

        colors = [color_for_correct_context, color_for_incorrect_context]

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


def get_dendritic_current_mean_and_std(
    net, use_non_gated=True, axes_for_currents_and_dendrites=None
):

    if axes_for_currents_and_dendrites is None:
        fig, (ax1, ax2) = plt.subplots(2, sharey=True)

    else:
        ax1, ax2, _, _ = axes_for_currents_and_dendrites

    colors = [
        "#f7fcf5",
        "#e5f5e0",
        "#c7e9c0",
        "#a1d99b",
        "#74c476",
        "#41ab5d",
        "#238b45",
        "#006d2c",
        "#00441b",
    ]
    name = "gated"
    for ii, values in enumerate(net.save_dict[f"voltage_dends_{name}"]):
        count = net.save_dict[f"counts_{ii}_{name}"]
        ax1.plot(
            net.save_dict["x_time"],
            values,
            label=f"{count} active inputs",
            color=colors[ii],
        )
    ax1.set_xlabel("Time in ms")
    ax1.set_ylabel("gated Dendritic voltage in mV")
    ax1.set_ylim([-80, 0])

    name = "non_gated"

    colors = [
        "#f7fbff",
        "#deebf7",
        "#c6dbef",
        "#9ecae1",
        "#6baed6",
        "#4292c6",
        "#2171b5",
        "#08519c",
        "#08306b",
    ]
    for ii, values in enumerate(net.save_dict[f"voltage_dends_{name}"]):
        count = net.save_dict[f"counts_{ii}_{name}"]
        ax2.plot(
            net.save_dict["x_time"],
            values,
            label=f"{count} active inputs",
            color=colors[ii],
        )
    ax2.set_xlabel("Time in ms")
    ax2.set_ylabel("non-gated Dendritic voltage in mV")
    ax2.set_ylim([-80, 0])

    ax1.legend()
    ax2.legend()

    name = "non_gated"

    mean_val_of_dends = np.mean(net.save_dict[f"voltage_dends_{name}"])
    mean_val_of_somas = np.mean(net.save_dict[f"voltage_soma_{name}"])
    gEachCouple = net.parameters["gEachCouple_pyr"]

    print(mean_val_of_dends * mV)
    print(mean_val_of_somas * mV)

    mean_current_flowing = (
        -5 * gEachCouple * (mean_val_of_dends * mV - mean_val_of_somas * mV)
    )

    print(mean_current_flowing)

    all_currents = (
        -5
        * gEachCouple
        * (
            net.save_dict[f"voltage_dends_{name}"] * mV
            - net.save_dict[f"voltage_soma_{name}"] * mV
        )
    )

    mean = np.mean(all_currents)
    std = np.std(all_currents)
    print(mean, std)

    # - gLSoma_pyr*(V-vRest_pyr) + gEachCouple*(V_pre-V_post)

    # - gLSoma_pyr * V - gEachCouple * V + gLSoma_pyr * vRest_pyr + gEachCouple * V_pre

    # - ( V * x)

    # x = (gLSoma_pyr + gEachCouple)

    # - (V * x) + x * (a/x)

    x = net.parameters["gLSoma_pyr"] + 5 * net.parameters["gEachCouple_pyr"]

    a = (
        net.parameters["gLSoma_pyr"] * net.parameters["vRest_pyr"]
        + 5 * net.parameters["gEachCouple_pyr"] * mean_val_of_dends * mV
    )

    new_vrest = a / x

    # gL = - x * (V - new_vrest)

    print(x)
    print(a)
    print(new_vrest)


if __name__ == "__main__":
    Fig_3(only_load_results=True)
