from brian2.units import *
import numpy as np

from src.network_recall import (
    NetworkRecall,
)
from src.get_activity_metrics import get_activity_metrics_from_assembly_neurons
from src.utils import get_path_to_save_file_name

from Fig_3 import Fig_3


import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colorbar import ColorbarBase
from multiprocessing import Pool
import time

plt.style.use("../plots_style.txt")

mean_noise = 0.11 * nA - 91.75836 * pA
parameter_dict = {
    "n_dend_each": 1,
    "n_contexts": 0,
    "gLSoma_pyr": 22.5 * nS,
    "vRest_pyr": -73.5 * mV,
}

parameters_for_run = {
    "runtime_imprint": 40 * second,  # 40
    "runtime_baseline": 2.5 * second,
    "normalize": True,
    "seed": 11,
    "run_association": False,
    "monitor_dt_weights": 100 * ms,
}


def Fig_S2(only_load_results=True, run_id=None):
    (
        fig,
        ax_for_large_weight_matrix,
        ax_recall_example_firing_rate,
        ax_recall_example_n_active,
        ax_recall_example_colorbar,
        ax_recall_projection_fr,
        ax_recall_projection_na,
        axes_for_currents_and_dendrites,
    ) = create_figure_layout_Fig_S2()

    ########################
    # larg imprint with recall (start)
    ######################

    normalization_clocks = Fig_3(
        only_load_results=True,
        axes_for_currents_and_dendrites=axes_for_currents_and_dendrites,
    )

    all_seeds_for_large_imprint = run_large_imprint_with_recall_on_server(
        only_get_seeds=True
    )
    n_large_imprint_seeds = len(all_seeds_for_large_imprint)

    if run_id is not None:
        if run_id < n_large_imprint_seeds:
            all_seeds_for_large_imprint = all_seeds_for_large_imprint[
                run_id : run_id + 1
            ]
        else:
            all_seeds_for_large_imprint = []

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
        save_file_name="data_Fig_S2_large_imprint_single_dendrite",
        parameter_dict=parameter_dict,
        only_load_results=False,
        normalization_clocks=normalization_clocks,
        figure_name="Fig_S2",
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

        if seed_id == 0:
            net.show_weight_matrix(
                show_plot=False,
                matlab_export_name=get_path_to_save_file_name("Fig_S2", "weights_S2D"),
                axes_for_weights=(None, ax_for_large_weight_matrix),
                n_dends=1,
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

    for ii, save_name in enumerate(["avg_fr", "n_active"]):
        for key in recall_export_keys:
            values = np.array(recall_export[ii][key])
            path = get_path_to_save_file_name(
                "Fig_S2", "F_" + save_name + "_" + key + "_single_dendrite"
            )
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

    ########################
    # larg imprint with recall (end)
    ######################

    ########################
    # Recall (start)
    ######################

    all_network_seeds, _ = run_recall_for_multiple_instances_on_server(get_seeds=True)

    if run_id is not None:
        if run_id < n_large_imprint_seeds:
            return
        new_run_id = run_id - n_large_imprint_seeds
        if new_run_id < len(all_network_seeds):
            all_network_seeds = all_network_seeds[new_run_id : new_run_id + 1]

    normalization_clocks = [net.all_areas[0].normalization.clock]
    run_recall_for_multiple_instances(
        axes=ax_recall_projection_fr,
        change_firing_rate=True,
        run_association=False,
        all_network_seeds=all_network_seeds,
        normalization_clocks=normalization_clocks,
        only_load_results=only_load_results,
    )
    run_recall_for_multiple_instances(
        axes=ax_recall_projection_na,
        change_firing_rate=False,
        run_association=False,
        all_network_seeds=all_network_seeds,
        normalization_clocks=normalization_clocks,
        only_load_results=only_load_results,
    )

    if run_id is None:
        fig.savefig("../results/figures/Fig_S2.pdf", dpi=800)

    # all_within_assembly_weights = [syn_id for syn_id in synapse_ids_post if syn_id in synapses_from_assembly_neurons]
    # recurrent_weights = syn_rec.w[all_within_assembly_weights]

    # ax2.scatter([ii for _ in all_within_assembly_weights], recurrent_weights)

    # weights =

    # recurrent_weights = syn_rec[].w
    # ff_weights =


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
        print(f"Trying to load imprint: {ii}")
        if recall_after_imprint_id is not None:
            if ii != recall_after_imprint_id:
                original_assembly_ids_for_all_imprints.append([])
                firing_rate_of_assemblies_after_imprint.append([])
                continue
        try:
            filename_for_stored_network = net.save_dict["filename_for_stored_network"]
            if not type(filename_for_stored_network) is str:
                filename_for_stored_network = filename_for_stored_network.decode(
                    "utf-8"
                )
            filename_for_stored_network += f"_{ii}"
        except KeyError:
            print("DID NOT FIND > ", ii)
            time.sleep(1)
            continue

        print(f"Filename for stored network: {filename_for_stored_network}")
        filename = net.get_path_to_stored_networks(
            file_name=filename_for_stored_network
        )
        net.restore_network(filename)
        print("Restored Network")
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

        print("$$ ------------------ $$")
        print("assembly ids: ", original_assembly_ids_for_all_imprints[-1])
        # net.show_spike_rasters(
        #     show_plot=True,
        #     highlight_neuron_ids=[[0, original_assembly_ids_for_all_imprints[0]]],
        #     show_vertical_lines_at=[1000, 31000],
        # )

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
                    print("#### RUNNING RECALL NOW WITH THE FOLLOWING PARAMETERS")
                    print(net.parameters_for_run)
                    print("--- ", net.only_load_results)

                else:
                    if assembly_size_recall != 20:
                        continue

                if assembly_size_recall == 20:
                    if "assembly_size_recall" in net.parameters_for_run:
                        del net.parameters_for_run["assembly_size_recall"]

                net.run_recall(report_style="text")

                if not net.save_dict:
                    continue

                print("Fetching recall results")

                factor = len(all_assembly_ids_for_areas)
                if recall_after_imprint_id is not None:
                    factor = recall_after_imprint_id + 1

                start_time = bsl + factor * (bsl + rtm)
                end_time = start_time + rtm_rec

                # net.show_spike_rasters(
                #         show_plot=True,
                #         highlight_neuron_ids=None,
                #         show_vertical_lines_at=[start_time, end_time],
                #     )
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
        485,
        932,
        3523,
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
    all_network_seeds = [177, 1858, 3052, 1290, 3070, 4874, 1127, 4642, 323, 4972]
    all_network_seeds_association = []

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


def create_figure_layout_Fig_S2():
    fig = plt.figure(figsize=(53, 48))
    gs = fig.add_gridspec(18, 17, hspace=1, wspace=1)

    ax_dendritic_currents = fig.add_subplot(gs[:2, 2:8])
    ax_dendritic_weights = fig.add_subplot(gs[:2, 9:17])

    ax_for_large_weight_matrix = fig.add_subplot(gs[3:5, :4])
    ax_recall_example_firing_rate = fig.add_subplot(gs[3:5, 12:14])
    ax_recall_example_n_active = fig.add_subplot(gs[3:5, 14:16])
    ax_recall_example_colorbar = fig.add_subplot(gs[3:5, 16:17])

    ax_recall_projection_fr = [
        fig.add_subplot(gs[6:8, 5:7]),
        fig.add_subplot(gs[6:8, 7:9]),
    ]
    ax_recall_projection_na = [
        fig.add_subplot(gs[6:8, 10:12]),
        fig.add_subplot(gs[6:8, 13:15]),
    ]

    ax_dendritic_currents_over_time_gated = fig.add_subplot(gs[9:11, 7:12])
    ax_dendritic_currents_over_time_non_gated = fig.add_subplot(gs[12:14, 7:12])

    return (
        fig,
        ax_for_large_weight_matrix,
        ax_recall_example_firing_rate,
        ax_recall_example_n_active,
        ax_recall_example_colorbar,
        ax_recall_projection_fr,
        ax_recall_projection_na,
        (
            ax_dendritic_currents_over_time_gated,
            ax_dendritic_currents_over_time_non_gated,
            ax_dendritic_currents,
            ax_dendritic_weights,
        ),
    )


def run_recall_for_multiple_instances(
    axes=None,
    change_firing_rate=False,
    show_results=True,
    run_association=False,
    show_plot=False,
    specific_seed=None,
    only_run_imprint=False,
    all_network_seeds=None,
    normalization_clocks=None,
    clear_stored_networks=True,
    only_load_results=False,
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

    print("##### ///// ##### All network seeds> ", all_network_seeds)
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
    all_recall_contexts = [[(0, 0)]]
    all_recall_seeds = [0, 1]
    all_recall_sizes = [ii for ii in range(21)]

    runtime_recall = 2 * second

    if run_association:
        parameters_for_run["all_assembly_ids_for_areas"] = [[(0, 0, 0)]]
        all_recall_inputs = [[(0, 0, 0)]]

    save_file_name = "data_Fig_S2_multiple_instances"
    net = NetworkRecall(
        parameter_file_name="parameters",
        parameters_for_run=parameters_for_run,
        save_file_name=save_file_name,
        parameter_dict=parameter_dict,
        normalization_clocks=normalization_clocks,
        figure_name="Fig_S2",
    )
    net.only_load_results = only_load_results

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

        print("Running imprint for network seed> ", network_seed_id, network_seed)
        parameters_for_run["seed"] = network_seed

        if not only_load_results:
            net = NetworkRecall(
                parameter_file_name="parameters",
                parameters_for_run=parameters_for_run,
                save_file_name=save_file_name,
                parameter_dict=parameter_dict,
                normalization_clocks=normalization_clocks,
                figure_name="Fig_S2",
            )
            net.only_load_results = only_load_results

        net.parameters_for_run["seed"] = network_seed

        net.run_imprint(report_style="text", report_period=1000 * second)

        filename_for_stored_network = net.save_dict["filename_for_stored_network"]
        if not type(filename_for_stored_network) is str:
            filename_for_stored_network = filename_for_stored_network.decode("utf-8")
        filename_for_stored_network += "_0"
        net.restore_network(
            net.get_path_to_stored_networks(file_name=filename_for_stored_network)
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

            fig_raster, raster_axes = plt.subplots(3, sharex=True, figsize=(10, 8))
            net.show_spike_rasters(
                axes=[tuple(raster_axes)],
                show_vertical_lines_at=[bsl, bsl + rtm, bsl + rtm + bsl],
            )
            pdf_path = get_path_to_save_file_name(
                "Fig_S2", f"imprint_raster_seed_{network_seed}_{area.name}.pdf"
            )
            fig_raster.savefig(pdf_path, dpi=300)
            plt.close(fig_raster)

        for seed_id, recall_seed in enumerate(all_recall_seeds):

            print("### Running recall with recall seed> ", seed_id, recall_seed)
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

                        # net.run_imprint(report_style="text")
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

        seed_color = "#6baed6"
        mean_color = "#ec7014"

        data_fr = np.nanmean(np.nanmean(avg_firing_rates[:, :, :, :, 0, in_out], 1), 1)
        data_na = np.nanmean(np.nanmean(n_active_neurons[:, :, :, :, 0, in_out], 1), 1)
        for nn, network_seed in enumerate(all_network_seeds):
            ax1.plot(x_values, data_fr[nn], color=seed_color, alpha=0.3)
            ax2.plot(x_values, data_na[nn], color=seed_color, alpha=0.3)

        ax1.plot(x_values, np.nanmean(data_fr, axis=0), color=mean_color, label="mean")
        ax2.plot(x_values, np.nanmean(data_na, axis=0), color=mean_color, label="mean")

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
    Fig_S2(only_load_results=True, run_id=None)
