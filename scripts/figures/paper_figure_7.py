from brian2.units import *
import brian2 as br2
import numpy as np

from src.network_recall import (
    NetworkRecall,
)
from src.get_activity_metrics import get_activity_metrics_from_assembly_neurons
from figure_4 import run_large_imprint_with_recall
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colorbar import ColorbarBase
from multiprocessing import Pool
from matplotlib_venn import venn3

# Former figure_5.py - now removed case=1 (inputs into Y and Z instead of only into Y)


def paper_figure_7(only_load_results=False):
    (
        fig,
        axes_recall_with_orders,
        axes_recall_with_orders_normed,
        ax_recall_orders_avg,
        ax_dendrite_inputs_Y,
        ax_dendrite_inputs_Z,
        ax_recall_orders_venn_area_Y,
        ax_recall_orders_venn_area_Z,
        ax_recall_orders_venn_area_Y_association,
        ax_recall_orders_venn_area_Z_association,
    ) = create_figure_layout_paper_fig_7()

    net = get_network_for_investigation(seed=0)
    result_dict = setup_result_dict(case_id=0)
    seed = 5  # 5
    how_does_association_change_the_recall(
        net=net,
        seed=seed,
        only_load_results=True,
        result_dict=result_dict,
    )

    show_single_results_for_association_changes_the_recall(
        seed=seed,
        result_dict=result_dict,
        axes=axes_recall_with_orders,
        normalize_results=False,
        show_plot=False,
    )
    show_single_results_for_association_changes_the_recall(
        seed=seed,
        result_dict=result_dict,
        axes=axes_recall_with_orders_normed,
        normalize_results=True,
        show_plot=False,
    )

    show_results_for_association_changes_the_recall(
        net=net,
        case_id=0,
        axes=[
            ax_recall_orders_avg,
            ax_dendrite_inputs_Y,
            ax_dendrite_inputs_Z,
            ax_recall_orders_venn_area_Y,
            ax_recall_orders_venn_area_Z,
            ax_recall_orders_venn_area_Y_association,
            ax_recall_orders_venn_area_Z_association,
        ],
    )

    ax_recall_orders_avg.set_title("inputs only into area Y (X and X`)")

    fig.savefig("../../results/figures/paper_fig_7.pdf", dpi=800)


def create_figure_layout_paper_fig_7():
    fig = plt.figure(figsize=(88, 64))
    gs = fig.add_gridspec(9, 16, hspace=1, wspace=1)

    axes_recall_with_orders = np.array(
        [
            [fig.add_subplot(gs[ii * 2 : (ii + 1) * 2, jj * 2 : (jj + 1) * 2]) for jj in range(4)]
            for ii in range(2)
        ]
    )

    axes_recall_with_orders_normed = np.array(
        [
            [
                fig.add_subplot(gs[ii * 2 : (ii + 1) * 2, 8 + jj * 2 : 8 + (jj + 1) * 2])
                for jj in range(4)
            ]
            for ii in range(2)
        ]
    )

    ax_recall_orders_venn_area_Y = fig.add_subplot(gs[5:7, 2:4])
    ax_recall_orders_venn_area_Z = fig.add_subplot(gs[7:9, 2:4])

    ax_recall_orders_venn_area_Y_association = fig.add_subplot(gs[5:7, 4:6])
    ax_recall_orders_venn_area_Z_association = fig.add_subplot(gs[7:9, 4:6])

    ax_recall_orders_avg = fig.add_subplot(gs[5:7, 6:8])
    ax_dendrite_inputs_Y = fig.add_subplot(gs[5:7, 8:10])
    ax_dendrite_inputs_Z = fig.add_subplot(gs[7:9, 8:10])

    return (
        fig,
        axes_recall_with_orders,
        axes_recall_with_orders_normed,
        ax_recall_orders_avg,
        ax_dendrite_inputs_Y,
        ax_dendrite_inputs_Z,
        ax_recall_orders_venn_area_Y,
        ax_recall_orders_venn_area_Z,
        ax_recall_orders_venn_area_Y_association,
        ax_recall_orders_venn_area_Z_association,
    )


def get_network_for_investigation(seed=5):
    parameter_dict = {}
    parameters_for_run = {
        "runtime_imprint": 50 * second,  # 50
        "runtime_baseline": 1 * second,  # 5
        "all_assembly_ids_for_areas": None,
        "area_names": ["A", "B"],
        "all_context_ids_for_areas": [[]],  # this sets all contexts to 0
        "seed": seed,
    }

    net = NetworkRecall(
        parameter_file_name="parameters",
        parameters_for_run=parameters_for_run,
        save_file_name="association_changes",
        parameter_dict=parameter_dict,
        only_load_results=False,
        # new_file_to_save_to="association_changes",
    )

    return net


def get_simulated_network(net, filename_for_stored_network=None, all_assembly_ids_for_areas=None):
    net.parameters_for_run["all_assembly_ids_for_areas"] = all_assembly_ids_for_areas
    if "restore_from_save_name" in net.parameters_for_run:
        del net.parameters_for_run["restore_from_save_name"]
    if filename_for_stored_network is not None:
        net.parameters_for_run["restore_from_save_name"] = filename_for_stored_network
    save_dict = net.run_imprint(report_style="text", report_period=900 * second)
    print("#####", save_dict)
    if not save_dict:
        net.show_all_saved_dictionaries()
        return None
    filename_for_stored_network = save_dict["filename_for_stored_network"]
    if not type(filename_for_stored_network) is str:
        filename_for_stored_network = filename_for_stored_network.decode("utf-8")

    filename_for_stored_network += "_0"

    return filename_for_stored_network


def get_assembly_ids_and_distributions(
    net,
    order_id,
    imprint_id,
    result_dict,
):
    if net.save_dict:
        t_0 = (
            net.parameters_for_run["runtime_imprint"] / msecond
            + net.parameters_for_run["runtime_baseline"] / msecond
        ) * imprint_id

        current_vals_a = np.copy(net.parameters_for_run["all_assembly_ids_for_areas"]).tolist()
        current_vals_b = np.copy(net.parameters_for_run["all_context_ids_for_areas"]).tolist()

        net.parameters_for_run["all_assembly_ids_for_areas"] = [[] for _ in range(imprint_id + 1)]
        net.parameters_for_run["all_context_ids_for_areas"] = [
            [(0, 0)] for _ in range(imprint_id + 1)
        ]

        _, _, selected_ids_A = net.sort_neurons_by_firing_rate(
            area_name="A",
        )
        _, _, selected_ids_B = net.sort_neurons_by_firing_rate(
            area_name="B",
        )

        net.parameters_for_run["all_assembly_ids_for_areas"] = current_vals_a
        net.parameters_for_run["all_context_ids_for_areas"] = current_vals_b

        selected_ids = [selected_ids_A, selected_ids_B]

        dists = look_at_dendrite_statistics_within_assembly(
            net=net,
            selected_ids_A=selected_ids_A,
            selected_ids_B=selected_ids_B,
            order_id=order_id,
        )

        for area_id in range(2):
            params = {
                "seed": net.parameters_for_run["seed"],
                "area_id": area_id,
                "order_id": order_id,
                "imprint_id": imprint_id,
            }
            for input_id in range(2):
                key = get_key_for_result_dictionary(
                    **params,
                    input_id=input_id,
                    result_type="dendrite_distributions",
                )
                result_dict[key] = dists[area_id][input_id]
            key = get_key_for_result_dictionary(
                **params,
                result_type="selected_assembly_ids",
            )
            result_dict[key] = selected_ids[area_id]

        return selected_ids


def run_recall_for_loaded_net(
    net,
    selected_ids,
    assembly_ids_for_areas,
    change_firing_rate=True,
    runtime_recall=2 * second,
    n_of_imprints=1,
    run_recall_after_imprint=True,
    result_dict=None,
):
    all_recall_sizes = result_dict["all_recall_sizes"]

    # (firing_rate/n_active, area_id, recall_sizes)
    all_recall_responses = np.zeros((2, 2, len(all_recall_sizes))) * float("nan")
    x_values = []

    bsl = net.parameters_for_run["runtime_baseline"] / msecond
    rtm = net.parameters_for_run["runtime_imprint"] / msecond
    rtm_recall = runtime_recall / msecond

    start_time_after_imprint = n_of_imprints * (2 * bsl + rtm) - rtm_recall - bsl
    end_time_after_imprint = start_time_after_imprint + rtm_recall

    for active_id, recall_size in enumerate(all_recall_sizes):
        net.parameters_for_run.update(
            {
                "all_assembly_ids_for_areas_recall": assembly_ids_for_areas,
                "all_context_ids_for_areas_recall": [[(0, 0)]],  # sets all contexts to 0
                "runtime_baseline_recall": 0.1 * second,
                "runtime_recall": runtime_recall,
                "run_recall_after_imprint": run_recall_after_imprint,
                "recall_after_imprint_id": 0,
            }
        )
        if change_firing_rate:
            net.parameters_for_run["assembly_firing_rate_recall"] = (
                net.parameters["assembly_firing_rate"] * recall_size
            ) / net.parameters["assembly_size"]
            x_values.append(net.parameters_for_run["assembly_firing_rate_recall"])
        else:
            net.parameters_for_run["assembly_size_recall"] = recall_size
            x_values.append(net.parameters_for_run["assembly_size_recall"])

        # net.show_all_saved_dictionaries()
        net.run_recall(report_style="text")

        if not net.save_dict:
            continue

        start_time = end_time_after_imprint + bsl
        end_time = start_time + rtm_recall

        # if recall_size == 20:
        #     net.show_spike_rasters(
        #         show_plot=True,
        #         highlight_neuron_ids=None,
        #         sort_for_specific_imprint=None,
        #         show_vertical_lines_at=[start_time, end_time],
        #     )

        for area_id, area in enumerate(net.all_areas):
            active_threshold = result_dict["active_threshold"]
            (
                avg_firing_rate,
                n_active_neurons,
                _,
                _,
            ) = get_activity_metrics_from_assembly_neurons(
                active_threshold=active_threshold,
                net=net,
                area=area,
                selected_ids=selected_ids[area_id],
                start_time=start_time,
                end_time=end_time,
            )

            all_recall_responses[0, area_id, active_id] = avg_firing_rate
            all_recall_responses[1, area_id, active_id] = n_active_neurons

    if change_firing_rate:
        result_dict["x_values_firing_rate"] = x_values
    else:
        result_dict["x_values_n_active"] = x_values

    return all_recall_responses


def process_case(
    net,
    change_firing_rate,
    imprint_inputs,
    recall_inputs,
    order_id,
    seed,
    result_dict,
    only_run_imprint=False,
):
    filename_for_stored_network = None
    for imprint_id, im_in in enumerate(imprint_inputs):
        filename_for_stored_network = get_simulated_network(
            net=net,
            filename_for_stored_network=filename_for_stored_network,
            all_assembly_ids_for_areas=[im_in],
        )
        net.network.restore(
            filename=net.get_path_to_stored_networks(file_name=filename_for_stored_network)
        )

        selected_ids = get_assembly_ids_and_distributions(
            net=net, order_id=order_id, imprint_id=imprint_id, result_dict=result_dict
        )

    if only_run_imprint or not net.save_dict:
        return

    runtime_recall = 2 * second
    bsl = net.parameters_for_run["runtime_baseline"] / msecond
    rtm = net.parameters_for_run["runtime_imprint"] / msecond
    rtm_recall = runtime_recall / msecond

    start_time_after_imprint = (
        bsl + rtm - rtm_recall + (2 * bsl + rtm) * imprint_id
    )  # imprint_id * (2 * bsl + rtm) - rtm_recall - bsl
    end_time_after_imprint = start_time_after_imprint + rtm_recall

    # net.show_spike_rasters(
    #     show_plot=True,
    #     highlight_neuron_ids=None,
    #     sort_for_specific_imprint=None,
    #     show_vertical_lines_at=[start_time_after_imprint, end_time_after_imprint],
    # )

    for area_id, area in enumerate(net.all_areas):
        (
            avg_fr_after_imprint,
            n_act_n_after_imprint,
            _,
            _,
        ) = get_activity_metrics_from_assembly_neurons(
            active_threshold=result_dict["active_threshold"],
            net=net,
            area=area,
            selected_ids=selected_ids[area_id],
            start_time=start_time_after_imprint,
            end_time=end_time_after_imprint,
        )
        for fr_n_active_id in range(2):
            key = get_key_for_result_dictionary(
                seed=net.parameters_for_run["seed"],
                area_id=area_id,
                firing_rate_or_n_active_neurons=fr_n_active_id,
                order_id=order_id,
                result_type="imprint",
            )
            res = avg_fr_after_imprint
            if fr_n_active_id == 1:
                res = n_act_n_after_imprint

            result_dict[key] = res

    for stim_id, inputs in enumerate(recall_inputs):
        for run_recall_after_imprint in [True, False]:
            all_recall_responses = run_recall_for_loaded_net(
                net=net,
                change_firing_rate=change_firing_rate,
                assembly_ids_for_areas=[inputs],
                runtime_recall=runtime_recall,
                n_of_imprints=len(imprint_inputs),
                run_recall_after_imprint=run_recall_after_imprint,
                result_dict=result_dict,
                selected_ids=selected_ids,
            )

            for area_id in range(2):
                for fr_n_active_id in range(2):
                    key = get_key_for_result_dictionary(
                        seed=net.parameters_for_run["seed"],
                        stimulus_id_recall=stim_id,
                        run_recall_after_imprint=run_recall_after_imprint,
                        area_id=area_id,
                        firing_rate_or_n_active_neurons=fr_n_active_id,
                        order_id=order_id,
                        result_type="recall",
                    )

                    result_dict[key] = all_recall_responses[fr_n_active_id, area_id]


def get_key_for_result_dictionary(
    seed=None,
    input_id=None,
    area_id=None,
    order_id=None,
    imprint_id=None,
    result_type=None,
    stimulus_id_recall=None,
    run_recall_after_imprint=None,
    firing_rate_or_n_active_neurons=None,
):
    # all result types:
    # 'recall', 'imprint', 'dendrite_distributions', 'selected_assembly_ids'

    if seed is None:
        raise ValueError("You need to provide the seed for the network")

    if result_type == "recall":
        all_relevant_parameters = [
            seed,
            order_id,
            area_id,
            run_recall_after_imprint,
            stimulus_id_recall,
            firing_rate_or_n_active_neurons,
        ]

    if result_type == "imprint":
        all_relevant_parameters = [seed, order_id, area_id, firing_rate_or_n_active_neurons]

    if result_type == "dendrite_distributions":
        all_relevant_parameters = [seed, input_id, area_id, order_id, imprint_id]

    if result_type == "selected_assembly_ids":
        all_relevant_parameters = [seed, area_id, order_id, imprint_id]

    # first we check if all relevant parameters are provided
    key = result_type
    for par in all_relevant_parameters:
        if par is None:
            raise ValueError("You should provide information for all relevant paramters")

        key += f"{par}"

    return key


def show_single_results_for_association_changes_the_recall(
    seed,
    result_dict,
    axes=None,
    normalize_results=False,
    show_plot=False,
    change_firing_rate=True,
):
    case_id = result_dict["case_id"]
    # title = f"CASE {case_id + 1} "

    prime = "X`"
    if case_id == 1:
        prime = "Y`"

    title = f"X & {prime}"

    if axes is None:
        fig, axes = plt.subplots(2, 4)
    for firing_rate_n_active in [0, 1]:
        ylabel = "avg firing rate"
        if firing_rate_n_active:
            ylabel = "n active"

        order_colors = ["#377eb8", "#4daf4a", "#984ea3"]
        for order_id in range(3):
            for area_id in range(2):
                for stim_id_recall in range(2):
                    for run_recall_after_imprint in [True, False]:
                        ls = "-"
                        extra_label = ""
                        if not run_recall_after_imprint:
                            ls = "--"
                            extra_label = " before imprint"

                        if area_id == 0 and stim_id_recall == 0:
                            if order_id == 0:
                                label = f"First X then {prime}" + extra_label
                            if order_id == 1:
                                label = f"First {prime} then X" + extra_label
                            if order_id == 2:
                                label = "simultaneous" + extra_label
                        else:
                            label = None

                        if change_firing_rate:
                            x_values = result_dict["x_values_firing_rate"]
                        else:
                            x_values = result_dict["x_values_n_active"]
                        res = []
                        for recall_or_imprint in ["recall", "imprint"]:
                            key = get_key_for_result_dictionary(
                                seed=seed,
                                stimulus_id_recall=stim_id_recall,
                                run_recall_after_imprint=run_recall_after_imprint,
                                area_id=area_id,
                                firing_rate_or_n_active_neurons=firing_rate_n_active,
                                order_id=order_id,
                                result_type=recall_or_imprint,
                            )

                            try:
                                res.append(result_dict[key])
                                if order_id == 2:
                                    print("##$", result_dict[key])
                            except KeyError:
                                print("keyError", order_id, stim_id_recall, run_recall_after_imprint)
                                res.append([float("nan") for _ in x_values])

                        y_values = res[0]
                        if normalize_results:
                            y_values = np.array(y_values) / res[1]

                        ax = axes[stim_id_recall, area_id + 2 * firing_rate_n_active]
                        ax.plot(
                            x_values,
                            y_values,
                            color=order_colors[order_id],
                            label=label,
                            ls=ls,
                        )

                        if stim_id_recall == 0:
                            new_title = title + " >> Recall in X"
                        if stim_id_recall == 1:
                            new_title = title + f" >> Recall in {prime}"

                        if area_id == 0:
                            new_title += " [in Area Y]"
                        if area_id == 1:
                            new_title += " [in Area Z]"

                        if normalize_results:
                            new_title += "\n(normalized with respect to end of imprint)"
                        ax.set(
                            title=new_title,
                            xlabel="Firing rate",
                            ylabel=ylabel,
                        )

                        if not normalize_results:
                            ax.axhline(
                                y=res[1],
                                ls="--",
                                lw=0.5,
                                color=order_colors[order_id],
                            )
                        else:
                            ax.set(ylim=[-0.05, 1.05])

        for ax in axes.flatten():
            ax.legend()

    if show_plot:
        plt.show()


def how_does_association_change_the_recall(
    net=None,
    seed=5,
    change_firing_rate=True,
    only_load_results=False,
    only_run_imprint=False,
    result_dict=None,
):
    print(f"CREATE NETWORK {seed, change_firing_rate, only_run_imprint}")

    if net is None:
        net = get_network_for_investigation(seed=seed)
    else:
        net.parameters_for_run["seed"] = seed
    net.only_load_results = only_load_results

    all_case_recall_inputs = result_dict["all_case_recall_inputs"]
    all_case_imprint_inputs = result_dict["all_case_imprint_inputs"]

    for order_id in range(3):
        process_case(
            net,
            change_firing_rate=change_firing_rate,
            imprint_inputs=all_case_imprint_inputs[order_id],
            recall_inputs=all_case_recall_inputs,
            order_id=order_id,
            seed=seed,
            result_dict=result_dict,
            only_run_imprint=only_run_imprint,
        )

    return


def look_at_dendrite_statistics_within_assembly(net, selected_ids_A, selected_ids_B, order_id):
    # we now want to look at the statistics of how many highly active
    # arrived at the dendrites

    # Area A - 1

    subset_ids_for_a1 = []
    subset_ids_for_a2 = []
    subset_ids_for_b2 = []

    for assembly_ids in net.parameters_for_run["all_assembly_ids_for_areas"][0]:
        print(assembly_ids)
        if assembly_ids[0] == 0 and assembly_ids[1] >= 0:
            subset_ids_for_a1 = [ii + (assembly_ids[1] * 20) for ii in range(20)]
        if assembly_ids[0] == 0 and assembly_ids[2] >= 0:
            subset_ids_for_a2 = [ii + (assembly_ids[2] * 20) for ii in range(20)]
        if assembly_ids[0] == 1 and assembly_ids[2] >= 0:
            subset_ids_for_b2 = [ii + (assembly_ids[2] * 20) for ii in range(20)]

    input_subsets = [[subset_ids_for_a1, subset_ids_for_a2], [selected_ids_A, subset_ids_for_b2]]

    n_of_inputs_that_target_selected_ids = [[[], []], [[], []]]
    targeted_neurons = [selected_ids_A, selected_ids_B]

    print(subset_ids_for_a1)
    print(subset_ids_for_a2)
    print(subset_ids_for_b2)

    for area_id in range(2):
        area = net.all_areas[area_id]
        context_id = 0
        for target_id in targeted_neurons[area_id]:
            for jj in range(area.n_dends_each):
                dend_id = target_id * area.n_dends_each + jj
                if dend_id in area.dends_of_ctxt[context_id]:
                    for input_id in range(2):
                        syn_ids_that_target_dendrite = np.where(
                            area.input_synapses[input_id].j[:] == dend_id
                        )[0]
                        input_neurons_that_target_dendrite = area.input_synapses[input_id].i[
                            syn_ids_that_target_dendrite
                        ]

                        n_of_inputs_that_target_selected_ids[area_id][input_id].append(0)
                        for n_id in input_neurons_that_target_dendrite:
                            if n_id in input_subsets[area_id][input_id]:
                                n_of_inputs_that_target_selected_ids[area_id][input_id][-1] += 1

    return n_of_inputs_that_target_selected_ids


def run_how_does_association_change_the_recall_on_server(max_cores=2, get_seeds=False, case_id=0):
    result_dict = setup_result_dict(case_id=case_id)
    result_dict["all_recall_sizes"] = [20]
    all_seeds = [
        6427,
        5,
        723,
        495,
        852,
        138,
        593,
        952,
        953,
        82,
        981,
        623,
        7433,
        849,
        942,
        748,
        4738,
        543,
        7822,
        843,
    ]
    if get_seeds:
        return all_seeds

    params = []

    # iterate over seed
    change_firing_rate = True
    only_load_results = False
    # iterate over case_id
    only_run_imprint = True

    for seed in all_seeds:
        params.append(
            (
                seed,
                change_firing_rate,
                only_load_results,
                only_run_imprint,
                result_dict,
            )
        )

    n_cores = np.min([len(params), max_cores])
    with Pool(n_cores) as pool:
        _ = pool.starmap(
            how_does_association_change_the_recall,
            params,
        )

    only_run_imprint = False
    for change_firing_rate in [True, False]:
        for seed in all_seeds:
            params.append(
                (
                    seed,
                    change_firing_rate,
                    only_load_results,
                    case_id,
                    only_run_imprint,
                    result_dict,
                )
            )

    print(len(params))
    n_cores = np.min([len(params), max_cores])
    with Pool(n_cores) as pool:
        _ = pool.starmap(
            how_does_association_change_the_recall,
            params,
        )


def setup_result_dict(case_id):
    result_dict = {"case_id": case_id}

    all_possible_inputs = [[(0, 0, -1)], [(0, -1, 0)], [(1, -1, 0)], [(0, 0, 0)]]
    all_recall_sizes = [ii for ii in range(21) if ii % 2 == 0]

    result_dict["all_possible_inputs"] = all_possible_inputs
    result_dict["all_recall_sizes"] = all_recall_sizes
    result_dict["active_threshold"] = 4

    if case_id == 0:
        all_case_recall_inputs = [
            all_possible_inputs[0],
            all_possible_inputs[1],
            all_possible_inputs[0] + all_possible_inputs[1],
        ]
        all_case_imprint_inputs = [
            [all_possible_inputs[0], all_possible_inputs[0] + all_possible_inputs[1]],
            [all_possible_inputs[1], all_possible_inputs[1] + all_possible_inputs[0]],
            [all_possible_inputs[3]],
        ]
    elif case_id == 1:
        all_case_recall_inputs = [
            all_possible_inputs[0],
            all_possible_inputs[2],
            all_possible_inputs[0] + all_possible_inputs[2],
        ]
        all_case_imprint_inputs = [
            [all_possible_inputs[0], all_possible_inputs[0] + all_possible_inputs[2]],
            [all_possible_inputs[2], all_possible_inputs[2] + all_possible_inputs[0]],
            [all_possible_inputs[0] + all_possible_inputs[2]],
        ]

    result_dict["all_case_recall_inputs"] = all_case_recall_inputs
    result_dict["all_case_imprint_inputs"] = all_case_imprint_inputs

    return result_dict


def show_results_for_association_changes_the_recall(net=None, case_id=0, axes=None, show_plot=False):
    result_dict = setup_result_dict(case_id=case_id)
    all_seeds = run_how_does_association_change_the_recall_on_server(get_seeds=True)

    prime = "X`"
    if case_id == 1:
        prime = "Y`"

    # all_seeds = [5]
    # all_seeds = all_seeds
    if net is None:
        net = get_network_for_investigation(seed=0)

    for seed_id, seed in enumerate(all_seeds):
        how_does_association_change_the_recall(
            net=net,
            seed=seed,
            only_load_results=True,
            result_dict=result_dict,
        )

    if axes is None:
        fig, axes = plt.subplots(2, 4)
        (ax1, ax2, ax3, _, ax4, ax5, ax6, ax7) = axes.flatten()
    else:
        ax1, ax2, ax3, ax4, ax5, ax6, ax7 = axes

    xticklabels = []
    xticks = []
    y_values = []
    y_values_bars = []

    def get_recall_res_for_seeds(
        order_id,
        run_recall_after_imprint,
        stim_id_recall,
        firing_rate_or_n_active_neurons,
        area_id,
        result_type="recall",
        normalize_results=False,
    ):
        all_res = []
        for seed in all_seeds:
            key = get_key_for_result_dictionary(
                seed=seed,
                order_id=order_id,
                stimulus_id_recall=stim_id_recall,
                run_recall_after_imprint=run_recall_after_imprint,
                area_id=area_id,
                firing_rate_or_n_active_neurons=firing_rate_or_n_active_neurons,
                result_type=result_type,
            )
            res = result_dict[key]

            if normalize_results:
                key_to_normalize = get_key_for_result_dictionary(
                    seed=seed,
                    area_id=area_id,
                    firing_rate_or_n_active_neurons=firing_rate_or_n_active_neurons,
                    order_id=order_id,
                    result_type="imprint",
                )
                res_to_normalize = result_dict[key_to_normalize]
                res /= res_to_normalize

            all_res.append(res)

        return np.vstack(all_res)

    for area_id in range(2):
        recall_results = [[], [], []]

        for ii, recall_ids in enumerate(
            [
                # [order_id, stim_recall_id]
                [[0, 0], [1, 1]],  # first (same order as stimulus)
                [[0, 1], [1, 0]],  # last (different order than stimulus)
                [[2, 0], [2, 1]],  # same
            ]
        ):
            for o_id, s_id in recall_ids:
                x = get_recall_res_for_seeds(
                    order_id=o_id,
                    stim_id_recall=s_id,
                    firing_rate_or_n_active_neurons=0,
                    area_id=area_id,
                    run_recall_after_imprint=True,
                    normalize_results=True,
                )

                recall_results[ii].append(x[:, -1:])
                print("_______________")
                print(x)
                print(recall_results)

            recall_results[ii] = np.vstack(recall_results[ii])

        area_label = "Y"
        if area_id == 1:
            area_label = "Z"

        new_xtick_labels = [
            f"recall (first)\narea {area_label} /{result_dict['x_values_firing_rate'][-1]}",
            f"recall (last)\narea {area_label} /{result_dict['x_values_firing_rate'][-1]}",
            f"recall (same)\narea {area_label} /{result_dict['x_values_firing_rate'][-1]}",
        ]
        xticklabels += new_xtick_labels
        xticks += [ii + area_id * (len(new_xtick_labels) + 1) for ii in range(len(new_xtick_labels))]
        y_values_bars += [np.nanmean(x) for x in recall_results]
        y_values += [x.flatten() for x in recall_results]

    ax1.bar(xticks, y_values_bars)
    for x, y in zip(xticks, y_values):
        ax1.scatter([x for _ in y], y, color="k")
    ax1.set(
        xticks=xticks, xticklabels=xticklabels, ylabel="avg firing rate, normalized for last imprint"
    )

    def get_dend_dist_res_for_seeds(
        order_id,
        input_id,
        area_id,
        imprint_id,
        result_type="dendrite_distributions",
    ):
        if result_dict["case_id"] == 0:
            if area_id == 1 and input_id == 1:
                return []
        all_res = []
        for seed in all_seeds:
            key = get_key_for_result_dictionary(
                seed=seed,
                order_id=order_id,
                input_id=input_id,
                area_id=area_id,
                imprint_id=imprint_id,
                result_type=result_type,
            )
            x = result_dict[key]
            all_res.append(x)
            print("___oo___")
            print(x)

        return np.hstack(all_res).tolist()

    for area_id, ax in enumerate([ax2, ax3]):
        all_dend_results = [[], [], [], [], []]
        all_labels = [
            "first trained inputs",
            "last trained inputs",
            "inputs trained at the same time (X)",
            "inputs trained at the same time (X`)",
            "trained alone",
        ]
        all_colors = ["#fe9929", "#0570b0", "#6a51a3", "#238443", "#e5446d"]
        for ii, dend_res_ids in enumerate(
            [
                # [order_id, input_id, imprint_id]
                [[0, 0, 1], [1, 1, 1]],  # first (same order as stimulus)
                [[0, 1, 1], [1, 0, 1]],  # last (different order than stimulus)
                [[2, 0, 0]],  # same (X first)
                [[2, 1, 0]],  # same (X` first)
                [[0, 0, 0], [1, 1, 0]],  # alone
            ]
        ):
            for o_id, in_id, im_id in dend_res_ids:
                all_dend_results[ii] += get_dend_dist_res_for_seeds(
                    order_id=o_id,
                    input_id=in_id,
                    area_id=area_id,
                    imprint_id=im_id,
                )

        if area_id == 1:  # Area B
            # Combine first and last trained inputs into "sequential"
            sequential_data = all_dend_results[0] + all_dend_results[1]
            all_dend_results = [sequential_data] + all_dend_results[2:]

            # Update labels and colors
            all_labels = ["sequential inputs"] + all_labels[2:]
            all_colors = ["#7f7f7f"] + all_colors[2:]  # Use gray for sequential

        bins = np.arange(0, 10) - 0.5

        x_values = np.floor(bins)[-1:]

        width = 0.2
        for ii, (data, label, color) in enumerate(zip(all_dend_results, all_labels, all_colors)):
            counts, bin_edges = np.histogram(data, bins=bins, density=True)
            bin_centers = (
                bin_edges[:-1] + bin_edges[1:]
            ) / 2  # Calculate bin centers for correct x-axis placement
            ax.bar(
                bin_centers + (ii - 1.5) * width,
                counts,
                width=0.25,
                label=label,
                color=color,
                alpha=0.7,
            )

            # counts, bin_edges = np.histogram(data, bins=bins, density=True)
            # ax.bar(x_values + (ii - 1.5) * width, counts, width=width, label=label, color=color)

            # ax.hist(
            #     data,
            #     bins=bins + (ii - 1.5) * width,
            #     width=width,
            #     density=True,
            #     label=label,
            #     color=color,
            # )
        area_label = "Y"
        if area_id == 1:
            area_label = "Z"
        ax.set(xlabel="n connections from inputs", ylabel="density", title=f"Area {area_label}")
        ax.legend()

    def get_assembly_neuron_ids_for_seeds(
        seed,
        order_id,
        area_id,
        imprint_id,
        result_type="selected_assembly_ids",
    ):
        all_res = []
        key = get_key_for_result_dictionary(
            seed=seed,
            order_id=order_id,
            area_id=area_id,
            imprint_id=imprint_id,
            result_type=result_type,
        )

        return result_dict[key]

    # now we create the average venn diagramm
    # Function to calculate intersections
    def calculate_intersections(sets):
        set1, set2, set3 = sets
        set1 = np.array(set1)
        set2 = np.array(set2)
        set3 = np.array(set3)

        if np.all(np.isnan(set1)) or np.all(np.isnan(set2)) or np.all(np.isnan(set3)):
            print("DID NOT USE A SET OF SETS")
            return [float("nan")] * 7

        set1 = set(set1[~np.isnan(set1)].astype(int))
        set2 = set(set2[~np.isnan(set2)].astype(int))
        set3 = set(set3[~np.isnan(set3)].astype(int))

        return (
            len(set1 & set2 & set3),
            len(set1 & set2 - set3),
            len(set1 & set3 - set2),
            len(set2 & set3 - set1),
            len(set1 - set2 - set3),
            len(set2 - set1 - set3),
            len(set3 - set1 - set2),
        )

    # First scenario Venn
    labels = ["X Alone", f"{prime} alone", f"X - then {prime}"]
    for venn_ax, area_id in zip([ax4, ax5], range(2)):
        # Calculate intersections for each set
        intersections = np.array(
            [
                calculate_intersections(
                    # X alone
                    [
                        get_assembly_neuron_ids_for_seeds(
                            seed, order_id=0, area_id=area_id, imprint_id=0
                        ),
                        # Y alone
                        get_assembly_neuron_ids_for_seeds(
                            seed, order_id=1, area_id=area_id, imprint_id=0
                        ),
                        # First X then Y
                        get_assembly_neuron_ids_for_seeds(
                            seed, order_id=0, area_id=area_id, imprint_id=1
                        ),
                    ]
                )
                for seed in all_seeds
            ]
            + [
                calculate_intersections(
                    # X alone
                    [
                        get_assembly_neuron_ids_for_seeds(
                            seed, order_id=1, area_id=area_id, imprint_id=0
                        ),
                        # Y alone
                        get_assembly_neuron_ids_for_seeds(
                            seed, order_id=0, area_id=area_id, imprint_id=0
                        ),
                        # First X then Y
                        get_assembly_neuron_ids_for_seeds(
                            seed, order_id=1, area_id=area_id, imprint_id=1
                        ),
                    ]
                )
                for seed in all_seeds
            ]
        )
        print(intersections)
        # Calculate average intersection sizes
        average_intersections = np.round(np.nanmean(intersections, axis=0), 2)

        # Plot the average Venn diagram on the existing axis
        venn3(
            subsets={
                "100": average_intersections[4],
                "010": average_intersections[5],
                "110": average_intersections[1],
                "001": average_intersections[6],
                "101": average_intersections[2],
                "011": average_intersections[3],
                "111": average_intersections[0],
            },
            set_labels=labels,
            ax=venn_ax,
        )
        area_label = "Y"
        if area_id == 1:
            area_label = "Z"
        venn_ax.set_title(f"Area: {area_label}")

    # Second scenario Venn
    labels = ["X Alone", f"{prime} alone", "Association"]
    for venn_ax, area_id in zip([ax6, ax7], range(2)):
        # Calculate intersections for each set
        intersections = np.array(
            [
                calculate_intersections(
                    # X alone
                    [
                        get_assembly_neuron_ids_for_seeds(
                            seed, order_id=0, area_id=area_id, imprint_id=0
                        ),
                        # Y alone
                        get_assembly_neuron_ids_for_seeds(
                            seed, order_id=1, area_id=area_id, imprint_id=0
                        ),
                        # First X then Y
                        get_assembly_neuron_ids_for_seeds(
                            seed, order_id=2, area_id=area_id, imprint_id=0
                        ),
                    ]
                )
                for seed in all_seeds
            ]
        )
        print(intersections)
        # Calculate average intersection sizes
        average_intersections = np.round(np.nanmean(intersections, axis=0), 2)

        # Plot the average Venn diagram on the existing axis
        venn3(
            subsets={
                "100": average_intersections[4],
                "010": average_intersections[5],
                "110": average_intersections[1],
                "001": average_intersections[6],
                "101": average_intersections[2],
                "011": average_intersections[3],
                "111": average_intersections[0],
            },
            set_labels=labels,
            ax=venn_ax,
        )
        area_label = "Y"
        if area_id == 1:
            area_label = "Z"
        venn_ax.set_title(f"Area: {area_label}")

    if show_plot:
        plt.show()


if __name__ == "__main__":
    paper_figure_7(only_load_results=True)
