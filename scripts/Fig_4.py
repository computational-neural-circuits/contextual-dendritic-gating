from brian2.units import *
import numpy as np

from src.network_recall import (
    NetworkRecall,
)
from src.utils import get_path_to_save_file_name
from src.get_activity_metrics import get_activity_metrics_from_assembly_neurons
from Fig_3 import run_large_imprint_with_recall_on_server, run_large_imprint_with_recall


import matplotlib.pyplot as plt
import time

plt.style.use("../plots_style.txt")

parameter_dict = {}

# only these successive-imprint steps are ever read by the Fig_4 plot, so
# recall probes for the other steps are skipped in run_overlap_simulations
MULTIPLE_IMPRINT_IDS_USED_FOR_PLOT = [0, 1, 3, 7, 11]


def get_figure_parameters():

    all_assembly_recall_sizes = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    all_id_shifts_for_new_assemblies = [0, 2, 4, 5, 6, 8, 10, 12, 14, 15, 16, 18, 20]

    shifts_for_deeper_recall_run = [0, 5, 10, 15, 20]

    all_imprint_orders = [0, 2, 4, 6, 8, 9, 18]

    primary_imprint_orders = [0, 18, 9]

    all_seeds = run_large_imprint_with_recall_on_server(only_get_seeds=True)

    all_context_ids = [0, 1]

    multiple_overlap_shifts = [15, 10]
    imprint_orders_for_multiple_imprints = [0, 9]

    return (
        all_seeds,
        all_assembly_recall_sizes,
        all_id_shifts_for_new_assemblies,
        all_context_ids,
        all_imprint_orders,
        shifts_for_deeper_recall_run,
        primary_imprint_orders,
        multiple_overlap_shifts,
        imprint_orders_for_multiple_imprints,
    )


def get_current_parameters_for_cluster_run(
    run_id, multiple_overlaps=False, only_get_deep_runs=False
):

    (
        all_seeds,
        all_assembly_recall_sizes,
        all_id_shifts_for_new_assemblies,
        all_context_ids,
        all_imprint_orders,
        shifts_for_deeper_recall_run,
        primary_imprint_orders,
        multiple_overlap_shifts,
        imprint_orders_for_multiple_imprints,
    ) = get_figure_parameters()

    def get_values(run_id_counter, the_seeds, the_imprint_orders, the_shifts):

        seed_id = None
        assembly_recall_sizes = None
        id_shift_for_new_assembly = None
        context_id = None
        selected_imprint_order = None

        found_id = False

        for seed in the_seeds:
            for context in all_context_ids:
                for imprint_order in the_imprint_orders:
                    for shift in the_shifts:

                        if shift in shifts_for_deeper_recall_run or multiple_overlaps:
                            the_recall_sizes = all_assembly_recall_sizes
                        else:
                            the_recall_sizes = [20]

                        if run_id_counter == run_id:

                            seed_id = seed
                            id_shift_for_new_assembly = shift
                            assembly_recall_sizes = the_recall_sizes
                            selected_imprint_order = imprint_order
                            context_id = context
                            found_id = True
                            break

                        run_id_counter += 1

                    if found_id:
                        break

                if found_id:
                    break
            if found_id:
                break

        res = (
            seed_id,
            context_id,
            id_shift_for_new_assembly,
            assembly_recall_sizes,
            selected_imprint_order,
        )

        return (
            run_id_counter,
            found_id,
            res,
        )

    run_id_counter = 0

    if multiple_overlaps:

        run_id_counter, found_id, res = get_values(
            run_id_counter,
            the_seeds=all_seeds[:],
            the_imprint_orders=imprint_orders_for_multiple_imprints[:1],
            the_shifts=multiple_overlap_shifts,
        )

        if not found_id:
            run_id_counter, found_id, res = get_values(
                run_id_counter,
                the_seeds=all_seeds[:],
                the_imprint_orders=imprint_orders_for_multiple_imprints[1:],
                the_shifts=multiple_overlap_shifts,
            )

        return res, run_id_counter

    run_id_counter, found_id, res = get_values(
        run_id_counter,
        the_seeds=all_seeds[:5],
        the_imprint_orders=primary_imprint_orders,
        the_shifts=shifts_for_deeper_recall_run,
    )

    if not found_id:
        run_id_counter, found_id, res = get_values(
            run_id_counter,
            the_seeds=all_seeds[:5],
            the_imprint_orders=primary_imprint_orders,
            the_shifts=[
                ii
                for ii in all_id_shifts_for_new_assemblies
                if ii not in shifts_for_deeper_recall_run
            ],
        )
        if only_get_deep_runs and found_id:
            return (None, None, None, None, None), run_id_counter

    if not found_id:
        run_id_counter, found_id, res = get_values(
            run_id_counter,
            the_seeds=all_seeds[5:],
            the_imprint_orders=primary_imprint_orders,
            the_shifts=shifts_for_deeper_recall_run,
        )

    if not found_id:
        run_id_counter, found_id, res = get_values(
            run_id_counter,
            the_seeds=all_seeds[5:],
            the_imprint_orders=primary_imprint_orders,
            the_shifts=[
                ii
                for ii in all_id_shifts_for_new_assemblies
                if ii not in shifts_for_deeper_recall_run
            ],
        )
        if only_get_deep_runs and found_id:
            return (None, None, None, None, None), run_id_counter

    if not found_id:
        run_id_counter, found_id, res = get_values(
            run_id_counter,
            the_seeds=all_seeds,
            the_imprint_orders=[
                ii for ii in all_imprint_orders if ii not in primary_imprint_orders
            ],
            the_shifts=all_id_shifts_for_new_assemblies,
        )

    return res, run_id_counter


def get_network_for_original_sim(net=None, seed_id=None):

    if seed_id is None:
        seed_id = 0

    parameter_dict = {}
    all_assembly_ids_for_areas = [[(0, ii, -1)] for ii in range(20)]
    parameters_for_run_large_imprint = {
        "runtime_imprint": 30 * second,  # 40
        "runtime_baseline": 1 * second,
        "seed": seed_id,
        "all_assembly_ids_for_areas": all_assembly_ids_for_areas,
        "area_names": ["A"],
        "all_context_ids_for_areas": [[(0, 0)] for _ in all_assembly_ids_for_areas],
        "save_network_after_each_imprint": True,
    }

    if net is None:

        net = NetworkRecall(
            parameter_file_name="parameters",
            parameters_for_run=parameters_for_run_large_imprint,
            save_file_name="data_Fig_4_large_imprint",
            parameter_dict=parameter_dict,
            only_load_results=False,
            figure_name="Fig_4",
        )

    else:
        net.parameters_for_run.update(parameters_for_run_large_imprint)

    if "restore_from_save_name" in net.parameters_for_run:
        del net.parameters_for_run["restore_from_save_name"]

    if "presynaptic_sources_1" in net.parameters_for_run:
        del net.parameters_for_run["presynaptic_sources_1"]

    keys = list(net.parameters_for_run.keys())

    for key in keys:
        if "recall" in key:
            del net.parameters_for_run[key]

    net.save_file_name = "data_Fig_4_large_imprint"

    return net


def Fig_4(only_load_results=False, multiple_overlaps=False):

    net = get_network_for_original_sim(net=None, seed_id=0)

    (
        all_seeds,
        all_assembly_recall_sizes,
        all_id_shifts_for_new_assemblies,
        all_context_ids,
        all_imprint_orders,
        shifts_for_deeper_recall_run,
        primary_imprint_orders,
        multiple_overlap_shifts,
        imprint_orders_for_multiple_imprints,
    ) = get_figure_parameters()

    n_multiple_imprints = 12  # this is because the highest count is for shift 15 where we have 4 sucessive imprints in total

    # the 4 is for the multiple overlap shifts,
    avg_firing_rate_after_new_imprint = np.ones(
        (
            2,
            len(all_seeds),
            len(all_context_ids),
            len(all_imprint_orders),
            len(all_id_shifts_for_new_assemblies),
            len(all_assembly_recall_sizes),
            n_multiple_imprints,
        )
    ).astype(float) * float("nan")
    avg_firing_rate_after_original_imprint_for_all_recall_sizes = np.ones(
        (2, len(all_seeds), len(all_imprint_orders), len(all_assembly_recall_sizes))
    ) * float("nan")
    avg_n_active_after_new_imprint = np.ones(
        (
            2,
            len(all_seeds),
            len(all_context_ids),
            len(all_imprint_orders),
            len(all_id_shifts_for_new_assemblies),
            len(all_assembly_recall_sizes),
            n_multiple_imprints,
        )
    ).astype(float) * float("nan")
    avg_n_active_after_original_imprint_for_all_recall_sizes = np.ones(
        (2, len(all_seeds), len(all_imprint_orders), len(all_assembly_recall_sizes))
    ) * float("nan")

    range_n = 1559
    steps = 1

    if multiple_overlaps:
        range_n = 80
        steps = 2

    for run_id in range(0, range_n, steps):

        res = run_simulation_for_run_id(
            run_id,
            net=net,
            only_load_results=only_load_results,
            multiple_overlaps=multiple_overlaps,
            only_get_deep_runs=False,
        )

        if res is None:
            continue

        all_recall_results, original_recall_result, parameters_for_run = res

        (
            seed_id,
            context_id,
            id_shift_for_new_assembly,
            assembly_recall_sizes,
            selected_imprint_order,
        ) = parameters_for_run

        seed_index = all_seeds.index(seed_id)
        shift_index = all_id_shifts_for_new_assemblies.index(id_shift_for_new_assembly)
        order_index = all_imprint_orders.index(selected_imprint_order)

        for multiple_imprint_id in range(n_multiple_imprints):

            for res, recall_size in zip(
                all_recall_results[0, multiple_imprint_id], assembly_recall_sizes
            ):
                recall_size_index = all_assembly_recall_sizes.index(recall_size)

                for bck in [0, 1]:
                    avg_firing_rate_after_new_imprint[
                        bck,
                        seed_index,
                        context_id,
                        order_index,
                        shift_index,
                        recall_size_index,
                        multiple_imprint_id,
                    ] = res[bck * 2 + 0]
                    avg_n_active_after_new_imprint[
                        bck,
                        seed_index,
                        context_id,
                        order_index,
                        shift_index,
                        recall_size_index,
                        multiple_imprint_id,
                    ] = res[bck * 2 + 1]

                    if original_recall_result is not None:
                        bck_index = 1
                        if bck == 1:
                            bck_index = 0
                        avg_firing_rate_after_original_imprint_for_all_recall_sizes[
                            bck, seed_index, order_index, recall_size_index
                        ] = original_recall_result[0][bck_index, recall_size_index]
                        # here 0 is bck, 1 is with context 0 and 2 is with context 1 (but never simulated currently)
                        avg_n_active_after_original_imprint_for_all_recall_sizes[
                            bck, seed_index, order_index, recall_size_index
                        ] = original_recall_result[1][bck_index, recall_size_index]
                        # here 0 is bck, 1 is with context 0 and 2 is with context 1 (but never simulated currently)

    avg_firing_rate_after_original_imprint = (
        avg_firing_rate_after_original_imprint_for_all_recall_sizes[:, :, :, -1]
    )
    avg_n_active_after_original_imprint = (
        avg_n_active_after_original_imprint_for_all_recall_sizes[:, :, :, -1]
    )
    # bck, seed, ctxt, order, shift, size
    avg_firing_rate_after_new_imprint
    # bck, seed, order
    avg_firing_rate_after_original_imprint
    avg_n_active_after_new_imprint
    avg_n_active_after_original_imprint

    fig, axes = plt.subplots(2, 2, figsize=(12, 16))

    ax1, ax2, ax3, ax4 = axes.flatten()

    colored_curves = [
        ["#9ecae1", "#6baed6", "#4292c6", "#2171b5", "#084594"],
        ["#fdd0a2", "#fdae6b", "#fd8d3c", "#f16913", "d94801"],
        ["#dadaeb", "#bcbddc", "#9e9ac8", "#807dba", "6a51a3"],
        ["#fcbba1", "#fc9272", "#fb6a4a", "#ef3b2c", "cb181d"],
        ["#c7e9c0", "#a1d99b", "#74c476", "#41ab5d", "238b45"],
    ]

    special_selection = MULTIPLE_IMPRINT_IDS_USED_FOR_PLOT
    for multiple_imprint_id in range(n_multiple_imprints):
        if multiple_imprint_id not in special_selection:
            continue
        for ii, order in enumerate(all_imprint_orders):
            if order not in [0]:
                continue

            label = None
            for color_id, shift in enumerate([15]):

                label = f"Order: {order}, shift: {shift}, #imprint: {multiple_imprint_id + 1}"
                shift_id = all_id_shifts_for_new_assemblies.index(shift)

                for bck_index, color in enumerate(
                    [
                        colored_curves[
                            color_id * 2
                            + imprint_orders_for_multiple_imprints.index(order)
                        ][special_selection.index(multiple_imprint_id)],
                        "k",
                    ]
                ):

                    if bck_index == 1:
                        if multiple_imprint_id == 0:
                            label = "bck"
                        else:
                            label = None
                    ax1.plot(
                        all_assembly_recall_sizes,
                        np.nanmean(
                            avg_firing_rate_after_new_imprint[
                                bck_index, :, 0, ii, shift_id, :, multiple_imprint_id
                            ]
                            / (avg_firing_rate_after_original_imprint[0, :, ii])[
                                :, None
                            ],
                            axis=0,
                        ),
                        color=color,
                        label=label,
                    )
                    ax2.plot(
                        all_assembly_recall_sizes,
                        np.nanmean(
                            avg_n_active_after_new_imprint[
                                bck_index, :, 0, ii, shift_id, :, multiple_imprint_id
                            ]
                            / (avg_n_active_after_original_imprint[0, :, ii])[:, None],
                            axis=0,
                        ),
                        color=color,
                        label=label,
                    )
                    ax3.plot(
                        all_assembly_recall_sizes,
                        np.nanmean(
                            avg_firing_rate_after_new_imprint[
                                bck_index, :, 1, ii, shift_id, :, multiple_imprint_id
                            ]
                            / (avg_firing_rate_after_original_imprint[0, :, ii])[
                                :, None
                            ],
                            axis=0,
                        ),
                        color=color,
                        label=label,
                    )
                    ax4.plot(
                        all_assembly_recall_sizes,
                        np.nanmean(
                            avg_n_active_after_new_imprint[
                                bck_index, :, 1, ii, shift_id, :, multiple_imprint_id
                            ]
                            / (avg_n_active_after_original_imprint[0, :, ii])[:, None],
                            axis=0,
                        ),
                        color=color,
                        label=label,
                    )

        ax1.plot(
            all_assembly_recall_sizes,
            np.nanmean(
                avg_firing_rate_after_original_imprint_for_all_recall_sizes[0, :, 0, :]
                / (avg_firing_rate_after_original_imprint[0, :, 0])[:, None],
                axis=0,
            ),
            color="r",
            label="recall_after_imprint",
        )
        ax2.plot(
            all_assembly_recall_sizes,
            np.nanmean(
                avg_n_active_after_original_imprint_for_all_recall_sizes[0, :, 0, :]
                / (avg_n_active_after_original_imprint[0, :, 0])[:, None],
                axis=0,
            ),
            color="r",
            label="recall_after_imprint",
        )
        ax3.plot(
            all_assembly_recall_sizes,
            np.nanmean(
                avg_firing_rate_after_original_imprint_for_all_recall_sizes[0, :, 0, :]
                / (avg_firing_rate_after_original_imprint[0, :, 0])[:, None],
                axis=0,
            ),
            color="r",
            label="recall_after_imprint",
        )
        ax4.plot(
            all_assembly_recall_sizes,
            np.nanmean(
                avg_n_active_after_original_imprint_for_all_recall_sizes[0, :, 0, :]
                / (avg_n_active_after_original_imprint[0, :, 0])[:, None],
                axis=0,
            ),
            color="r",
            label="recall_after_imprint",
        )

        ax4.legend()

        ax1.set(
            ylim=[0, 1.4],
            title="Context 0",
            ylabel="avg firing rate (normed)",
            xlabel="Active recall units",
        )
        ax2.set(
            ylim=[0, 1.4],
            title="Context 0",
            ylabel="n active (normed)",
            xlabel="Active recall units",
        )
        ax3.set(
            ylim=[0, 1.4],
            title="Context 1",
            ylabel="avg firing rate (normed)",
            xlabel="Active recall units",
        )
        ax4.set(
            ylim=[0, 1.4],
            title="Context 1",
            ylabel="n active (normed)",
            xlabel="Active recall units",
        )

    fig.savefig("../results/figures/Fig_4.pdf", dpi=800)


def get_recall_results_for_this_assembly(net, selected_imprint_order):

    (
        recall_firing,
        recall_active,
        assembly_sizes_for_all_imprints,
        original_assembly_ids_for_all_imprints,
        firing_rate_of_assemblies_after_imprint,
    ) = run_large_imprint_with_recall(
        net=net, recall_after_imprint_id=selected_imprint_order
    )

    return (
        recall_firing[selected_imprint_order, :, :],
        recall_active[selected_imprint_order, :, :],
        original_assembly_ids_for_all_imprints[selected_imprint_order],
    )


def run_simulation_for_run_id(
    run_id,
    net=None,
    only_load_results=False,
    multiple_overlaps=False,
    only_get_deep_runs=False,
):

    parameters_for_run, _ = get_current_parameters_for_cluster_run(
        run_id,
        multiple_overlaps=multiple_overlaps,
        only_get_deep_runs=only_get_deep_runs,
    )

    (
        seed_id,
        context_id,
        id_shift_for_new_assembly,
        assembly_recall_sizes,
        selected_imprint_order,
    ) = parameters_for_run

    if id_shift_for_new_assembly != 15:
        return None

    if selected_imprint_order != 0:
        return None

    if seed_id is None:
        return None

    if seed_id == 31 and id_shift_for_new_assembly == 15:
        return None  # this seems to be a bad request currently

    net = get_network_for_original_sim(net=net, seed_id=seed_id)
    net.only_load_results = only_load_results
    original_recall_result = None

    all_seeds, _, _, _, all_imprint_orders, _, _, _, _ = get_figure_parameters()

    net.only_load_results = only_load_results

    recall_firing, recall_active, original_assembly_neuron_ids = (
        get_recall_results_for_this_assembly(net, selected_imprint_order)
    )

    original_recall_result = [recall_firing, recall_active]

    net = get_network_for_original_sim(net=net, seed_id=seed_id)
    net.only_load_results = only_load_results

    net.run_imprint(
        report_style="text",
        report_period=900 * second,
        restore_beginning=False,
    )

    net.save_file_name = "data_Fig_4"

    all_recall_results = run_overlap_simulations(
        net,
        seed_id=seed_id,
        context_id=context_id,
        id_shift_for_new_inputs=id_shift_for_new_assembly,
        assembly_recall_sizes=assembly_recall_sizes,
        selected_imprint_order=selected_imprint_order,
        original_assembly_neuron_ids=original_assembly_neuron_ids,
        multiple_overlaps=multiple_overlaps,
    )

    return all_recall_results, original_recall_result, parameters_for_run


def run_overlap_simulations(
    net,
    seed_id,
    context_id,
    id_shift_for_new_inputs,
    assembly_recall_sizes,
    selected_imprint_order=0,
    original_assembly_neuron_ids=None,
    multiple_overlaps=False,
):

    id_start_for_original_assembly = selected_imprint_order * 20

    stored_name = net.save_dict["filename_for_stored_network"]
    if type(stored_name) is not str:
        stored_name = stored_name.decode("utf-8")
    filename_of_baseline_network = stored_name

    network_filename = filename_of_baseline_network + f"_{selected_imprint_order}"
    current_baseline = net.parameters_for_run["runtime_baseline"] / second

    presynaptic_sources_1_for_new_assembly = [
        id_start_for_original_assembly + id_shift_for_new_inputs + pp
        for pp in range(20)
    ]
    presynaptic_sources_1_for_original_assembly = [
        id_start_for_original_assembly + pp for pp in range(20)
    ]

    all_additional_overlaps = [presynaptic_sources_1_for_new_assembly]
    all_additional_contexts = [context_id + context_id * 0]

    if multiple_overlaps:
        if id_shift_for_new_inputs == 15:
            all_additional_overlaps.append(
                [id_start_for_original_assembly + ii + 10 for ii in range(5)]
                + [id_start_for_original_assembly + 20 + 15 + pp for pp in range(15)]
            )
            all_additional_contexts.append(context_id + context_id * 0)
            all_additional_overlaps.append(
                [id_start_for_original_assembly + ii + 5 for ii in range(5)]
                + [
                    id_start_for_original_assembly + 20 + 15 * 2 + pp
                    for pp in range(15)
                ]
            )
            all_additional_contexts.append(context_id + context_id * 0)
            all_additional_overlaps.append(
                [id_start_for_original_assembly + ii for ii in range(5)]
                + [
                    id_start_for_original_assembly + 20 + 15 * 3 + pp
                    for pp in range(15)
                ]
            )
            all_additional_contexts.append(context_id + context_id * 0)

            all_additional_overlaps.append(
                [id_start_for_original_assembly + ii for ii in [0, 4, 8, 12, 16]]
                + [
                    id_start_for_original_assembly + 20 + 15 * 4 + pp
                    for pp in range(15)
                ]
            )
            all_additional_contexts.append(context_id + context_id * 1)
            all_additional_overlaps.append(
                [id_start_for_original_assembly + ii for ii in [1, 5, 9, 13, 17]]
                + [
                    id_start_for_original_assembly + 20 + 15 * 5 + pp
                    for pp in range(15)
                ]
            )
            all_additional_contexts.append(context_id + context_id * 1)
            all_additional_overlaps.append(
                [id_start_for_original_assembly + ii for ii in [2, 6, 10, 14, 18]]
                + [
                    id_start_for_original_assembly + 20 + 15 * 6 + pp
                    for pp in range(15)
                ]
            )
            all_additional_contexts.append(context_id + context_id * 1)
            all_additional_overlaps.append(
                [id_start_for_original_assembly + ii for ii in [3, 7, 11, 15, 19]]
                + [
                    id_start_for_original_assembly + 20 + 15 * 7 + pp
                    for pp in range(15)
                ]
            )
            all_additional_contexts.append(context_id + context_id * 1)

            all_additional_overlaps.append(
                [id_start_for_original_assembly + ii for ii in [3, 4, 5, 6, 7]]
                + [
                    id_start_for_original_assembly + 20 + 15 * 8 + pp
                    for pp in range(15)
                ]
            )
            all_additional_contexts.append(context_id + context_id * 2)
            all_additional_overlaps.append(
                [id_start_for_original_assembly + ii for ii in [8, 9, 10, 11, 12]]
                + [
                    id_start_for_original_assembly + 20 + 15 * 9 + pp
                    for pp in range(15)
                ]
            )
            all_additional_contexts.append(context_id + context_id * 2)
            all_additional_overlaps.append(
                [id_start_for_original_assembly + ii for ii in [13, 14, 15, 16, 17]]
                + [
                    id_start_for_original_assembly + 20 + 15 * 10 + pp
                    for pp in range(15)
                ]
            )
            all_additional_contexts.append(context_id + context_id * 2)
            all_additional_overlaps.append(
                [id_start_for_original_assembly + ii for ii in [18, 19, 0, 1, 2]]
                + [
                    id_start_for_original_assembly + 20 + 15 * 11 + pp
                    for pp in range(15)
                ]
            )
            all_additional_contexts.append(context_id + context_id * 2)

        if id_shift_for_new_inputs == 10:
            all_additional_overlaps.append(
                [id_start_for_original_assembly + ii for ii in range(10)]
                + [id_start_for_original_assembly + 20 + 10 + pp for pp in range(10)]
            )
            all_additional_contexts.append(context_id + context_id * 0)

    # the first one is the assembly we try to recall
    all_recall_results = np.ones((13, 12, len(assembly_recall_sizes), 4)) * float("nan")

    sorted_neuron_ids_original, all_assembly_ids, _ = net.sort_neurons_by_firing_rate(
        area=net.all_areas[0],
        sort_for_specific_imprint=selected_imprint_order,
        shuffle_rest=True,
        print_top_rates=True,
        return_rates_for_imprint=None,
    )

    path = get_path_to_save_file_name(
        "Fig_4", f"seed_{net.parameters_for_run['seed']}_0"
    )
    net.save_weight_matrix(
        export_name=path,
        show_plot=False,
        specific_area_id=0,
        context_ids=[0],
        sorted_neuron_ids=sorted_neuron_ids_original,
    )

    for ps_id, (presyn_sources, this_context_id) in enumerate(
        zip(all_additional_overlaps, all_additional_contexts)
    ):

        parameters_for_run_overlap = {
            "runtime_imprint": 30 * second,  # 40
            "runtime_baseline": 0 * second,
            "seed": seed_id,
            "all_assembly_ids_for_areas": [[(0, 1992, -1)]],
            "area_names": ["A"],
            "all_context_ids_for_areas": [[(0, this_context_id)]],
            "save_network_after_each_imprint": True,
            "presynaptic_sources_1": presyn_sources,
        }

        if current_baseline == 1:
            parameters_for_run_overlap["runtime_baseline"] = 0 * second
        if current_baseline == 0:
            parameters_for_run_overlap["runtime_baseline"] = 1 * second

        net.parameters_for_run.update(parameters_for_run_overlap)
        net.parameters_for_run["restore_from_save_name"] = network_filename

        for ii, assembly_size_recall in enumerate(assembly_recall_sizes):

            for kk, (this_presyn, this_recall_context_id) in enumerate(
                zip(
                    [presynaptic_sources_1_for_original_assembly]
                    + all_additional_overlaps,
                    [0] + all_additional_contexts,
                )
            ):

                # the plot only ever reads results for
                # MULTIPLE_IMPRINT_IDS_USED_FOR_PLOT, so for the other
                # successive-imprint steps we still need exactly one
                # net.run_imprint call (at ii == 0) to advance the network
                # to the next imprint, but can skip the repeated
                # re-imprint + recall probe for the remaining recall sizes
                skip_unused_imprint_step = (
                    kk == 0
                    and ps_id not in MULTIPLE_IMPRINT_IDS_USED_FOR_PLOT
                    and ii != 0
                )
                if skip_unused_imprint_step:
                    continue

                if kk > 0 and (
                    ii != len(assembly_recall_sizes) - 1
                    or ps_id != len(all_additional_overlaps) - 1
                ):
                    # we want these addtioanl overlaps only at the very end
                    continue
                net.run_imprint(report_style="text", report_period=900 * second)

                if not net.save_dict:
                    break

                if multiple_overlaps:
                    stored_name = net.save_dict["filename_for_stored_network"]
                    if type(stored_name) is not str:
                        stored_name = stored_name.decode("utf-8")
                    filename_of_baseline_network = stored_name

                    network_filename = filename_of_baseline_network + f"_0"

                current_baseline = net.parameters_for_run["runtime_baseline"] / second

                if kk == 0 and ps_id not in MULTIPLE_IMPRINT_IDS_USED_FOR_PLOT:
                    # network has been advanced to the next imprint above;
                    # the recall probe itself is never read by the plot
                    continue

                res = run_recall_and_get_results(
                    net,
                    presynaptic_sources_1=this_presyn,
                    context_id=this_recall_context_id,
                    assembly_size_recall=assembly_size_recall,
                    assembly_neuron_selection_seed_recall=0,
                    assembly_ids=original_assembly_neuron_ids,
                    selected_imprint_order=selected_imprint_order,
                    show_plot=False,
                    ommit_baseline=current_baseline != 0,
                    additional_imprint=ps_id,
                    sorted_neuron_ids_original=sorted_neuron_ids_original,
                    id_shift_for_new_inputs=id_shift_for_new_inputs,
                    imprint_context_id=this_context_id,
                )

                if res is not None:

                    all_recall_results[kk, ps_id, ii, 0] = res[
                        0
                    ]  # avg_firing_rate_recall
                    all_recall_results[kk, ps_id, ii, 1] = res[
                        1
                    ]  # n_active_neurons_recall
                    all_recall_results[kk, ps_id, ii, 2] = res[
                        2
                    ]  # avg_firing_rate_recall_bck
                    all_recall_results[kk, ps_id, ii, 3] = res[
                        3
                    ]  # n_active_neurons_recall_bck

    return all_recall_results


def run_recall_and_get_results(
    net,
    presynaptic_sources_1,
    context_id=0,
    assembly_size_recall=20,
    assembly_neuron_selection_seed_recall=0,
    assembly_ids=None,
    selected_imprint_order=None,
    show_plot=False,
    ommit_baseline=False,
    additional_imprint=0,
    sorted_neuron_ids_original=None,
    id_shift_for_new_inputs=None,
    imprint_context_id=None,
):
    stored_name = net.save_dict["filename_for_stored_network"]
    if type(stored_name) is not str:
        stored_name = stored_name.decode("utf-8")
    network_filename = stored_name + f"_0"
    # this is correct, we mean here the extra imprint we are doing, which is oth order
    filename = net.get_path_to_stored_networks(file_name=network_filename)
    if not net.only_load_results:
        net.network.restore(filename=filename)
        try:

            if (
                id_shift_for_new_inputs == 15
                and assembly_size_recall == 20
                and additional_imprint in [0, 1, 3, 7, 11]
            ):
                try:
                    path = get_path_to_save_file_name(
                        "Fig_4",
                        f"seed_{net.parameters_for_run['seed']}_{additional_imprint + 1}_context_{imprint_context_id}",
                    )
                    net.save_weight_matrix(
                        export_name=path,
                        show_plot=False,
                        specific_area_id=0,
                        context_ids=[0],
                        sorted_neuron_ids=sorted_neuron_ids_original,
                    )
                except FileNotFoundError as e:
                    print(e)
                    time.sleep(10)

        except FileNotFoundError:
            return None

    runtime_recall = 2 * second
    rtm_rec = runtime_recall / msecond

    runtime_baseline_recall = 1 * second
    if ommit_baseline:
        runtime_baseline_recall = 0 * second

    net.parameters_for_run.update(
        {
            "all_assembly_ids_for_areas_recall": [[(0, 1992, -1)]],
            "all_context_ids_for_areas_recall": [[(0, context_id)]],
            "runtime_baseline_recall": runtime_baseline_recall,
            "runtime_recall": runtime_recall,
            "assembly_neuron_selection_seed_recall": assembly_neuron_selection_seed_recall,
            "presynaptic_sources_1_recall": presynaptic_sources_1,
            "assembly_size_recall": assembly_size_recall,
        }
    )

    net.run_recall_for_overlap(report_style="text", report_period=900 * second)

    if not net.save_dict:
        return None

    baseline = 1 * second
    start_time = (
        baseline
        + (selected_imprint_order + additional_imprint + 1)
        * (baseline + net.parameters_for_run["runtime_imprint"])
    ) / msecond
    start_time += (
        net.parameters_for_run["runtime_imprint"]
    ) / msecond + baseline / msecond
    end_time = start_time + rtm_rec

    (
        avg_firing_rate_recall,
        n_active_neurons_recall,
        avg_firing_rate_recall_bck,
        n_active_neurons_recall_bck,
    ) = get_activity_metrics_from_assembly_neurons(
        active_threshold=4,
        net=net,
        area=net.all_areas[0],
        selected_ids=assembly_ids,
        start_time=start_time,
        end_time=end_time,
        select_randomly_for_background=True,
    )

    if assembly_size_recall == 20 and show_plot:
        net.show_spike_rasters(
            show_plot=True,
            highlight_neuron_ids=None,
            show_vertical_lines_at=[start_time, end_time],
            sort_specific_neurons_to_front=assembly_ids,
        )

    return (
        avg_firing_rate_recall,
        n_active_neurons_recall,
        avg_firing_rate_recall_bck,
        n_active_neurons_recall_bck,
    )


if __name__ == "__main__":
    Fig_4(only_load_results=True, multiple_overlaps=True)
