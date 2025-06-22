from brian2.units import *
import numpy as np

from src.network_multiple_contexts_over_time_with_association import (
    NetworMultipleContextsOverTimeWithAssociation,
)
from src.network_recall import (
    NetworkRecall,
)
from src.get_activity_metrics import get_activity_metrics_from_assembly_neurons
from figure_3 import load_recurrent_inhibition_comparison

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colorbar import ColorbarBase
import math
from multiprocessing import Pool
from collections import Counter

plt.style.use("../../plots_style.txt")

parameter_dict = {}

# former figure 4
# specifically - new version to get promising inputs


def paper_figure_4(only_load_results=True, seed=927, order_id=0, use_same_context=False, case=None):
    (
        fig,
        axes_for_weights_over_time,
    ) = create_figure_layout_paper_fig_4()

    ########################
    # weights over time (start)
    ######################

    parameters_for_run = {
        "runtime_imprint": 32 * second,  # 33
        "runtime_baseline": 2 * second,  # 2
        "seed": seed,  # 4928
        "all_assembly_ids": [(0, -1), (2, 2), (1992, -1), (1992, 1992), (0, -1), (2, 2)],
        "all_context_ids": [0, 0, 1, 1, 2, 2],
        "monitor_dt_weights": 50 * ms,
    }

    if order_id == 1:
        parameters_for_run["all_assembly_ids"] = [
            (0, -1),
            (2, 2),
            (0, -1),
            (2, 2),
            (1992, -1),
            (1992, 1992),
        ]  # order 1

        parameters_for_run["all_context_ids"] = [0, 0, 2, 2, 1, 1]

    if use_same_context:
        parameters_for_run["all_context_ids"] = [0, 0, 1, 1, 1, 1]

    parameter_dict = {}

    if seed == 927:
        parameters_for_run["presynaptic_sources"] = [
            67,
            379,
            25,
            37,
            274,
            29,
            128,
            144,
            299,
            42,
            149,
            30,
            302,
            23,
            137,
            331,
            358,
            269,
            284,
            158,
        ]
    if seed == 111:
        parameters_for_run["presynaptic_sources"] = [
            85,
            38,
            84,
            266,
            358,
            287,
            34,
            294,
            353,
            301,
            127,
            155,
            369,
            209,
            338,
            158,
            49,
            337,
            171,
            78,
        ]
        parameters_for_run["presynaptic_sources_1"] = [
            53,
            291,
            394,
            65,
            383,
            16,
            371,
            46,
            325,
            359,
            200,
            217,
            37,
            195,
            254,
            279,
            20,
            140,
            146,
            62,
        ]
        parameters_for_run["presynaptic_sources_2"] = [
            154,
            187,
            395,
            202,
            138,
            275,
            322,
            28,
            348,
            356,
            144,
            375,
            181,
            198,
            363,
            176,
            30,
            377,
            103,
            383,
        ]

        if case is not None:
            if case == 0:
                parameters_for_run["presynaptic_sources_1"] = [
                    53,
                    291,
                    16,
                    46,
                    65,
                    371,
                    383,
                    394,
                    20,
                    140,
                    146,
                    62,
                    304,
                    44,
                    300,
                    335,
                    311,
                    269,
                    352,
                    47,
                ]
                parameters_for_run["presynaptic_sources_2"] = [
                    154,
                    187,
                    202,
                    395,
                    181,
                    198,
                    363,
                    176,
                    30,
                    377,
                    103,
                    383,
                    245,
                    280,
                    257,
                    171,
                    392,
                    55,
                    28,
                    138,
                ]

            if case == 1:
                parameters_for_run["presynaptic_sources_1"] = [
                    53,
                    291,
                    371,
                    394,
                    16,
                    46,
                    65,
                    383,
                    37,
                    20,
                    140,
                    200,
                    335,
                    352,
                    325,
                    300,
                    217,
                    44,
                    269,
                    62,
                ]
                parameters_for_run["presynaptic_sources_2"] = [
                    154,
                    395,
                    202,
                    187,
                    275,
                    322,
                    392,
                    245,
                    55,
                    138,
                    356,
                    257,
                    280,
                    363,
                    30,
                    28,
                    176,
                    348,
                    375,
                    198,
                ]
            if case == 2:
                parameters_for_run["presynaptic_sources_1"] = [
                    394,
                    16,
                    300,
                    53,
                    291,
                    65,
                    383,
                    46,
                    325,
                    217,
                    339,
                    62,
                    44,
                    374,
                    368,
                    1,
                    37,
                    168,
                    398,
                    359,
                ]
                parameters_for_run["presynaptic_sources_2"] = [
                    154,
                    187,
                    395,
                    322,
                    138,
                    275,
                    363,
                    181,
                    388,
                    375,
                    303,
                    278,
                    352,
                    252,
                    75,
                    245,
                    396,
                    248,
                    313,
                    257,
                ]
            if case == 3:
                parameters_for_run["presynaptic_sources_1"] = [
                    53,
                    383,
                    368,
                    394,
                    16,
                    44,
                    300,
                    1,
                    374,
                    82,
                    291,
                    390,
                    398,
                    217,
                    62,
                    65,
                    37,
                    140,
                    359,
                    254,
                ]
                parameters_for_run["presynaptic_sources_2"] = [
                    395,
                    181,
                    154,
                    138,
                    275,
                    252,
                    313,
                    197,
                    187,
                    260,
                    143,
                    245,
                    363,
                    388,
                    303,
                    354,
                    396,
                    322,
                    352,
                    248,
                ]

    same_context = ""
    if use_same_context:
        same_context = "_same_context_for_new"

    extra_end = ""
    if case is not None:
        extra_end = f"_{case}"

    net = NetworMultipleContextsOverTimeWithAssociation(
        parameter_file_name="parameters",
        parameters_for_run=parameters_for_run,
        save_file_name=f"paper_fig_4_weights_over_time_order_{order_id}{same_context}{extra_end}",
        parameter_dict=parameter_dict,
        only_load_results=False,
    )

    net.run(report_style="text")

    net.show_weight_matrix(
        matlab_export_name=f"../../results/figures/figure_weights/paper_figure_4_C_order_{order_id}{same_context}{extra_end}"
    )
    net.generate_results(axes=np.array(axes_for_weights_over_time), save_fig=False)

    ########################
    # weights over time (end)
    ######################

    fig.savefig(
        f"../../results/figures/paper_fig_4_order_{order_id}{same_context}{extra_end}.pdf", dpi=800
    )


def create_figure_layout_paper_fig_4():
    fig = plt.figure(figsize=(53, 48))
    gs = fig.add_gridspec(7, 9, hspace=1, wspace=1)

    axes_for_weights_over_time = np.array(
        [
            [fig.add_subplot(gs[ii * 2 : (ii + 1) * 2, 4 * jj : 4 + 4 * jj]) for jj in range(2)]
            for ii in range(3)
        ]
    )
    return (
        fig,
        axes_for_weights_over_time,
    )


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


def run_look_for_large_overlap_on_server(max_cores=4, seed=927):
    params = []

    all_seeds = [
        432,
        562,
        732,
        823,
        3424,
        73,
        845,
        384,
        234,
        66,
        335,
        78,
        235,
        288,
        352,
        313,
        7342,
        632,
        8236,
        344,
        5167,
        74,
        26,
        1356,
        8142,
        51,
        6671,
        424,
        5892,
        351,
        623,
        663,
        32,
    ]
    for new_version_seed in all_seeds:
        params.append([seed, new_version_seed])

    n_cores = np.min([len(params), max_cores])

    with Pool(n_cores) as pool:
        _ = pool.starmap(
            look_for_large_overlap,
            params,
        )


def get_selective_presynaptic_group(
    area,
    targeted_neurons,
    new_context_id,
    inputs_id=0,
    required_ids=20,
    threshold=5,  # Added threshold parameter back
    random_seed=None,
    verbose=True,
    max_attempts=1000,
):
    """
    Find a group of presynaptic neurons that maximizes inputs to target dendrites
    while minimizing inputs to non-target dendrites, using threshold-relative scoring.
    """
    import numpy as np
    from collections import defaultdict, Counter
    import time

    start_time = time.time()

    # Set random seed if provided
    if random_seed is not None:
        np.random.seed(random_seed)

    # Step 1: Identify target dendrites and non-target dendrites
    open_dendrites = set(area.dends_of_ctxt[new_context_id])
    target_dendrites = []

    for target_id in targeted_neurons:
        for jj in range(area.n_dends_each):
            dend_id = target_id * area.n_dends_each + jj
            if dend_id in open_dendrites:
                target_dendrites.append(dend_id)

    target_dendrites_set = set(target_dendrites)
    non_target_dendrites = open_dendrites - target_dendrites_set

    if verbose:
        print(
            f"Found {len(target_dendrites_set)} target dendrites and {len(non_target_dendrites)} non-target dendrites"
        )

    if not target_dendrites:
        if verbose:
            print(f"WARNING: No target dendrites found in this context")
        return []

    # Step 2: Map each presynaptic neuron to its target and non-target connections
    neuron_to_dendrites = defaultdict(lambda: {"target": set(), "non_target": set()})

    for i, j in zip(area.input_synapses[inputs_id].i[:], area.input_synapses[inputs_id].j[:]):
        pre_id, post_id = int(i), int(j)

        if post_id not in open_dendrites:
            continue  # Skip connections to dendrites outside the current context

        if post_id in target_dendrites_set:
            neuron_to_dendrites[pre_id]["target"].add(post_id)
        else:
            neuron_to_dendrites[pre_id]["non_target"].add(post_id)

    # Filter to neurons that connect to at least one target dendrite
    available_neurons = [n for n in neuron_to_dendrites.keys() if neuron_to_dendrites[n]["target"]]

    if verbose:
        print(f"Found {len(available_neurons)} presynaptic neurons connecting to target dendrites")

    # Improved scoring function using threshold-relative scoring
    def score_solution(dendrite_inputs, target_dendrites, non_target_dendrites):
        """
        Threshold-relative scoring:
        - For target dendrites: penalties for below threshold, rewards for above
        - For non-target dendrites: rewards for below threshold, penalties for above
        """
        score = 0

        # Target dendrite scoring
        target_values = [dendrite_inputs.get(d, 0) for d in target_dendrites]
        for input_val in target_values:
            relative_to_threshold = input_val - threshold
            if relative_to_threshold < 0:
                # Penalty for being below threshold, quadratic to emphasize
                score -= 5000 * (relative_to_threshold**2)
            else:
                # Reward for being at or above threshold
                score += 2000 * (relative_to_threshold + 1)

        # Additional reward for uniform high distribution
        target_min = min(target_values) if target_values else 0
        score += 10000 * target_min  # Strongly encourage raising the minimum

        # Non-target dendrite scoring
        non_target_values = [dendrite_inputs.get(d, 0) for d in non_target_dendrites]
        for input_val in non_target_values:
            relative_to_threshold = input_val - threshold
            if relative_to_threshold >= 0:
                # Penalty for being at or above threshold, quadratic
                score -= 5000 * ((relative_to_threshold + 1) ** 2)
            else:
                # Reward for being below threshold
                score += 1000 * abs(relative_to_threshold)

        # Additional penalty for high non-target inputs
        non_target_sum = sum(non_target_values)
        score -= 50 * non_target_sum

        return score

    # Helper function to calculate inputs from a set of neurons
    def calculate_inputs(selected_neurons):
        inputs = {d: 0 for d in target_dendrites_set | non_target_dendrites}
        for neuron in selected_neurons:
            for d in neuron_to_dendrites[neuron]["target"]:
                inputs[d] += 1
            for d in neuron_to_dendrites[neuron]["non_target"]:
                inputs[d] += 1
        return inputs

    # Improved swapping function with adaptive sampling
    def optimize_by_swapping(selected, available, max_swaps=100):
        inputs = calculate_inputs(selected)
        current_score = score_solution(inputs, target_dendrites_set, non_target_dendrites)

        swaps_made = 0
        improvement_found = True
        consecutive_failures = 0

        # Target dendrites with low inputs - prioritize these
        low_input_targets = set()

        while improvement_found and swaps_made < max_swaps:
            improvement_found = False
            swaps_made += 1

            # Update our knowledge of problematic dendrites
            target_values = {d: inputs.get(d, 0) for d in target_dendrites_set}
            low_input_targets = {d for d, v in target_values.items() if v < threshold}

            # Adaptive sampling: focus more on removing neurons that target few low-input dendrites
            out_weights = []
            for neuron in selected:
                # Lower weight for neurons that target many low-input dendrites
                low_targets_hit = len(
                    neuron_to_dendrites[neuron]["target"].intersection(low_input_targets)
                )
                # Higher weight for neurons with many non-target connections
                non_target_conns = len(neuron_to_dendrites[neuron]["non_target"])
                # Lower weight = more likely to be considered for removal
                weight = non_target_conns / (1 + 3 * low_targets_hit)
                out_weights.append(weight)

            # Normalize weights if needed
            if out_weights and sum(out_weights) > 0:
                out_weights = [w / sum(out_weights) for w in out_weights]
                out_candidates = np.random.choice(
                    selected, min(10, len(selected)), p=out_weights, replace=False
                )
            else:
                out_candidates = np.random.choice(selected, min(10, len(selected)), replace=False)

            # For in-candidates, prioritize neurons that target low-input dendrites
            in_weights = []
            for neuron in available:
                # Higher weight for neurons that target many low-input dendrites
                low_targets_hit = len(
                    neuron_to_dendrites[neuron]["target"].intersection(low_input_targets)
                )
                # Lower weight for neurons with many non-target connections
                non_target_conns = len(neuron_to_dendrites[neuron]["non_target"])
                # Higher weight = more likely to be considered for addition
                weight = (1 + 3 * low_targets_hit) / (1 + non_target_conns)
                in_weights.append(weight)

            # Normalize weights if needed
            if in_weights and sum(in_weights) > 0:
                in_weights = [w / sum(in_weights) for w in in_weights]
                in_candidates = np.random.choice(
                    available, min(20, len(available)), p=in_weights, replace=False
                )
            else:
                in_candidates = np.random.choice(available, min(20, len(available)), replace=False)

            best_swap = None
            best_swap_score = current_score

            for out_neuron in out_candidates:
                # Remove connections from out_neuron
                for d in neuron_to_dendrites[out_neuron]["target"]:
                    inputs[d] -= 1
                for d in neuron_to_dendrites[out_neuron]["non_target"]:
                    inputs[d] -= 1

                for in_neuron in in_candidates:
                    # Add connections from in_neuron
                    for d in neuron_to_dendrites[in_neuron]["target"]:
                        inputs[d] += 1
                    for d in neuron_to_dendrites[in_neuron]["non_target"]:
                        inputs[d] += 1

                    # Score this swap
                    swap_score = score_solution(inputs, target_dendrites_set, non_target_dendrites)

                    if swap_score > best_swap_score:
                        best_swap_score = swap_score
                        best_swap = (out_neuron, in_neuron)

                    # Undo the addition
                    for d in neuron_to_dendrites[in_neuron]["target"]:
                        inputs[d] -= 1
                    for d in neuron_to_dendrites[in_neuron]["non_target"]:
                        inputs[d] -= 1

                # Restore connections from out_neuron
                for d in neuron_to_dendrites[out_neuron]["target"]:
                    inputs[d] += 1
                for d in neuron_to_dendrites[out_neuron]["non_target"]:
                    inputs[d] += 1

            # Apply the best swap if it improves the solution
            if best_swap and best_swap_score > current_score:
                out_neuron, in_neuron = best_swap

                # Update sets of selected/available neurons
                selected.remove(out_neuron)
                selected.append(in_neuron)
                available.remove(in_neuron)
                available.append(out_neuron)

                # Update inputs
                for d in neuron_to_dendrites[out_neuron]["target"]:
                    inputs[d] -= 1
                for d in neuron_to_dendrites[out_neuron]["non_target"]:
                    inputs[d] -= 1

                for d in neuron_to_dendrites[in_neuron]["target"]:
                    inputs[d] += 1
                for d in neuron_to_dendrites[in_neuron]["non_target"]:
                    inputs[d] += 1

                current_score = best_swap_score
                improvement_found = True
                consecutive_failures = 0
            else:
                consecutive_failures += 1
                # Early termination if multiple consecutive failures
                if consecutive_failures > 20:
                    break

        return selected, available, inputs, current_score, swaps_made

    # Main optimization
    best_solution = None
    best_score = float("-inf")
    best_inputs = None
    no_improvement_count = 0
    total_swaps = 0
    plateau_counter = 0
    last_improvement_score = float("-inf")

    for attempt in range(max_attempts):
        # Early stopping if we've hit a plateau
        if plateau_counter >= 200:
            if verbose:
                print(f"Stopping early after {attempt} attempts - solution has plateaued")
            break

        # Display progress occasionally
        if verbose and (attempt % 50 == 0 or attempt == max_attempts - 1):
            elapsed = time.time() - start_time
            print(
                f"Attempt {attempt+1}/{max_attempts} ({elapsed:.1f}s elapsed, {total_swaps} total swaps)"
            )

        # Build initial solution - either fresh or based on previous best
        if attempt < 200 or no_improvement_count > 50 or best_solution is None:
            # Fresh start
            selected = []
            available = available_neurons.copy()

            # Improved selection strategy
            if attempt % 3 == 0:
                # Strategy 1: Ratio-based (same as before but improved ratio calculation)
                potential_neurons = [
                    (
                        n,
                        len(neuron_to_dendrites[n]["target"]) ** 1.5
                        / (1 + len(neuron_to_dendrites[n]["non_target"])),
                    )
                    for n in available
                ]
                potential_neurons.sort(key=lambda x: x[1], reverse=True)

                # Select some best and some random for diversity
                top_count = min(required_ids // 2, len(potential_neurons))
                if top_count > 0:
                    for i in range(top_count):
                        neuron = potential_neurons[i][0]
                        selected.append(neuron)
                        available.remove(neuron)

            elif attempt % 3 == 1:
                # Strategy 2: Coverage-based (select neurons that cover many uncovered target dendrites)
                remaining_slots = required_ids
                uncovered = set(target_dendrites_set)

                while remaining_slots > 0 and uncovered and available:
                    # For each available neuron, calculate how many uncovered targets it hits
                    coverage_scores = []
                    for neuron in available:
                        targets_covered = len(
                            neuron_to_dendrites[neuron]["target"].intersection(uncovered)
                        )
                        non_targets_hit = len(neuron_to_dendrites[neuron]["non_target"])
                        if targets_covered > 0:
                            score = targets_covered / (1 + 0.2 * non_targets_hit)
                            coverage_scores.append((neuron, score))

                    if not coverage_scores:
                        break

                    # Pick the best neuron
                    coverage_scores.sort(key=lambda x: x[1], reverse=True)
                    best_neuron = coverage_scores[0][0]

                    # Add it to our selection
                    selected.append(best_neuron)
                    available.remove(best_neuron)
                    remaining_slots -= 1

                    # Update uncovered set
                    uncovered -= neuron_to_dendrites[best_neuron]["target"]

            else:
                # Strategy 3: Pure random selection for exploration
                pass  # Will be filled randomly below

            # Fill remaining slots randomly
            remaining = required_ids - len(selected)
            if remaining > 0 and available:
                random_selection = np.random.choice(
                    available, min(remaining, len(available)), replace=False
                )
                for neuron in random_selection:
                    selected.append(neuron)
                    available.remove(neuron)

            no_improvement_count = 0

        else:
            # Start from best solution with adaptive perturbation
            selected = best_solution.copy()
            available = [n for n in available_neurons if n not in selected]

            # Adaptive perturbation based on lack of improvement
            replace_ratio = 0.1 + (0.2 * min(no_improvement_count / 30, 1.0))
            replace_count = max(1, int(len(selected) * replace_ratio))

            if selected and available:
                # Identify underperforming neurons to replace
                if best_inputs:
                    # Find target dendrites with lowest inputs
                    target_values = {d: best_inputs.get(d, 0) for d in target_dendrites_set}
                    low_input_targets = {d for d, v in target_values.items() if v < threshold}

                    # Score each selected neuron by how useful it is
                    neuron_scores = []
                    for neuron in selected:
                        low_targets_hit = len(
                            neuron_to_dendrites[neuron]["target"].intersection(low_input_targets)
                        )
                        non_targets_hit = len(neuron_to_dendrites[neuron]["non_target"])
                        # Lower score = more likely to be replaced
                        score = low_targets_hit / (1 + non_targets_hit)
                        neuron_scores.append((neuron, score))

                    # Sort by ascending score (replace least useful first)
                    neuron_scores.sort(key=lambda x: x[1])
                    to_replace = [n for n, _ in neuron_scores[:replace_count]]
                else:
                    # If no best_inputs yet, choose randomly
                    to_replace = np.random.choice(
                        selected, min(replace_count, len(selected)), replace=False
                    )

                for neuron in to_replace:
                    selected.remove(neuron)
                    available.append(neuron)

                # Add replacements, focusing on neurons that target low-input dendrites
                if best_inputs:
                    # Find neurons that target low-input dendrites
                    candidate_scores = []
                    for neuron in available:
                        low_targets_hit = len(
                            neuron_to_dendrites[neuron]["target"].intersection(low_input_targets)
                        )
                        non_targets_hit = len(neuron_to_dendrites[neuron]["non_target"])
                        score = low_targets_hit / (1 + non_targets_hit)
                        candidate_scores.append((neuron, score))

                    # Sort by descending score (add most useful first)
                    candidate_scores.sort(key=lambda x: x[1], reverse=True)
                    replacements = [n for n, _ in candidate_scores[:replace_count]]

                    # If not enough candidates, add some random ones
                    if len(replacements) < replace_count:
                        remaining = replace_count - len(replacements)
                        remaining_candidates = [n for n in available if n not in replacements]
                        if remaining_candidates:
                            random_additions = np.random.choice(
                                remaining_candidates,
                                min(remaining, len(remaining_candidates)),
                                replace=False,
                            )
                            replacements.extend(random_additions)
                else:
                    # If no best_inputs yet, choose randomly
                    replacements = np.random.choice(
                        available, min(replace_count, len(available)), replace=False
                    )

                for neuron in replacements[:replace_count]:  # Ensure we don't exceed replace_count
                    if neuron in available:  # Safety check
                        selected.append(neuron)
                        available.remove(neuron)

        # Optimize solution through swapping
        selected, available, inputs, score, swaps = optimize_by_swapping(
            selected, available, max_swaps=100
        )
        total_swaps += swaps

        # Update best solution if improved
        if score > best_score:
            improvement_amount = score - best_score
            best_score = score
            best_solution = selected.copy()
            best_inputs = inputs.copy()
            no_improvement_count = 0

            # Check if we're seeing diminishing returns
            if last_improvement_score > float("-inf"):
                if improvement_amount < (best_score * 0.001):  # Less than 0.1% improvement
                    plateau_counter += 1
                else:
                    plateau_counter = 0

            last_improvement_score = score

            if verbose:
                # Calculate statistics
                target_values = [inputs.get(d, 0) for d in target_dendrites_set]
                target_min = min(target_values) if target_values else 0
                target_avg = sum(target_values) / len(target_values) if target_values else 0
                below_threshold = sum(1 for v in target_values if v < threshold)

                non_target_values = [inputs.get(d, 0) for d in non_target_dendrites]
                non_target_max = max(non_target_values) if non_target_values else 0
                non_target_avg = (
                    sum(non_target_values) / len(non_target_values) if non_target_values else 0
                )
                above_threshold = sum(1 for v in non_target_values if v >= threshold)

                print(f"Attempt {attempt+1}: New best solution (score: {best_score:.0f})")
                print(
                    f"  Target: min={target_min}, avg={target_avg:.2f}, below threshold={below_threshold}"
                )
                print(
                    f"  Non-target: max={non_target_max}, avg={non_target_avg:.2f}, above threshold={above_threshold}"
                )
        else:
            no_improvement_count += 1
            # Increment plateau counter if we haven't improved in a while
            if no_improvement_count > 50:
                plateau_counter += 1

    # Display final statistics
    if verbose and best_solution:
        display_solution_statistics(
            area, inputs_id, best_solution, target_dendrites_set, non_target_dendrites, threshold
        )

    return [int(n) for n in best_solution]


def display_solution_statistics(
    area, inputs_id, selected_neurons, target_dendrites, non_target_dendrites, threshold
):
    """Display detailed statistics about the solution with threshold information"""
    import numpy as np
    from collections import Counter

    # Calculate input distribution
    target_inputs = {d: 0 for d in target_dendrites}
    non_target_inputs = {d: 0 for d in non_target_dendrites}

    for neuron_id in selected_neurons:
        indices = np.where(area.input_synapses[inputs_id].i[:] == neuron_id)[0]
        post_targets = area.input_synapses[inputs_id].j[indices]

        for target in post_targets:
            target = int(target)
            if target in target_dendrites:
                target_inputs[target] += 1
            elif target in non_target_dendrites:
                non_target_inputs[target] += 1

    # Calculate statistics
    target_values = list(target_inputs.values())
    target_min = min(target_values) if target_values else 0
    target_max = max(target_values) if target_values else 0
    target_avg = sum(target_values) / len(target_values) if target_values else 0
    below_threshold = sum(1 for v in target_values if v < threshold)

    non_target_values = list(non_target_inputs.values())
    non_target_min = min(non_target_values) if non_target_values else 0
    non_target_max = max(non_target_values) if non_target_values else 0
    non_target_avg = sum(non_target_values) / len(non_target_values) if non_target_values else 0
    above_threshold = sum(1 for v in non_target_values if v >= threshold)

    # Calculate distribution
    target_dist = Counter(target_inputs.values())
    non_target_dist = Counter(non_target_inputs.values())

    print("\n====== Solution Statistics ======")
    print(f"Selected {len(selected_neurons)} presynaptic neurons")

    print(f"\nTarget dendrites ({len(target_dendrites)}):")
    print(f"  Input range: {target_min} to {target_max}")
    print(f"  Average inputs: {target_avg:.2f}")
    print(
        f"  Below threshold (<{threshold}): {below_threshold} ({100*below_threshold/len(target_dendrites):.1f}%)"
    )
    print(
        f"  At or above threshold (≥{threshold}): {len(target_dendrites)-below_threshold} ({100*(len(target_dendrites)-below_threshold)/len(target_dendrites):.1f}%)"
    )

    print(f"\nNon-target dendrites ({len(non_target_dendrites)}):")
    print(f"  Input range: {non_target_min} to {non_target_max}")
    print(f"  Average inputs: {non_target_avg:.2f}")
    print(
        f"  At or above threshold (≥{threshold}): {above_threshold} ({100*above_threshold/len(non_target_dendrites):.1f}%)"
    )
    print(
        f"  Below threshold (<{threshold}): {len(non_target_dendrites)-above_threshold} ({100*(len(non_target_dendrites)-above_threshold)/len(non_target_dendrites):.1f}%)"
    )

    print("\nDistribution of inputs per target dendrite:")
    for count in sorted(target_dist.keys()):
        print(
            f"  {count} inputs: {target_dist[count]} dendrites ({100*target_dist[count]/len(target_dendrites):.1f}%)"
        )

    print("\nDistribution of inputs per non-target dendrite:")
    for count in sorted(non_target_dist.keys()):
        print(
            f"  {count} inputs: {non_target_dist[count]} dendrites ({100*non_target_dist[count]/len(non_target_dendrites):.1f}%)"
        )


def get_selective_presynaptic_groups_both(
    area,
    targeted_neurons,
    new_context_id,
    required_ids=20,
    threshold=5,  # Added threshold parameter like in the reference function
    random_seed=None,
    verbose=True,
    max_attempts=500,
):
    """
    Find presynaptic neurons from both input groups that maximize inputs to target dendrites
    while minimizing inputs to non-target dendrites, using threshold-relative scoring.
    """
    import numpy as np
    import time
    from collections import defaultdict, Counter

    start_time = time.time()

    if verbose:
        print("\n===== Finding selective presynaptic groups from both input streams =====")

    # Set random seed if provided
    if random_seed is not None:
        np.random.seed(random_seed)

    # Step 1: Identify target dendrites and non-target dendrites
    open_dendrites = set(area.dends_of_ctxt[new_context_id])
    target_dendrites = []

    for target_id in targeted_neurons:
        for jj in range(area.n_dends_each):
            dend_id = target_id * area.n_dends_each + jj
            if dend_id in open_dendrites:
                target_dendrites.append(dend_id)

    target_dendrites_set = set(target_dendrites)
    non_target_dendrites = open_dendrites - target_dendrites_set

    if verbose:
        print(
            f"Found {len(target_dendrites_set)} target dendrites and {len(non_target_dendrites)} non-target dendrites"
        )

    if not target_dendrites:
        if verbose:
            print(f"WARNING: No target dendrites found in this context")
        return {0: [], 1: []}

    # Step 2: Map each presynaptic neuron to its target and non-target connections
    neuron_to_dendrites = {
        input_id: defaultdict(lambda: {"target": set(), "non_target": set()}) for input_id in [0, 1]
    }

    for input_id in [0, 1]:
        for i, j in zip(area.input_synapses[input_id].i[:], area.input_synapses[input_id].j[:]):
            pre_id, post_id = int(i), int(j)

            if post_id not in open_dendrites:
                continue  # Skip connections to dendrites outside the current context

            if post_id in target_dendrites_set:
                neuron_to_dendrites[input_id][pre_id]["target"].add(post_id)
            else:
                neuron_to_dendrites[input_id][pre_id]["non_target"].add(post_id)

    # Filter to neurons that connect to at least one target dendrite
    available_neurons = {
        input_id: [
            n
            for n in neuron_to_dendrites[input_id].keys()
            if neuron_to_dendrites[input_id][n]["target"]
        ]
        for input_id in [0, 1]
    }

    if verbose:
        for input_id in [0, 1]:
            print(
                f"Found {len(available_neurons[input_id])} presynaptic neurons connecting to target dendrites in input stream {input_id}"
            )

    # Threshold-relative scoring function similar to the reference function
    def score_solution(dendrite_inputs):
        """
        Threshold-relative scoring:
        - For target dendrites: penalties for below threshold, rewards for above
        - For non-target dendrites: rewards for below threshold, penalties for above
        """
        score = 0

        # Target dendrite scoring
        target_values = [dendrite_inputs.get(d, 0) for d in target_dendrites_set]
        for input_val in target_values:
            relative_to_threshold = input_val - threshold
            if relative_to_threshold < 0:
                # Penalty for being below threshold, quadratic to emphasize
                score -= 5000 * (relative_to_threshold**2)
            else:
                # Reward for being at or above threshold
                score += 2000 * (relative_to_threshold + 1)

        # Additional reward for uniform high distribution
        target_min = min(target_values) if target_values else 0
        score += 10000 * target_min  # Strongly encourage raising the minimum

        # Non-target dendrite scoring
        non_target_values = [dendrite_inputs.get(d, 0) for d in non_target_dendrites]
        for input_val in non_target_values:
            relative_to_threshold = input_val - threshold
            if relative_to_threshold >= 0:
                # Penalty for being at or above threshold, quadratic
                score -= 5000 * ((relative_to_threshold + 1) ** 2)
            else:
                # Reward for being below threshold
                score += 1000 * abs(relative_to_threshold)

        # Additional penalty for high non-target inputs
        non_target_sum = sum(non_target_values)
        score -= 50 * non_target_sum

        return score

    # Calculate combined inputs from selected neurons across both streams
    def calculate_inputs(selected_neurons):
        inputs = {d: 0 for d in target_dendrites_set | non_target_dendrites}

        for input_id in [0, 1]:
            for neuron in selected_neurons[input_id]:
                for d in neuron_to_dendrites[input_id][neuron]["target"]:
                    inputs[d] += 1
                for d in neuron_to_dendrites[input_id][neuron]["non_target"]:
                    inputs[d] += 1

        return inputs

    # Improved swapping function for both input streams
    def optimize_by_swapping(selected, available, max_swaps=100):
        inputs = calculate_inputs(selected)
        current_score = score_solution(inputs)

        swaps_made = 0
        improvement_found = True
        consecutive_failures = 0

        # Target dendrites with low inputs - prioritize these
        low_input_targets = set()

        while improvement_found and swaps_made < max_swaps:
            improvement_found = False
            swaps_made += 1

            # Update problematic dendrites info
            target_values = {d: inputs.get(d, 0) for d in target_dendrites_set}
            low_input_targets = {d for d, v in target_values.items() if v < threshold}

            # Try swapping for each input stream independently
            for input_id in [0, 1]:
                # Skip if we don't have enough neurons to swap
                if not selected[input_id] or not available[input_id]:
                    continue

                # Adaptive sampling: prioritize removing neurons that don't help low-input targets
                out_weights = []
                for neuron in selected[input_id]:
                    # Lower weight for neurons that target many low-input dendrites
                    low_targets_hit = len(
                        neuron_to_dendrites[input_id][neuron]["target"].intersection(
                            low_input_targets
                        )
                    )
                    # Higher weight for neurons with many non-target connections
                    non_target_conns = len(neuron_to_dendrites[input_id][neuron]["non_target"])
                    # Lower weight = more likely to be considered for removal
                    weight = non_target_conns / (1 + 3 * low_targets_hit)
                    out_weights.append(weight)

                # Normalize weights and select candidates
                if out_weights and sum(out_weights) > 0:
                    out_weights = [w / sum(out_weights) for w in out_weights]
                    out_candidates = np.random.choice(
                        selected[input_id],
                        min(10, len(selected[input_id])),
                        p=out_weights,
                        replace=False,
                    )
                else:
                    out_candidates = np.random.choice(
                        selected[input_id], min(10, len(selected[input_id])), replace=False
                    )

                # For in-candidates, prioritize neurons that target low-input dendrites
                in_weights = []
                for neuron in available[input_id]:
                    # Higher weight for neurons that target many low-input dendrites
                    low_targets_hit = len(
                        neuron_to_dendrites[input_id][neuron]["target"].intersection(
                            low_input_targets
                        )
                    )
                    # Lower weight for neurons with many non-target connections
                    non_target_conns = len(neuron_to_dendrites[input_id][neuron]["non_target"])
                    # Higher weight = more likely to be considered for addition
                    weight = (1 + 3 * low_targets_hit) / (1 + non_target_conns)
                    in_weights.append(weight)

                # Normalize weights and select candidates
                if in_weights and sum(in_weights) > 0:
                    in_weights = [w / sum(in_weights) for w in in_weights]
                    in_candidates = np.random.choice(
                        available[input_id],
                        min(20, len(available[input_id])),
                        p=in_weights,
                        replace=False,
                    )
                else:
                    in_candidates = np.random.choice(
                        available[input_id], min(20, len(available[input_id])), replace=False
                    )

                # Find best swap
                best_swap = None
                best_swap_score = current_score

                for out_neuron in out_candidates:
                    # Remove connections from out_neuron temporarily
                    for d in neuron_to_dendrites[input_id][out_neuron]["target"]:
                        inputs[d] -= 1
                    for d in neuron_to_dendrites[input_id][out_neuron]["non_target"]:
                        inputs[d] -= 1

                    for in_neuron in in_candidates:
                        # Add connections from in_neuron temporarily
                        for d in neuron_to_dendrites[input_id][in_neuron]["target"]:
                            inputs[d] += 1
                        for d in neuron_to_dendrites[input_id][in_neuron]["non_target"]:
                            inputs[d] += 1

                        # Score this swap
                        swap_score = score_solution(inputs)

                        if swap_score > best_swap_score:
                            best_swap_score = swap_score
                            best_swap = (out_neuron, in_neuron)

                        # Undo the addition
                        for d in neuron_to_dendrites[input_id][in_neuron]["target"]:
                            inputs[d] -= 1
                        for d in neuron_to_dendrites[input_id][in_neuron]["non_target"]:
                            inputs[d] -= 1

                    # Restore connections from out_neuron
                    for d in neuron_to_dendrites[input_id][out_neuron]["target"]:
                        inputs[d] += 1
                    for d in neuron_to_dendrites[input_id][out_neuron]["non_target"]:
                        inputs[d] += 1

                # Apply the best swap if it improves the solution
                if best_swap and best_swap_score > current_score:
                    out_neuron, in_neuron = best_swap

                    # Update sets of selected/available neurons
                    selected[input_id].remove(out_neuron)
                    selected[input_id].append(in_neuron)
                    available[input_id].remove(in_neuron)
                    available[input_id].append(out_neuron)

                    # Update inputs
                    for d in neuron_to_dendrites[input_id][out_neuron]["target"]:
                        inputs[d] -= 1
                    for d in neuron_to_dendrites[input_id][out_neuron]["non_target"]:
                        inputs[d] -= 1

                    for d in neuron_to_dendrites[input_id][in_neuron]["target"]:
                        inputs[d] += 1
                    for d in neuron_to_dendrites[input_id][in_neuron]["non_target"]:
                        inputs[d] += 1

                    current_score = best_swap_score
                    improvement_found = True
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1
                    if consecutive_failures > 40:
                        break

        return selected, available, inputs, current_score, swaps_made

    # Main optimization
    best_solution = None
    best_score = float("-inf")
    best_inputs = None
    no_improvement_count = 0
    total_swaps = 0
    plateau_counter = 0

    for attempt in range(max_attempts):
        # Early stopping if we've hit a plateau
        if plateau_counter >= 150:
            if verbose:
                print(f"Stopping early after {attempt} attempts - solution has plateaued")
            break

        # Display progress occasionally
        if verbose and (attempt % 50 == 0 or attempt == max_attempts - 1):
            elapsed = time.time() - start_time
            print(
                f"Attempt {attempt+1}/{max_attempts} ({elapsed:.1f}s elapsed, {total_swaps} total swaps)"
            )

        # Build initial solution - either fresh or based on previous best
        if attempt < 200 or no_improvement_count > 50 or best_solution is None:
            # Fresh start for both input streams
            selected = {input_id: [] for input_id in [0, 1]}
            available = {input_id: available_neurons[input_id].copy() for input_id in [0, 1]}

            # For each input stream, select initial neurons using one of three strategies
            for input_id in [0, 1]:
                if attempt % 3 == 0:
                    # Strategy 1: Ratio-based selection
                    potential_neurons = [
                        (
                            n,
                            len(neuron_to_dendrites[input_id][n]["target"]) ** 1.5
                            / (1 + len(neuron_to_dendrites[input_id][n]["non_target"])),
                        )
                        for n in available[input_id]
                    ]
                    potential_neurons.sort(key=lambda x: x[1], reverse=True)

                    # Select top neurons
                    top_count = min(required_ids // 2, len(potential_neurons))
                    if top_count > 0:
                        for i in range(top_count):
                            neuron = potential_neurons[i][0]
                            selected[input_id].append(neuron)
                            available[input_id].remove(neuron)

                elif attempt % 3 == 1:
                    # Strategy 2: Coverage-based selection
                    remaining_slots = required_ids
                    uncovered = set(target_dendrites_set)

                    while remaining_slots > 0 and uncovered and available[input_id]:
                        coverage_scores = []
                        for neuron in available[input_id]:
                            targets_covered = len(
                                neuron_to_dendrites[input_id][neuron]["target"].intersection(
                                    uncovered
                                )
                            )
                            non_targets_hit = len(
                                neuron_to_dendrites[input_id][neuron]["non_target"]
                            )
                            if targets_covered > 0:
                                score = targets_covered / (1 + 0.2 * non_targets_hit)
                                coverage_scores.append((neuron, score))

                        if not coverage_scores:
                            break

                        # Pick the best neuron
                        coverage_scores.sort(key=lambda x: x[1], reverse=True)
                        best_neuron = coverage_scores[0][0]

                        # Add it to our selection
                        selected[input_id].append(best_neuron)
                        available[input_id].remove(best_neuron)
                        remaining_slots -= 1

                        # Update uncovered set
                        uncovered -= neuron_to_dendrites[input_id][best_neuron]["target"]

                # Fill remaining slots randomly
                remaining = required_ids - len(selected[input_id])
                if remaining > 0 and available[input_id]:
                    random_selection = np.random.choice(
                        available[input_id], min(remaining, len(available[input_id])), replace=False
                    )
                    for neuron in random_selection:
                        selected[input_id].append(neuron)
                        available[input_id].remove(neuron)

            no_improvement_count = 0

        else:
            # Start from best solution with adaptive perturbation
            selected = {input_id: best_solution[input_id].copy() for input_id in [0, 1]}
            available = {
                input_id: [n for n in available_neurons[input_id] if n not in selected[input_id]]
                for input_id in [0, 1]
            }

            # For each input stream, replace some neurons
            for input_id in [0, 1]:
                if not selected[input_id] or not available[input_id]:
                    continue

                # Adaptive perturbation based on lack of improvement
                replace_ratio = 0.1 + (0.2 * min(no_improvement_count / 30, 1.0))
                replace_count = max(1, int(len(selected[input_id]) * replace_ratio))

                # Identify underperforming neurons to replace
                if best_inputs:
                    # Find target dendrites with lowest inputs
                    target_values = {d: best_inputs.get(d, 0) for d in target_dendrites_set}
                    low_input_targets = {d for d, v in target_values.items() if v < threshold}

                    # Score each selected neuron by how useful it is
                    neuron_scores = []
                    for neuron in selected[input_id]:
                        low_targets_hit = len(
                            neuron_to_dendrites[input_id][neuron]["target"].intersection(
                                low_input_targets
                            )
                        )
                        non_targets_hit = len(neuron_to_dendrites[input_id][neuron]["non_target"])
                        # Lower score = more likely to be replaced
                        score = low_targets_hit / (1 + non_targets_hit)
                        neuron_scores.append((neuron, score))

                    # Sort by score and replace worst neurons
                    neuron_scores.sort(key=lambda x: x[1])
                    to_replace = [n for n, _ in neuron_scores[:replace_count]]
                else:
                    # If no best_inputs yet, choose randomly
                    to_replace = np.random.choice(
                        selected[input_id],
                        min(replace_count, len(selected[input_id])),
                        replace=False,
                    )

                for neuron in to_replace:
                    selected[input_id].remove(neuron)
                    available[input_id].append(neuron)

                # Add replacements, focusing on neurons that target low-input dendrites
                if best_inputs and low_input_targets:
                    # Find neurons that target low-input dendrites
                    candidate_scores = []
                    for neuron in available[input_id]:
                        low_targets_hit = len(
                            neuron_to_dendrites[input_id][neuron]["target"].intersection(
                                low_input_targets
                            )
                        )
                        non_targets_hit = len(neuron_to_dendrites[input_id][neuron]["non_target"])
                        score = low_targets_hit / (1 + non_targets_hit)
                        candidate_scores.append((neuron, score))

                    # Sort by score and add best replacements
                    candidate_scores.sort(key=lambda x: x[1], reverse=True)
                    replacements = [n for n, _ in candidate_scores[:replace_count]]
                else:
                    # Choose randomly
                    replacements = np.random.choice(
                        available[input_id],
                        min(replace_count, len(available[input_id])),
                        replace=False,
                    )

                for neuron in replacements[:replace_count]:
                    if neuron in available[input_id]:
                        selected[input_id].append(neuron)
                        available[input_id].remove(neuron)

        # Optimize solution through swapping
        selected, available, inputs, score, swaps = optimize_by_swapping(
            selected, available, max_swaps=100
        )
        total_swaps += swaps

        # Update best solution if improved
        if score > best_score:
            improvement_amount = score - best_score
            best_score = score
            best_solution = {input_id: selected[input_id].copy() for input_id in [0, 1]}
            best_inputs = inputs.copy()
            no_improvement_count = 0

            # Check if we're seeing diminishing returns
            if improvement_amount < (abs(best_score) * 0.001):  # Less than 0.1% improvement
                plateau_counter += 1
            else:
                plateau_counter = 0

            if verbose:
                # Calculate statistics
                target_values = [inputs.get(d, 0) for d in target_dendrites_set]
                target_min = min(target_values) if target_values else 0
                target_avg = sum(target_values) / len(target_values) if target_values else 0
                below_threshold = sum(1 for v in target_values if v < threshold)

                non_target_values = [inputs.get(d, 0) for d in non_target_dendrites]
                non_target_max = max(non_target_values) if non_target_values else 0
                non_target_avg = (
                    sum(non_target_values) / len(non_target_values) if non_target_values else 0
                )
                above_threshold = sum(1 for v in non_target_values if v >= threshold)

                print(f"Attempt {attempt+1}: New best solution (score: {best_score:.0f})")
                print(
                    f"  Target: min={target_min}, avg={target_avg:.2f}, below threshold={below_threshold}"
                )
                print(
                    f"  Non-target: max={non_target_max}, avg={non_target_avg:.2f}, above threshold={above_threshold}"
                )
        else:
            no_improvement_count += 1
            if no_improvement_count > 50:
                plateau_counter += 1

    # Display final statistics
    # if verbose and best_solution:
    #     display_solution_statistics(
    #         area, [0, 1], best_solution, target_dendrites_set, non_target_dendrites, threshold
    #     )

    return [[int(n) for n in best_solution[input_id]] for input_id in [0, 1]]


def look_for_large_overlap(seed, new_version_seed=None, only_load_results=False, show_results=False):
    parameters_for_run = {
        "runtime_imprint": 32 * second,  # 33
        "runtime_baseline": 2 * second,  # 2
        "seed": seed,
        "area_names": ["A"],
        "all_assembly_ids_for_areas": [[(0, 0, -1)], [(0, 2, 2)]],  # [(0, 1, 1)], [(0, 0, -1)]
        "all_context_ids_for_areas": [[(0, 0)], [(0, 0)]],
        "save_network_after_each_imprint": True,
    }
    # parameters_for_run = {
    #     "runtime_imprint": 32 * second,  # 33
    #     "runtime_baseline": 2 * second,  # 2
    #     "seed": seed,
    #     "area_names": ["A"],
    #     "all_assembly_ids_for_areas": [[(0, 0, -1)], [(0, 1, 1)], [(0, 0, -1)]],
    #     "all_context_ids_for_areas": [[(0, 0)], [(0, 0)], [(0, 1)]],
    #     "save_network_after_each_imprint": True,
    # }

    net = NetworkRecall(
        parameter_file_name="parameters",
        parameters_for_run=parameters_for_run,
        # save_file_name="paper_search_for_large_overlap",
        save_file_name="look_for_large_overlap",
        parameter_dict=parameter_dict,
        only_load_results=False,
    )

    save_dict = net.run_imprint(report_style="text")
    net.only_load_results = only_load_results

    # return

    all_seeds = [
        0,
        # 1,
        # 2,
        # 3,
        # 4,
        # 5,
        # 6,
        # 7,
    ]

    file_name = save_dict["filename_for_stored_network"]
    if type(file_name) == str:
        network_filename_base = file_name + "_1"
    else:
        network_filename_base = file_name.decode("utf-8") + "_1"
    net.network.restore(filename=net.get_path_to_stored_networks(file_name=network_filename_base))

    # _, _, selected_ids = net.sort_neurons_by_firing_rate(
    #     sort_for_specific_imprint=0,
    #     area_name="A",
    # )

    _, _, selected_ids_association = net.sort_neurons_by_firing_rate(
        sort_for_specific_imprint=1,
        area_name="A",
    )

    new_context_id = 1

    run_association = True

    for random_seed in all_seeds:
        if not run_association:
            print("selected ids: ", selected_ids, random_seed)

            presynaptic_sources = get_selective_presynaptic_group(
                area=net.all_areas[0],
                targeted_neurons=selected_ids,
                new_context_id=new_context_id,
                inputs_id=0,
                required_ids=20,
                # activation_threshold=4,
                random_seed=None,
                verbose=True,
                # optimization_attempts=200,
                # lambda_penalty=1.0,
                max_attempts=10,
            )

            #### sanity check
            area = net.all_areas[0]
            all_input_synapses = area.input_synapses[0]

            ### control for other groups
            all_counts = []
            for dend_id in range(area.n_dends_each * area.n_somas):
                if dend_id in area.dends_of_ctxt[new_context_id]:
                    syn_ids_that_target_dendrite = np.where(area.input_synapses[0].j[:] == dend_id)[
                        0
                    ]
                    count = 0
                    for source in area.input_synapses[0].i[syn_ids_that_target_dendrite]:
                        if source in presynaptic_sources:
                            count += 1
                    all_counts.append(count)
                    if count > 3 and dend_id // area.n_dends_each not in selected_ids:
                        print(
                            "not selected higher than 5: ",
                            dend_id // area.n_dends_each,
                            "  with count:",
                            count,
                        )

                    if dend_id // area.n_dends_each in selected_ids:
                        print("selected dend", dend_id // area.n_dends_each, "  with count:", count)

            print(len(all_counts), np.sort(all_counts))

            net.network.restore(
                filename=net.get_path_to_stored_networks(file_name=network_filename_base)
            )

            print("presyn: ", presynaptic_sources, type(presynaptic_sources))
            net.parameters_for_run["all_assembly_ids_for_areas"] = [[(0, 1992, -1)]]
            net.parameters_for_run["all_context_ids_for_areas"] = [[(0, new_context_id)]]
            net.parameters_for_run["presynaptic_sources_1"] = presynaptic_sources
            net.parameters_for_run["restore_from_save_name"] = network_filename_base
            net.parameters_for_run["runtime_baseline"] = 0 * second  # 2

            net.run_imprint(report_style="text")

            file_name = save_dict["filename_for_stored_network"]
            if type(file_name) == str:
                network_filename = file_name + "_0"
            else:
                network_filename = file_name.decode("utf-8") + "_0"

            net.network.restore(filename=net.get_path_to_stored_networks(file_name=network_filename))

            if net.save_dict and show_results:
                net.parameters_for_run["all_assembly_ids_for_areas"] = [
                    [(0, 0, -1)],
                    [(0, 1, 1)],
                    [(0, 1992, -1)],
                ]
                net.parameters_for_run["all_context_ids_for_areas"] = [
                    [(0, 0)],
                    [(0, 0)],
                    [(0, new_context_id)],
                ]
                net.parameters_for_run["runtime_baseline"] = 2 * second  # 2

                print(net.save_dict["filename_for_stored_network"])

                _, _, selected_ids1 = net.sort_neurons_by_firing_rate(
                    sort_for_specific_imprint=0,
                    area_name="A",
                    print_top_rates=False,
                )
                _, _, selected_ids2 = net.sort_neurons_by_firing_rate(
                    sort_for_specific_imprint=2,
                    area_name="A",
                    print_top_rates=False,
                )
                overlap = []
                for sid in selected_ids1:
                    if sid in selected_ids2:
                        overlap.append(sid)
                print("OVERLAP ", len(overlap), overlap)
                net.show_spike_rasters(
                    axes=None, show_plot=True, print_top_rates=False, sort_for_specific_imprint=0
                )

        else:
            net.network.restore(
                filename=net.get_path_to_stored_networks(file_name=network_filename_base)
            )

            presynaptic_sources = get_selective_presynaptic_groups_both(
                area=net.all_areas[0],
                targeted_neurons=selected_ids_association,
                new_context_id=new_context_id,
                required_ids=20,
                threshold=8,
                random_seed=None,
                max_attempts=500,
            )

            print("#########")
            print(presynaptic_sources)

            #### sanity check
            area = net.all_areas[0]
            all_input_synapses = area.input_synapses[0]

            ### control for other groups
            all_counts = []
            for dend_id in range(area.n_dends_each * area.n_somas):
                if dend_id in area.dends_of_ctxt[new_context_id]:
                    count = 0

                    for kk in range(2):
                        syn_ids_that_target_dendrite = np.where(
                            area.input_synapses[kk].j[:] == dend_id
                        )[0]
                        for source in area.input_synapses[kk].i[syn_ids_that_target_dendrite]:
                            if source in presynaptic_sources[kk]:
                                count += 1
                    all_counts.append(count)
                    if count > 5 and dend_id // area.n_dends_each not in selected_ids_association:
                        print(
                            "not selected higher than 5: ",
                            dend_id // area.n_dends_each,
                            "  with count:",
                            count,
                        )

                    if dend_id // area.n_dends_each in selected_ids_association:
                        print("selected dend", dend_id // area.n_dends_each, "  with count:", count)

            raise ValueError

            net.parameters_for_run["all_assembly_ids_for_areas"] = [[(0, 1992, 1992)]]
            net.parameters_for_run["all_context_ids_for_areas"] = [[(0, new_context_id)]]
            net.parameters_for_run["presynaptic_sources_1"] = presynaptic_sources[0]
            net.parameters_for_run["presynaptic_sources_2"] = presynaptic_sources[1]
            net.parameters_for_run["restore_from_save_name"] = network_filename_base
            net.parameters_for_run["runtime_baseline"] = 0 * second  #

            net.run_imprint(report_style="text")

            file_name = save_dict["filename_for_stored_network"]
            if type(file_name) == str:
                network_filename = file_name + "_1"
            else:
                network_filename = file_name.decode("utf-8") + "_1"

            net.network.restore(filename=net.get_path_to_stored_networks(file_name=network_filename))

            if net.save_dict and show_results:
                net.parameters_for_run["all_assembly_ids_for_areas"] = [
                    [(0, 0, -1)],
                    [(0, 1, 1)],
                    [(0, 1992, 1992)],
                ]
                net.parameters_for_run["all_context_ids_for_areas"] = [
                    [(0, 0)],
                    [(0, 0)],
                    [(0, new_context_id)],
                ]
                net.parameters_for_run["runtime_baseline"] = 2 * second  # 2

                _, _, selected_ids1 = net.sort_neurons_by_firing_rate(
                    sort_for_specific_imprint=1,
                    area_name="A",
                    print_top_rates=False,
                )
                _, _, selected_ids2 = net.sort_neurons_by_firing_rate(
                    sort_for_specific_imprint=2,
                    area_name="A",
                    print_top_rates=False,
                )
                overlap = []
                for sid in selected_ids1:
                    if sid in selected_ids2:
                        overlap.append(sid)
                print("OVERLAP ", len(overlap), overlap)
                net.show_spike_rasters(
                    axes=None, show_plot=True, print_top_rates=False, sort_for_specific_imprint=1
                )


if __name__ == "__main__":
    # look_for_large_overlap(seed=927, show_results=True)
    # look_for_large_overlap(seed=111, show_results=True)
    # paper_figure_4(only_load_results=False, seed=111, order_id=1, use_same_context=False)
    # paper_figure_4(only_load_results=False, seed=111, order_id=1, use_same_context=True)
    # paper_figure_4(only_load_results=False, seed=111, order_id=1, use_same_context=True, case=0)
    # paper_figure_4(only_load_results=False, seed=111, order_id=1, use_same_context=True, case=1)
    # paper_figure_4(only_load_results=False, seed=111, order_id=1, use_same_context=False, case=1)
    # paper_figure_4(only_load_results=False, seed=111, order_id=1, use_same_context=True, case=3)
    # paper_figure_4(only_load_results=False, seed=111, order_id=1, use_same_context=False, case=2)
    paper_figure_4(only_load_results=False, seed=111, order_id=1, use_same_context=False, case=3)
