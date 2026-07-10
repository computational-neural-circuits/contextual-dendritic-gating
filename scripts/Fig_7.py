from brian2.units import *
import numpy as np

from src.network_recall import (
    NetworkRecall,
)
from src.get_activity_metrics import get_activity_metrics_from_assembly_neurons
from src.utils import get_path_to_save_file_name
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colorbar import ColorbarBase
from multiprocessing import Pool

plt.style.use("../plots_style.txt")


def Fig_7(only_load_results=False):
    (
        fig,
        ax_recall_multi_area,
        ax_recall_multi_area_avg_firing_rate,
        ax_recall_multi_area_n_active,
    ) = create_figure_layout_Fig_7()

    normalization_clocks = None

    normalization_clocks = show_multi_layer_recall_for_many_responses(
        axes=ax_recall_multi_area,
        only_load_results=only_load_results,
        seed=843,
        all_assembly_ids_for_areas=[[(0, -1, 0)]],
        normalize_results=True,
        show_plot=False,
        normalization_clocks=normalization_clocks,
    )

    normalization_clocks = show_multi_layer_recall_max_response(
        axes=[ax_recall_multi_area_avg_firing_rate, ax_recall_multi_area_n_active],
        show_plot=False,
        normalization_clocks=normalization_clocks,
    )
    fig.savefig("../results/figures/Fig_7.pdf", dpi=800)


def create_figure_layout_Fig_7():
    fig = plt.figure(figsize=(20, 8))
    gs = fig.add_gridspec(4, 12, hspace=1, wspace=1)

    ax_recall_multi_area = (
        [fig.add_subplot(gs[ii * 2 : (ii + 1) * 2, 2:4]) for ii in range(2)]
        + [fig.add_subplot(gs[ii * 2 : (ii + 1) * 2, 4:6]) for ii in range(2)]
        + [fig.add_subplot(gs[:4, 1:2])]
    )
    ax_recall_multi_area_avg_firing_rate = fig.add_subplot(gs[1:3, 6:9])
    ax_recall_multi_area_n_active = fig.add_subplot(gs[1:3, 9:12])

    return (
        fig,
        ax_recall_multi_area,
        ax_recall_multi_area_avg_firing_rate,
        ax_recall_multi_area_n_active,
    )


def run_multilayer_recall_and_get_results(
    net,
    all_network_seeds,
    list_of_all_assembly_ids_for_areas,
    all_recall_seeds,
    all_deleted_neurons,
    all_recall_sizes,
    change_firing_rate=True,
    normalize_results=False,
    only_load_results=True,
):
    all_avg_firing_rates = np.zeros(
        (
            len(all_network_seeds),
            len(list_of_all_assembly_ids_for_areas),
            len(net.all_areas),
            len(all_recall_seeds),
            len(all_deleted_neurons),
            len(all_recall_sizes),
            2,
        )
    ) * float("nan")
    all_n_active_neurons = np.zeros_like(all_avg_firing_rates) * float("nan")

    all_avg_firing_rates_end_of_imprint = np.zeros(
        (
            len(all_network_seeds),
            len(list_of_all_assembly_ids_for_areas),
            len(net.all_areas),
            2,
        )
    ) * float("nan")
    all_n_active_neurons_end_of_imprint = np.zeros_like(
        all_avg_firing_rates_end_of_imprint
    ) * float("nan")

    for network_seed_id, network_seed in enumerate(all_network_seeds[:]):
        for aids_id, all_assembly_ids_for_areas in enumerate(
            list_of_all_assembly_ids_for_areas
        ):
            (
                [avg_firing_rates, n_active_neurons],
                [
                    avg_firing_rates_end_of_imprint,
                    n_active_neurons_end_of_imprint,
                ],
            ) = multi_layer_recall(
                net,
                network_seed,
                only_load_results,
                all_assembly_ids_for_areas,
                all_recall_seeds,
                all_deleted_neurons,
                all_recall_sizes,
                change_firing_rate,
            )

            all_avg_firing_rates[network_seed_id, aids_id] = avg_firing_rates
            all_avg_firing_rates_end_of_imprint[network_seed_id, aids_id] = (
                avg_firing_rates_end_of_imprint
            )
            all_n_active_neurons[network_seed_id, aids_id] = n_active_neurons
            all_n_active_neurons_end_of_imprint[network_seed_id, aids_id] = (
                n_active_neurons_end_of_imprint
            )

    if normalize_results:
        a, b, c, _ = all_avg_firing_rates_end_of_imprint.shape
        reshaped = all_avg_firing_rates_end_of_imprint[:, :, :, 0].reshape(
            a, b, c, 1, 1, 1, 1
        )
        all_avg_firing_rates /= reshaped

        reshaped = all_n_active_neurons_end_of_imprint[:, :, :, 0].reshape(
            a, b, c, 1, 1, 1, 1
        )
        all_n_active_neurons /= reshaped

    return (
        all_avg_firing_rates,
        all_avg_firing_rates_end_of_imprint,
        all_n_active_neurons,
        all_n_active_neurons_end_of_imprint,
    )


def show_multi_layer_recall_for_many_responses(
    axes=None,
    only_load_results=False,
    seed=5,
    all_assembly_ids_for_areas=[[(0, -1, 0)]],
    normalize_results=True,
    show_plot=True,
    normalization_clocks=None,
):
    net, normalization_clocks = get_network_for_investigation(
        seed=seed, normalization_clocks=normalization_clocks
    )
    all_recall_seeds = [0, 1, 2, 3, 4, 5]
    all_deleted_neurons = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
    all_recall_sizes = [ii for ii in range(21)]

    change_firing_rate = True

    (
        # (network seeds, assemblyIDs, areas, recall seeds, deleted, recall sizes, in/bck)
        all_avg_firing_rates,
        all_avg_firing_rates_end_of_imprint,
        all_n_active_neurons,
        all_n_active_neurons_end_of_imprint,
    ) = run_multilayer_recall_and_get_results(
        net=net,
        all_network_seeds=[seed],
        list_of_all_assembly_ids_for_areas=[all_assembly_ids_for_areas],
        all_recall_seeds=all_recall_seeds,
        all_deleted_neurons=all_deleted_neurons,
        all_recall_sizes=all_recall_sizes,
        change_firing_rate=change_firing_rate,
        normalize_results=normalize_results,
        only_load_results=only_load_results,
    )

    if axes is None:
        fig, axes = plt.subplots(3, 2)
        axes = axes.flatten()
    cmap = plt.cm.Greys_r
    bounds = np.arange(0, len(all_deleted_neurons))
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    cb = ColorbarBase(
        axes[4],
        cmap=cmap,
        norm=norm,
        spacing="proportional",
        ticks=np.arange(len(all_deleted_neurons)),
        boundaries=bounds,
        format="%1i",
    )
    tick_labels = [str(val) for val in all_deleted_neurons]
    cb.set_ticks(np.arange(len(all_deleted_neurons)) + 0.5)
    cb.set_ticklabels(tick_labels)
    for ax, res, ylabel in zip(
        [[axes[0], axes[1]], [axes[2], axes[3]]],
        [all_avg_firing_rates, all_n_active_neurons],
        ["avg. firing rate of assembly", "n active neurons of assembly"],
    ):
        for area_id, area_name in enumerate(["A", "B"]):
            ax[area_id].set(title=f"Area {area_name}", xlabel="inputs", ylabel=ylabel)
            for deleted_neurons in all_deleted_neurons:
                y_values = res[
                    0,  # there is only one network seed
                    0,  # there is only one type of all_assembly_ids_for_areas
                    area_id,
                    :,  # take mean of recall seeds
                    all_deleted_neurons.index(deleted_neurons),
                    :,  # look at all recall sizes
                    0,  # we look at the assembly neurons
                ]

                y_values = np.nanmean(y_values, axis=0)

                ax[area_id].plot(
                    all_recall_sizes,
                    y_values,
                    color=cmap(norm(all_deleted_neurons.index(deleted_neurons))),
                )

            ax[area_id].set(ylim=[0, 1.06])
    if show_plot:
        plt.show()

    return normalization_clocks


def show_multi_layer_recall_max_response(
    axes=None,
    only_load_results=True,
    normalize_results=True,
    # contains the network seed to highlight and the assembly_ids_for_all_areas for that network
    highlight_point_for=[843, [[(0, -1, 0)]]],
    show_plot=True,
    normalization_clocks=None,
):
    all_seeds = run_how_does_association_change_the_recall_on_server(get_seeds=True)

    net, normalization_clocks = get_network_for_investigation(
        seed=all_seeds[0], normalization_clocks=normalization_clocks
    )

    all_recall_seeds = [0]
    all_deleted_neurons = [0, 10]
    all_recall_sizes = [20]

    list_of_all_assembly_ids_for_areas = [[[(0, 0, -1)]], [[(0, -1, 0)]]]

    (
        all_avg_firing_rates,
        all_avg_firing_rates_end_of_imprint,
        all_n_active_neurons,
        all_n_active_neurons_end_of_imprint,
    ) = run_multilayer_recall_and_get_results(
        net=net,
        all_network_seeds=all_seeds,
        list_of_all_assembly_ids_for_areas=list_of_all_assembly_ids_for_areas,
        all_recall_seeds=all_recall_seeds,
        all_deleted_neurons=all_deleted_neurons,
        all_recall_sizes=all_recall_sizes,
        change_firing_rate=True,
        normalize_results=normalize_results,
        only_load_results=only_load_results,
    )

    highlight_seed_id = all_seeds.index(highlight_point_for[0])
    for a_id, aa in enumerate(list_of_all_assembly_ids_for_areas):
        if np.all(np.array(aa) == highlight_point_for[1]):
            highlight_all_assembly_ids_for_areas_id = a_id

    if axes is None:
        fig, axes = plt.subplots(1, 2)

    xticks = []
    xticklabels = []

    all_area_names = ["Y", "Z"]
    assembly_background_name_list = ["assembly", "bck"]
    recall_export_keys = []
    for area_name in all_area_names:
        for assembly_background_name in assembly_background_name_list:
            for deleted_neurons in all_deleted_neurons:
                recall_export_keys.append(
                    f"{area_name}_{assembly_background_name}_{deleted_neurons}_silenced"
                )

    recall_export = [{}, {}]
    for ii in range(2):
        for key in recall_export_keys:
            recall_export[ii][key] = []

    for n_id, (ax, res, ylabel) in enumerate(
        zip(
            axes,
            [all_avg_firing_rates, all_n_active_neurons],
            ["avg. firing rate of assembly", "n active neurons of assembly"],
        )
    ):
        for area_id, area_name in enumerate(all_area_names):
            for deleted_neurons in all_deleted_neurons:
                for assembly_background, assembly_background_name in enumerate(
                    assembly_background_name_list
                ):
                    firing_rates_full_recall = np.nanmean(
                        res[
                            :,
                            :,
                            area_id,
                            :,  # take mean
                            all_deleted_neurons.index(
                                deleted_neurons
                            ),  # we are at full recall
                            all_recall_sizes.index(20),
                            assembly_background,  # we look at the assembly neurons
                        ],
                        axis=2,
                    ).flatten()

                    for e_s_id, e_seed_id in enumerate(all_seeds):
                        for l_id, _ in enumerate(list_of_all_assembly_ids_for_areas):
                            recall_export[n_id][
                                f"{area_name}_{assembly_background_name}_{deleted_neurons}_silenced"
                            ].append(
                                [
                                    e_seed_id,
                                    l_id,
                                    np.nanmean(
                                        res[
                                            e_s_id,
                                            l_id,
                                            area_id,
                                            :,
                                            all_deleted_neurons.index(deleted_neurons),
                                            all_recall_sizes.index(20),
                                            assembly_background,
                                        ]
                                    ),
                                ]
                            )

                    initial_xval = (
                        area_id + deleted_neurons * 0.25 - 0.5 * assembly_background
                    )
                    xticks.append(initial_xval)

                    x_val = (
                        initial_xval
                        + np.random.rand(len(firing_rates_full_recall)) * 0.3
                        - 0.15
                    )
                    y_val = firing_rates_full_recall

                    point_id = (
                        highlight_seed_id * len(list_of_all_assembly_ids_for_areas)
                        + highlight_all_assembly_ids_for_areas_id
                    )
                    highlight_point = (x_val[point_id], y_val[point_id])

                    if assembly_background == 1:
                        ax.text(
                            initial_xval + 0.25, -0.25, f"area {area_name}", ha="center"
                        )

                        if area_id == 0:
                            ax.text(
                                initial_xval + 0.25 + 0.5,
                                -0.35,
                                f"{deleted_neurons} deleted",
                                ha="center",
                            )

                    xticklabels.append(f"{assembly_background_name}")
                    np.random.seed(12)
                    ax.scatter(
                        x_val,
                        y_val,
                    )

                    ax.annotate(
                        "",
                        xy=highlight_point,
                        xytext=(10, 10),
                        textcoords="offset points",
                        arrowprops=dict(facecolor="black", shrink=0.05),
                    )

        ax.set(xticks=xticks, xticklabels=xticklabels, ylabel=ylabel)
    if show_plot:
        plt.show()

    for ii, save_name in enumerate(["avg_fr", "n_active"]):
        for key in recall_export_keys:
            values = np.array(recall_export[ii][key])
            path = get_path_to_save_file_name("Fig_7", "B_" + save_name + "_" + key)
            np.savetxt(path, values)

    return normalization_clocks


def multi_layer_recall(
    net,
    network_seed,
    only_load_results=True,
    all_assembly_ids_for_areas=[[(0, 0, -1)]],
    all_recall_seeds=[0],
    all_deleted_neurons=[0, 10],
    all_recall_sizes=[20],
    #
    change_firing_rate=True,
):
    net.only_load_results = only_load_results
    net.parameters_for_run["seed"] = network_seed

    net.parameters_for_run["all_assembly_ids_for_areas"] = all_assembly_ids_for_areas

    runtime_recall = 2 * second
    bsl = net.parameters_for_run["runtime_baseline"] / msecond
    rtm = net.parameters_for_run["runtime_imprint"] / msecond
    rtm_recall = runtime_recall / msecond

    avg_firing_rates = np.zeros(
        (
            len(net.all_areas),
            len(all_recall_seeds),
            len(all_deleted_neurons),
            len(all_recall_sizes),
            2,
        )
    ) * float("nan")
    n_active_neurons = np.zeros_like(avg_firing_rates) * float("nan")

    avg_firing_rates_end_of_imprint = np.zeros(
        (
            len(net.all_areas),
            2,
        )
    ) * float("nan")
    n_active_neurons_end_of_imprint = np.zeros_like(
        avg_firing_rates_end_of_imprint
    ) * float("nan")

    net.run_imprint(report_style="text", report_period=900 * second)
    network_filename = (
        net.save_dict["filename_for_stored_network"].decode("utf-8") + "_0"
    )
    net.network.restore(
        filename=net.get_path_to_stored_networks(file_name=network_filename)
    )

    sorted_neuron_ids = []
    selected_ids = []

    if not net.save_dict:
        return (
            avg_firing_rates,
            n_active_neurons,
            avg_firing_rates_end_of_imprint,
            n_active_neurons_end_of_imprint,
        )

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

        avg_firing_rates_end_of_imprint[area_id, 0] = avg_fr_after_imprint
        avg_firing_rates_end_of_imprint[area_id, 1] = (
            avg_firing_rate_not_in_assembly_after_imprint
        )
        n_active_neurons_end_of_imprint[area_id, 0] = n_act_n_after_imprint
        n_active_neurons_end_of_imprint[area_id, 1] = (
            n_active_neurons_not_in_assembly_after_imprint
        )

    for seed_id, recall_seed in enumerate(all_recall_seeds):
        for delete_id, remove_n_neurons_from_first_assembly in enumerate(
            all_deleted_neurons
        ):
            x_values = []
            for active_id, recall_size in enumerate(all_recall_sizes):
                run_recall_after_imprint = True

                np.random.seed(recall_seed)
                if remove_n_neurons_from_first_assembly > 0:
                    neurons_to_silence = np.random.choice(
                        selected_ids[0],
                        remove_n_neurons_from_first_assembly,
                        replace=False,
                    ).tolist()
                else:
                    neurons_to_silence = []

                net.parameters_for_run.update(
                    {
                        "silence_neurons_with_ids_for_recall": [
                            [0] + neurons_to_silence
                        ],  # first part is the area
                        "all_assembly_ids_for_areas_recall": all_assembly_ids_for_areas,
                        "all_context_ids_for_areas_recall": [[(0, 0)]],
                        "runtime_baseline_recall": 0.1 * second,
                        "runtime_recall": runtime_recall,
                        "run_recall_after_imprint": run_recall_after_imprint,
                        "recall_after_imprint_id": 0,
                        "assembly_neuron_selection_seed_recall": recall_seed,
                    }
                )

                if change_firing_rate:
                    net.parameters_for_run.update(
                        {
                            "assembly_firing_rate_recall": (
                                net.parameters["assembly_firing_rate"] * recall_size
                            )
                            / net.parameters["assembly_size"]
                        }
                    )
                else:
                    net.parameters_for_run.update({"assembly_size_recall": recall_size})

                if change_firing_rate:
                    x_values.append(
                        net.parameters_for_run["assembly_firing_rate_recall"]
                    )
                else:
                    x_values.append(net.parameters_for_run["assembly_size_recall"])

                net.run_recall(report_style="text")

                if not net.save_dict:
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

                    avg_firing_rates[area_id, seed_id, delete_id, active_id, 0] = avg_fr
                    avg_firing_rates[area_id, seed_id, delete_id, active_id, 1] = (
                        avg_firing_rate_not_in_assembly
                    )

                    n_active_neurons[area_id, seed_id, delete_id, active_id, 0] = (
                        n_act_n
                    )
                    n_active_neurons[area_id, seed_id, delete_id, active_id, 1] = (
                        n_active_neurons_not_in_assembly
                    )

    return [avg_firing_rates, n_active_neurons], [
        avg_firing_rates_end_of_imprint,
        n_active_neurons_end_of_imprint,
    ]


def get_network_for_investigation(seed=5, normalization_clocks=None):
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
        save_file_name="data_Fig_7",
        parameter_dict=parameter_dict,
        only_load_results=False,
        normalization_clocks=normalization_clocks,
        figure_name="Fig_7",
    )

    if normalization_clocks is None:
        normalization_clocks = [area.normalization.clock for area in net.all_areas]

    return net, normalization_clocks


def get_simulated_network(
    net, filename_for_stored_network=None, all_assembly_ids_for_areas=None
):
    net.parameters_for_run["all_assembly_ids_for_areas"] = all_assembly_ids_for_areas
    if "restore_from_save_name" in net.parameters_for_run:
        del net.parameters_for_run["restore_from_save_name"]
    if filename_for_stored_network is not None:
        net.parameters_for_run["restore_from_save_name"] = filename_for_stored_network
    save_dict = net.run_imprint(report_style="text", report_period=900 * second)
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
        current_vals_a = np.copy(
            net.parameters_for_run["all_assembly_ids_for_areas"]
        ).tolist()
        current_vals_b = np.copy(
            net.parameters_for_run["all_context_ids_for_areas"]
        ).tolist()

        net.parameters_for_run["all_assembly_ids_for_areas"] = [
            [] for _ in range(imprint_id + 1)
        ]
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
                "all_context_ids_for_areas_recall": [
                    [(0, 0)]
                ],  # sets all contexts to 0
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

        net.run_recall(report_style="text")

        if not net.save_dict:
            continue

        start_time = end_time_after_imprint + bsl
        end_time = start_time + rtm_recall

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
            filename=net.get_path_to_stored_networks(
                file_name=filename_for_stored_network
            )
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
        all_relevant_parameters = [
            seed,
            order_id,
            area_id,
            firing_rate_or_n_active_neurons,
        ]

    if result_type == "dendrite_distributions":
        all_relevant_parameters = [seed, input_id, area_id, order_id, imprint_id]

    if result_type == "selected_assembly_ids":
        all_relevant_parameters = [seed, area_id, order_id, imprint_id]

    key = result_type
    for par in all_relevant_parameters:
        if par is None:
            raise ValueError(
                "You should provide information for all relevant paramters"
            )

        key += f"{par}"

    return key


def how_does_association_change_the_recall(
    net=None,
    seed=5,
    change_firing_rate=True,
    only_load_results=False,
    only_run_imprint=False,
    result_dict=None,
    normalization_clocks=None,
):
    if net is None:
        net, normalization_clocks = get_network_for_investigation(
            seed=seed, normalization_clocks=normalization_clocks
        )
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

    return normalization_clocks


def look_at_dendrite_statistics_within_assembly(
    net, selected_ids_A, selected_ids_B, order_id
):
    subset_ids_for_a1 = []
    subset_ids_for_a2 = []
    subset_ids_for_b2 = []

    for assembly_ids in net.parameters_for_run["all_assembly_ids_for_areas"][0]:
        if assembly_ids[0] == 0 and assembly_ids[1] >= 0:
            subset_ids_for_a1 = [ii + (assembly_ids[1] * 20) for ii in range(20)]
        if assembly_ids[0] == 0 and assembly_ids[2] >= 0:
            subset_ids_for_a2 = [ii + (assembly_ids[2] * 20) for ii in range(20)]
        if assembly_ids[0] == 1 and assembly_ids[2] >= 0:
            subset_ids_for_b2 = [ii + (assembly_ids[2] * 20) for ii in range(20)]

    input_subsets = [
        [subset_ids_for_a1, subset_ids_for_a2],
        [selected_ids_A, subset_ids_for_b2],
    ]

    n_of_inputs_that_target_selected_ids = [[[], []], [[], []]]
    targeted_neurons = [selected_ids_A, selected_ids_B]

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
                        input_neurons_that_target_dendrite = area.input_synapses[
                            input_id
                        ].i[syn_ids_that_target_dendrite]

                        n_of_inputs_that_target_selected_ids[area_id][input_id].append(
                            0
                        )
                        for n_id in input_neurons_that_target_dendrite:
                            if n_id in input_subsets[area_id][input_id]:
                                n_of_inputs_that_target_selected_ids[area_id][input_id][
                                    -1
                                ] += 1

    return n_of_inputs_that_target_selected_ids


def run_how_does_association_change_the_recall_on_server(
    max_cores=2, get_seeds=False, case_id=0
):
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

    change_firing_rate = True
    only_load_results = False
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


if __name__ == "__main__":
    Fig_7(only_load_results=True)
