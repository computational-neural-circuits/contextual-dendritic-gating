from brian2.units import *
import numpy as np

from src.network_multiple_contexts_over_time_with_association import (
    NetworMultipleContextsOverTimeWithAssociation,
)
from src.get_activity_metrics import get_activity_metrics_from_assembly_neurons
from src.utils import get_path_to_save_file_name

import matplotlib.pyplot as plt

plt.style.use("../plots_style.txt")

parameter_dict = {}


def Fig_5(
    only_load_results=True,
    seed=927,
    order_id=0,
    use_same_context=False,
    case=None,
    include_recall=False,
):
    (
        fig,
        axes_for_weights_over_time,
        neuron_to_neuron_connectivity_ax,
        neuron_to_dendrite_connectivity_ax,
        recall_rates_ax,
        recall_neurons_ax,
    ) = create_figure_layout_Fig_5()

    ########################
    # weights over time (start)
    ######################

    parameters_for_run = {
        "runtime_imprint": 32 * second,  # 32
        "runtime_baseline": 2 * second,  # 2
        "seed": seed,  # 4928
        "all_assembly_ids": [
            (0, -1),
            (2, 2),
            (1992, -1),
            (1992, 1992),
            (0, -1),
            (2, 2),
        ],
        "all_context_ids": [0, 0, 1, 1, 2, 2],
        "monitor_dt_weights": 50 * ms,
    }

    parameters_for_run["all_assembly_ids"] = [
        (0, -1),
        (2, 2),
        (0, -1),
        (2, 2),
        (1992, -1),
        (1992, 1992),
    ]  # order 1

    parameters_for_run["all_context_ids"] = [0, 0, 2, 2, 1, 1]

    parameter_dict = {}

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

    if include_recall:
        parameters_for_run["save_network_after_each_imprint"] = True

    net = NetworMultipleContextsOverTimeWithAssociation(
        parameter_file_name="parameters",
        parameters_for_run=parameters_for_run,
        save_file_name="data_Fig_5",
        parameter_dict=parameter_dict,
        only_load_results=False,
        figure_name="Fig_5",
    )

    net.run(report_style="text")

    net.show_weight_matrix(
        matlab_export_name=get_path_to_save_file_name("Fig_5", "weights_5_B_and_C"),
        axes=[[None, neuron_to_neuron_connectivity_ax], [None, None], [None, None]],
        axes_for_dendrite_matrix=[neuron_to_dendrite_connectivity_ax, None, None, None],
    )

    neuron_to_neuron_connectivity_ax.set_title("Neuron to Neuron weights (Context 1)")
    neuron_to_dendrite_connectivity_ax.set_title(
        "All Neuron to Dendrite weights (sorted by Context 1)"
    )

    net.generate_results(axes=np.array(axes_for_weights_over_time), save_fig=False)

    ########################
    # weights over time (end)
    ######################

    all_recall_sizes = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

    # bck, avg_fr/n_active, after_imprint, assembly_id, recall_size
    recall_res = np.ones((2, 2, 2, 6, len(all_recall_sizes))) * float("nan")

    filename_for_stored_network = net.save_dict["filename_for_stored_network"]
    if not type(filename_for_stored_network) is str:
        filename_for_stored_network = filename_for_stored_network.decode("utf-8")
    filename_for_stored_network += f"_5"
    # Add recall

    for ii, (assembly_ids, context_id) in enumerate(
        zip(
            parameters_for_run["all_assembly_ids"],
            parameters_for_run["all_context_ids"],
        )
    ):

        net.network.restore(
            filename=net.get_path_to_stored_networks(
                file_name=filename_for_stored_network
            )
        )
        ids_of_the_assembly = net.get_assembly_neuron_ids(context_id, assembly_ids)

        for rs, recall_size in enumerate(all_recall_sizes):

            for run_directly_after_imprint in [True, False]:

                recall_after_imprint_id = ii

                if run_directly_after_imprint:
                    if recall_size != 20:
                        continue
                else:
                    recall_after_imprint_id = (
                        len(parameters_for_run["all_assembly_ids"]) - 1
                    )

                net.parameters_for_run.update(
                    {
                        "runtime_baseline_recall": 0 * second,
                        "runtime_recall": 2 * second,
                        "all_context_ids_for_recall": [context_id],
                        "all_assembly_ids_for_recall": [assembly_ids],
                        "run_recall_after_imprint": True,
                        "recall_after_imprint_id": recall_after_imprint_id,
                        "assembly_neuron_selection_seed_recall": 0,
                        "assembly_size_recall": recall_size,
                    }
                )

                net.run_recall()

                imp = parameters_for_run["runtime_imprint"]
                bsl = parameters_for_run["runtime_baseline"]
                start_time = (bsl + (1 + recall_after_imprint_id) * (bsl + imp)) / ms
                end_time = start_time + net.parameters_for_run["runtime_recall"] / ms

                (
                    avg_firing_rate_recall,
                    n_active_neurons_recall,
                    avg_firing_rate_recall_bck,
                    n_active_neurons_recall_bck,
                ) = get_activity_metrics_from_assembly_neurons(
                    active_threshold=4,
                    net=net,
                    area=None,
                    selected_ids=ids_of_the_assembly,
                    start_time=start_time,
                    end_time=end_time,
                    select_randomly_for_background=True,
                )

                # bck, avg_fr/n_active, after_imprint, assembly_id, recall_size
                recall_res[0, 0, int(run_directly_after_imprint), ii, rs] = (
                    avg_firing_rate_recall
                )
                recall_res[0, 1, int(run_directly_after_imprint), ii, rs] = (
                    n_active_neurons_recall
                )
                recall_res[1, 0, int(run_directly_after_imprint), ii, rs] = (
                    avg_firing_rate_recall_bck
                )
                recall_res[1, 1, int(run_directly_after_imprint), ii, rs] = (
                    n_active_neurons_recall_bck
                )

    colors = [
        ["#82C780", "#046937", "#4292c6", "#9ecae1", "#deebf7"],
        ["#BDAED5", "#67338E", "#41ab5d", "#a1d99b", "#e5f5e0"],
        ["#FCC085", "#B37129", "#fd8d3c", "#fdd0a2", "#fee6ce"],
    ]

    for ii in range(6):
        recall_rates_ax.plot(
            all_recall_sizes,
            recall_res[0, 0, 0, ii],
            label=f"Imprint # {ii} assembly neurons",
            color=colors[ii // 2][ii % 2],
        )
        recall_rates_ax.plot(
            all_recall_sizes,
            recall_res[1, 0, 0, ii],
            label=f"Imprint # {ii} bck",
            color=colors[ii // 2][ii % 2],
            ls="--",
        )

        recall_neurons_ax.plot(
            all_recall_sizes,
            recall_res[0, 1, 0, ii],
            label=f"Imprint # {ii} assembly neurons",
            color=colors[ii // 2][ii % 2],
        )
        recall_neurons_ax.plot(
            all_recall_sizes,
            recall_res[1, 1, 0, ii],
            label=f"Imprint # {ii} bck",
            color=colors[ii // 2][ii % 2],
            ls="--",
        )

    recall_rates_ax.set(xlabel="n active pre", ylabel="Avg firing rate")
    recall_neurons_ax.set(xlabel="n active pre", ylabel="n active")

    recall_rates_ax.legend()
    recall_neurons_ax.legend()

    fig.savefig(f"../results/figures/Fig_5.pdf", dpi=800)


def create_figure_layout_Fig_5():
    fig = plt.figure(figsize=(53, 48))
    gs = fig.add_gridspec(12, 9, hspace=1, wspace=1)

    axes_for_weights_over_time = np.array(
        [
            [
                fig.add_subplot(gs[ii * 2 : (ii + 1) * 2, 4 * jj : 4 + 4 * jj])
                for jj in range(2)
            ]
            for ii in range(3)
        ]
    )

    neuron_to_neuron_connectivity_ax = fig.add_subplot(gs[8:12, :3])
    neuron_to_dendrite_connectivity_ax = fig.add_subplot(gs[7:12, 3:5])
    recall_rates_ax = fig.add_subplot(gs[8:10, 6:9])
    recall_neurons_ax = fig.add_subplot(gs[10:12, 6:9])

    return (
        fig,
        axes_for_weights_over_time,
        neuron_to_neuron_connectivity_ax,
        neuron_to_dendrite_connectivity_ax,
        recall_rates_ax,
        recall_neurons_ax,
    )


if __name__ == "__main__":
    Fig_5(
        only_load_results=False,
        seed=111,
        order_id=1,
        use_same_context=False,
        case=3,
        include_recall=True,
    )
