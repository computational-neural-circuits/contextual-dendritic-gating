import brian2 as br
from brian2.units import *
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import community as community_louvain

from src.handle_parameters_and_results import HandleParametersAndResults
from src.area import Area
from src.utils import get_firing_rate_for_single_neuron, get_assembly_neuron_ids_by_weight_and_rate

import scipy
import os


class NetworkTask(HandleParametersAndResults):
    def __init__(
        self,
        parameter_file_name,
        save_file_name,
        parameters_for_run={},
        parameter_dict={},
        equation_file_name="equations",
        setup_new_task=False,
        **kwargs,
    ):
        super().__init__(
            parameter_dict=parameter_dict,
            parameter_file_name=parameter_file_name,
            save_file_name=save_file_name,
            equation_file_name=equation_file_name,
            parameters_for_run=parameters_for_run,
            **kwargs,
        )

        self.setup_new_task = setup_new_task

        if self.create_network:
            self.network = br.Network()
            self.setup_network()
            self.create_monitors()

    def setup_network(self):
        print("Setup the network for the task")
        br.seed(self.parameters_for_run["seed"])
        np.random.seed(self.parameters_for_run["seed"])

        self.all_areas = []
        all_area_names = ["A", "B", "C"]
        if self.setup_new_task:
            all_area_names = ["A", "B", "C", "D"]
        for name in all_area_names:
            inputs_1 = None
            inputs_2 = None
            if name == "C":
                inputs_1 = self.all_areas[0].somas
                inputs_2 = self.all_areas[1].somas
            if name == "D":
                inputs_1 = self.all_areas[2].somas
                inputs_2 = self.all_areas[1].input_units_1

            self.all_areas.append(
                Area(
                    network=self.network,
                    eqs=self.equations,
                    input_units_1=inputs_1,
                    input_units_2=inputs_2,
                    params={**self.parameters, **self.parameters_for_run},
                    name=name,
                )
            )

    def create_monitors(self):
        self.spM_somas = []
        self.spM_inputs = []
        for ii, area in enumerate(self.all_areas):
            self.spM_somas.append(br.SpikeMonitor(area.somas, name=f"somata_monitor_{area.name}"))
            self.network.add(self.spM_somas[-1])
            if ii < 2:
                spM_inputs = []
                for iu, input_units in enumerate([area.input_units_1, area.input_units_2]):
                    spm = br.SpikeMonitor(input_units, name=f"input_monitor_{iu}_{area.name}")
                    spM_inputs.append(spm)
                    self.network.add(spm)
                self.spM_inputs.append(spM_inputs)

    def create_save_dict(self):
        self.save_dict = {}

        for nn, area in enumerate(self.all_areas):
            area_name = area.name

            weights_recurrent = np.zeros(shape=(area.n_somas, area.n_dends))
            weights_recurrent[area.srcs, area.tgts] = area.synapses_E.w

            additional_dict = {
                f"{area_name}_spikes_somas_t": self.spM_somas[nn].t / ms,
                f"{area_name}_spikes_somas_i": self.spM_somas[nn].i,
                f"{area_name}_weights": weights_recurrent,
            }
            self.save_dict = {**self.save_dict, **additional_dict}
            if nn < 2:
                additional_dict = {
                    f"{area_name}_spikes_inputs_t_1": self.spM_inputs[nn][0].t / ms,
                    f"{area_name}_spikes_inputs_t_2": self.spM_inputs[nn][1].t / ms,
                    f"{area_name}_spikes_inputs_i_1": self.spM_inputs[nn][0].i,
                    f"{area_name}_spikes_inputs_i_2": self.spM_inputs[nn][1].i,
                }
                self.save_dict = {**self.save_dict, **additional_dict}

    def set_network_state(
        self, all_assembly_neuron_ids=None, all_assembly_inputs=None, net_state_id=0, set_bck=False
    ):
        if set_bck:
            for ii, area in enumerate(self.all_areas):
                area.start_context(0)  # does not matter which one here its just for baseline

                if ii < 2:
                    area.input_units_1[:].rates = self.parameters["ff_bck"]
                    area.input_units_2[:].rates = self.parameters["ff_bck"]
            return

        context_1, assembly_id_1, context_2, assembly_id_2, context_3 = net_state_id
        self.all_areas[0].start_context(context_1)
        self.all_areas[1].start_context(context_2)
        self.all_areas[2].start_context(context_3)

        if self.setup_new_task:
            self.all_areas[1].start_context(0)
            self.all_areas[3].start_context(0)

        if assembly_id_1 >= 0:
            if all_assembly_inputs is None:
                for neuron_id in all_assembly_neuron_ids[assembly_id_1]:
                    self.all_areas[0].input_units_1[
                        neuron_id : neuron_id + 1
                    ].rates = self.parameters["assembly_firing_rate"]
            else:
                print(len(all_assembly_inputs), assembly_id_1)
                self.all_areas[0].input_units_1[:].rates = all_assembly_inputs[assembly_id_1] * Hz
        if assembly_id_2 >= 0:
            for neuron_id in all_assembly_neuron_ids[assembly_id_2]:
                self.all_areas[1].input_units_1[neuron_id : neuron_id + 1].rates = self.parameters[
                    "assembly_firing_rate"
                ]

    def run_imprint(
        self,
        report_style=None,
        report_period=10 * second,
        all_assembly_inputs=None,
        ignore_all_keys_with_keywords=["recall"],
    ):
        filename_for_stored_network = f"stored_imprint_{self.get_unique_paramter_and_equation_key(ignore_all_keys_with_keywords=['recall'])}"
        path = self.get_path_to_stored_networks(file_name=filename_for_stored_network + f"_-1")

        if not os.path.isfile(path):
            self.network.store(filename=path)

        runtime_baseline = self.parameters_for_run["runtime_baseline"]
        runtime_imprint = self.parameters_for_run["runtime_imprint"]
        all_imprint_ids = self.parameters_for_run["all_imprint_ids"]

        all_assembly_neuron_ids = self.parameters_for_run["all_assembly_neuron_ids"]

        if self.check_for_results(ignore_all_keys_with_keywords=ignore_all_keys_with_keywords):
            return self.save_dict
        elif self.only_load_results:
            return None

        self.set_network_state(set_bck=True)
        self.network.run(runtime_baseline, report=report_style, report_period=report_period)

        for ii, net_state_id in enumerate(all_imprint_ids):
            print(f"IMPRINT {ii}")
            self.set_network_state(
                all_assembly_neuron_ids=all_assembly_neuron_ids,
                all_assembly_inputs=all_assembly_inputs,
                net_state_id=net_state_id,
            )
            self.network.run(
                runtime_imprint,
                report=report_style,
                report_period=report_period,
            )

            self.set_network_state(set_bck=True)
            self.network.run(runtime_baseline, report=report_style, report_period=report_period)

            self.create_save_dict()
            self.save_results()

        self.create_save_dict()
        filename_for_stored_network = f"stored_imprint_{self.get_unique_paramter_and_equation_key(ignore_all_keys_with_keywords=['recall'])}"

        self.save_dict["filename_for_stored_network"] = filename_for_stored_network
        self.network.store(
            filename=self.get_path_to_stored_networks(file_name=filename_for_stored_network)
        )
        self.save_results(ignore_all_keys_with_keywords=["recall"])

    def run_recall(self, report_style=None, report_period=10 * second, recall_inputs=None):
        runtime_recall = self.parameters_for_run["runtime_recall"]
        recall_id = self.parameters_for_run["recall_id"]
        run_after_imprint = self.parameters_for_run["run_recall_after_imprint"]
        runtime_baseline_recall = self.parameters_for_run["runtime_baseline_recall"]

        all_assembly_neuron_ids = self.parameters_for_run["all_assembly_neuron_ids"]

        if recall_inputs is None:
            all_assembly_inputs = None
        else:
            all_assembly_inputs = [recall_inputs]

        if self.check_for_results():
            return self.save_dict

        elif self.only_load_results:
            return None

        self.only_load_results = True
        self.run_imprint()
        self.only_load_results = False
        print(self.save_dict["filename_for_stored_network"])

        filename_for_stored_network = self.save_dict["filename_for_stored_network"]
        if not type(filename_for_stored_network) is str:
            filename_for_stored_network = filename_for_stored_network.decode("utf-8")

        if not run_after_imprint:
            filename_for_stored_network += "_-1"

        self.network.restore(
            filename=self.get_path_to_stored_networks(file_name=filename_for_stored_network)
        )

        print(f"START RECALL")

        self.set_network_state(set_bck=True)
        self.set_network_state(
            all_assembly_neuron_ids=all_assembly_neuron_ids,
            all_assembly_inputs=all_assembly_inputs,
            net_state_id=recall_id,
        )
        self.network.run(
            runtime_recall,
            report=report_style,
            report_period=report_period,
        )
        self.set_network_state(set_bck=True)
        self.network.run(runtime_baseline_recall, report=report_style, report_period=report_period)

        self.create_save_dict()
        self.save_results()

    def sort_neurons_by_weights(self, area, context_id=0):
        weights_loaded = self.save_dict[f"{area.name}_weights"]
        weights = weights_loaded[:, context_id::6]
        G = nx.from_numpy_array(weights, create_using=nx.DiGraph)
        partition = community_louvain.best_partition(G.to_undirected(), weight="weight")
        node_community_map = {node: community for node, community in enumerate(partition.values())}
        sorted_nodes = sorted(node_community_map, key=node_community_map.get)
        W_reordered = weights[np.ix_(sorted_nodes, sorted_nodes)]

        fig, ax = plt.subplots()
        ax.imshow(W_reordered)
        # plt.show()
        return sorted_nodes

    def sort_neurons_by_firing_rate(
        self,
        area,
        use_input_units=False,
        sort_for_specific_imprint=None,
        t_0=0,
        context_id=0,
    ):
        if use_input_units == 0:
            somas_time = self.save_dict[f"{area.name}_spikes_somas_t"]
            somas_i = self.save_dict[f"{area.name}_spikes_somas_i"]
        elif use_input_units == 1:
            somas_time = self.save_dict[f"{area.name}_spikes_inputs_t_1"]
            somas_i = self.save_dict[f"{area.name}_spikes_inputs_i_1"]
        else:
            somas_time = self.save_dict[f"{area.name}_spikes_inputs_t_2"]
            somas_i = self.save_dict[f"{area.name}_spikes_inputs_i_2"]

        all_firing_rates_somas = []
        highest_firing_rate_somas = []
        for neuron_index in range(self.parameters["n_somas"]):
            spike_times_for_neuron = somas_time[somas_i == neuron_index]

            bsl = self.parameters_for_run["runtime_baseline"] / msecond
            rtm = self.parameters_for_run["runtime_imprint"] / msecond
            neuron_firing_rate = []

            for tt in range(len(self.parameters_for_run["all_imprint_ids"])):
                start = bsl + (rtm + bsl) * tt + t_0
                end = start + rtm

                firing_rate = get_firing_rate_for_single_neuron(
                    start=start, end=end, spike_times_for_neuron=spike_times_for_neuron
                )
                neuron_firing_rate.append(firing_rate)
            all_firing_rates_somas.append(neuron_firing_rate)
            highest_firing_rate_somas.append(np.max(neuron_firing_rate))
        all_firing_rates_somas = np.array(all_firing_rates_somas)

        sorted_neuron_ids = []

        area_names = [aa.name for aa in self.all_areas]
        for tt, imprint_ids in enumerate(self.parameters_for_run["all_imprint_ids"]):
            if sort_for_specific_imprint is not None:
                if tt != sort_for_specific_imprint:
                    continue
                else:
                    context_id = imprint_ids[area_names.index(area.name) * 2]

            all_rates = all_firing_rates_somas[:, tt]
            selected_ids = get_assembly_neuron_ids_by_weight_and_rate(
                net=self,
                all_rates=all_rates,
                context_id=context_id,
                area=area,
            )

            # we only add those ids that are not yet part of the sorted ids
            new_selected_ids = [si for si in selected_ids if si not in sorted_neuron_ids]

            sorted_neuron_ids += new_selected_ids

        sorted_neuron_ids += [
            ni for ni in range(self.parameters["n_somas"]) if ni not in sorted_neuron_ids
        ]

        return sorted_neuron_ids, all_firing_rates_somas[sorted_neuron_ids, 0], selected_ids

    def show_spike_rasters(
        self,
        show_plot=False,
        axes=None,
        sort_by_new_algorithm=False,
        context_id=0,
        sorted_ids=None,
        highlight_neuron_ids=None,
        highlight_neuron_ids_between_specific_time_points=None,
        show_vertical_lines_at=None,
        save_file_name=None,
        only_show_spikes_after=None,
    ):
        for aa, area in enumerate(self.all_areas):
            somas_time = self.save_dict[f"{area.name}_spikes_somas_t"]
            somas_i = self.save_dict[f"{area.name}_spikes_somas_i"]
            if aa < 2:
                inputs_1_time = self.save_dict[f"{area.name}_spikes_inputs_t_1"]
                inputs_1_i = self.save_dict[f"{area.name}_spikes_inputs_i_1"]
                inputs_2_time = self.save_dict[f"{area.name}_spikes_inputs_t_2"]
                inputs_2_i = self.save_dict[f"{area.name}_spikes_inputs_i_2"]
            elif aa == 2:
                inputs_1_time = self.save_dict[f"A_spikes_somas_t"]
                inputs_1_i = self.save_dict[f"A_spikes_somas_i"]
                inputs_2_time = self.save_dict[f"B_spikes_somas_t"]
                inputs_2_i = self.save_dict[f"B_spikes_somas_i"]
            else:
                nputs_1_time = self.save_dict[f"C_spikes_somas_t"]
                inputs_1_i = self.save_dict[f"C_spikes_somas_i"]
                inputs_1_time = self.save_dict[f"B_spikes_inputs_t_1"]
                inputs_1_i = self.save_dict[f"B_spikes_inputs_i_1"]

            if only_show_spikes_after is not None:
                somas_i = somas_i[somas_time >= only_show_spikes_after]
                somas_time = somas_time[somas_time >= only_show_spikes_after]
                inputs_1_i = inputs_1_i[inputs_1_time >= only_show_spikes_after]
                inputs_1_time = inputs_1_time[inputs_1_time >= only_show_spikes_after]
                inputs_2_i = inputs_2_i[inputs_2_time >= only_show_spikes_after]
                inputs_2_time = inputs_2_time[inputs_2_time >= only_show_spikes_after]

            if axes is None:
                fig, (ax_inputs_1, ax_inputs_2, ax_somas) = plt.subplots(
                    3, sharex=True, figsize=(14, 11)
                )
            else:
                (ax_inputs, ax_somas) = axes[aa]

            if sorted_ids is None:
                if sort_by_new_algorithm:
                    sorted_neuron_ids = self.sort_neurons_by_weights(
                        area=area, context_id=context_id
                    )
                else:
                    area_names = [aa.name for aa in self.all_areas]
                    last_imprint_ids = self.parameters_for_run["all_imprint_ids"][-1]
                    context_id = ontext_id = last_imprint_ids[area_names.index(area.name) * 2]
                    sorted_neuron_ids, _, _ = self.sort_neurons_by_firing_rate(area=area)
            else:
                sorted_neuron_ids = sorted_ids[aa]

            for neuron_index in range(self.parameters["n_somas"]):
                spike_times_for_neuron = inputs_1_time[inputs_1_i == neuron_index]
                ax_inputs_1.vlines(
                    spike_times_for_neuron,
                    ymin=neuron_index - 0.5,
                    ymax=neuron_index + 0.5,
                    colors="k",
                )
                spike_times_for_neuron = inputs_2_time[inputs_2_i == neuron_index]
                ax_inputs_2.vlines(
                    spike_times_for_neuron,
                    ymin=neuron_index - 0.5,
                    ymax=neuron_index + 0.5,
                    colors="k",
                )

            for ii, neuron_index in enumerate(sorted_neuron_ids):
                spike_times_for_neuron = somas_time[somas_i == neuron_index]
                ax_somas.vlines(spike_times_for_neuron, ymin=ii - 0.5, ymax=ii + 0.5, colors="k")

            ax_inputs_1.set_title(f"Inputs (1) for {area.name}")
            ax_inputs_2.set_title(f"Inputs (2) for {area.name}")
            ax_somas.set_title(f"{area.name}")
            ax_somas.set(xlabel="Time in ms", ylim=ax_inputs_1.get_ylim())

            for ax in [ax_somas, ax_inputs_1, ax_inputs_2]:
                ax.set(ylabel="Neuron Number")

            if highlight_neuron_ids is not None:
                bsl = self.parameters_for_run["runtime_baseline"] / msecond
                rtm = self.parameters_for_run["runtime_imprint"] / msecond

                if highlight_neuron_ids_between_specific_time_points is None:
                    highlight_start = 2 * bsl + rtm
                    highlight_end = (
                        2 * bsl + rtm + self.parameters_for_run["runtime_recall"] / msecond
                    )
                else:
                    (
                        highlight_start,
                        highlight_end,
                    ) = highlight_neuron_ids_between_specific_time_points

                for area_id, neuron_ids in highlight_neuron_ids:
                    if area_id == aa:
                        for n_id in neuron_ids:
                            position = list(sorted_neuron_ids).index(n_id)
                            ax_somas.fill_between(
                                [
                                    highlight_start,
                                    highlight_end,
                                ],
                                position - 0.5,
                                position + 0.5,
                                color="#a50f15",
                                alpha=0.5,
                            )

            if show_vertical_lines_at is not None:
                colors = ["#238b45", "#cb181d", "#225ea8", "#54278f"]
                for ax in [ax_inputs_1, ax_inputs_2, ax_somas]:
                    for ii, xx in enumerate(show_vertical_lines_at):
                        ax.axvline(x=xx, color=colors[ii % len(colors)], label=f"{ii}")

                    ax.legend()

            if save_file_name is not None:
                fig.savefig(save_file_name + f"_area_{area.name}.pdf", dpi=600)

        if show_plot and save_file_name is None:
            plt.show()

    def show_end_of_imprint_vs_end_of_recall_in_spike_raster(self, show_plot=False):
        fig, ax = plt.subplots()

        area = self.all_areas[-1]

        somas_time = self.save_dict[f"{area.name}_spikes_somas_t"]
        somas_i = self.save_dict[f"{area.name}_spikes_somas_i"]

        # First we need to get the sorted neurons of each assembly
        sorted_neuron_ids, _, _ = self.sort_neurons_by_firing_rate(area=area)

        all_rates_before = [[] for _ in self.parameters_for_run["all_imprint_ids"]]
        all_rates_end = [[] for _ in self.parameters_for_run["all_imprint_ids"]]
        all_rates_recall = [[] for _ in self.parameters_for_run["all_imprint_ids"]]

        for ii, neuron_index in enumerate(sorted_neuron_ids):
            spike_times_for_neuron = somas_time[somas_i == neuron_index]

            bsl = self.parameters_for_run["runtime_baseline"] / msecond
            rtm = self.parameters_for_run["runtime_imprint"] / msecond
            rcl = self.parameters_for_run["runtime_recall"] / msecond

            for tt in range(len(self.parameters_for_run["all_imprint_ids"])):
                start_imprint = bsl + (rtm + bsl) * tt
                end_imprint = start_imprint + rtm

                start_recall = (
                    bsl
                    + (rtm + bsl) * len(self.parameters_for_run["all_imprint_ids"])
                    + (rcl + bsl) * tt
                )
                end_recall = start_recall + rcl

                previous_time = tt * 3 * rcl + 0.5

                spike_times_start_of_imprint = spike_times_for_neuron[
                    np.logical_and(
                        spike_times_for_neuron > start_imprint,
                        spike_times_for_neuron < start_imprint + rcl,
                    )
                ]

                spike_times_start_of_imprint += previous_time - start_imprint

                spike_times_end_of_imprint = spike_times_for_neuron[
                    np.logical_and(
                        spike_times_for_neuron > end_imprint - rcl,
                        spike_times_for_neuron < end_imprint,
                    )
                ]

                spike_times_end_of_imprint += previous_time - (end_imprint - rcl) + 0.1 + rcl

                spike_times_recall = spike_times_for_neuron[
                    np.logical_and(
                        spike_times_for_neuron > start_recall,
                        spike_times_for_neuron < end_recall,
                    )
                ]

                spike_times_recall += previous_time - start_recall + 2 * (0.1 + rcl)

                ax.vlines(spike_times_start_of_imprint, ymin=ii - 0.5, ymax=ii + 0.5, colors="r")
                ax.vlines(spike_times_end_of_imprint, ymin=ii - 0.5, ymax=ii + 0.5, colors="g")
                ax.vlines(spike_times_recall, ymin=ii - 0.5, ymax=ii + 0.5, colors="b")

                all_rates_before[tt].append(len(spike_times_start_of_imprint) / (rcl / 1000.0))
                all_rates_end[tt].append(len(spike_times_end_of_imprint) / (rcl / 1000.0))
                all_rates_recall[tt].append(len(spike_times_recall) / (rcl / 1000.0))

        avg_rate_end_of_learning = []
        avg_rate_before = []
        avg_rate_recall = []
        avg_rate_bck_recall = []
        avg_rate_bck_end_of_imprint = []
        avg_rate_bck_before = []

        for tt in range(len(self.parameters_for_run["all_imprint_ids"])):
            all_rates = np.array(all_rates_end[tt])

            sorted_rates = np.sort(all_rates)
            sorted_rates_ids = np.argsort(all_rates)
            threshold = np.mean(sorted_rates[-8:]) / 3.0  # 8 is chosen randomly here

            # print(sorted_rates.shape)
            cutoff_id = np.searchsorted(sorted_rates, threshold)

            selected_ids = list(sorted_rates_ids[cutoff_id:])

            mean_rate_end = np.mean(np.array(all_rates_end[tt])[selected_ids])
            mean_rate_before = np.mean(np.array(all_rates_before[tt])[selected_ids])
            mean_rate_recall = np.mean(np.array(all_rates_recall[tt])[selected_ids])

            avg_rate_end_of_learning.append(mean_rate_end)
            avg_rate_recall.append(mean_rate_recall)
            avg_rate_before.append(mean_rate_before)

            print(
                f"rate end of imprint: {np.round(mean_rate_end,2)} Hz | before: {np.round(mean_rate_before,2)} Hz | recall: {np.round(mean_rate_recall,2)}"
            )

            # now we take the same number of neurons but randomly selected from the rest of the cells

            print(len(selected_ids))

            label_name = "bck"
            use_second_highest_as_comparison = False
            if use_second_highest_as_comparison:
                label_name = "next highest"
                selected_ids = list(sorted_rates_ids[cutoff_id - len(selected_ids) : cutoff_id])
            else:
                selected_ids = list(sorted_rates_ids[: -len(selected_ids)])

            mean_rate_end = np.mean(np.array(all_rates_end[tt])[selected_ids])
            mean_rate_before = np.mean(np.array(all_rates_before[tt])[selected_ids])
            mean_rate_recall = np.mean(np.array(all_rates_recall[tt])[selected_ids])

            avg_rate_bck_recall.append(mean_rate_recall)
            avg_rate_bck_end_of_imprint.append(mean_rate_end)
            avg_rate_bck_before.append(mean_rate_before)

            print(
                f"{label_name} --  rate end of imprint: {np.round(mean_rate_end,2)} Hz | before: {np.round(mean_rate_before,2)} Hz | recall: {np.round(mean_rate_recall,2)}"
            )

            # print(len(selected_ids))

        fig, ax = plt.subplots()
        ax.bar(
            [0, 7, 14, 21],
            avg_rate_before,
            color="#1f78b4",
            label="assembly neurons - before imprint",
        )
        ax.bar(
            np.array([1, 8, 15, 22]) - 0.5,
            avg_rate_bck_before,
            color="#a6cee3",
            label=f"{label_name} neurons - before imprint",
        )
        ax.bar(
            [2, 9, 16, 23],
            avg_rate_end_of_learning,
            color="#33a02c",
            label="assembly neurons - end of imprint",
        )
        ax.bar(
            np.array([3, 10, 17, 24]) - 0.5,
            avg_rate_bck_end_of_imprint,
            color="#b2df8a",
            label=f"{label_name} neurons - end of imprint",
        )
        ax.bar([4, 11, 18, 25], avg_rate_recall, color="#ff7f00", label="assembly neurons - recall")
        ax.bar(
            np.array([5, 12, 19, 26]) - 0.5,
            avg_rate_bck_recall,
            color="#fdbf6f",
            label=f"{label_name} neurons - recall",
        )

        # fb9a99
        # e31a1c

        ax.set(
            xticks=[2, 9, 16, 23],
            xticklabels=["0", "1", "O", "I"],
            ylabel="avg. firing rate",
            xlabel="Visual stimulus",
        )
        ax.legend()

        if show_plot:
            plt.show()

    def save_weight_matrix_for_matlab(self, show_plot=False, matlab_export_name=None):
        matlab_save_dict = {}

        for nn, area in enumerate(self.all_areas):
            for context_id in range(2):
                area_name = area.name

                weights = self.save_dict[f"{area_name}_weights"]
                matlab_save_dict[f"all_weights_area_{area_name}"] = weights.T
                weights = weights[:, context_id::6]
                matlab_save_dict[f"weights_area_{area_name}_of_context_{context_id}"] = weights.T

                sorted_neuron_ids, _, _ = self.sort_neurons_by_firing_rate(area=area)
                weights = weights[np.ix_(sorted_neuron_ids, sorted_neuron_ids)]
                matlab_save_dict[
                    f"weights_sorted_area_{area_name}_of_context_{context_id}"
                ] = weights.T

                if show_plot:
                    fig, ax = plt.subplots()
                    im = ax.imshow(weights.T, cmap="Greys")
                    ax.set(aspect=1 / 6.0)
                    plt.show()

        # scipy.io.savemat("task_weights.mat", mdict=save_dict)
        scipy.io.savemat(f"{matlab_export_name}.mat", mdict=matlab_save_dict)
        for key, val in matlab_save_dict.items():
            np.savetxt(f"{matlab_export_name}_{key}.txt", val)


if __name__ == "__main__":
    pass
