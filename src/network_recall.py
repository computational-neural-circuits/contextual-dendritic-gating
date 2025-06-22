import brian2 as br
from brian2.units import *

import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import community as community_louvain

from src.handle_parameters_and_results import HandleParametersAndResults
from src.area import Area
from src.utils import get_firing_rate_for_single_neuron, get_assembly_neuron_ids_by_weight_and_rate


import scipy


class NetworkRecall(HandleParametersAndResults):
    def __init__(
        self,
        parameter_file_name,
        save_file_name,
        parameters_for_run={},
        parameter_dict={},
        equation_file_name="equations",
        multiple_areas=False,
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

        self.multiple_areas = multiple_areas

        if self.create_network:
            self.network = br.Network()
            self.setup_network()
            self.create_monitors()
        else:
            self.setup_network(only_setup_basics=True)

    def setup_network(self, only_setup_basics=False):
        print("Setup the network for recall")
        br.seed(self.parameters_for_run["seed"])
        np.random.seed(self.parameters_for_run["seed"])

        self.all_areas = []
        for name in self.parameters_for_run["area_names"]:
            inputs = None
            if len(self.all_areas) > 0 and not only_setup_basics:
                inputs = self.all_areas[-1].somas

            self.all_areas.append(
                Area(
                    network=self.network,
                    eqs=self.equations,
                    input_units_1=inputs,
                    params={**self.parameters, **self.parameters_for_run},
                    name=name,
                    only_setup_basics=only_setup_basics,
                )
            )

    def create_monitors(self):
        # monitor the spikes of the somas and the inputs

        self.spM_somas = []
        self.spM_inputs = []

        for area in self.all_areas:
            self.spM_somas.append(br.SpikeMonitor(area.somas, name=f"somata_monitor_{area.name}"))
            self.network.add(self.spM_somas[-1])
            self.spM_inputs.append([])
            for iu, input_units in enumerate([area.input_units_1, area.input_units_2]):
                spm = br.SpikeMonitor(input_units, name=f"input_monitor_{iu}_{area.name}")
                self.spM_inputs[-1].append(spm)
                self.network.add(spm)

    def create_save_dict(self, imprint=False):
        self.save_dict = {}
        for ii, area in enumerate(self.all_areas):
            self.save_dict.update(
                {
                    f"spikes_somas_t_{area.name}": self.spM_somas[ii].t / ms,
                    f"spikes_somas_i_{area.name}": self.spM_somas[ii].i,
                    f"spikes_inputs_t_1_{area.name}": self.spM_inputs[ii][0].t / ms,
                    f"spikes_inputs_t_2_{area.name}": self.spM_inputs[ii][1].t / ms,
                    f"spikes_inputs_i_1_{area.name}": self.spM_inputs[ii][0].i,
                    f"spikes_inputs_i_2_{area.name}": self.spM_inputs[ii][1].i,
                }
            )

    def run_imprint(
        self,
        report_period=10 * second,
        report_style=None,
        restore_beginning=True,
    ):
        filename_for_baseline_network = f"stored_imprint_{self.get_unique_paramter_and_equation_key(ignore_all_keys_with_keywords=['recall', 'all_assembly_ids_for_areas','runtime_baseline','all_context_ids_for_areas','runtime_imprint','save_network_after_each_imprint'])}"
        filename_for_stored_network = f"stored_imprint_{self.get_unique_paramter_and_equation_key(ignore_all_keys_with_keywords=['recall'])}"
        path_to_baseline_network = self.get_path_to_stored_networks(
            file_name=filename_for_baseline_network
        )

        presynaptic_sources_1 = None
        presynaptic_sources_2 = None
        if "presynaptic_sources_1" in self.parameters_for_run:
            presynaptic_sources_1 = self.parameters_for_run["presynaptic_sources_1"]
        if "presynaptic_sources_2" in self.parameters_for_run:
            presynaptic_sources_2 = self.parameters_for_run["presynaptic_sources_2"]

        save_network_after_each_imprint = False
        if "save_network_after_each_imprint" in self.parameters_for_run:
            save_network_after_each_imprint = self.parameters_for_run[
                "save_network_after_each_imprint"
            ]
        all_assembly_ids_for_areas = self.parameters_for_run["all_assembly_ids_for_areas"]
        runtime_baseline = self.parameters_for_run["runtime_baseline"]
        all_context_ids_for_areas = self.parameters_for_run["all_context_ids_for_areas"]
        runtime_imprint = self.parameters_for_run["runtime_imprint"]

        if self.check_for_results(ignore_all_keys_with_keywords=["recall"]):
            if not save_network_after_each_imprint:
                return self.save_dict

            if (
                len(self.save_dict["all_imprint_ids"]) == len(all_context_ids_for_areas)
                or self.only_load_results
            ):
                return self.save_dict

        elif self.only_load_results:
            return None

        if not os.path.isfile(path_to_baseline_network):
            self.network.store(filename=path_to_baseline_network)
        else:
            if restore_beginning:
                print("RESTORED NETWORK FOR IMPRINT")
                self.network.restore(filename=path_to_baseline_network)

        if "restore_from_save_name" in self.parameters_for_run:
            print("RESTORED FROM SAVE NAME")
            self.network.restore(
                filename=self.get_path_to_stored_networks(
                    file_name=self.parameters_for_run["restore_from_save_name"]
                )
            )

        all_imprint_ids = []
        for ii, (context_ids_for_areas, assembly_ids_for_areas) in enumerate(
            zip(all_context_ids_for_areas, all_assembly_ids_for_areas)
        ):
            all_imprint_ids.append(ii)

            if save_network_after_each_imprint and self.save_dict:
                # first we test whether this instances was already run

                print(self.save_dict["all_imprint_ids"])
                if ii in self.save_dict["all_imprint_ids"]:
                    if ii + 1 in self.save_dict["all_imprint_ids"]:
                        continue
                    else:
                        filename_for_stored_network = self.save_dict["filename_for_stored_network"]
                        if not type(filename_for_stored_network) is str:
                            filename_for_stored_network = filename_for_stored_network.decode("utf-8")

                        self.network.restore(
                            filename=self.get_path_to_stored_networks(
                                file_name=filename_for_stored_network + f"_{ii}"
                            )
                        )
                        continue

            for mm, area in enumerate(self.all_areas):
                area.start_context(0)
                if mm == 0:
                    area.input_units_1[:].rates = self.parameters["ff_bck"]
                area.input_units_2[:].rates = self.parameters["ff_bck"]

            if ii == 0:
                self.network.run(runtime_baseline, report=report_style, report_period=report_period)

            for area_index, area in enumerate(self.all_areas):
                # now we check whether this area should receive a specific context
                for context_id in context_ids_for_areas:
                    if context_id[0] == area_index:
                        area.start_context(context_id[1])

                # now we check whether this area should receive an input
                for assembly_ids in assembly_ids_for_areas:
                    if assembly_ids[0] != area_index:
                        continue

                    print(assembly_ids_for_areas, area_index, area.name, assembly_ids)

                    if assembly_ids[1] == 1992:
                        print("HERE: ", presynaptic_sources_1)
                        for n_id in presynaptic_sources_1:
                            self.all_areas[area_index].input_units_1[
                                n_id : n_id + 1
                            ].rates = self.parameters["assembly_firing_rate"]
                    elif assembly_ids[1] >= 0:
                        print(f"setting rates of input 1 {area.name}")
                        self.all_areas[area_index].input_units_1[
                            assembly_ids[1]
                            * self.parameters["assembly_size"] : (assembly_ids[1] + 1)
                            * self.parameters["assembly_size"]
                        ].rates = self.parameters["assembly_firing_rate"]

                    if assembly_ids[2] == 1992:
                        for n_id in presynaptic_sources_2:
                            self.all_areas[area_index].input_units_2[
                                n_id : n_id + 1
                            ].rates = self.parameters["assembly_firing_rate"]
                    elif assembly_ids[2] >= 0:
                        print(f"setting rates of input 2 {area.name}")
                        self.all_areas[area_index].input_units_2[
                            assembly_ids[2]
                            * self.parameters["assembly_size"] : (assembly_ids[2] + 1)
                            * self.parameters["assembly_size"]
                        ].rates = self.parameters["assembly_firing_rate"]

            self.network.run(runtime_imprint, report=report_style, report_period=report_period)

            for area_index, area in enumerate(self.all_areas):
                if area_index == 0:
                    area.input_units_1[:].rates = self.parameters["ff_bck"]
                area.input_units_2[:].rates = self.parameters["ff_bck"]
            self.network.run(runtime_baseline, report=report_style, report_period=report_period)

            if save_network_after_each_imprint or ii == (len(all_context_ids_for_areas) - 1):
                self.create_save_dict(imprint=True)
                self.save_dict["filename_for_stored_network"] = filename_for_stored_network
                self.save_dict["filename_for_baseline_network"] = filename_for_baseline_network
                self.save_dict["all_imprint_ids"] = [mm for mm in all_imprint_ids]
                final_save_name = filename_for_stored_network + f"_{ii}"
                self.network.store(
                    filename=self.get_path_to_stored_networks(file_name=final_save_name)
                )
                self.save_results(ignore_all_keys_with_keywords=["recall"])

        return self.save_dict

    def run_recall(self, report_period=10 * second, report_style=None):
        all_assembly_ids_for_areas = self.parameters_for_run["all_assembly_ids_for_areas_recall"]
        runtime_baseline = self.parameters_for_run["runtime_baseline_recall"]
        all_context_ids_for_areas = self.parameters_for_run["all_context_ids_for_areas_recall"]
        runtime_recall = self.parameters_for_run["runtime_recall"]
        run_after_imprint = self.parameters_for_run["run_recall_after_imprint"]
        recall_after_imprint_id = self.parameters_for_run["recall_after_imprint_id"]

        try:
            silence_neurons_with_ids_for_recall = self.parameters_for_run[
                "silence_neurons_with_ids_for_recall"
            ]
        except Exception as e:
            silence_neurons_with_ids_for_recall = None

        print("SILENCE ", silence_neurons_with_ids_for_recall)

        try:
            assembly_size_recall = self.parameters_for_run["assembly_size_recall"]
        except KeyError:
            assembly_size_recall = self.parameters["assembly_size"]

        try:
            assembly_firing_rate_recall = self.parameters_for_run["assembly_firing_rate_recall"]
        except KeyError:
            assembly_firing_rate_recall = self.parameters["assembly_firing_rate"]

        try:
            assembly_neuron_selection_seed_recall = self.parameters_for_run[
                "assembly_neuron_selection_seed_recall"
            ]
        except KeyError:
            assembly_neuron_selection_seed_recall = 0

        # print(runtime_baseline, all_context_ids_for_areas, runtime_recall, run_after_imprint)
        # print(
        #     assembly_size_recall, assembly_firing_rate_recall, assembly_neuron_selection_seed_recall
        # )

        if self.check_for_results():
            return self.save_dict

        elif self.only_load_results:
            return None

        self.only_load_results = True
        self.run_imprint(report_period=report_period, report_style=report_style)
        self.only_load_results = False
        # print(self.save_dict["filename_for_stored_network"])

        filename_for_stored_network = self.save_dict["filename_for_stored_network"]
        if not type(filename_for_stored_network) is str:
            filename_for_stored_network = filename_for_stored_network.decode("utf-8")
        if run_after_imprint:
            filename_for_stored_network += f"_{recall_after_imprint_id}"
        else:
            filename_for_stored_network = self.save_dict["filename_for_baseline_network"]

        if not type(filename_for_stored_network) is str:
            filename_for_stored_network = filename_for_stored_network.decode("utf-8")

        self.network.restore(
            filename=self.get_path_to_stored_networks(file_name=filename_for_stored_network)
        )

        print(all_context_ids_for_areas, all_assembly_ids_for_areas)
        for _, (context_ids_for_areas, assembly_ids_for_areas) in enumerate(
            zip(all_context_ids_for_areas, all_assembly_ids_for_areas)
        ):
            for area_index, area in enumerate(self.all_areas):
                area.start_context(0)
                # now we check whether this area should receive a specific context
                for context_id in context_ids_for_areas:
                    if context_id[0] == area_index:
                        area.start_context(context_id[1])

                if area_index == 0:
                    area.input_units_1[:].rates = self.parameters["ff_bck"]
                area.input_units_2[:].rates = self.parameters["ff_bck"]

                all_ids = np.arange(0, self.parameters["n_somas"], dtype=int)

                for assembly_ids in assembly_ids_for_areas:
                    if assembly_ids[0] != area_index:
                        continue
                    for kk in range(2):
                        this_assembly_size = assembly_size_recall

                        if assembly_ids[kk + 1] >= 0:
                            original_assembly_neuron_ids = all_ids[
                                assembly_ids[kk + 1]
                                * self.parameters["assembly_size"] : (assembly_ids[kk + 1] + 1)
                                * self.parameters["assembly_size"]
                            ]

                            np.random.seed(assembly_neuron_selection_seed_recall)

                            if this_assembly_size == 0:
                                # then we take n = assembly_size neurons that are not part of the origininal
                                original_assembly_neuron_ids = [
                                    ii
                                    for ii in range(0, self.parameters["n_somas"])
                                    if ii not in original_assembly_neuron_ids
                                ]
                                this_assembly_size = self.parameters["assembly_size"]

                            selected_assembly_neuron_ids = np.random.choice(
                                original_assembly_neuron_ids, this_assembly_size, replace=False
                            )

                            print("###", area.name)
                            print(selected_assembly_neuron_ids)

                            for neuron_id in selected_assembly_neuron_ids:
                                if kk == 0:
                                    area.input_units_1[
                                        neuron_id : neuron_id + 1
                                    ].rates = assembly_firing_rate_recall
                                if kk == 1:
                                    area.input_units_2[
                                        neuron_id : neuron_id + 1
                                    ].rates = assembly_firing_rate_recall

            if silence_neurons_with_ids_for_recall is not None:
                for ids in silence_neurons_with_ids_for_recall:
                    for n_id in ids[1:]:
                        self.all_areas[ids[0]].somas[n_id].theta = (
                            1e9 * volt
                        )  # Set a very high threshold
            self.network.run(runtime_recall, report=report_style, report_period=report_period)

            # just to be sure I set it back again
            if silence_neurons_with_ids_for_recall is not None:
                for ids in silence_neurons_with_ids_for_recall:
                    for n_id in ids[1:]:
                        self.all_areas[ids[0]].somas[n_id].theta = self.parameters["vThres_pyr"]

            for area_index, area in enumerate(self.all_areas):
                if area_index == 0:
                    area.input_units_1[:].rates = self.parameters["ff_bck"]
                area.input_units_2[:].rates = self.parameters["ff_bck"]
            self.network.run(runtime_baseline, report=report_style, report_period=report_period)
            self.create_save_dict()
            self.save_results()

    def sort_neurons_by_firing_rate(
        self,
        area=None,
        area_name=None,
        use_input_units=None,
        sort_for_specific_imprint=None,
        shuffle_rest=True,
        print_top_rates=True,
        return_rates_for_imprint=None,
    ):
        if area_name is None:
            area_name = area.name

        if use_input_units == 1:
            somas_time = self.save_dict[f"spikes_inputs_t_1_{area_name}"]
            somas_i = self.save_dict[f"spikes_inputs_i_1_{area_name}"]
        elif use_input_units == 2:
            somas_time = self.save_dict[f"spikes_inputs_t_2_{area_name}"]
            somas_i = self.save_dict[f"spikes_inputs_i_2_{area_name}"]
        else:
            somas_time = self.save_dict[f"spikes_somas_t_{area_name}"]
            somas_i = self.save_dict[f"spikes_somas_i_{area_name}"]

        all_firing_rates_somas = []
        for neuron_index in range(self.parameters["n_somas"]):
            spike_times_for_neuron = somas_time[somas_i == neuron_index]

            bsl = self.parameters_for_run["runtime_baseline"] / msecond
            rtm = self.parameters_for_run["runtime_imprint"] / msecond
            neuron_firing_rate = []

            for tt in range(len(self.parameters_for_run["all_assembly_ids_for_areas"])):
                start = bsl + (rtm + bsl) * tt
                end = start + rtm

                firing_rate = get_firing_rate_for_single_neuron(
                    start=start, end=end, spike_times_for_neuron=spike_times_for_neuron
                )
                neuron_firing_rate.append(firing_rate)
            all_firing_rates_somas.append(neuron_firing_rate)
        all_firing_rates_somas = np.array(all_firing_rates_somas)

        sorted_neuron_ids = []

        all_assembly_ids = []

        area_index = [aa.name for aa in self.all_areas].index(area_name)
        for tt in range(len(self.parameters_for_run["all_assembly_ids_for_areas"])):
            if sort_for_specific_imprint is not None:
                if tt != sort_for_specific_imprint:
                    continue

            context_id = 0
            for aa, cc in self.parameters_for_run["all_context_ids_for_areas"][tt]:
                if aa == area_index:
                    context_id = cc

            all_rates = all_firing_rates_somas[:, tt]

            selected_ids = get_assembly_neuron_ids_by_weight_and_rate(
                net=self,
                all_rates=all_rates,
                context_id=context_id,
                area=self.all_areas[area_index],
            )
            # we only add those ids that are not yet part of the sorted ids
            new_selected_ids = [si for si in selected_ids if si not in sorted_neuron_ids]

            all_assembly_ids.append(selected_ids)

            sorted_neuron_ids += new_selected_ids

        sorted_neuron_ids += [
            ni for ni in range(self.parameters["n_somas"]) if ni not in sorted_neuron_ids
        ]

        if return_rates_for_imprint is None:
            return sorted_neuron_ids, all_assembly_ids, selected_ids
        else:
            return (
                sorted_neuron_ids,
                all_assembly_ids,
                selected_ids,
                all_firing_rates_somas[:, return_rates_for_imprint],
            )

    def show_spike_rasters(
        self,
        show_plot=False,
        axes=None,
        start_with_last_imprint=False,
        highlight_neuron_ids=None,
        sort_for_specific_imprint=None,
        show_vertical_lines_at=None,
        print_top_rates=True,
    ):
        for aa, area in enumerate(self.all_areas):
            somas_time = self.save_dict[f"spikes_somas_t_{area.name}"]
            somas_i = self.save_dict[f"spikes_somas_i_{area.name}"]
            inputs_1_time = self.save_dict[f"spikes_inputs_t_1_{area.name}"]
            inputs_1_i = self.save_dict[f"spikes_inputs_i_1_{area.name}"]
            inputs_2_time = self.save_dict[f"spikes_inputs_t_2_{area.name}"]
            inputs_2_i = self.save_dict[f"spikes_inputs_i_2_{area.name}"]

            if axes is None:
                fig, (ax_inputs_1, ax_inputs_2, ax_somas) = plt.subplots(
                    3, sharex=True, figsize=(10, 8)
                )
            else:
                (ax_inputs_1, ax_inputs_2, ax_somas) = axes[aa]
            sorted_neuron_ids, _, _ = self.sort_neurons_by_firing_rate(
                area=area,
                sort_for_specific_imprint=sort_for_specific_imprint,
                print_top_rates=print_top_rates,
            )

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
                if show_vertical_lines_at is None:
                    bsl = self.parameters_for_run["runtime_baseline"] / msecond
                    rtm = self.parameters_for_run["runtime_imprint"] / msecond
                    show_vertical_lines_at = [
                        2 * bsl + rtm
                    ], 2 * bsl + rtm + self.parameters_for_run["runtime_recall"] / msecond

                for area_id, neuron_ids in highlight_neuron_ids:
                    if area_id == aa:
                        for n_id in neuron_ids:
                            position = list(sorted_neuron_ids).index(n_id)
                            ax_somas.fill_between(
                                [
                                    show_vertical_lines_at[0],
                                    show_vertical_lines_at[1],
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

        if show_plot:
            plt.show()

    def show_weight_matrix(self, show_plot=False, matlab_export_name=None, specific_area_id=0):
        area = self.all_areas[specific_area_id]
        weights_recurrent = np.zeros(shape=(area.n_somas, area.n_dends))
        weights_recurrent[area.srcs, area.tgts] = area.synapses_E.w

        if np.all(weights_recurrent == self.parameters["w0"]):
            # ensure that we actually have the weights loaded and are not workig with initial weights
            raise ValueError("seems like you are using starting weights")

        matlab_save_dict = {"all_weights": weights_recurrent}

        context_id = 0

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        weights = weights_recurrent[:, context_id::6]

        matlab_save_dict[f"weights_of_context_{context_id}"] = weights

        print("new shape, ", weights.shape)

        G = nx.from_numpy_array(weights, create_using=nx.DiGraph)

        # Community detection (convert to undirected for the community detection if necessary)
        partition = community_louvain.best_partition(G.to_undirected(), weight="weight")

        # Create a mapping of node index to community
        node_community_map = {node: community for node, community in enumerate(partition.values())}

        # Sort nodes by community
        sorted_nodes = sorted(node_community_map, key=node_community_map.get)

        # Reorder the matrix accordingly
        W_reordered = weights[np.ix_(sorted_nodes, sorted_nodes)]
        im = ax1.imshow(W_reordered.T, cmap="Greys")
        plt.colorbar(im)

        sorted_neuron_ids, _, _ = self.sort_neurons_by_firing_rate(area=area)

        weights = weights[np.ix_(sorted_neuron_ids, sorted_neuron_ids)]

        matlab_save_dict[f"weights_of_context_{context_id}_sorted_for_assemblies"] = weights

        im = ax2.imshow(weights.T, cmap="Greys")
        plt.colorbar(im)

        if matlab_export_name is not None:
            scipy.io.savemat(f"{matlab_export_name}.mat", mdict=matlab_save_dict)
            for key, val in matlab_save_dict.items():
                np.savetxt(f"{matlab_export_name}_{key}.txt", val)

        if show_plot:
            plt.show()


if __name__ == "__main__":
    pass
