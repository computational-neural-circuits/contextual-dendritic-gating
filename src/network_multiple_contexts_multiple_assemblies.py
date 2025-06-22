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


class NetworkMultipleContextsMultipleAssemblies(HandleParametersAndResults):
    def __init__(
        self,
        parameter_file_name,
        save_file_name,
        parameters_for_run={},
        parameter_dict={},
        equation_file_name="equations",
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

        if self.create_network and not self.only_load_results:
            self.network = br.Network()
            self.setup_network()
            self.create_monitors()
            print("Network setup finished")
        else:
            self.network = None
            self.setup_network(only_setup_basics=True)

    def setup_network(self, only_setup_basics=False):
        print("Setup the network for multiple Assemblies in multiple contexts")
        br.seed(self.parameters_for_run["seed"])
        np.random.seed(self.parameters_for_run["seed"])

        # print(self.parameters)
        self.area = Area(
            network=self.network,
            eqs=self.equations,
            params={**self.parameters, **self.parameters_for_run},
            only_setup_basics=only_setup_basics,
        )

    def create_monitors(self):
        self.record_gated_and_non_gated_synapses = True
        if "record_gated_and_non_gated_synapses" in self.parameters_for_run:
            self.record_gated_and_non_gated_synapses = self.parameters_for_run[
                "record_gated_and_non_gated_synapses"
            ]

        self.record_recurrent_inhibition = False
        if "record_recurrent_inhibition" in self.parameters_for_run:
            self.record_recurrent_inhibition = self.parameters_for_run["record_recurrent_inhibition"]

        self.record_new_recurrent_inhibition = False
        if "record_new_recurrent_inhibition" in self.parameters_for_run:
            self.record_new_recurrent_inhibition = self.parameters_for_run[
                "record_new_recurrent_inhibition"
            ]

        # monitor the spikes of the somas and the inputs
        area = self.area

        self.spM_somas = br.SpikeMonitor(area.somas, name=f"somata_monitor_{area.name}")
        self.network.add(self.spM_somas)
        self.spM_inputs = []
        for iu, input_units in enumerate([area.input_units_1, area.input_units_2]):
            spm = br.SpikeMonitor(input_units, name=f"input_monitor_{iu}_{area.name}")
            self.spM_inputs.append(spm)
            self.network.add(spm)

        area.count_number_of_inputs_from_subsets_in_context()

        first_imprinted_assembly = self.parameters_for_run["all_assembly_ids"][0]
        if first_imprinted_assembly[0] == -1 or first_imprinted_assembly[1] == -1:
            gated_counts = area.counts_gated[0]
            non_gated_counts = area.counts_non_gated[0]
            min_inputs_to_potentiate = 3

        else:
            gated_counts = area.counts_gated[0] + area.counts_gated[1]
            non_gated_counts = area.counts_non_gated[0] + area.counts_non_gated[1]
            min_inputs_to_potentiate = 5

        unique_counts = np.sort(np.unique(gated_counts))[:-1].astype(int)  # remove the NaN value
        unique_counts_non_gated = np.sort(np.unique(non_gated_counts))[:-1].astype(
            int
        )  # remove the NaN value

        # print(unique_counts)
        # print(unique_counts_non_gated)

        # randomly select samples to visualize
        select_ids_gated = []
        self.select_ids_gated_counts = []
        for uc in np.sort(unique_counts):
            select_ids_gated.append(np.argmax(gated_counts == uc))
            self.select_ids_gated_counts.append(uc)

        self.potenitially_potenitated_dendrites = []
        for n_syn in range(min_inputs_to_potentiate, 18):
            self.potenitially_potenitated_dendrites += list(np.where(gated_counts == n_syn)[0])
        # print("Potentially potentiated for context 0:")
        # print(len(self.potenitially_potenitated_dendrites))
        # print(self.potenitially_potenitated_dendrites)

        select_ids_non_gated = []
        self.select_ids_non_gated_counts = []
        for uc in np.sort(unique_counts_non_gated):
            select_ids_non_gated.append(np.argmax(non_gated_counts == uc))
            self.select_ids_non_gated_counts.append(uc)

        self.select_ids_gated = select_ids_gated
        self.select_ids_non_gated = select_ids_non_gated

        if self.record_gated_and_non_gated_synapses:
            self.Mdends_V = {}
            self.Msomas_V = {}
            self.Msyns_w = {}
            self.Mff_w = {}
            for name, id_list in zip(
                ["gated", "non_gated"], [select_ids_gated, select_ids_non_gated]
            ):
                mon = br.StateMonitor(
                    area.dends, "V", record=id_list, dt=self.parameters["monitor_dt"]
                )
                self.Mdends_V[name] = mon
                self.network.add(mon)

                neighbour_ids = [
                    s_id + 1 if (s_id + 1) % self.parameters["n_dend_each"] != 0 else s_id - 1
                    for s_id in id_list
                ]
                mon = br.StateMonitor(
                    area.dends, "V", record=neighbour_ids, dt=self.parameters["monitor_dt"]
                )
                self.Mdends_V[name + "_neighbour"] = mon
                self.network.add(mon)

                mon = br.StateMonitor(
                    area.somas,
                    "V",
                    record=[s_id // self.parameters["n_dend_each"] for s_id in id_list],
                    dt=self.parameters["monitor_dt"],
                )
                self.Msomas_V[name] = mon
                self.network.add(mon)

                to_monitor = "w"
                monitor_list = []
                for kk in id_list:
                    mon = br.StateMonitor(
                        area.synapses_E,
                        to_monitor,
                        record=area.synapses_E[np.where((area.synapses_E.j)[:] == kk)[0]],
                        dt=self.parameters["monitor_dt_weights"],
                    )
                    monitor_list.append(mon)
                    self.network.add(mon)

                self.Msyns_w[name] = monitor_list

                to_monitor = "w"
                monitor_list = []
                for kk in id_list:
                    for ii in range(2):
                        mon = br.StateMonitor(
                            area.input_synapses[ii],
                            to_monitor,
                            record=(area.input_synapses[ii])[
                                np.where(area.input_synapses[ii].j == kk)[0]
                            ],
                            dt=self.parameters["monitor_dt_weights"],
                        )
                        monitor_list.append(mon)
                        self.network.add(mon)
                self.Mff_w[name] = monitor_list

            if self.parameters_for_run["debug_mode"]:
                mon = br.StateMonitor(
                    area.dends,
                    "V",
                    record=self.potenitially_potenitated_dendrites,
                    dt=self.parameters["monitor_dt"],
                )
                self.Mdends_V["pot_pot"] = mon
                self.network.add(mon)

                to_monitor = "w"
                monitor_list = []
                for kk in self.potenitially_potenitated_dendrites:
                    mon = br.StateMonitor(
                        area.synapses_E,
                        to_monitor,
                        record=area.synapses_E[np.where((area.synapses_E.j)[:] == kk)[0]],
                        dt=self.parameters["monitor_dt_weights"],
                    )
                    monitor_list.append(mon)
                    self.network.add(mon)

                self.Msyns_w["pot_pot"] = monitor_list

                to_monitor = "w"
                monitor_list = []
                for kk in self.potenitially_potenitated_dendrites:
                    for ii in range(2):
                        mon = br.StateMonitor(
                            area.input_synapses[ii],
                            to_monitor,
                            record=(area.input_synapses[ii])[
                                np.where(area.input_synapses[ii].j == kk)[0]
                            ],
                            dt=self.parameters["monitor_dt_weights"],
                        )
                        monitor_list.append(mon)
                        self.network.add(mon)
                self.Mff_w["pot_pot"] = monitor_list

        self.Minhibition = br.StateMonitor(
            area.ff_inhibitors, "rate", record=True, dt=self.parameters["monitor_dt"]
        )
        self.network.add(self.Minhibition)

        self.Mx = br.StateMonitor(
            area.population_firing_rate_estimator,
            "x_pop",
            record=True,
            dt=self.parameters["monitor_dt"],
        )
        self.network.add(self.Mx)

        if self.parameters_for_run["debug_mode"]:
            self.M_n_active = br.StateMonitor(
                area.rec_inihib_pop, "n_active", record=True, dt=self.parameters["monitor_dt"]
            )
            self.network.add(self.M_n_active)
            self.M_somas_x = br.StateMonitor(
                area.somas,
                ["x_estimator"],
                record=True,
                dt=self.parameters["monitor_dt"],
            )
            self.network.add(self.M_somas_x)

        # if self.record_new_recurrent_inhibition:

        if self.record_recurrent_inhibition:
            self.spM_rec_inhib = br.SpikeMonitor(area.rec_inihib_pop)
            self.network.add(self.spM_rec_inhib)
            self.M_syn_to_rec_inhib = br.StateMonitor(
                area.syn_to_rec_inhib_pop,
                ["close"],
                record=True,
                dt=self.parameters["monitor_dt"],
            )
            self.network.add(self.M_syn_to_rec_inhib)
            self.M_somas_x = br.StateMonitor(
                area.somas,
                ["x_estimator"],
                record=True,
                dt=self.parameters["monitor_dt"],
            )
            self.network.add(self.M_somas_x)

        print("created monitors")

    def create_save_dict(self):
        area = self.area
        weights_recurrent = np.zeros(shape=(area.n_somas, area.n_dends))
        weights_recurrent[area.srcs, area.tgts] = area.synapses_E.w

        self.save_dict = {
            f"spikes_somas_t": self.spM_somas.t / ms,
            f"spikes_somas_i": self.spM_somas.i,
            f"spikes_inputs_t_1": self.spM_inputs[0].t / ms,
            f"spikes_inputs_t_2": self.spM_inputs[1].t / ms,
            f"spikes_inputs_i_1": self.spM_inputs[0].i,
            f"spikes_inputs_i_2": self.spM_inputs[1].i,
        }

        self.save_dict["inhibitory_rate"] = self.Minhibition.rate / Hz
        self.save_dict["inhibitory_time"] = self.Minhibition.t / ms
        self.save_dict["x_time"] = self.Mx.t / ms
        self.save_dict["x_value"] = self.Mx.x_pop

        if self.record_recurrent_inhibition:
            self.save_dict["spikes_rec_inhib_t"] = self.spM_rec_inhib.t / ms
            self.save_dict["spikes_rec_inhib_i"] = self.spM_rec_inhib.i
            self.save_dict["syn_to_rec_inhib_time"] = self.M_syn_to_rec_inhib.t / ms
            self.save_dict["syn_to_rec_inhib_close"] = self.M_syn_to_rec_inhib.close
            self.save_dict["somas_x"] = self.M_somas_x.x_estimator

        if self.parameters_for_run["debug_mode"]:
            self.save_dict["rec_inhib_time"] = self.M_n_active.t / ms
            self.save_dict["rec_inhib_n_active"] = self.M_n_active.n_active

        if self.parameters_for_run["save_weights"]:
            if "save_most_active_neuron_weights" in self.parameters_for_run:
                if self.parameters_for_run["save_most_active_neuron_weights"]:
                    weights_recurrent = weights_recurrent[:, 0::6]
                    sorted_neuron_ids, selected_ids, _ = self.sort_neurons_by_firing_rate(
                        shuffle_rest=False, reverse_order=True
                    )

                    weights_recurrent = weights_recurrent[
                        np.ix_(sorted_neuron_ids[0], sorted_neuron_ids[0])
                    ]
                    weights_recurrent = weights_recurrent[
                        : len(selected_ids) + 25, : len(selected_ids) + 25
                    ]

            self.save_dict["weights"] = weights_recurrent

        if self.record_gated_and_non_gated_synapses:
            self.save_dict[f"voltage_dends_t"] = self.Mdends_V["gated"][0].t / ms
            self.save_dict[f"voltag_somas_t"] = self.Msomas_V["gated"][0].t / ms
            self.save_dict[f"voltage_weights_t"] = self.Msyns_w["gated"][0].t / ms
            for name, counts in zip(
                ["gated", "non_gated"],
                [self.select_ids_gated_counts, self.select_ids_non_gated_counts],
            ):
                self.save_dict[f"voltage_dends_{name}"] = self.Mdends_V[name].V / mV
                self.save_dict[f"voltage_dends_{name}_neighbour"] = (
                    self.Mdends_V[name + "_neighbour"].V / mV
                )
                self.save_dict[f"voltage_soma_{name}"] = self.Msomas_V[name].V / mV

                for ii, val in enumerate(counts):
                    self.save_dict[f"counts_{ii}_{name}"] = val
                    ww = self.Msyns_w[name][ii].w
                    self.save_dict[f"weight_w_{ii}_{name}"] = ww

                    ww_1 = self.Mff_w[name][ii * 2].w
                    ww_2 = self.Mff_w[name][ii * 2 + 1].w
                    self.save_dict[f"weight_w_ff_1_{ii}_{name}"] = ww_1
                    self.save_dict[f"weight_w_ff_2_{ii}_{name}"] = ww_2

        if self.parameters_for_run["debug_mode"]:
            self.save_dict["voltage_dends_potentially_potentiated"] = self.Mdends_V["pot_pot"].V / mV

            for ii, _ in enumerate(self.potenitially_potenitated_dendrites):
                ww = self.Msyns_w["pot_pot"][ii].w
                self.save_dict[f"weight_w_{ii}_pot_pot"] = ww

                ww_1 = self.Mff_w["pot_pot"][ii * 2].w
                ww_2 = self.Mff_w["pot_pot"][ii * 2 + 1].w
                self.save_dict[f"weight_w_ff_1_{ii}_pot_pot"] = ww_1
                self.save_dict[f"weight_w_ff_2_{ii}_pot_pot"] = ww_2
        # for key in self.save_dict:
        # print(key, type(self.save_dict[key]))

    def run(self, report_period=10 * second, report_style=None):
        all_assembly_ids = self.parameters_for_run["all_assembly_ids"]
        runtime_baseline = self.parameters_for_run["runtime_baseline"]
        all_context_ids = self.parameters_for_run["all_context_ids"]
        runtime_imprint = self.parameters_for_run["runtime_imprint"]

        if self.check_for_results():
            return self.save_dict

        elif self.only_load_results:
            return None

        for ii, (context_id, assembly_ids) in enumerate(zip(all_context_ids, all_assembly_ids)):
            self.area.stop_context()
            self.area.start_context(context_id)

            self.area.input_units_1[:].rates = self.parameters["ff_bck"]
            self.area.input_units_2[:].rates = self.parameters["ff_bck"]

            self.network.run(runtime_baseline, report=report_style, report_period=report_period)

            if assembly_ids[0] >= 0:
                self.area.input_units_1[
                    assembly_ids[0]
                    * self.parameters["assembly_size"] : (assembly_ids[0] + 1)
                    * self.parameters["assembly_size"]
                ].rates = self.parameters["assembly_firing_rate"]

            if assembly_ids[1] >= 0:
                self.area.input_units_2[
                    assembly_ids[1]
                    * self.parameters["assembly_size"] : (assembly_ids[1] + 1)
                    * self.parameters["assembly_size"]
                ].rates = self.parameters["assembly_firing_rate"]

            self.network.run(runtime_imprint, report=report_style, report_period=report_period)

            self.create_save_dict()
            self.save_results()

            self.area.input_units_1[:].rates = self.parameters["ff_bck"]
            self.area.input_units_2[:].rates = self.parameters["ff_bck"]

            if ii == np.min([len(all_context_ids), len(all_assembly_ids)]) - 1:
                self.network.run(runtime_baseline, report=report_style, report_period=report_period)

        self.create_save_dict()
        self.save_results()

        if "no_recall" in self.parameters_for_run:
            return

        for ii, (context_id, assembly_ids) in enumerate(zip(all_context_ids, all_assembly_ids)):
            self.area.stop_context()
            self.area.start_context(context_id)

            self.area.input_units_1[:].rates = self.parameters["ff_bck"]
            self.area.input_units_2[:].rates = self.parameters["ff_bck"]

            self.network.run(0.5 * second, report=report_style, report_period=report_period)

            if assembly_ids[0] >= 0:
                self.area.input_units_1[
                    assembly_ids[0]
                    * self.parameters["assembly_size"] : (assembly_ids[0] + 1)
                    * self.parameters["assembly_size"]
                ].rates = self.parameters["assembly_firing_rate"]

            if assembly_ids[1] >= 0:
                self.area.input_units_2[
                    assembly_ids[1]
                    * self.parameters["assembly_size"] : (assembly_ids[1] + 1)
                    * self.parameters["assembly_size"]
                ].rates = self.parameters["assembly_firing_rate"]

            self.network.run(1.5 * second, report=report_style, report_period=report_period)

            self.create_save_dict()
            self.save_results()

    def show_weight_matrix(
        self,
        show_plot=False,
        matlab_export_name=None,
    ):
        if "save_most_active_neuron_weights" in self.parameters_for_run:
            if self.parameters_for_run["save_most_active_neuron_weights"]:
                fig, ax = plt.subplots()
                im = ax.imshow(self.save_dict["weights"].T, cmap="Greys")
                plt.colorbar(im)

                return

        weights_loaded = self.save_dict["weights"]
        print(self.save_dict["weights"].shape)

        matlab_save_dict = {"all_weights": weights_loaded}

        if self.area.non_overlapping_ctxt:
            for ii, context_id in enumerate(
                np.sort(np.unique(self.parameters_for_run["all_context_ids"]))
            ):
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

                weights = weights_loaded[:, context_id::6]

                matlab_save_dict[f"weights_of_context_{context_id}"] = weights

                print("new shape, ", weights.shape)

                G = nx.from_numpy_array(weights, create_using=nx.DiGraph)

                # Community detection (convert to undirected for the community detection if necessary)
                partition = community_louvain.best_partition(G.to_undirected(), weight="weight")

                # Create a mapping of node index to community
                node_community_map = {
                    node: community for node, community in enumerate(partition.values())
                }

                # Sort nodes by community
                sorted_nodes = sorted(node_community_map, key=node_community_map.get)

                # Reorder the matrix accordingly
                W_reordered = weights[np.ix_(sorted_nodes, sorted_nodes)]
                im = ax1.imshow(W_reordered.T, cmap="Greys")
                plt.colorbar(im)

                sorted_neuron_ids, _, _ = self.sort_neurons_by_firing_rate()

                weights = weights[np.ix_(sorted_neuron_ids[ii], sorted_neuron_ids[ii])]

                im = ax2.imshow(weights.T, cmap="Greys")
                plt.colorbar(im)

            if matlab_export_name is not None:
                scipy.io.savemat(f"{matlab_export_name}.mat", mdict=matlab_save_dict)
                for key, val in matlab_save_dict.items():
                    np.savetxt(f"{matlab_export_name}_{key}.txt", val)

        else:
            fig, ax1 = plt.subplots(figsize=(12, 6))
            weights = np.copy(weights_loaded)

            only_neuron_weights = np.zeros((400, 400))
            # Add edges from the non-square matrix
            m, n = weights.shape
            for i in range(m):
                for j in range(n):
                    if weights[i, j] != 0:
                        if only_neuron_weights[i, j // 6] == 0:
                            only_neuron_weights[i, j // 6] = weights[i, j]
                        else:
                            only_neuron_weights[i, j // 6] = (
                                weights[i, j] + only_neuron_weights[i, j // 6]
                            ) / 2.0

            G = nx.from_numpy_array(only_neuron_weights, create_using=nx.DiGraph)
            # Community detection (convert to undirected for the community detection if necessary)
            partition = community_louvain.best_partition(G.to_undirected(), weight="weight")

            # Create a mapping of node index to community
            node_community_map = {
                node: community for node, community in enumerate(partition.values())
            }

            # Sort nodes by community
            sorted_nodes = sorted(node_community_map, key=node_community_map.get)

            print(len(sorted_nodes))

            den_ids = []
            for neuron_id in sorted_nodes:
                den_ids += [neuron_id * 6 + ii for ii in range(6)]

            # Reorder the matrix accordingly
            W_reordered = weights[np.ix_(sorted_nodes, den_ids)]
            im = ax1.imshow(W_reordered.T, cmap="Greys")
            plt.colorbar(im)

            ax_counter = 0

            sorted_neuron_ids, _, _ = self.sort_neurons_by_firing_rate()

            for context_id in np.sort(np.unique(self.parameters_for_run["all_context_ids"])):
                if ax_counter == 0:
                    fig = plt.figure(figsize=(16, 7), constrained_layout=True)
                    gs = fig.add_gridspec(23, 20)
                    ax_1 = fig.add_subplot(gs[:10, :5])
                    ax_2 = fig.add_subplot(gs[:10, 5:10])
                    ax_3 = fig.add_subplot(gs[:10, 10:15])
                    ax_4 = fig.add_subplot(gs[:10, 15:20])
                    ax_5 = fig.add_subplot(gs[10:20, :5])
                    ax_6 = fig.add_subplot(gs[10:20, 5:10])
                    ax_7 = fig.add_subplot(gs[10:20, 10:15])
                    ax_8 = fig.add_subplot(gs[10:20, 15:20])
                    colorbar_ax = fig.add_subplot(gs[21:22, 6:14])

                    axes = [ax_1, ax_2, ax_3, ax_4, ax_5, ax_6, ax_7, ax_8]

                ax = axes[ax_counter]
                weights = weights_loaded[:, self.area.dends_of_ctxt[context_id]]
                weights = weights[
                    np.ix_(sorted_neuron_ids[context_id], sorted_neuron_ids[context_id])
                ]
                im = ax.imshow(
                    weights.T,
                    cmap="Greys",
                    vmin=0,
                    vmax=self.parameters["w_max_rec"],
                )
                ax.set(
                    ylabel="post",
                    xlabel="pre",
                    title=f"context {context_id}",
                )

                plt.colorbar(im, cax=colorbar_ax, location="bottom")
                ax_counter = (ax_counter + 1) % len(axes)

        if show_plot:
            plt.show()

    def sort_neurons_by_weights(self, context_id=0, show_plot=False, title=None):
        weights = self.save_dict["weights"]
        G = nx.from_numpy_array(weights, create_using=nx.DiGraph)
        partition = community_louvain.best_partition(G.to_undirected(), weight="weight")
        node_community_map = {node: community for node, community in enumerate(partition.values())}
        sorted_nodes = sorted(node_community_map, key=node_community_map.get)

        # Define a threshold for strong connections
        threshold = 2.5  # This can be adjusted based on your specific needs

        # Create a binary matrix of strong connections
        binary_matrix = weights > threshold

        # Create a graph from the binary matrix
        G = nx.from_numpy_array(binary_matrix)

        # Find connected components
        components = list(nx.connected_components(G))

        # Assembly sizes are the sizes of the connected components
        assembly_sizes = [len(component) for component in components if len(component) > 1]

        if show_plot:
            fig, (ax1, ax2) = plt.subplots(1, 2)
            W_reordered = weights[np.ix_(sorted_nodes, sorted_nodes)]
            ax1.imshow(binary_matrix.T, cmap="Greys")
            ax2.imshow(weights.T, cmap="Greys")
            ax1.set(xlabel="Pre", ylabel="Post")
            ax2.set(xlabel="Pre", ylabel="Post")
            # [area.srcs, area.tgts]
            fig.suptitle(title + f"{assembly_sizes}")
            plt.show()
        return sorted_nodes, assembly_sizes

    def sort_neurons_by_firing_rate(self, shuffle_rest=True, reverse_order=False, sort_by_rate=True):
        somas_time = np.copy(self.save_dict["spikes_somas_t"])
        somas_i = np.copy(self.save_dict["spikes_somas_i"])

        unique_context_ids = list(np.sort(np.unique(self.parameters_for_run["all_context_ids"])))
        all_firing_rates = [[] for nn in unique_context_ids]

        bsl = self.parameters_for_run["runtime_baseline"] / msecond
        rtm = self.parameters_for_run["runtime_imprint"] / msecond
        for tt in range(len(self.parameters_for_run["all_assembly_ids"])):
            start = (rtm + bsl) * tt + bsl * int(tt > 0)
            end = start + rtm + bsl * int(tt < 1)

            context_id = self.parameters_for_run["all_context_ids"][tt]
            contex_id_index = unique_context_ids.index(context_id)

            all_firing_rates_somas = []
            for neuron_index in range(self.parameters["n_somas"]):
                spike_times_for_neuron = somas_time[somas_i == neuron_index]

                firing_rate = get_firing_rate_for_single_neuron(
                    start=start, end=end, spike_times_for_neuron=spike_times_for_neuron
                )
                all_firing_rates_somas.append(firing_rate)
            all_firing_rates[contex_id_index].append(all_firing_rates_somas)

        all_context_ids = self.parameters_for_run["all_context_ids"]
        all_assembly_ids = self.parameters_for_run["all_assembly_ids"]
        unique_context_ids = np.unique(all_context_ids)
        sorted_neuron_ids = [[] for nn in unique_context_ids]
        # all_assembly_ids_per_context = [
        #     [
        #         assembly_ids
        #         for assembly_ids, context_id in zip(all_assembly_ids, all_context_ids)
        #         if context_id == this_context
        #     ]
        #     for this_context in unique_context_ids
        # ]
        # # print("aaids:", all_assembly_ids_per_context)

        for ii, context_id in enumerate(unique_context_ids):
            for tt, all_rates in enumerate(np.array(all_firing_rates[ii])):
                selected_ids = get_assembly_neuron_ids_by_weight_and_rate(
                    net=self,
                    all_rates=all_rates,
                    context_id=context_id,
                    area=self.area,
                )

                # we only add those ids that are not yet part of the sorted ids
                new_selected_ids = [si for si in selected_ids if si not in sorted_neuron_ids[ii]]

                sorted_neuron_ids[ii] += new_selected_ids

            sorted_neuron_ids[ii] += [
                ni for ni in range(self.parameters["n_somas"]) if ni not in sorted_neuron_ids[ii]
            ]

        return sorted_neuron_ids, selected_ids, all_rates

    def show_recall(self, show_plot=False):
        somas_time = self.save_dict["spikes_somas_t"]
        somas_i = self.save_dict["spikes_somas_i"]
        bsl = self.parameters_for_run["runtime_baseline"] / msecond
        rtm = self.parameters_for_run["runtime_imprint"] / msecond
        inputs_2_time = self.save_dict["spikes_inputs_t_2"]
        inputs_1_time = self.save_dict["spikes_inputs_t_1"]
        inputs_1_i = self.save_dict["spikes_inputs_i_1"]
        inputs_2_i = self.save_dict["spikes_inputs_i_2"]
        bsl_recall = 0.5 * second / msecond
        rtm_recall = 1.5 * second / msecond

        sorted_neuron_ids, _, _ = self.sort_neurons_by_firing_rate(
            reverse_order=True, shuffle_rest=True
        )
        fig, (ax_inputs_1, ax_inputs_2, ax_recall) = plt.subplots(3)
        end_of_imprint = (rtm + bsl) * len(self.parameters_for_run["all_assembly_ids"]) + bsl
        unique_context_ids = list(np.sort(np.unique(self.parameters_for_run["all_context_ids"])))
        for tt in range(len(self.parameters_for_run["all_assembly_ids"])):
            context_id = self.parameters_for_run["all_context_ids"][tt]
            contex_id_index = unique_context_ids.index(context_id)

            start_imprint = (rtm + bsl) * tt + bsl
            end_imprint = start_imprint + rtm
            start_imprint_late = (
                end_imprint - rtm_recall
            )  # we make the time span the same as for the recall
            end_imprint_early = start_imprint + rtm_recall

            start_recall = end_of_imprint + bsl_recall + (bsl_recall + rtm_recall) * tt
            end_recall = start_recall + rtm_recall

            for (start, end), color, set_time in zip(
                [
                    (start_imprint_late, end_imprint),
                    (start_imprint, end_imprint_early),
                    (start_recall, end_recall),
                ],
                ["k", "g", "r"],
                [
                    tt * 3.1 * rtm_recall,
                    tt * 3.1 * rtm_recall + rtm_recall,
                    tt * 3.1 * rtm_recall + 2 * rtm_recall,
                ],
            ):
                for ii, neuron_index in enumerate(sorted_neuron_ids[contex_id_index]):
                    spike_times_for_neuron = somas_time[somas_i == neuron_index]

                    spikes_are_after = spike_times_for_neuron > start
                    spikes_are_before = spike_times_for_neuron < end
                    spike_times_for_neuron_selection = (
                        spike_times_for_neuron[np.logical_and(spikes_are_before, spikes_are_after)]
                        - start
                    )

                    ax_recall.vlines(
                        spike_times_for_neuron_selection + set_time,
                        ymin=ii - 0.5,
                        ymax=ii + 0.5,
                        colors=color,
                    )

                    for ax, inputs_time, inputs_i in zip(
                        [ax_inputs_1, ax_inputs_2],
                        [inputs_1_time, inputs_2_time],
                        [inputs_1_i, inputs_2_i],
                    ):
                        spike_times_for_neuron = inputs_time[inputs_i == ii]
                        spikes_are_after = spike_times_for_neuron > start
                        spikes_are_before = spike_times_for_neuron < end

                        ax.vlines(
                            spike_times_for_neuron[
                                np.logical_and(spikes_are_before, spikes_are_after)
                            ]
                            - start
                            + set_time,
                            ymin=ii - 0.5,
                            ymax=ii + 0.5,
                            colors=color,
                        )

            ylim = ax.get_ylim()
            ax.annotate(
                f"#{context_id}",
                xy=(tt * 2.1 * rtm_recall + rtm_recall, ylim[1] + (ylim[1] - ylim[0]) * 0.05),
                ha="center",
                va="center",
            )

        if show_plot:
            plt.show()

    def show_spike_rasters(
        self,
        show_plot=False,
        order=None,
        axes=None,
        show_SOM_activity=False,
        shuffle_rest=True,
        show_recall=False,
    ):
        somas_time = self.save_dict["spikes_somas_t"]
        somas_i = self.save_dict["spikes_somas_i"]
        inputs_2_time = self.save_dict["spikes_inputs_t_2"]
        inputs_1_time = self.save_dict["spikes_inputs_t_1"]
        inputs_1_i = self.save_dict["spikes_inputs_i_1"]
        inputs_2_i = self.save_dict["spikes_inputs_i_2"]
        bsl = self.parameters_for_run["runtime_baseline"] / msecond
        rtm = self.parameters_for_run["runtime_imprint"] / msecond

        def plot_spikes_of_inputs(ax1, ax2, context_colors=None):
            for tt in range(len(self.parameters_for_run["all_assembly_ids"])):
                start = (rtm + bsl) * tt + bsl * int(tt > 0)
                end = start + rtm + bsl + bsl * int(tt < 1)

                context_id = self.parameters_for_run["all_context_ids"][tt]
                for neuron_index in range(self.parameters["n_somas"]):
                    for ax, inputs_time, inputs_i in zip(
                        [ax1, ax2],
                        [inputs_1_time, inputs_2_time],
                        [inputs_1_i, inputs_2_i],
                    ):
                        spike_times_for_neuron = inputs_time[inputs_i == neuron_index]
                        spikes_are_after = spike_times_for_neuron > start
                        spikes_are_before = spike_times_for_neuron < end

                        color = "k"
                        if context_colors is not None:
                            color = context_colors[context_id]

                        ax.vlines(
                            spike_times_for_neuron[
                                np.logical_and(spikes_are_before, spikes_are_after)
                            ],
                            ymin=neuron_index - 0.5,
                            ymax=neuron_index + 0.5,
                            colors=color,
                        )
                ax1.set_title("Inputs (1)")
                ax2.set_title("Inputs (2)")

        if self.area.non_overlapping_ctxt:
            context_colors = [
                "#e41a1c",
                "#377eb8",
                "#4daf4a",
                "#984ea3",
                "#ff7f00",
                "#ffff33",
            ]
            if axes is None:
                n_axes_needed = (
                    len(np.unique(self.parameters_for_run["all_context_ids"])) + 2
                )  # +2 for the inputs

                # we want 2 cols

                n_rows = int(np.ceil(n_axes_needed / 2.0))

                fig, axes = plt.subplots(n_rows, 2, sharex=True, figsize=(10, 8))

            plot_spikes_of_inputs(
                ax1=axes.flatten()[0], ax2=axes.flatten()[1], context_colors=context_colors
            )

            sorted_neuron_ids, _, _ = self.sort_neurons_by_firing_rate(
                reverse_order=True, shuffle_rest=shuffle_rest
            )

            if order is not None:
                sorted_neuron_ids = order

            print(somas_i.shape, somas_time.shape)

            for jj, context_id in enumerate(
                np.sort(np.unique(self.parameters_for_run["all_context_ids"]))
            ):
                for ii, neuron_index in enumerate(sorted_neuron_ids[jj]):
                    spike_times_for_neuron = somas_time[somas_i == neuron_index]
                    ax = axes.flatten()[jj + 2]
                    ax.vlines(spike_times_for_neuron, ymin=ii - 0.5, ymax=ii + 0.5, colors="k")
                    ax.set_title(
                        f"Sorted for context {context_id}", color=context_colors[context_id]
                    )

                    if show_SOM_activity:
                        spike_times_SOM = self.save_dict["spikes_rec_inhib_t"][
                            self.save_dict["spikes_rec_inhib_i"] == neuron_index
                        ]
                        ax.vlines(
                            spike_times_SOM, ymin=ii - 0.3, ymax=ii + 0.3, colors="r", alpha=0.3
                        )

            if show_SOM_activity:
                somas_x = self.save_dict["somas_x"]
                time_x = self.save_dict["syn_to_rec_inhib_time"]
                ax.imshow(
                    somas_x[sorted_neuron_ids[0], :] > self.parameters["theta_som_rate_estimator"],
                    cmap="Blues",
                    alpha=0.4,
                    extent=[
                        np.min(time_x),
                        np.max(time_x),
                        0,
                        self.parameters["n_somas"],
                    ],
                )
                ax.set(aspect=somas_x.shape[1] / somas_x.shape[0])

                fig, ax = plt.subplots()
                ax.plot(
                    time_x,
                    np.sum(
                        somas_x[sorted_neuron_ids[0], :]
                        > self.parameters["theta_som_rate_estimator"],
                        axis=0,
                    ),
                )

        else:
            if axes == None:
                fig, axes = plt.subplots(2, 2, sharex=True, figsize=(10, 8))

                ax_inputs_1 = axes.flatten()[0]
                ax_inputs_2 = axes.flatten()[1]
                ax_sorted_all = axes.flatten()[2]
                ax_sorted_by_context = axes.flatten()[3]

            plot_spikes_of_inputs(ax1=ax_inputs_1, ax2=ax_inputs_2)

            unique_context_ids = list(np.sort(np.unique(self.parameters_for_run["all_context_ids"])))
            sorted_neuron_ids, _, _ = self.sort_neurons_by_firing_rate(
                reverse_order=True, shuffle_rest=shuffle_rest
            )
            for tt in range(len(self.parameters_for_run["all_assembly_ids"])):
                context_id = self.parameters_for_run["all_context_ids"][tt]
                contex_id_index = unique_context_ids.index(context_id)

                start = (rtm + bsl) * tt + bsl * int(tt > 0)
                end = start + rtm + bsl + bsl * int(tt < 1)
                print(f"START ({tt}) : ", start)
                print("END: ", end)

                for ii, neuron_index in enumerate(sorted_neuron_ids[contex_id_index]):
                    spike_times_for_neuron = somas_time[somas_i == neuron_index]

                    spikes_are_after = spike_times_for_neuron > start
                    spikes_are_before = spike_times_for_neuron < end
                    spike_times_for_neuron = spike_times_for_neuron[
                        np.logical_and(spikes_are_before, spikes_are_after)
                    ]

                    ax = ax_sorted_all
                    ax.vlines(spike_times_for_neuron, ymin=ii - 0.5, ymax=ii + 0.5, colors="k")
                    ax.set_title(
                        f"Sorted for corresponding context",
                    )

                ylim = ax.get_ylim()
                ax.annotate(
                    f"#{context_id}",
                    xy=((start + end) / 2, ylim[1] + (ylim[1] - ylim[0]) * 0.05),
                    ha="center",
                    va="center",
                )

            ylim = ax.get_ylim()
            ylim = [ylim[0], ylim[1] + (ylim[1] - ylim[0]) * 0.12]
            ax.set(ylim=ylim)

            sort_for_context = 0
            for ii, neuron_index in enumerate(sorted_neuron_ids[sort_for_context]):
                spike_times_for_neuron = somas_time[somas_i == neuron_index]

                ax = ax_sorted_by_context
                ax.vlines(spike_times_for_neuron, ymin=ii - 0.5, ymax=ii + 0.5, colors="k")
                ax.set_title(
                    f"Sorted for context {sort_for_context}",
                )

                # if show_SOM_activity:
                #     spike_times_SOM = self.save_dict["spikes_rec_inhib_t"][
                #         self.save_dict["spikes_rec_inhib_i"] == neuron_index
                #     ]
                #     ax.vlines(
                #         spike_times_SOM, ymin=ii - 0.3, ymax=ii + 0.3, colors="r", alpha=0.3
                #     )

        fig, ax = plt.subplots(2)
        ax[0].plot(self.save_dict["inhibitory_time"], self.save_dict["inhibitory_rate"][0, :])
        ax[1].plot(self.save_dict["x_time"], self.save_dict["x_value"][0, :])

        if show_plot:
            plt.show()

    def show_traces_for_potentially_potentiated_dendrites(self, show_plot=False):
        if not self.parameters_for_run["debug_mode"]:
            print("cannot show these details in non-debug mode")
            return

        pot_pot = self.potenitially_potenitated_dendrites

        fig_dim = int(np.ceil(np.sqrt(len(pot_pot) / 2)))

        print(fig_dim, 2 * fig_dim)

        fig, axes_dendrites = plt.subplots(fig_dim, 2 * fig_dim, sharex=True, sharey=True)
        fig, axes_weights = plt.subplots(fig_dim, 2 * fig_dim, sharex=True, sharey=True)
        fig, axes_dist = plt.subplots(fig_dim, 2 * fig_dim, sharex=True, sharey=True)
        for ii in range(len(pot_pot)):
            dend_id = pot_pot[ii]
            pre_som_neurons = self.area.synapse_som_to_dend.i[
                np.where(self.area.synapse_som_to_dend.j == dend_id)[0]
            ]

            print(pre_som_neurons)

            rec_inhib_n_active = self.save_dict["rec_inhib_n_active"][pre_som_neurons, :]

            # print(rec_inhib_n_active.shape)
            # print((self.save_dict["rec_inhib_n_active"][pre_som_neurons, :]).shape)
            # return

            ax = axes_dendrites.flatten()[ii]
            for mm in range(rec_inhib_n_active.shape[0]):
                rec_inhib_time = self.save_dict["rec_inhib_time"]
                rec_inhib_time = rec_inhib_time[np.where(rec_inhib_n_active[mm] > 7)[0]]
                rec_inhib_n_active_selection = [np.where(rec_inhib_n_active[mm] > 7)[0]]
                ax.scatter(
                    rec_inhib_time,
                    np.ones_like(rec_inhib_n_active_selection) + mm,
                    color="#fe9929",
                    s=6,
                    alpha=(1 + mm) / ((rec_inhib_n_active.shape[0])),
                )

            ax.plot(
                self.save_dict["voltage_dends_t"],
                self.save_dict[f"voltage_dends_potentially_potentiated"][ii],
            )

            title = (
                f"{self.area.counts_gated[0][pot_pot[ii]] + self.area.counts_gated[1][pot_pot[ii]]}"
            )

            if self.parameters_for_run["all_assembly_ids"][0][1] == -1:
                title = f"{self.area.counts_gated[0][pot_pot[ii]]}"
            if self.parameters_for_run["all_assembly_ids"][0][0] == -1:
                title = f"{self.area.counts_gated[1][pot_pot[ii]]}"

            ax.set_title(title)

            if ii % (2 * fig_dim) == 0:
                ax.set_ylabel("Dendritic voltage in mV")
            if ii // (2 * fig_dim) == fig_dim - 1:
                ax.set_xlabel("Time in mS")

            ax = axes_weights.flatten()[ii]
            ax.set_title(title)
            w_sum = []
            gather_weights = []
            for w in self.save_dict[f"weight_w_{ii}_pot_pot"]:
                gather_weights.append(w[-1])
                ax.plot(self.save_dict["voltage_weights_t"], w, color="k", alpha=0.2)
                w_sum.append(w)

            for w in self.save_dict[f"weight_w_ff_1_{ii}_pot_pot"]:
                ax.plot(self.save_dict["voltage_weights_t"], w, color="#1d91c0", alpha=0.2)
                w_sum.append(w)

            for w in self.save_dict[f"weight_w_ff_2_{ii}_pot_pot"]:
                ax.plot(self.save_dict["voltage_weights_t"], w, color="#54278f", alpha=0.2)
                w_sum.append(w)

            ax = axes_dist.flatten()[ii]
            ax.set_title(title)
            ax.hist(gather_weights, bins=np.linspace(0, self.parameters["w_max_rec"], 50), color="k")
            ax.set_ylim([0, 35])

            if ii % (2 * fig_dim) == 0:
                ax.set_ylabel("weight")

            if ii // (2 * fig_dim) == fig_dim - 1:
                ax.set_xlabel("Time in mS")

        if show_plot:
            plt.show()

    def show_traces_for_example_neurons(self, gated=True, show_plot=False):
        for ii in range(10):
            print(f"######### Final weights above {ii}")
            print(np.sum(self.save_dict["weights"] > ii))
        fig, ax = plt.subplots()
        ax.hist(self.save_dict["weights"].flatten(), bins=np.linspace(0, 12, 40))

        counts = self.select_ids_gated_counts
        name = "gated"
        main_color = "#2171b5"
        neighbour_color = "#fd8d3c"
        if not gated:
            counts = self.select_ids_non_gated_counts
            name = "non_gated"
            main_color = "#fd8d3c"
            neighbour_color = "#6a51a3"

        fig, ax_w_sum = plt.subplots(1, len(counts))
        fig, axes = plt.subplots(3, len(counts), sharex=True)
        for ii in range(len(counts)):
            ax = axes[0, ii]
            if ii > 0:
                ax.set_yticklabels([])
                ax.sharey(axes[0, 0])

            ax.plot(
                self.save_dict["voltage_dends_t"],
                self.save_dict[f"voltage_dends_{name}"][ii],
                label=f"{name}",
            )
            ax.plot(
                self.save_dict["voltage_dends_t"],
                self.save_dict[f"voltage_dends_{name}_neighbour"][ii],
                label="random neighbour",
            )
            ax.set_title(f"n act. syn: {counts[ii]}")
            if ii == 0:
                ax.set_ylabel("Dendritic voltage in mV")

            ax = axes[1, ii]
            if ii > 0:
                ax.set_yticklabels([])
                ax.sharey(axes[1, 0])

            w_sum = []
            for w in self.save_dict[f"weight_w_{ii}_{name}"]:
                ax.plot(self.save_dict["voltage_weights_t"], w, color="k", alpha=0.2)
                w_sum.append(w)

            try:
                for w in self.save_dict[f"weight_w_ff_1_{ii}_{name}"]:
                    ax.plot(self.save_dict["voltage_weights_t"], w, color="#1d91c0", alpha=0.2)
                    w_sum.append(w)

                for w in self.save_dict[f"weight_w_ff_2_{ii}_{name}"]:
                    ax.plot(self.save_dict["voltage_weights_t"], w, color="#54278f", alpha=0.2)
                    w_sum.append(w)
            except KeyError:
                # This means we have an old version of the save dict
                for w in self.save_dict[f"weight_w_ff_{ii}_{name}"]:
                    ax.plot(self.save_dict["voltage_weights_t"], w, color="#1d91c0", alpha=0.2)
                    w_sum.append(w)

            if ii == 0:
                ax.set_ylabel("weight")
                ax_w_sum[ii].set_ylabel("Summed Weight")
            ax_w_sum[ii].plot(self.save_dict["voltage_weights_t"], np.sum(w_sum, axis=0))
            ax_w_sum[ii].set_xlabel("Time in ms")
            ax_w_sum[ii].set_title(self.area.dends[self.select_ids_gated[ii]].w_tot[:])

            ax = axes[2, ii]
            if ii > 0:
                ax.set_yticklabels([])
                ax.sharey(axes[2, 0])

            ax.plot(self.save_dict["voltag_somas_t"], self.save_dict[f"voltage_soma_{name}"][0])
            ax.set_xlabel("Time in mS")
            if ii == 0:
                ax.set_ylabel("Somatic voltage in mV")

        if show_plot:
            plt.show()


if __name__ == "__main__":
    pass
