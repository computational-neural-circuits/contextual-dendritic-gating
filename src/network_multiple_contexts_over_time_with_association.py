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


class NetworMultipleContextsOverTimeWithAssociation(HandleParametersAndResults):
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

        self.monitor_everything = True
        if "monitor_everything" in self.parameters_for_run:
            self.monitor_everything = self.parameters_for_run["monitor_everything"]

        if self.create_network:
            self.network = br.Network()
            self.setup_network()
            self.create_monitors()

    def setup_network(self):
        print("Setup the network for multiple contexts over time")
        br.seed(self.parameters_for_run["seed"])
        np.random.seed(self.parameters_for_run["seed"])
        self.area = Area(
            network=self.network,
            eqs=self.equations,
            params={**self.parameters, **self.parameters_for_run},
        )

    def create_monitors(self):
        # monitor the spikes of the somas and the inputs
        area = self.area

        self.spM_somas = br.SpikeMonitor(area.somas, name=f"somata_monitor_{area.name}")
        self.network.add(self.spM_somas)
        self.spM_inputs = []
        for iu, input_units in enumerate([area.input_units_1, area.input_units_2]):
            spm = br.SpikeMonitor(input_units, name=f"input_monitor_{iu}_{area.name}")
            self.network.add(spm)
            self.spM_inputs.append(spm)

        if not self.monitor_everything:
            return
        self.potenitially_potenitated_dendrites = {}
        all_ids = []
        for context_id, assembly_ids in zip(
            self.parameters_for_run["all_context_ids"], self.parameters_for_run["all_assembly_ids"]
        ):
            assembly_id_0 = assembly_ids[0]
            assembly_id_1 = assembly_ids[1]

            if assembly_id_0 == 1992:
                subset_0 = self.parameters_for_run["presynaptic_sources"]
            elif assembly_id_0 > -1:
                subset_0 = [ii + 20 * assembly_id_0 for ii in range(20)]
            else:
                subset_0 = []

            if assembly_id_1 == 1992:
                subset_1 = self.parameters_for_run["presynaptic_sources"]
            elif assembly_id_1 > -1:
                subset_1 = [ii + 20 * assembly_id_1 for ii in range(20)]
            else:
                subset_1 = []

            area.count_number_of_inputs_from_subsets_in_context(
                context_id=context_id, subsets=[subset_0, subset_1]
            )

            if assembly_id_0 > -1 and assembly_id_1 == -1:
                name = f"context_{context_id}_assembly_{assembly_id_0}"
                self.potenitially_potenitated_dendrites[name] = []
                for n_syn in range(3, 10):
                    X = list(np.where(area.counts_gated[0] == n_syn)[0])
                    self.potenitially_potenitated_dendrites[name] += X
                    all_ids += X

            if assembly_id_0 == -1 and assembly_id_1 > -1:
                name = f"context_{context_id}_assembly_{assembly_id_1}"
                self.potenitially_potenitated_dendrites[name] = []
                for n_syn in range(3, 10):
                    X = list(np.where(area.counts_gated[1] == n_syn)[0])
                    self.potenitially_potenitated_dendrites[name] += X
                    all_ids += X

            if assembly_id_0 > -1 and assembly_id_1 > -1:
                name = f"context_{context_id}_assembly_{assembly_id_0}_{assembly_id_1}"
                self.potenitially_potenitated_dendrites[name] = []
                for n_syn in range(5, 12):
                    X = list(np.where((area.counts_gated[0] + area.counts_gated[1]) == n_syn)[0])
                    self.potenitially_potenitated_dendrites[name] += X
                    all_ids += X

        self.rec_list = []
        for pot_pot in all_ids:
            if pot_pot not in self.rec_list:
                self.rec_list.append(pot_pot)

        monitor_list = np.array([np.where((area.synapses_E.j)[:] == kk)[0] for kk in self.rec_list])

        print("This is what we monitor")
        print(np.array(monitor_list).shape)

        self.Msyns_w = br.StateMonitor(
            area.synapses_E,
            "w",
            record=monitor_list.flatten(),
            dt=self.parameters["monitor_dt_weights"],
        )

        self.network.add(self.Msyns_w)

        monitor_list = []
        for kk in self.rec_list:
            monitor_list += list(np.where((area.input_synapses[0].j)[:] == kk)[0])

        self.Mff_w = br.StateMonitor(
            area.input_synapses[0],
            "w",
            record=monitor_list,
            dt=self.parameters["monitor_dt_weights"],
        )
        self.network.add(self.Mff_w)

        print("Done with setup")

    def create_save_dict(self):
        area = self.area
        weights_recurrent = np.zeros(shape=(area.n_somas, area.n_dends))
        weights_recurrent[area.srcs, area.tgts] = area.synapses_E.w

        weights_ff_1 = np.zeros(shape=(area.n_somas, area.n_dends))
        weights_ff_2 = np.zeros(shape=(area.n_somas, area.n_dends))

        weights_ff_1[area.input_synapses[0].i[:], area.input_synapses[0].j[:]] = area.input_synapses[
            0
        ].w
        weights_ff_2[area.input_synapses[1].i[:], area.input_synapses[1].j[:]] = area.input_synapses[
            1
        ].w

        self.save_dict = {
            f"spikes_somas_t": self.spM_somas.t / ms,
            f"spikes_somas_i": self.spM_somas.i,
            f"spikes_inputs_t_1": self.spM_inputs[0].t / ms,
            f"spikes_inputs_t_2": self.spM_inputs[1].t / ms,
            f"spikes_inputs_i_1": self.spM_inputs[0].i,
            f"spikes_inputs_i_2": self.spM_inputs[1].i,
        }

        if not self.monitor_everything:
            return

        self.save_dict["weights"] = weights_recurrent
        self.save_dict["weights_ff_1"] = weights_ff_1
        self.save_dict["weights_ff_2"] = weights_ff_2

        for key, val in self.potenitially_potenitated_dendrites.items():
            self.save_dict[f"pot_pot_{key}"] = val

        self.save_dict["weights_t"] = self.Msyns_w.t / ms
        self.save_dict[f"pot_pot_recurrent_w"] = self.Msyns_w.w
        self.save_dict["pot_pot_ff_w"] = self.Mff_w.w

        self.save_dict["weights"] = weights_recurrent
        self.save_dict["weights_ff_1"] = weights_ff_1
        self.save_dict["weights_ff_2"] = weights_ff_2

    def run(self, report_period=10 * second, report_style=None):
        runtime_baseline = self.parameters_for_run["runtime_baseline"]
        runtime_imprint = self.parameters_for_run["runtime_imprint"]
        all_context_ids = self.parameters_for_run["all_context_ids"]
        all_assembly_ids = self.parameters_for_run["all_assembly_ids"]

        presynaptic_sources_single = None
        if "presynaptic_sources" in self.parameters_for_run:
            presynaptic_sources_single = self.parameters_for_run["presynaptic_sources"]

        presynaptic_sources = [None, None]
        if "presynaptic_sources_1" in self.parameters_for_run:
            presynaptic_sources[0] = self.parameters_for_run["presynaptic_sources_1"]
        if "presynaptic_sources_2" in self.parameters_for_run:
            presynaptic_sources[1] = self.parameters_for_run["presynaptic_sources_2"]

        if self.check_for_results():
            return self.save_dict

        elif self.only_load_results:
            return None

        self.area.start_context(0)
        self.area.input_units_1[:].rates = self.parameters["ff_bck"]
        self.area.input_units_2[:].rates = self.parameters["ff_bck"]

        self.network.run(runtime_baseline, report=report_style, report_period=report_period)
        for context_id, assembly_ids in zip(all_context_ids, all_assembly_ids):
            self.area.stop_context()
            self.area.start_context(context_id)

            if assembly_ids[0] == 1992:
                ps = presynaptic_sources_single
                if assembly_ids[1] == 1992:
                    ps = presynaptic_sources[0]
                for n_id in ps:
                    self.area.input_units_1[n_id : n_id + 1].rates = self.parameters[
                        "assembly_firing_rate"
                    ]
            elif assembly_ids[0] >= 0:
                self.area.input_units_1[
                    assembly_ids[0]
                    * self.parameters["assembly_size"] : (assembly_ids[0] + 1)
                    * self.parameters["assembly_size"]
                ].rates = self.parameters["assembly_firing_rate"]

            if assembly_ids[1] == 1992:
                ps = presynaptic_sources_single
                if assembly_ids[0] == 1992:
                    ps = presynaptic_sources[1]
                for n_id in ps:
                    self.area.input_units_2[n_id : n_id + 1].rates = self.parameters[
                        "assembly_firing_rate"
                    ]

            elif assembly_ids[1] >= 0:
                self.area.input_units_2[
                    assembly_ids[1]
                    * self.parameters["assembly_size"] : (assembly_ids[1] + 1)
                    * self.parameters["assembly_size"]
                ].rates = self.parameters["assembly_firing_rate"]

            self.network.run(runtime_imprint, report=report_style, report_period=report_period)

            self.area.input_units_1[:].rates = self.parameters["ff_bck"]
            self.area.input_units_2[:].rates = self.parameters["ff_bck"]

            self.network.run(runtime_baseline, report=report_style, report_period=report_period)

            self.create_save_dict()
            self.save_results()
            if self.monitor_everything:
                self.generate_results()

    def show_weight_matrix(self, show_plot=False, matlab_export_name=None):
        weights_loaded = self.save_dict["weights"]
        print(self.save_dict["weights"].shape)

        matlab_save_dict = {"all_weights": weights_loaded}

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

            sorted_neuron_ids, _ = self.sort_neurons_by_firing_rate(reverse_order=False)

            weights = weights[np.ix_(sorted_neuron_ids[ii], sorted_neuron_ids[ii])]

            im = ax2.imshow(weights.T, cmap="Greys")
            plt.colorbar(im)

        sorted_neuron_ids, _ = self.sort_neurons_by_firing_rate(reverse_order=False)

        matlab_save_dict = {"all_weights_not_sorted": weights_loaded.T}
        for context_id in [-1, 0, 1, 2]:
            sorted_dendrite_ids = []
            for snid in sorted_neuron_ids[0]:
                for ii in range(self.parameters["n_dend_each"]):
                    if ii != context_id:
                        if context_id != -1:
                            continue

                    sorted_dendrite_ids.append(snid * self.parameters["n_dend_each"] + ii)

            # weights_reduced = weights_loaded[::, ::]
            # if context_id != -1:
            #     weights_reduced = weights_loaded[::, context_id :: self.parameters["n_dend_each"]]
            weights_sorted = weights_loaded[np.ix_(sorted_neuron_ids[0], sorted_dendrite_ids[:])]

            print("asd:, ", context_id, weights_sorted.shape)
            save_name = "all_weights_sorted_by_context_0"
            if context_id != -1:
                save_name = f"weights_of_context_{context_id}_sorted_by_context_{context_id}"

            matlab_save_dict[save_name] = weights_sorted.T
        if matlab_export_name is not None:
            scipy.io.savemat(f"{matlab_export_name}.mat", mdict=matlab_save_dict)
            for key, val in matlab_save_dict.items():
                np.savetxt(f"{matlab_export_name}_{key}.txt", val)
        if show_plot:
            plt.show()

    def sort_neurons_by_firing_rate(
        self,
        reverse_order=True,
        shuffle_rest=True,
        sort_only_for_contexts=None,
        sort_for_specific_imprint=None,
    ):
        somas_time = np.copy(self.save_dict["spikes_somas_t"])
        somas_i = np.copy(self.save_dict["spikes_somas_i"])

        all_context_ids = self.parameters_for_run["all_context_ids"]
        all_assembly_ids = self.parameters_for_run["all_assembly_ids"]

        unique_context_ids = list(np.sort(np.unique(all_context_ids)))
        all_firing_rates = [{} for nn in unique_context_ids]

        bsl = self.parameters_for_run["runtime_baseline"] / msecond
        rtm = self.parameters_for_run["runtime_imprint"] / msecond

        for ii, (context_id, assembly_ids) in enumerate(zip(all_context_ids, all_assembly_ids)):
            contex_id_index = unique_context_ids.index(context_id)

            start = bsl + (rtm + bsl) * ii
            end = start + rtm

            all_firing_rates_somas = []
            for neuron_index in range(self.parameters["n_somas"]):
                spike_times_for_neuron = somas_time[somas_i == neuron_index]
                firing_rate = get_firing_rate_for_single_neuron(
                    start=start, end=end, spike_times_for_neuron=spike_times_for_neuron
                )
                all_firing_rates_somas.append(firing_rate)
            all_firing_rates[contex_id_index][
                f"{assembly_ids[0]}_{assembly_ids[1]}"
            ] = all_firing_rates_somas

        sorted_neuron_ids = [[] for nn in unique_context_ids]
        selected_ids = [{} for nn in unique_context_ids]
        for ii, context_id in enumerate(unique_context_ids):
            for tt, (assembly_id, all_rates) in enumerate(all_firing_rates[ii].items()):
                if sort_for_specific_imprint is not None:
                    if tt != sort_for_specific_imprint:
                        continue

                all_rates = np.array(all_rates)
                ids_of_assembly = get_assembly_neuron_ids_by_weight_and_rate(
                    net=self,
                    all_rates=all_rates,
                    context_id=context_id,
                    area=self.area,
                )

                print("#############_____________________________________________")
                print("context_id", context_id)
                print("assembly_id", assembly_id)
                print("ids_of_assembly", ids_of_assembly)
                print("_____________________________________________")

                new_assembly_ids = [si for si in ids_of_assembly if si not in sorted_neuron_ids[ii]]
                selected_ids[ii][assembly_id] = ids_of_assembly
                sorted_neuron_ids[ii] += new_assembly_ids

            sorted_neuron_ids[ii] += [
                ni for ni in range(self.parameters["n_somas"]) if ni not in sorted_neuron_ids[ii]
            ]

        print("Final selected_ids")
        print(selected_ids)

        if not sort_for_specific_imprint:
            sorted_neuron_ids = np.array(sorted_neuron_ids)
        return sorted_neuron_ids, selected_ids

    def show_spike_rasters(self, show_plot=False, order=None, axes=None):
        somas_time = self.save_dict["spikes_somas_t"]
        somas_i = self.save_dict["spikes_somas_i"]
        inputs_2_time = self.save_dict["spikes_inputs_t_2"]
        inputs_1_time = self.save_dict["spikes_inputs_t_1"]
        inputs_1_i = self.save_dict["spikes_inputs_i_1"]
        inputs_2_i = self.save_dict["spikes_inputs_i_2"]

        all_context_ids = self.parameters_for_run["all_context_ids"]
        all_assembly_ids = self.parameters_for_run["all_assembly_ids"]

        colors = [
            ["#08306b", "#08519c", "#4292c6", "#9ecae1", "#deebf7"],
            ["#00441b", "#006d2c", "#41ab5d", "#a1d99b", "#e5f5e0"],
            ["#7f2704", "#d94801", "#fd8d3c", "#fdd0a2", "#fee6ce"],
        ]

        unique_context_ids = list(np.sort(np.unique(self.parameters_for_run["all_context_ids"])))

        if axes is None:
            n_axes_needed = len(unique_context_ids) + 2  # +2 for the inputs

            # we want 2 cols

            n_rows = int(np.ceil(n_axes_needed / 2.0))

            fig, axes = plt.subplots(n_rows, 2, sharex=True, figsize=(10, 8))

        ax_inputs_1 = axes.flatten()[0]
        ax_inputs_2 = axes.flatten()[1]
        sorted_neuron_ids, _ = self.sort_neurons_by_firing_rate(reverse_order=False)

        if order is not None:
            sorted_neuron_ids = order

        bsl = self.parameters_for_run["runtime_baseline"] / msecond
        rtm = self.parameters_for_run["runtime_imprint"] / msecond

        n_assembly = [-1 for ii in unique_context_ids]
        for tt, (context_id, assembly_id) in enumerate(zip(all_context_ids, all_assembly_ids)):
            start = bsl * int(tt > 0) + (rtm + bsl) * tt
            end = start + rtm + bsl + bsl * int(tt < 1)

            n_assembly[unique_context_ids.index(context_id)] += 1

            for neuron_index in range(self.parameters["n_somas"]):
                for ax, inputs_time, inputs_i in zip(
                    [ax_inputs_1, ax_inputs_2],
                    [inputs_1_time, inputs_2_time],
                    [inputs_1_i, inputs_2_i],
                ):
                    spike_times_for_neuron = inputs_time[inputs_i == neuron_index]
                    spikes_are_after = spike_times_for_neuron > start
                    spikes_are_before = spike_times_for_neuron < end

                    ax.vlines(
                        spike_times_for_neuron[np.logical_and(spikes_are_before, spikes_are_after)],
                        ymin=neuron_index - 0.5,
                        ymax=neuron_index + 0.5,
                        colors=colors[unique_context_ids.index(context_id)][
                            n_assembly[unique_context_ids.index(context_id)]
                        ],
                    )

        print(somas_i.shape, somas_time.shape)
        for jj, context_id in enumerate(
            np.sort(np.unique(self.parameters_for_run["all_context_ids"]))
        ):
            for ii, neuron_index in enumerate(sorted_neuron_ids[jj]):
                spike_times_for_neuron = somas_time[somas_i == neuron_index]
                ax = axes.flatten()[jj + 2]
                ax.vlines(spike_times_for_neuron, ymin=ii - 0.5, ymax=ii + 0.5, colors="k")
                ax.set_title(
                    f"Sorted for context {context_id}",
                    color=colors[unique_context_ids.index(context_id)][0],
                )

        ax_inputs_1.set_title("Inputs (1)")
        ax_inputs_2.set_title("Inputs (2)")
        # ax_somas.set_title("Area")
        # ax_somas.set(xlabel="Time in ms", ylim=ax_inputs.get_ylim())

        if show_plot:
            plt.show()

    def show_traces_for_potentially_potentiated_dendrites(self, show_plot=False):
        pot_pot = self.potenitially_potenitated_dendrites

        fig_dim = int(np.ceil(np.sqrt(len(pot_pot) / 2)))

        print(fig_dim, 2 * fig_dim)

        fig, axes_dendrites = plt.subplots(fig_dim, 2 * fig_dim, sharex=True, sharey=True)
        fig, axes_weights = plt.subplots(fig_dim, 2 * fig_dim, sharex=True, sharey=True)
        fig, axes_dist = plt.subplots(fig_dim, 2 * fig_dim, sharex=True, sharey=True)
        for ii in range(len(pot_pot)):
            ax = axes_dendrites.flatten()[ii]
            ax.plot(
                self.save_dict["voltage_dends_t"],
                self.save_dict[f"voltage_dends_potentially_potentiated"][ii],
            )
            ax.set_title(f" ({self.area.counts_gated[0][pot_pot[ii]]})")

            if ii % (2 * fig_dim) == 0:
                ax.set_ylabel("Dendritic voltage in mV")
            if ii // (2 * fig_dim) == fig_dim - 1:
                ax.set_xlabel("Time in mS")

            ax = axes_weights.flatten()[ii]
            ax.set_title(f"{self.area.counts_gated[0][pot_pot[ii]]}")
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
            ax.set_title(
                f"#{pot_pot[ii]} ({self.area.counts_gated[0][pot_pot[ii]]}/{self.area.dends.w_tot[ii]})"
            )
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

    def generate_results(self, axes=None, show_plot=False, save_fig=True):
        # self.show_weight_matrix(
        #     matlab_export_name=f"../../results/figures/simulation_results/fig_D_weights_seed_{self.parameters_for_run['seed']}"
        # )

        if axes is None:
            fig, axes = plt.subplots(3, 2, figsize=(24, 16), sharex=True)

        self.show_spike_rasters(show_plot=False, axes=axes)

        all_assembly_weights = []

        unique_contexts = list(np.sort(np.unique(self.parameters_for_run["all_context_ids"])))
        # unique_assembly_ids = list(np.sort(np.unique(self.parameters_for_run["all_assembly_ids"])))

        # all_weights = [[[] for ii in unique_assembly_ids] for jj in unique_contexts]

        ax = axes.flatten()[-1]

        colors = [
            ["#08306b", "#08519c", "#4292c6", "#9ecae1", "#deebf7"],
            ["#00441b", "#006d2c", "#41ab5d", "#a1d99b", "#e5f5e0"],
            ["#7f2704", "#d94801", "#fd8d3c", "#fdd0a2", "#fee6ce"],
        ]

        n_assembly = [-1 for ii in unique_contexts]

        for context_id, assembly_ids in zip(
            self.parameters_for_run["all_context_ids"], self.parameters_for_run["all_assembly_ids"]
        ):
            assembly_neuron_ids = self.get_assembly_neuron_ids(context_id, assembly_ids)
            print("ASSEMBLY NEURON IDS", assembly_neuron_ids)

            aa = unique_contexts.index(context_id)
            # bb = unique_assembly_ids.index(assembly_id)

            n_assembly[aa] += 1
            all_weights = []

            all_post_dend_ids = []

            for ii, post_dend_id in enumerate(self.rec_list):
                post_neuron_id = post_dend_id // self.parameters["n_dend_each"]

                if post_neuron_id in assembly_neuron_ids:
                    if post_dend_id % self.parameters["n_dend_each"] == context_id:
                        # now we go through all weights and see if they come from within assembly neurons

                        synapse_ids = np.where((self.area.synapses_E.j)[:] == post_dend_id)[0]

                        synapse_sources = self.area.synapses_E.i[synapse_ids]

                        synapses_from_assembly = [
                            mm
                            for mm, source in enumerate(synapse_sources)
                            if source in assembly_neuron_ids
                        ]

                        print(
                            "#",
                            len(synapses_from_assembly),
                            len(assembly_neuron_ids),
                            post_dend_id,
                            post_neuron_id,
                        )

                        weights = self.save_dict["pot_pot_recurrent_w"][
                            ii
                            * (self.parameters["n_somas"] - 1) : (ii + 1)
                            * (self.parameters["n_somas"] - 1),
                            :,
                        ]  # n_somas - 1, since we have no self connection

                        weights_from_within = weights[synapses_from_assembly, :]

                        # all_weights[aa][bb].append(weights_from_within)
                        all_weights.append(weights_from_within)

            try:
                # all_weights[aa][bb] = np.vstack(all_weights[aa][bb])
                # these_weights = all_weights[aa][bb]
                all_weights = np.vstack(all_weights)
                these_weights = all_weights
                print("$$, ", these_weights.shape)
                ax.plot(
                    self.save_dict["weights_t"],
                    np.mean(these_weights, axis=0),
                    label=f"ctxt:{context_id} | assembly:{assembly_ids[0]} - {assembly_ids[1]}",
                    color=colors[aa][n_assembly[aa]],
                )

                # ax.plot(
                #     net.save_dict["weights_t"],
                #     (these_weights.T)[:, :40],
                #     color=colors[aa][bb],
                #     lw=0.8,
                #     alpha=0.4,
                # )
            except ValueError:
                pass

        ax.legend()
        ax.set(xlabel="Time in ms", ylabel="average weight within assembly")

        if save_fig:
            fig.savefig(
                f"../../results/figures/intermediate_result_seed_{self.parameters_for_run['seed']}.pdf"
            )

        if show_plot:
            plt.show()

    def get_assembly_neuron_ids(self, context_id, assembly_ids):
        _, selected_ids = self.sort_neurons_by_firing_rate()

        assembly_neuron_ids = selected_ids[context_id][f"{assembly_ids[0]}_{assembly_ids[1]}"]

        return assembly_neuron_ids


if __name__ == "__main__":
    pass
