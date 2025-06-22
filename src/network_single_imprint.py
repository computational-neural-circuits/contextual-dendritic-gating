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


class NetworkSingleImprint(HandleParametersAndResults):
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

        if self.create_network:
            self.network = br.Network()
            self.setup_network()
            self.create_monitors()

    def setup_network(self):
        print("Setup the network for single imprint")
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
            self.spM_inputs.append(spm)
            self.network.add(spm)

        area.count_number_of_inputs_from_subsets_in_context()

        unique_counts = np.sort(np.unique(area.counts_gated[0]))[:-1].astype(
            int
        )  # remove the NaN value
        unique_counts_non_gated = np.sort(np.unique(area.counts_non_gated[0]))[:-1].astype(
            int
        )  # remove the NaN value

        # print(unique_counts)
        # print(unique_counts_non_gated)

        # randomly select samples to visualize
        select_ids_gated = []
        self.select_ids_gated_counts = []
        for uc in np.sort(unique_counts):
            select_ids_gated.append(np.argmax(area.counts_gated[0] == uc))
            self.select_ids_gated_counts.append(uc)

        self.potenitially_potentiated_dendrites = []
        if self.parameters_for_run["run_association"]:
            for n_syn in range(5, 12):
                self.potenitially_potentiated_dendrites += list(
                    np.where(area.counts_gated[0] == n_syn)[0]
                )
        else:
            for n_syn in range(3, 10):
                self.potenitially_potentiated_dendrites += list(
                    np.where(area.counts_gated[0] == n_syn)[0]
                )
        # print("Potentially potentiated for context 0:")
        # print(len(self.potenitially_potentiated_dendrites))
        # print(self.potenitially_potentiated_dendrites)

        select_ids_non_gated = []
        self.select_ids_non_gated_counts = []
        for uc in np.sort(unique_counts_non_gated):
            select_ids_non_gated.append(np.argmax(area.counts_non_gated[0] == uc))
            self.select_ids_non_gated_counts.append(uc)

        self.select_ids_gated = select_ids_gated
        self.select_ids_non_gated = select_ids_non_gated

        self.Mdends_V = {}
        self.Msomas_V = {}
        self.Msyns_w = {}
        self.Mff_w = {}
        for name, id_list in zip(["gated", "non_gated"], [select_ids_gated, select_ids_non_gated]):
            mon = br.StateMonitor(area.dends, "V", record=id_list, dt=self.parameters["monitor_dt"])
            self.Mdends_V[name] = mon
            self.network.add(mon)

            neighbour_ids = [
                s_id + 1 if (s_id + 1) % self.parameters["n_dend_each"] != 0 else s_id - 1
                for s_id in id_list
            ]
            print(neighbour_ids)
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

        mon = br.StateMonitor(
            area.dends,
            "V",
            record=self.potenitially_potentiated_dendrites,
            dt=self.parameters["monitor_dt"],
        )
        self.Mdends_V["pot_pot"] = mon
        self.network.add(mon)

        to_monitor = "w"
        monitor_list = []
        for kk in self.potenitially_potentiated_dendrites:
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
        for kk in self.potenitially_potentiated_dendrites:
            for ii in range(2):
                mon = br.StateMonitor(
                    area.input_synapses[ii],
                    to_monitor,
                    record=(area.input_synapses[ii])[np.where(area.input_synapses[ii].j == kk)[0]],
                    dt=self.parameters["monitor_dt_weights"],
                )
                monitor_list.append(mon)
                self.network.add(mon)
        self.Mff_w["pot_pot"] = monitor_list

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

        self.spM_rec_inhib = br.SpikeMonitor(area.rec_inihib_pop)
        self.network.add(self.spM_rec_inhib)

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
            f"voltage_dends_t": self.Mdends_V["gated"][0].t / ms,
            f"voltag_somas_t": self.Msomas_V["gated"][0].t / ms,
            f"voltage_weights_t": self.Msyns_w["gated"][0].t / ms,
        }

        self.save_dict["inhibitory_rate"] = self.Minhibition.rate / Hz
        self.save_dict["inhibitory_time"] = self.Minhibition.t / ms
        self.save_dict["x_time"] = self.Mx.t / ms
        self.save_dict["x_value"] = self.Mx.x_pop

        self.save_dict["rec_inhib_time"] = self.M_n_active.t / ms
        self.save_dict["rec_inhib_n_active"] = self.M_n_active.n_active

        self.save_dict["somas_x"] = self.M_somas_x.x_estimator

        self.save_dict["spikes_rec_inhib_t"] = self.spM_rec_inhib.t / ms
        self.save_dict["spikes_rec_inhib_i"] = self.spM_rec_inhib.i

        self.save_dict["weights"] = weights_recurrent
        self.save_dict["weights_ff_1"] = weights_ff_1
        self.save_dict["weights_ff_2"] = weights_ff_2

        for name, counts in zip(
            ["gated", "non_gated"], [self.select_ids_gated_counts, self.select_ids_non_gated_counts]
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

        self.save_dict["voltage_dends_potentially_potentiated"] = self.Mdends_V["pot_pot"].V / mV

        for ii, _ in enumerate(self.potenitially_potentiated_dendrites):
            ww = self.Msyns_w["pot_pot"][ii].w
            self.save_dict[f"weight_w_{ii}_pot_pot"] = ww

            ww_1 = self.Mff_w["pot_pot"][ii * 2].w
            ww_2 = self.Mff_w["pot_pot"][ii * 2 + 1].w
            self.save_dict[f"weight_w_ff_1_{ii}_pot_pot"] = ww_1
            self.save_dict[f"weight_w_ff_2_{ii}_pot_pot"] = ww_2

    def run(self, report_period=10 * second, report_style=None):
        runtime_baseline = self.parameters_for_run["runtime_baseline"]
        runtime_imprint = self.parameters_for_run["runtime_imprint"]
        run_association = self.parameters_for_run["run_association"]

        context_id = 0
        assembly_id = (0, 0)

        if self.check_for_results():
            return self.save_dict

        elif self.only_load_results:
            return None

        self.area.stop_context()
        self.area.start_context(context_id)

        self.area.input_units_1[:].rates = self.parameters["ff_bck"]
        self.area.input_units_2[:].rates = self.parameters["ff_bck"]

        self.network.run(runtime_baseline, report=report_style, report_period=report_period)

        self.area.input_units_1[: self.parameters["assembly_size"]].rates = self.parameters[
            "assembly_firing_rate"
        ]

        if run_association:
            self.area.input_units_2[: self.parameters["assembly_size"]].rates = self.parameters[
                "assembly_firing_rate"
            ]

        self.network.run(runtime_imprint, report=report_style, report_period=report_period)

        self.area.input_units_1[:].rates = self.parameters["ff_bck"]
        self.area.input_units_2[:].rates = self.parameters["ff_bck"]

        self.network.run(runtime_baseline, report=report_style, report_period=report_period)

        self.create_save_dict()
        self.save_results()

    def show_weight_matrix(self, show_plot=False, matlab_export_name=None):
        weights_loaded = self.save_dict["weights"]
        print(self.save_dict["weights"].shape)

        context_id = 0
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 6))

        # weights = weights_loaded[:, context_id::6]
        weights = weights_loaded[:, :]

        matlab_save_dict = {"all_weights": weights_loaded}

        im = ax1.imshow(weights.T, cmap="Greys", origin="lower")
        ax1.set(xlabel="Pre", ylabel="Post")

        print("new shape, ", weights.shape)

        sorted_neuron_ids, _ = self.sort_neurons_by_firing_rate(reverse_order=False)

        sorted_dendrite_ids = []
        for snid in sorted_neuron_ids:
            for ii in range(self.parameters["n_dend_each"]):
                sorted_dendrite_ids.append(snid * self.parameters["n_dend_each"] + ii)

        weights = weights[np.ix_(sorted_neuron_ids, sorted_dendrite_ids[:])]

        im = ax2.imshow(weights.T, cmap="Greys", origin="lower")

        matlab_save_dict[f"weights_sorted"] = weights

        ax2.set(xlabel="pre", ylabel="post")
        plt.colorbar(im)

        # G = nx.from_numpy_array(weights, create_using=nx.DiGraph)

        # # Community detection (convert to undirected for the community detection if necessary)
        # partition = community_louvain.best_partition(G.to_undirected(), weight="weight")

        # # Create a mapping of node index to community
        # node_community_map = {node: community for node, community in enumerate(partition.values())}

        # # Sort nodes by community
        # sorted_nodes = sorted(node_community_map, key=node_community_map.get)

        # Reorder the matrix accordingly
        weights = weights_loaded[:, context_id::6]
        weights = weights[np.ix_(sorted_neuron_ids, sorted_neuron_ids)]
        im = ax3.imshow(weights.T, cmap="Greys", origin="lower")

        matlab_save_dict[f"weights_in_context_sorted"] = weights
        ax3.set(xlabel="Pre", ylabel="Post", title="Sorted in context 0")
        plt.colorbar(im)

        if matlab_export_name is not None:
            scipy.io.savemat(f"{matlab_export_name}.mat", mdict=matlab_save_dict)

            for key, val in matlab_save_dict.items():
                np.savetxt(f"{matlab_export_name}_{key}.txt", val)

        if show_plot:
            plt.show()

    def sort_neurons_by_firing_rate(self, shuffle_rest=True, reverse_order=True):
        somas_time = np.copy(self.save_dict["spikes_somas_t"])
        somas_i = np.copy(self.save_dict["spikes_somas_i"])

        bsl = self.parameters_for_run["runtime_baseline"] / msecond
        rtm = self.parameters_for_run["runtime_imprint"] / msecond

        start = bsl
        end = start + rtm

        context_id = 0
        contex_id_index = 0

        all_firing_rates_somas = []
        for neuron_index in range(self.parameters["n_somas"]):
            spike_times_for_neuron = somas_time[somas_i == neuron_index]
            firing_rate = get_firing_rate_for_single_neuron(
                start=start, end=end, spike_times_for_neuron=spike_times_for_neuron
            )
            all_firing_rates_somas.append(firing_rate)

        sorted_neuron_ids = []

        print(all_firing_rates_somas)
        # get assembly ids
        selected_ids = get_assembly_neuron_ids_by_weight_and_rate(
            net=self,
            all_rates=np.array(all_firing_rates_somas),
            context_id=context_id,
            area=self.area,
        )

        sorted_neuron_ids = selected_ids + [
            si for si in range(self.parameters["n_somas"]) if si not in selected_ids
        ]

        return sorted_neuron_ids, selected_ids

    def show_spike_rasters(self, show_plot=False, order=None, axes=None, show_range=None):
        somas_time = self.save_dict["spikes_somas_t"]
        somas_i = self.save_dict["spikes_somas_i"]
        inputs_2_time = self.save_dict["spikes_inputs_t_2"]
        inputs_1_time = self.save_dict["spikes_inputs_t_1"]
        inputs_1_i = self.save_dict["spikes_inputs_i_1"]
        inputs_2_i = self.save_dict["spikes_inputs_i_2"]

        if axes is None:
            fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 8))

        ax_inputs_1 = axes.flatten()[0]
        ax_inputs_2 = axes.flatten()[1]
        ax_somas = axes.flatten()[2]
        sorted_neuron_ids, _ = self.sort_neurons_by_firing_rate(reverse_order=False)

        if order is not None:
            sorted_neuron_ids = order

        bsl = self.parameters_for_run["runtime_baseline"] / msecond
        rtm = self.parameters_for_run["runtime_imprint"] / msecond

        start = bsl
        end = start + rtm

        context_id = 0
        for neuron_index in range(self.parameters["n_somas"]):
            for ax, inputs_time, inputs_i in zip(
                [ax_inputs_1, ax_inputs_2],
                [inputs_1_time, inputs_2_time],
                [inputs_1_i, inputs_2_i],
            ):
                spike_times_for_neuron = inputs_time[inputs_i == neuron_index]

                ax.vlines(
                    spike_times_for_neuron,
                    ymin=neuron_index - 0.5,
                    ymax=neuron_index + 0.5,
                    colors="k",
                )

        print(somas_i.shape, somas_time.shape)

        for ii, neuron_index in enumerate(sorted_neuron_ids):
            spike_times_for_neuron = somas_time[somas_i == neuron_index]
            ax_somas.vlines(spike_times_for_neuron, ymin=ii - 0.5, ymax=ii + 0.5, colors="k")
            ax_somas.set_title(f"Sorted for context {context_id}", color="k")

        ax_inputs_1.set_title("Inputs (1)")
        ax_inputs_2.set_title("Inputs (2)")
        # ax_somas.set_title("Area")
        # ax_somas.set(xlabel="Time in ms", ylim=ax_inputs.get_ylim())

        # for ax in [ax_somas, ax_inputs]:
        #     ax.set(ylabel="Neuron Number")

        fig, inhib_axes = plt.subplots(2)
        inhib_axes[0].plot(
            self.save_dict["inhibitory_time"], self.save_dict["inhibitory_rate"][0, :]
        )
        inhib_axes[1].plot(self.save_dict["x_time"], self.save_dict["x_value"][0, :])

        if show_range is not None:
            for ax in [ax_inputs_1, ax_inputs_2, ax_somas, inhib_axes[0], inhib_axes[1]]:
                ax.set_xlim(show_range)

        if show_plot:
            plt.show()

    def show_traces_for_potentially_potentiated_dendrites(self, show_plot=False):
        pot_pot = self.potenitially_potentiated_dendrites

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


if __name__ == "__main__":
    pass
