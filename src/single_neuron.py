import brian2 as br
from brian2.units import *
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import community as community_louvain

from src.handle_parameters_and_results import HandleParametersAndResults
from src.area import Area


class SingleNeuron(HandleParametersAndResults):
    def __init__(
        self,
        parameter_file_name,
        save_file_name,
        parameters_for_run={},
        parameter_dict={},
        equation_file_name="equations",
        **kwargs,
    ):
        self.record_all = False
        if "record_all" in parameters_for_run:
            self.record_all = parameters_for_run["record_all"]

        super().__init__(
            parameter_dict=parameter_dict,
            parameter_file_name=parameter_file_name,
            save_file_name=save_file_name,
            equation_file_name=equation_file_name,
            parameters_for_run=parameters_for_run,
            **kwargs,
        )

        self.ff_p = self.parameters["ff_p"]
        self.n_somas = self.parameters["n_somas"]

        self.parameters_for_run["normalize"] = False

        if self.create_network:
            self.network = br.Network()
            self.setup_network()
            self.create_monitors()

    def setup_network(self):
        # Set random seeds
        br.seed(self.parameters_for_run["seed"])
        np.random.seed(self.parameters_for_run["seed"])

        # Define the potential presynaptic pool size (2*n_somas)
        potential_presynaptic_pool_size = 2 * self.n_somas

        # Calculate number of dendrites
        n_dendrites = self.parameters["n_dend_each"]

        # Calculate connections per dendrite
        self.n_inputs_ff_per_dendrite = round(self.ff_p * potential_presynaptic_pool_size)

        # Step 1: Generate the random connections to determine which presynaptic neurons are used
        random_connections = []
        for dendrite_idx in range(n_dendrites):
            # Randomly connect neurons from the potential pool to this dendrite
            connected_neurons = np.random.choice(
                potential_presynaptic_pool_size, size=self.n_inputs_ff_per_dendrite, replace=False
            )
            for neuron_idx in connected_neurons:
                random_connections.append((int(neuron_idx), dendrite_idx))

        # Step 2: Extract unique neurons that participate in connections
        unique_neurons = sorted(set(neuron_idx for neuron_idx, _ in random_connections))
        self.n_inputs_ff = len(unique_neurons)

        # Step 3: Create the presynaptic neurons (only the ones we need)
        self.input_units_ff = br.PoissonGroup(self.n_inputs_ff, rates=self.parameters["ff_bck"])

        # Step 4: Create sequential connections (first n to dendrite 0, etc.)
        sources = []
        targets = []
        new_neuron_ids = {}

        new_id = 0
        for target_id in range(n_dendrites):
            for neuron_id, t_id in random_connections:
                if target_id == t_id:
                    if f"{neuron_id}" not in new_neuron_ids.keys():
                        new_neuron_ids[f"{neuron_id}"] = new_id
                        new_id += 1

                    sources.append(new_neuron_ids[f"{neuron_id}"])
                    targets.append(target_id)

        # Create recurrent input units
        self.input_units_rec = br.PoissonGroup(self.n_somas - 1, rates=self.parameters["ff_bck"])

        if "prevent_plasticity" in self.parameters_for_run:
            # print("#######, PREVENT PLASTICITY")
            if self.parameters_for_run["prevent_plasticity"]:
                self.equations[
                    "Synapse_net"
                ] = """dsNMDA/dt = -sNMDA/tauNMDADecay + sNMDARise*(1-sNMDA)*alphaNMDA : 1 (clock-driven)
                       dsNMDARise/dt = -sNMDARise/tauNMDARise : 1 (clock-driven)
                       iTotNMDA1_post = -w*gNMDA*sNMDA*(V_post-vE_pyr)/(1+exp(-(V_post-vHalfNMDA)/vSpreadNMDA)) : amp (summed)
                        w : 1"""
                self.equations[
                    "on_pre"
                ] = """sNMDARise = 1
                    gTotAMPA_post += w * gAMPA"""

                if "make_nmda_spikes_linear" in self.parameters_for_run:
                    if self.parameters_for_run["make_nmda_spikes_linear"]:
                        self.equations[
                            "Synapse_net"
                        ] = """dsNMDA/dt = -sNMDA/tauNMDADecay + sNMDARise*(1-sNMDA)*alphaNMDA : 1 (clock-driven)
                            dsNMDARise/dt = -sNMDARise/tauNMDARise : 1 (clock-driven)
                            iTotNMDA1_post = -w*gNMDA*sNMDA*(V_post-vE_pyr)*0.2 : amp (summed)
                            w : 1"""

        if "make_nmda_spikes_linear" in self.parameters_for_run:
            if self.parameters_for_run["make_nmda_spikes_linear"]:
                self.equations[
                    "Synapse_net"
                ] = """dsNMDA/dt = -sNMDA/tauNMDADecay + sNMDARise*(1-sNMDA)*alphaNMDA : 1 (clock-driven)
                        dsNMDARise/dt = -sNMDARise/tauNMDARise : 1 (clock-driven)
                        iTotNMDA1_post = -w*gNMDA*sNMDA*(V_post-vE_pyr)*0.2 : amp (summed)
                        dx/dt = -x/tau_x : 1 (clock-driven)
                        dw/dt = A_LTP*(V_post-theta_plus)*int(V_post>theta_plus)*(u_plus_post-theta_minus)*int(u_plus_post>theta_minus)*x*int(w<w_max_post) : 1 (clock-driven)
                    """

        if "inhibitory_rate" in self.parameters_for_run:
            # print("set inhibitory rate")
            self.parameters["ff_inhib_gain"] = 0
            self.parameters["ff_inhib_intercept"] = -1
            self.parameters["ff_inhib_baseline"] = self.parameters_for_run["inhibitory_rate"]
        else:
            raise ValueError

        # make sure that there is no other inhibition on the dendrites
        self.parameters["rec_inhib_rate"] = 0 * Hz

        # now we only want to initialize a single neuron

        new_params = {**self.parameters}
        new_params["n_somas"] = 1
        new_params["conn_prob"] = 0
        new_params["ff_p"] = 0

        self.area = Area(
            network=self.network,
            eqs=self.equations,
            input_units_1=self.input_units_ff,
            input_units_2=self.input_units_rec,
            params={**new_params, **self.parameters_for_run},
        )

        self.area.input_synapses[0].connect(i=sources, j=targets)
        self.area.input_synapses[0].w[:] = self.parameters["ff_w"]
        self.area.input_synapses[1].connect(p=1)
        self.area.input_synapses[1].w[:] = self.parameters["w0"]

        self.area.dends.w_min_ff[:] = 0  # to not clip the recurrent weights to the min_ff

        self.area.add_silent_synapses = False

        if "add_silent_synapses" in self.parameters_for_run:
            if self.parameters_for_run["add_silent_synapses"]:
                self.__add_silent_synapses()

    def __add_silent_synapses(self):
        area = self.area
        area.add_silent_synapses = True

        all_rates = [self.parameters["assembly_firing_rate"] * (0.5 + ii / (9)) for ii in range(10)]
        area.silent_pre_population = br.PoissonGroup(len(all_rates), rates=all_rates)
        area.silent_synapses = br.Synapses(
            area.silent_pre_population,
            area.dends,
            model=""" dx/dt = -x/tau_x : 1 (clock-driven)
                      dw/dt = A_LTP*(V_post-theta_plus)*int(V_post>theta_plus)*(u_plus_post-theta_minus)*int(u_plus_post>theta_minus)*x*int(w<25) : 1 (clock-driven)""",
            on_pre="""w = clip(w - A_LTD*(u_minus_post-theta_minus)*int(u_minus_post>theta_minus), 0, 25)
                    x += 1""",
            method=area.integration_method,
            name=f"silent_synapses_{area.name}",
            namespace=area.params,
        )

        sources = area.n_dends_each * [ii for ii in range(len(all_rates))]
        targets = [ii // len(all_rates) for ii in range(area.n_dends * len(all_rates))]

        area.silent_synapses.connect(i=sources, j=targets)
        area.silent_synapses.w = self.parameters_for_run["silent_synapse_starting_weight"]

        self.network.add(area.silent_pre_population, area.silent_synapses)

    def create_monitors(self):
        # monitor the spikes of the somas and the inputs
        area = self.area

        if not area.add_silent_synapses or self.record_all:
            self.spM_somas = br.SpikeMonitor(area.somas, name=f"somata_monitor_{area.name}")
            self.spM_inputs = br.SpikeMonitor(area.input_units_1, name=f"input_monitor_{area.name}")
            self.spM_inhibition = br.SpikeMonitor(area.ff_inhibitors, name=f"ff_monitor_{area.name}")
            self.network.add(self.spM_somas, self.spM_inputs, self.spM_inhibition)

            self.Mdends_V = br.StateMonitor(
                area.dends,
                ["V", "u_plus", "u_minus"],
                record=[0, 1],
                dt=self.parameters["monitor_dt"],
            )
            self.network.add(self.Mdends_V)

            self.Msomas_V = br.StateMonitor(
                area.somas,
                "V",
                record=True,
                dt=self.parameters["monitor_dt"],
            )
            self.network.add(self.Msomas_V)

            synapse_ids_that_target_the_dendrite_0 = np.where(area.input_synapses[0].j == 0)[0]

            synapse_ids_that_target_the_dendrite_1 = np.where(area.input_synapses[0].j == 1)[0]

            self.Mff_w = br.StateMonitor(
                area.input_synapses[0],
                "w",
                record=list(synapse_ids_that_target_the_dendrite_0)
                + list(synapse_ids_that_target_the_dendrite_1),
                dt=self.parameters["monitor_dt_weights"],
            )
            self.network.add(self.Mff_w)

            if self.area.add_silent_synapses:
                self.Msilent = br.StateMonitor(
                    area.silent_synapses,
                    "w",
                    record=True,
                    dt=self.parameters["monitor_dt_weights"],
                )
                self.network.add(self.Msilent)

    def create_save_dict(self):
        if not self.area.add_silent_synapses or self.record_all:
            self.save_dict = {
                f"spikes_somas_t": self.spM_somas.t / ms,
                f"spikes_somas_i": self.spM_somas.i,
                f"spikes_inputs_t": self.spM_inputs.t / ms,
                f"spikes_inputs_i": self.spM_inputs.i,
                f"spikes_inhibition_t": self.spM_inhibition.t / ms,
                f"spikes_inhibition_i": self.spM_inhibition.i,
                f"voltage_dends_t": self.Mdends_V.t / ms,
                f"voltage_dends_v": self.Mdends_V.V / mV,
                f"u_plus_dends": self.Mdends_V.u_plus / mV,
                f"u_minus_dends": self.Mdends_V.u_minus / mV,
                f"ff_weights_dends_t": self.Mff_w.t / ms,
                f"ff_weights_dends_w": self.Mff_w.w,
                f"voltage_soma_t": self.Msomas_V.t / ms,
                f"voltage_soma_v": self.Msomas_V.V / mV,
            }
            if self.area.add_silent_synapses:
                self.save_dict["silent_synapses_weight"] = self.Msilent.w
        else:
            self.save_dict = {"final_silent_weights": self.area.silent_synapses.w}

    def run(self, report_period=10 * second, report_style=None):
        runtime = self.parameters_for_run["runtime"]
        n_active_inputs = self.parameters_for_run["n_active"]

        if self.check_for_results():
            return self.save_dict

        elif self.only_load_results:
            return None

        self.area.start_context(0)
        self.area.input_units_1[:].rates = self.parameters["ff_bck"]
        if n_active_inputs > 0:
            self.area.input_units_1[:n_active_inputs].rates = self.parameters["assembly_firing_rate"]

        self.network.run(runtime, report=report_style, report_period=report_period)

        self.create_save_dict()
        self.save_results()

    def run_different_contexts(self, report_period=10 * second, report_style=None):
        runtime = self.parameters_for_run["runtime"]
        n_active_inputs_per_dendrite = self.parameters_for_run["n_active_inputs_per_dendrite"]
        all_context_ids = self.parameters_for_run["all_context_ids"]

        if self.check_for_results():
            return self.save_dict

        elif self.only_load_results:
            return None

        for context_id in all_context_ids:
            self.area.start_context(context_id)
            self.area.input_units_1[:].rates = self.parameters["ff_bck"]
            for ii, n_active_inputs in enumerate(n_active_inputs_per_dendrite):
                self.area.input_units_1[
                    self.n_inputs_ff_per_dendrite * ii : self.n_inputs_ff_per_dendrite * ii
                    + n_active_inputs
                ].rates = self.parameters["assembly_firing_rate"]

            self.network.run(runtime, report=report_style, report_period=report_period)

        self.create_save_dict()
        self.save_results()

    def run_scan(self, report_style=None):
        all_inhibitory_rates = self.parameters_for_run["all_inhibitory_rates"]
        all_n_active = self.parameters_for_run["all_n_active"]
        target_dendrite = self.parameters_for_run["target_dendrite"]

        if self.check_for_results():
            return self.save_dict

        elif self.only_load_results:
            return None

        all_weight_changes = np.zeros((10, len(all_n_active), len(all_inhibitory_rates)))

        new_pars_for_run = {}
        new_pars_for_run.update(self.parameters_for_run)
        del new_pars_for_run["all_inhibitory_rates"]
        del new_pars_for_run["all_n_active"]
        del new_pars_for_run["target_dendrite"]

        for ii, n_active in enumerate(all_n_active):
            new_pars_for_run["n_active"] = n_active
            for jj, inhibitory_rate in enumerate(all_inhibitory_rates):
                new_pars_for_run["inhibitory_rate"] = inhibitory_rate

                print(n_active, inhibitory_rate)
                neuron = SingleNeuron(
                    parameter_file_name=self.parameter_file_name,
                    parameters_for_run=new_pars_for_run,
                    save_file_name=self.save_file_name,
                    save_parameters=False,
                    parameter_dict=self.parameters,
                )
                neuron.run(report_style="text")

                res = neuron.save_dict
                synapse_ids_that_target_the_dendrite = np.where(
                    neuron.area.silent_synapses.j == target_dendrite
                )[0]
                synapses_to_look_at = res["final_silent_weights"][
                    synapse_ids_that_target_the_dendrite
                ]

                all_weight_changes[:, ii, jj] = np.mean(
                    synapses_to_look_at[:] - new_pars_for_run["silent_synapse_starting_weight"],
                    axis=0,
                )

        self.save_dict = {"all_weight_changes": all_weight_changes}
        self.save_results()


if __name__ == "__main__":
    pass
