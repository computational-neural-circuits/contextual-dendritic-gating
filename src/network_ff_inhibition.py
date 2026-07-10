import brian2 as br
from brian2.units import *
import numpy as np

from src.handle_parameters_and_results import HandleParametersAndResults
from src.area import Area


class NetworkFFInhibition(HandleParametersAndResults):
    def __init__(
        self,
        parameter_file_name,
        save_file_name,
        parameters_for_run={},
        parameter_dict={},
        equation_file_name="equations",
        prevent_plasticity=False,
        **kwargs,
    ):
        self.prevent_plasticity = prevent_plasticity
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

            if not self.only_load_results:
                self.create_monitors()

                self.area.count_number_of_inputs_from_subsets_in_context(
                    subsets=[
                        [
                            ii
                            for ii in range(self.parameters_for_run["n_active_inputs"])
                        ],
                        [],
                    ]
                )

    def setup_network(self):
        print("Setup the network for ff inhib investigation")
        br.seed(self.parameters_for_run["seed"])
        np.random.seed(self.parameters_for_run["seed"])

        if self.prevent_plasticity:
            # prevent actual plasticity
            print("Prevent Plasticity")
            self.equations[
                "Synapse_net"
            ] = """dsNMDA/dt = -sNMDA/tauNMDADecay + sNMDARise*(1-sNMDA)*alphaNMDA : 1 (clock-driven)
                   dsNMDARise/dt = -sNMDARise/tauNMDARise : 1 (clock-driven)
                   iTotNMDA1_post = -w*gNMDA*sNMDA*(V_post-vE_pyr)/(1+exp(-(V_post-vHalfNMDA)/vSpreadNMDA)) : amp (summed)
                    w : 1"""
            self.equations["on_pre"] = """sNMDARise = 1
                gTotAMPA_post += w * gAMPA"""

        if self.only_load_results:
            return

        self.area = Area(
            network=self.network,
            eqs=self.equations,
            params={**self.parameters, **self.parameters_for_run},
        )

        self.__add_silent_synapses()

    def __add_silent_synapses(self):
        area = self.area
        area.add_silent_synapses = True

        all_rates = [self.parameters["assembly_firing_rate"] for ii in range(10)]
        area.silent_pre_population = br.PoissonGroup(len(all_rates), rates=all_rates)
        area.silent_synapses = br.Synapses(
            area.silent_pre_population,
            area.dends,
            model=""" dx/dt = -x/tau_x : 1 (clock-driven)
                      dw/dt = A_LTP*(V_post-theta_plus)*int(V_post>theta_plus)*(u_plus_post-theta_minus)*int(u_plus_post>theta_minus)*x*int(w<silent_synapse_max) : 1 (clock-driven)""",
            on_pre="""w = clip(w - A_LTD*(u_minus_post-theta_minus)*int(u_minus_post>theta_minus), silent_synapse_min, silent_synapse_max)
                    x += 1""",
            method=area.integration_method,
            name=f"silent_synapses_{area.name}",
            namespace=area.params,
        )

        area.silent_synapses.connect(p=1)
        area.silent_synapses.w = self.parameters_for_run[
            "silent_synapse_starting_weight"
        ]

        self.network.add(area.silent_pre_population, area.silent_synapses)

    def create_monitors(self):
        # monitor the spikes of the somas and the inputs
        area = self.area

        self.Msilent = br.StateMonitor(
            area.silent_synapses,
            "w",
            record=True,
            dt=self.parameters["monitor_dt_weights"],
        )
        self.network.add(self.Msilent)

    def create_save_dict(self):
        self.save_dict = {
            "silent_synapses_weight": self.Msilent.w,
            "counts_gated": self.area.counts_neurons_gated,
        }

    def run(self, report_period=10 * second, report_style=None):
        runtime_imprint = self.parameters_for_run["runtime_imprint"]
        n_active_inputs = self.parameters_for_run["n_active_inputs"]

        context_id = 0

        if self.check_for_results():
            return self.save_dict

        elif self.only_load_results:
            return None

        self.area.start_context(context_id)

        self.area.input_units_1[:].rates = self.parameters["ff_bck"]
        self.area.input_units_2[:].rates = self.parameters["ff_bck"]

        self.area.input_units_1[:n_active_inputs].rates = self.parameters[
            "assembly_firing_rate"
        ]

        self.network.run(
            runtime_imprint, report=report_style, report_period=report_period
        )

        self.create_save_dict()
        self.save_results()


def run_sim(
    seed=33,
    n_active_inputs=20,
    include_adaptive_feedforward_inhibition=True,
    prevent_plasticity=True,
    return_results=False,
    only_load_results=False,
    save_file_name="investigate_ff_inhibition",
):
    parameter_dict = {
        "rec_inhib_rate": 0 * Hz,
        # this ensures that there is no recurrent inhibition
    }

    parameters_for_run = {
        "seed": seed,
        "runtime_imprint": 30 * second,
        "n_active_inputs": n_active_inputs,
        "silent_synapse_starting_weight": 10,
        "silent_synapse_min": 0,
        "silent_synapse_max": 20,
        "monitor_dt_weights": 5 * second,  # timestep for monitors
    }
    if not include_adaptive_feedforward_inhibition:
        parameters_for_run.update(
            {
                "ff_inhib_gain": 0,
                "ff_inhib_intercept": -1,
            }
        )

    if prevent_plasticity:
        parameters_for_run.update(
            {
                "runtime_imprint": 15 * second,
            }
        )
    net = NetworkFFInhibition(
        parameter_file_name="parameters",
        parameters_for_run=parameters_for_run,
        save_file_name=save_file_name,
        parameter_dict=parameter_dict,
        only_load_results=only_load_results,
        prevent_plasticity=prevent_plasticity,
    )

    net.run(report_style="text", report_period=900 * second)

    if return_results:
        return net


if __name__ == "__main__":
    pass
