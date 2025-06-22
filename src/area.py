import brian2 as br
from brian2.units import *
import numpy as np
import time

import matplotlib.pyplot as plt

# import brian2tools as b2t
br.prefs.codegen.target = "cython"


class Area:
    def __init__(
        self,
        eqs,
        params,
        network=None,
        name="A",
        input_units_1=None,
        input_units_2=None,
        only_setup_basics=False,
    ):
        """A class representing one area of the neural network.

        Args:
            network (brian2.Network): A Network object running the simulation.
            conn_prob (float): E to E connection probability.
            n_contexts (int): number of inhibitory neurons to be created.
            eqsfile (string): A path to the file with the equations.
            paramsfile (string): A path to the file with the parameters.
            normalize (bool, optional): Include heterosynaptic weight normalization.
                Defaults to True.
            non_overlapping_ctxt (bool, optional): Whether context should gate
                random dendrites or subsequent ones. Defaults to False (random).
            **params_update: Parameters overwriting the ones in the paramsfile.
        """
        self.net = network
        self.name = name

        # Read equations and parameters
        self.eqs = eqs
        self.params = params
        self.n_somas = self.params["n_somas"]
        self.n_dends_each = self.params["n_dend_each"]
        self.n_dends = self.n_somas * self.n_dends_each
        self.params["gTotCouple_pyr"] = self.params["gEachCouple_pyr"] * self.n_dends_each

        self.integration_method = "euler"

        self.normalize = True
        if "normalize" in self.params:
            self.normalize = self.params["normalize"]

        self.non_overlapping_ctxt = True
        if "non_overlapping_ctxt" in self.params:
            self.non_overlapping_ctxt = self.params["non_overlapping_ctxt"]

        if only_setup_basics:
            return
        # Set up excitatory population
        self.__create_neurons()
        if self.params["conn_prob"] > 0:
            self.__create_synapses_E(
                conn_prob=self.params["conn_prob"],
            )

        if self.normalize:
            self.normalization = br.NetworkOperation(
                self.__normalize, dt=self.params["norm_dt"], name=f"normalization_{self.name}"
            )
            self.net.add(self.normalization)

        if input_units_1 is None:
            self.input_units_1 = br.PoissonGroup(
                self.n_somas, rates=0 * Hz, name=f"inputs_1_to_area_{self.name}"
            )

        else:
            self.input_units_1 = input_units_1

        if input_units_2 is None:
            self.input_units_2 = br.PoissonGroup(
                self.n_somas, rates=0 * Hz, name=f"inputs_2_to_area_{self.name}"
            )
        else:
            self.input_units_2 = input_units_2

        self.__connect_inputs()

        # Set up inhibitory population
        assert self.params["n_contexts"] >= 0, "Number of contexts must be a non-negative integer."
        self.n_contexts = self.params["n_contexts"]
        self.contexts_used = np.zeros(self.params["n_contexts"], dtype=bool)

        self.__add_context_inhibition()

        self.__add_feedforward_inhibition()

        self.__add_somatic_inhibition()

        self.__add_recurrent_inhibition()

    def __create_neurons(self):
        self.somas = br.NeuronGroup(
            self.n_somas,
            model=self.eqs["Soma"] + "\ntheta : volt",
            # we add this way of implementing the threshold to be able
            # to silence single neurons for the recall
            method=self.integration_method,
            threshold="V > theta",
            reset=self.eqs["Soma_reset"],
            refractory=self.params["refracTime_pyr"],
            namespace=self.params,
            name=f"somas_area_{self.name}",
        )
        self.somas.theta = self.params["vThres_pyr"]
        self.dends = br.NeuronGroup(
            self.n_dends,
            model=self.eqs["Dend"],
            # method=self.integration_method,
            method="rk4",
            namespace=self.params,
            name=f"dendrites_area_{self.name}",
        )

        # Setting parameters of the somas and dendrites
        self.somas.V = self.params["vRest_pyr"]
        self.somas.vShadow = self.params["vRest_pyr"]
        self.dends.V = self.params["vRest_pyr"]
        self.dends.u1 = self.params["vRest_pyr"]
        self.dends.u_plus = self.params["vRest_pyr"]
        self.dends.u_minus = self.params["vRest_pyr"]
        self.dends.w_max_ff = self.params["w_max_ff"]
        self.dends.w_max_rec = self.params["w_max_rec"]
        self.dends.w_min_rec = self.params["w_min_rec"]
        self.dends.iTotNMDA1 = 0 * amp
        self.dends.iTotNMDA2 = 0 * amp
        self.dends.iTotNMDA3 = 0 * amp

        # Attributes representing coupling (not synapses) between somas and their dendrites
        # # Which soma (soma_dend_map[i]) connects to which dendrites
        # self.soma_dend_map = np.reshape(np.array(range(self.n_dends), dtype=int),
        #                        (self.n_somas, self.params['n_dend_each']))

        self.coupling_dend_to_soma = br.Synapses(
            self.dends,
            self.somas,
            self.eqs["dend_to_soma_model"],
            # clock=self.params["sim_dt"],
            name=f"couple_dend_to_soma_{self.name}",
        )

        self.coupling_soma_to_dend = br.Synapses(
            self.somas,
            self.dends,
            self.eqs["soma_to_dend_model"],
            # clock=self.params["sim_dt"],
            name=f"couple_soma_to_dend_{self.name}",
            namespace=self.params,
            method=self.integration_method,
        )

        dend_each = self.params["n_dend_each"]

        i = np.linspace(0, self.n_dends - 1, self.n_dends).astype(int)

        self.coupling_dend_to_soma.connect(i=i, j=i // dend_each)
        self.coupling_soma_to_dend.connect(i=i // dend_each, j=i)
        self.coupling_soma_to_dend.gEachCouple = self.params["gEachCouple_pyr"]
        self.coupling_dend_to_soma.gEachCouple = self.params["gEachCouple_pyr"]

        self.net.add(self.coupling_dend_to_soma, self.coupling_soma_to_dend)

        self.net.add(self.somas, self.dends)

    def __connect_inputs(self):
        self.input_synapses = []
        for iu, inputs in enumerate([self.input_units_1, self.input_units_2]):
            syn = br.Synapses(
                inputs,
                self.dends,
                self.eqs["Synapse_net"]
                .replace("iTotNMDA1", f"iTotNMDA{iu+2}")
                .replace("w_max", "w_max_ff"),
                on_pre=self.eqs["on_pre"].replace("w_max", "w_max_ff").replace("w_min", "w_min_ff"),
                namespace=self.params,
                method=self.integration_method,
                name=f"synapse_input_{iu}_to_{self.name}",
            )
            syn.connect(p=self.params["ff_p"])
            syn.w = self.params["ff_w"]

            self.net.add(syn, inputs)
            self.input_synapses.append(syn)

        def calculate_w_min_ff(n_syn_ff, w_tot_start, n_strong_ff=5):
            y = (
                w_tot_start
                - n_strong_ff * self.params["w_max_ff"]
                - (self.params["assembly_size"] - 1) * self.params["w_max_rec"]
                - (
                    (self.params["n_somas"] - self.params["assembly_size"])
                    * self.params["w_min_rec"]
                )
            ) / (n_syn_ff - n_strong_ff)
            return y

        all_w_tot = []
        all_w_min = []
        for dend_id in range(self.n_dends):
            n_inputs = np.sum(self.input_synapses[0].j == dend_id) + np.sum(
                self.input_synapses[1].j == dend_id
            )
            w_tot = n_inputs * self.params["ff_w"] + self.params["w0"] * (self.n_somas - 1)
            all_w_tot.append(w_tot)
            self.dends[dend_id].w_tot = w_tot

            n_strong_ff = self.params["n_strong_ff"]
            self.dends[dend_id].w_min_ff = calculate_w_min_ff(
                n_syn_ff=n_inputs, w_tot_start=w_tot, n_strong_ff=n_strong_ff
            )
            all_w_min.append(self.dends[dend_id].w_min_ff[:])

    def soma_dend_map(self, soma):
        """
        Returns an array containing indices of dendrites corresponding to the
        given soma.
        Args:
            soma (int): The index of the soma for which the dendrites are sought.
        """
        assert (
            soma >= 0 and soma < self.n_somas
        ), f"Soma of id {soma} doesn't exist.\
            There are {self.n_somas} somas."
        return np.array(range(soma * self.n_dends_each, (soma + 1) * self.n_dends_each))

    def dend_soma_map(self, dend):
        """
        Returns the ID of the neuron to which the given dendrite is coupled.
        Args:
            dend (int): The index of the soma for which the dendrites are sought.
        """
        assert (
            dend >= 0 and dend < self.n_dends
        ), f"Dend of id {dend} doesn't exist.\
            There are {self.n_dends} somas."
        return dend // self.n_dends_each

    def __create_synapses_E(self, conn_prob):
        """
        Create synpases within the area. Every neuron projects to all except its
        own dendrites with probability `conn_prob`.
        """

        self.synapses_E = br.Synapses(
            self.somas,
            self.dends,
            self.eqs["Synapse_net"].replace("w_max", "w_max_rec"),
            on_pre=self.eqs["on_pre"].replace("w_max", "w_max_rec").replace("w_min", "w_min_rec"),
            namespace=self.params,
            method=self.integration_method,
            name=f"recurrent_synapses_area_{self.name}",
        )

        # Explicitly declaring i (srcs) and j (tgts) for S.connect()
        full_srcs = np.repeat(
            np.arange(self.n_somas, dtype=int),
            self.n_dends - self.params["n_dend_each"],
        )
        full_tgts = np.array([], dtype=int)
        for soma in range(self.n_somas):
            temp = np.arange(self.n_dends, dtype=int)
            full_tgts = np.append(full_tgts, temp[: soma * self.params["n_dend_each"]])
            full_tgts = np.append(full_tgts, temp[(soma + 1) * self.params["n_dend_each"] :])
        assert len(full_srcs) == len(full_tgts)

        self.synapses_E.connect(i=full_srcs, j=full_tgts, p=conn_prob)
        self.synapses_E.w = self.params["w0"]

        self.srcs = self.synapses_E.i
        self.tgts = self.synapses_E.j
        self.syns_on_dendrites = np.asarray(
            [np.asarray(self.tgts == j).nonzero()[0] for j in np.unique(self.tgts)]
        )
        # self.normalisation_time = np.zeros(1)

        self.net.add(self.synapses_E)

    def __add_feedforward_inhibition(self):
        ##### Feedforward inhibtion

        self.population_firing_rate_estimator = br.NeuronGroup(
            1,
            model=self.eqs["rate_estimator_of_ff_inhibition_model"],
            method=self.integration_method,
            namespace=self.params,
            name=f"ff_firing_rate_estimator_area_{self.name}",
        )

        for population, name in zip(
            [self.input_units_1, self.input_units_2],
            ["in_1", "in_2"],
        ):
            connect_to_estimator = br.Synapses(
                population,
                self.population_firing_rate_estimator,
                on_pre="x_pop+=1",
                name=f"{name}_connect_to_ff_estimator_from_{self.name}",
            )
            connect_to_estimator.connect(p=1)
            self.net.add(connect_to_estimator)

        self.ff_inhibitors = br.NeuronGroup(
            self.n_somas,
            model=self.eqs["ff_inhibition_model"],
            threshold="rand()<rate*dt",
            name=f"ff_inhibition_in_area_{self.name}",
        )

        self.connect_to_ff_inhibition = br.Synapses(
            self.population_firing_rate_estimator,
            self.ff_inhibitors,
            model=self.eqs["syn_to_ff_inhibition"],
            namespace=self.params,
            name=f"synapse_of_ff_estimator_to_ff_inhibitors_in_area_{self.name}",
        )

        self.connect_to_ff_inhibition.connect(p=1)

        srcs = np.array([ii // self.n_dends_each for ii in range(self.n_somas * self.n_dends_each)])
        tgts = [ii for ii in range(self.n_somas * self.n_dends_each)]
        np.random.shuffle(tgts)

        self.synapses_I_ff_dend = br.Synapses(
            self.ff_inhibitors,
            self.dends,
            model="",
            on_pre=self.eqs["on_pre_EI_ff"],
            namespace=self.params,
            method=self.integration_method,
            name=f"synapse_from_ff_inhibition_to_dendrites_in_area_{self.name}",
        )
        self.synapses_I_ff_dend.connect(
            i=srcs,
            j=tgts,
        )

        self.net.add(
            self.population_firing_rate_estimator,
            self.connect_to_ff_inhibition,
            self.ff_inhibitors,
            self.synapses_I_ff_dend,
        )

    def __add_context_inhibition(self):
        ##### Context Inhibtion
        self.context_inhibitors = br.PoissonGroup(
            self.n_somas * self.n_contexts,
            rates=0 * Hz,
            name=f"context_inhibitors_in_area_{self.name}",
        )
        self.synapses_I_context = br.Synapses(
            self.context_inhibitors,
            self.dends,
            model="",
            on_pre=self.eqs["on_pre_EI_context"],
            namespace=self.params,
            method=self.integration_method,
            name=f"synapses_from_context_inhibitors_to_dendrites_in_area_{self.name}",
        )

        srcs, tgts, skipped = self.__get_context_inh_srcs_and_tgts()
        # An array holding dendrite ids which get inputs from subsequent context neurons
        self.tgts_of_ctxt = np.reshape(
            tgts, (self.n_contexts, self.n_somas * (self.n_dends_each - 1))
        )

        # An array holding dendrite ids which don't get inhibited by subsequent context neurons
        self.dends_of_ctxt = np.reshape(skipped, (self.n_contexts, self.n_somas))
        # print("Dends of ctxt:")
        # print(self.dends_of_ctxt)
        self.synapses_I_context.connect(i=srcs, j=tgts)

        self.net.add(
            self.context_inhibitors,
            self.synapses_I_context,
        )

    def __add_somatic_inhibition(self):
        ##### Somatic inhibtion
        self.inhibitors_soma = br.PoissonGroup(
            self.n_somas,
            rates=self.params["soma_inhib_rate"] * Hz,
            name=f"soma_inhibition_in_area_{self.name}",
        )
        self.synapses_I_soma = br.Synapses(
            self.inhibitors_soma,
            self.somas,
            model="",
            on_pre=self.eqs["on_pre_EI_soma"],
            namespace=self.params,
            method=self.integration_method,
            name=f"synapses_from_soma_inhibition_to_soma_in_area_{self.name}",
        )
        self.synapses_I_soma.connect("i==j")

        self.net.add(
            self.synapses_I_soma,
            self.inhibitors_soma,
        )

    def __add_recurrent_inhibition(self):
        self.rec_inihib_pop = br.NeuronGroup(
            self.n_somas,
            model=self.eqs["recurrent_inhibition_model"],
            method=self.integration_method,
            threshold="rand()<rate*dt * int(n_active >= theta) ",
            name=f"recurrent_inhibition_in_area_{self.name}",
        )
        self.rec_inihib_pop.rate = self.params["rec_inhib_rate"]
        self.rec_inihib_pop.theta = self.params["rec_inhib_input_threshold"]
        self.rec_inihib_pop.n_active = 0

        self.syn_to_rec_inhib_pop = br.Synapses(
            self.somas,
            self.rec_inihib_pop,
            model=""" theta_syn : 1
                n_active_post = int(x_estimator_pre > theta_syn) : 1 (summed)""",
            method=self.integration_method,
            name=f"synapses_to_recurrent_inhibition_in_area_{self.name}",
        )

        self.syn_to_rec_inhib_pop.connect(p=self.params["rec_inhib_connect_pro_to_population"])
        # syn_to_rec_inhib_pop.open = 0
        self.syn_to_rec_inhib_pop.theta_syn = self.params["rec_inhib_rate_estimator_threshold"]

        self.synapse_som_to_dend = br.Synapses(
            self.rec_inihib_pop,
            self.dends,
            model="",
            on_pre=self.eqs["on_pre_EI_recurrent"],
            namespace=self.params,
            method=self.integration_method,
            name=f"synapses_from_recurrent_inhibition_in_area_{self.name}",
        )
        self.synapse_som_to_dend.connect(p=self.params["rec_inhib_connect_pro_to_dendrites"])

        self.net.add(
            self.rec_inihib_pop,
            self.syn_to_rec_inhib_pop,
            self.synapse_som_to_dend,
        )

    def __get_context_inh_srcs_and_tgts(self):
        """
        A function returning the i and j indices for Synapses.connect(), as
        well as a list of dendrite indices that were not targeted by inhibition.
        """
        tgts = np.array([], dtype=int)
        skipped = np.array([], dtype=int)
        for c in range(self.n_contexts):
            tgts_of_ctxt = np.array([], dtype=int)
            for s in range(self.n_somas):
                # Which of the 6 dendrites does not get inhibition
                if self.non_overlapping_ctxt:
                    assert self.n_dends_each >= self.n_contexts, (
                        "More contexts" f" ({self.n_contexts}) than dendrites ({self.n_dends_each})!"
                    )
                    uninhibited = c
                else:
                    uninhibited = np.random.randint(self.n_dends_each)
                former = self.soma_dend_map(s)[:uninhibited]
                latter = self.soma_dend_map(s)[uninhibited + 1 :]

                dend_tgts = np.append(former, latter)
                skipped = np.append(skipped, self.soma_dend_map(s)[uninhibited])
                tgts_of_ctxt = np.append(tgts_of_ctxt, dend_tgts)
                # print(tgts)
            np.random.shuffle(tgts_of_ctxt)
            tgts = np.append(tgts, tgts_of_ctxt)
        n_independent_sources = self.n_somas * self.n_contexts
        srcs = np.linspace(0, n_independent_sources - 1, n_independent_sources).astype(int)
        srcs = np.repeat(srcs, (self.n_dends_each - 1))
        assert len(srcs) == len(tgts)
        assert (
            len(tgts) + len(skipped) == self.n_dends * self.n_contexts
        ), f"{len(tgts)} + {len(skipped)} is not equal to {self.n_dends} * {self.n_contexts}"
        return srcs, tgts, skipped

    def __normalize(self):
        """
        A function executed every normalize_dt, normalizing all weights on
        dendrites with too much summaric input wieghts.
        """

        # Copying the weights from the synapses object because brian indexing is weird
        weights = np.zeros(shape=(self.n_somas, self.n_dends))
        weights[self.srcs, self.tgts] = self.synapses_E.w

        w_tot = self.dends.w_tot[:]

        for ii in range(2):
            weights_ff = np.zeros(shape=(self.n_somas, self.n_dends))
            weights_ff[self.input_synapses[ii].i, self.input_synapses[ii].j] = self.input_synapses[
                ii
            ].w

            weights = np.vstack([weights, weights_ff])

        weights = np.subtract(weights, self.params["eta"] * (np.sum(weights, axis=0) - w_tot))

        for ii in [1, 0]:  # going in reverse order!
            weights_ff = weights[-self.n_somas :, :]
            weights = weights[: -self.n_somas, :]

            self.input_synapses[ii].w = np.clip(
                weights_ff[self.input_synapses[ii].i, self.input_synapses[ii].j],
                self.dends.w_min_ff[self.input_synapses[ii].j],  # we need to substitue w_min_ff
                self.params["w_max_ff"],
            )

        self.synapses_E.w = np.clip(
            weights[self.srcs, self.tgts],
            self.params["w_min_rec"],
            self.params["w_max_rec"],
        )

    def count_number_of_inputs_from_subsets_in_context(
        self,
        context_id=0,
        subsets: list = [[ii for ii in range(20)] for ii in range(2)],
    ):
        self.counts_gated = np.zeros((2, self.n_dends_each * self.n_somas)) * float("NaN")
        self.counts_non_gated = np.zeros((2, self.n_dends_each * self.n_somas)) * float("NaN")
        self.counts_neurons_gated = np.zeros((2, self.n_somas)) * float("NaN")

        select_arrays = []
        for ii in range(2):
            this_array = np.array(
                [
                    True if this_index in subsets[ii] else False
                    for this_index in self.input_synapses[ii].i
                ]
            )
            select_arrays.append(this_array)

        for ii in range(self.counts_gated.shape[1]):
            if ii in self.dends_of_ctxt[context_id]:
                for nn in range(2):
                    self.counts_gated[nn, ii] = np.count_nonzero(
                        (self.input_synapses[nn].j[:])[select_arrays[nn]] == ii
                    )

                    self.counts_neurons_gated[nn, ii // self.n_dends_each] = self.counts_gated[
                        nn, ii
                    ]
            else:
                for nn in range(2):
                    self.counts_non_gated[nn, ii] = np.count_nonzero(
                        (self.input_synapses[nn].j[:])[select_arrays[nn]] == ii
                    )

    def start_context(self, context_id, rate=None):
        """Function activating one of the inhibitory neurons.

        Args:
            context_id (int): Id of the inhibitory neuron to be activated.
            inhib (float*Hz, optional): Rate of the inhibitory spikes.
                Defaults to 'inhib_rate' from parameters.txt.
        """
        assert (
            context_id < self.n_contexts
        ), f"Invalid context id {context_id}.\
            Only {self.n_contexts} available."
        if rate is None:
            rate = self.params["context_inhib_rate"]

        self.context_inhibitors.rates = 0 * Hz
        self.context_inhibitors[
            context_id * self.n_somas : (context_id + 1) * self.n_somas
        ].rates = rate
        # print(self.inhibitors.rates)
        self.contexts_used[context_id] = True

    def stop_context(self):
        """Stops all inhibitory generators."""
        self.context_inhibitors.rates = 0 * Hz

    def start_excitation(self, soma_ids, excitation):
        """Set external current I_ext to neurons specified in soma_ids.

        Args:
            soma_ids (list of ints): Ids of somas to have I_ext changed.
            excitation (float*amp or list of float*amp): Value(s) of I_ext.
        """
        self.somas[soma_ids].I_ext = excitation

    def stop_excitation(self):
        """Set external input I_ext to 0*amp for all somas."""
        self.somas.I_ext = 0 * amp
