import numpy as np
from brian2.units import *
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def get_firing_rate_for_single_neuron(start, end, spike_times_for_neuron):
    """
    start : float - start time in ms
    end   : float - end time in ms
    spike_times_for_neuron : np.array - cotains all spike times in ms

    """
    # we always use 2 seconds at the end to determine the firing rate
    part_time = 2000

    if end - start < part_time:
        raise ValueError("start and end are too close together")

    start = end - 2000  # only take the second half

    spikes_are_after = spike_times_for_neuron > start
    spikes_are_before = spike_times_for_neuron < end
    firing_rate = 1000 * np.sum(np.logical_and(spikes_are_before, spikes_are_after)) / part_time

    return firing_rate


def get_assembly_neuron_ids_by_weight_and_rate(
    net,
    all_rates,
    context_id,
    area,
):
    """
    all_rates : np.array - all rates of the neurons

    """
    if np.any(np.isnan(all_rates)):
        raise ValueError("There are NaN values in the firing rates")

    firing_rates_reshaped = all_rates.reshape(-1, 1)

    # we have 2 clusters, one for high and one for low
    kmeans = KMeans(n_clusters=2, random_state=1992)
    kmeans.fit(firing_rates_reshaped)

    high_cluster_label = np.argmax(kmeans.cluster_centers_)

    high_firing_indices = np.where(kmeans.labels_ == high_cluster_label)[0]
    low_firing_indices = np.where(kmeans.labels_ != high_cluster_label)[0]

    selected_ids_from_firing_rate = list(high_firing_indices)

    additional_rate_ids = [
        ii for ii in np.argsort(all_rates) if ii not in selected_ids_from_firing_rate
    ][-10:]

    selected_ids_from_firing_rate += additional_rate_ids

    # we now compare this assembly to the assembly we find by weights
    # first we need to get the weight matrix

    weights_recurrent = np.ones((area.n_somas, area.n_dends)) * area.params["w0"]
    try:
        weights_recurrent[area.srcs, area.tgts] = area.synapses_E.w
    except AttributeError:
        # it could be that we use saved weights (see below) instad of initialized networks
        pass

    if np.all(weights_recurrent == area.params["w0"]):
        for key in [f"{area.name}_weight", "weights", f"{area.name}_weights"]:
            try:
                weights_recurrent_loaded = net.save_dict[key]

                if weights_recurrent_loaded.shape[0] < area.n_somas:
                    # this means we have saved a subset of weights
                    # a subset of the highest firing rates
                    sorted_neuron_ids, selected_ids, _ = old_way_to_sort(
                        net, shuffle_rest=False, reverse_order=True
                    )
                    sorted_neuron_ids = np.array(sorted_neuron_ids[0])
                    weights_recurrent = np.ones((area.n_somas, area.n_somas)) * area.params["w0"]
                    weights_recurrent[
                        np.ix_(
                            sorted_neuron_ids[: len(selected_ids) + 25],
                            sorted_neuron_ids[: len(selected_ids) + 25],
                        )
                    ] = weights_recurrent_loaded
                else:
                    weights_recurrent = weights_recurrent_loaded

            except KeyError:
                pass

    if np.all(weights_recurrent == net.parameters["w0"]):
        # ensure that we actually have the weights loaded and are not workig with initial weights
        raise ValueError("You are using startig weights to detect assmebly size")

    # get the dendrite shape
    print("recurrent weights, shape: ", weights_recurrent.shape)
    if not (weights_recurrent.shape[0] == weights_recurrent.shape[1]):
        dend_dim = np.argmax([weights_recurrent.shape])
        if weights_recurrent.shape[dend_dim] == area.n_dends:
            if dend_dim == 0:
                weights_recurrent = weights_recurrent[np.sort(area.dends_of_ctxt[context_id]), :]
            else:
                weights_recurrent = weights_recurrent[:, np.sort(area.dends_of_ctxt[context_id])]

    # now we reduce the weights on the rate-sorted weights

    cut_weights_recurrent = weights_recurrent[
        np.ix_(selected_ids_from_firing_rate, selected_ids_from_firing_rate)
    ]

    n_clusters = 2

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=1992,
    )

    sum_connectivity = np.sum(cut_weights_recurrent, axis=0).reshape(-1, 1)
    combined_features = np.hstack((cut_weights_recurrent, sum_connectivity))
    kmeans.fit(combined_features)
    # kmeans.fit(weights_recurrent)

    labels = kmeans.labels_

    clusters = []
    mean_weights = []
    for cluster_id in range(n_clusters):
        neuron_ids = np.where(labels == cluster_id)[0]
        mean_weights.append(np.mean(cut_weights_recurrent[np.ix_(neuron_ids, neuron_ids)]))
        clusters.append(neuron_ids)

    selected_ids = clusters[np.argmax(mean_weights)]
    print("? ", mean_weights, len(selected_ids), context_id)

    final_selected_ids = [selected_ids_from_firing_rate[ii] for ii in selected_ids]

    # fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    # ax1.imshow(weights_recurrent.T)

    # ax2.imshow(
    #     weights_recurrent[np.ix_(selected_ids_from_firing_rate, selected_ids_from_firing_rate[:])].T
    # )
    # ax3.imshow(cut_weights_recurrent[np.ix_(selected_ids, selected_ids)].T)
    # ax4.imshow(
    #     cut_weights_recurrent[
    #         np.ix_(clusters[np.argmin(mean_weights)], clusters[np.argmin(mean_weights)][:])
    #     ].T
    # )
    # ax2.set(xlabel="Pre", ylabel="Post")
    # ax3.set(xlabel="Pre", ylabel="Post")
    # ax1.set(xlabel="Pre", ylabel="Post")
    # plt.show()
    return final_selected_ids


def old_way_to_sort(net, shuffle_rest=True, reverse_order=False):
    somas_time = np.copy(net.save_dict["spikes_somas_t"])
    somas_i = np.copy(net.save_dict["spikes_somas_i"])

    unique_context_ids = list(np.sort(np.unique(net.parameters_for_run["all_context_ids"])))
    all_firing_rates = [[] for nn in unique_context_ids]

    bsl = net.parameters_for_run["runtime_baseline"] / msecond
    rtm = net.parameters_for_run["runtime_imprint"] / msecond
    for tt in range(len(net.parameters_for_run["all_assembly_ids"])):
        start = (rtm + bsl) * tt + bsl * int(tt > 0)
        end = start + rtm + bsl * int(tt < 1)

        context_id = net.parameters_for_run["all_context_ids"][tt]
        contex_id_index = unique_context_ids.index(context_id)

        part_time = rtm / 4.0
        start = start + (rtm - part_time)  # only take the second half

        all_firing_rates_somas = []
        for neuron_index in range(net.parameters["n_somas"]):
            spike_times_for_neuron = somas_time[somas_i == neuron_index]
            spikes_are_after = spike_times_for_neuron > start
            spikes_are_before = spike_times_for_neuron < end
            neuron_firing_rate = (
                1000 * np.sum(np.logical_and(spikes_are_before, spikes_are_after)) / part_time
            )

            # 1000 * to get to Hz
            all_firing_rates_somas.append(neuron_firing_rate)
        all_firing_rates[contex_id_index].append(all_firing_rates_somas)

    all_context_ids = net.parameters_for_run["all_context_ids"]
    all_assembly_ids = net.parameters_for_run["all_assembly_ids"]
    unique_context_ids = np.unique(all_context_ids)
    sorted_neuron_ids = [[] for nn in unique_context_ids]
    all_assembly_ids_per_context = [
        [
            assembly_ids
            for assembly_ids, context_id in zip(all_assembly_ids, all_context_ids)
            if context_id == this_context
        ]
        for this_context in unique_context_ids
    ]
    # print("aaids:", all_assembly_ids_per_context)

    for ii, context_id in enumerate(unique_context_ids):
        mask = np.zeros(net.parameters["n_somas"]).astype(bool)
        for tt, all_rates in enumerate(np.array(all_firing_rates[ii])):
            all_rates[mask] = float("nan")

            sorted_rates = np.sort(all_rates)

            sorted_rates_ids = np.argsort(all_rates)
            sorted_rates_ids = sorted_rates_ids[np.logical_not(np.isnan(sorted_rates))]
            sorted_rates = sorted_rates[np.logical_not(np.isnan(sorted_rates))]

            threshold = np.mean(sorted_rates[-24:]) / 3.0  # 8 is chosen randomly here
            cutoff_id = np.searchsorted(sorted_rates, threshold)

            selected_ids = list(sorted_rates_ids[cutoff_id:])
            if reverse_order:
                sorted_neuron_ids[ii] += selected_ids[::-1]
            else:
                sorted_neuron_ids[ii] += selected_ids
            mask[selected_ids] = True

            if tt == len(all_assembly_ids_per_context[ii]) - 1:
                rest = sorted_rates_ids[:cutoff_id]
                np.random.seed(4)
                if shuffle_rest:
                    np.random.shuffle(rest)

                sorted_neuron_ids[ii] += list(rest)[::-1]

    return sorted_neuron_ids, selected_ids, sorted_rates
