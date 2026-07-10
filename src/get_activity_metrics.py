import numpy as np


def get_activity_metrics_from_assembly_neurons(
    active_threshold,
    net,
    area,
    selected_ids,
    start_time,
    end_time,
    remove_fraction_of_start=0,
    select_randomly_for_background=True,
    run_for_task=False,
    sorted_neuron_ids=None,
):
    if run_for_task:
        somas_time = net.save_dict[f"{area.name}_spikes_somas_t"]
        somas_i = net.save_dict[f"{area.name}_spikes_somas_i"]
    else:

        if area is not None:
            somas_time = net.save_dict[f"spikes_somas_t_{area.name}"]
            somas_i = net.save_dict[f"spikes_somas_i_{area.name}"]
        else:
            somas_time = net.save_dict[f"spikes_somas_t"]
            somas_i = net.save_dict[f"spikes_somas_i"]

    avg_firing_rate = 0
    avg_firing_rate_not_in_assembly = 0
    n_active_neurons = 0
    n_active_neurons_not_in_assembly = 0

    time_window = end_time - (
        start_time + remove_fraction_of_start * (end_time - start_time)
    )
    time_window /= 1000  # we are in miliseconds

    firing_rates_not_in_assembly = []
    neurons_active_not_in_assembly = []
    if sorted_neuron_ids is None:
        sorted_neuron_ids = [ii for ii in range(net.parameters["n_somas"])]
    for ii, neuron_index in enumerate(sorted_neuron_ids):
        spike_times_for_neuron = somas_time[somas_i == neuron_index]

        spike_times_of_recall = spike_times_for_neuron[
            np.logical_and(
                spike_times_for_neuron
                > (start_time + remove_fraction_of_start * (end_time - start_time)),
                spike_times_for_neuron < end_time,
            )
        ]

        firing_rate = len(spike_times_of_recall) / (time_window)
        if neuron_index in selected_ids:
            avg_firing_rate += (firing_rate) / len(selected_ids)

            if firing_rate > active_threshold:
                n_active_neurons += 1

        else:
            firing_rates_not_in_assembly.append(firing_rate)
            neurons_active_not_in_assembly.append(firing_rate > active_threshold)

    # we look at the avg firing of non-assembly neurons
    if select_randomly_for_background:
        avg_firing_rate_not_in_assembly = np.mean(
            firing_rates_not_in_assembly[: len(selected_ids)]
        )
        n_active_neurons_not_in_assembly = np.sum(
            neurons_active_not_in_assembly[: len(selected_ids)]
        )
    else:
        avg_firing_rate_not_in_assembly = np.mean(
            np.sort(firing_rates_not_in_assembly)[-len(selected_ids) :]
        )
        n_active_neurons_not_in_assembly = np.sum(
            np.sort(neurons_active_not_in_assembly)[-len(selected_ids) :]
        )

    return (
        avg_firing_rate,
        n_active_neurons,
        avg_firing_rate_not_in_assembly,
        n_active_neurons_not_in_assembly,
    )
