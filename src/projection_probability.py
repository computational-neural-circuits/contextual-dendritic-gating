import numpy as np
import matplotlib.pyplot as plt
import math


def show_dendrite_distributions(
    axes=None,
    equal_or_more=4,
    n_somas=400,
    n_dends_each=6,
    assembly_size=20,
    show_plot=True,
    original_p=0.07,
    new_p=0.08,
    max_x=55,
    prob_for_equal_to_potentiate=1,
):
    X = np.linspace(0, max_x, 1000)

    if axes is None:
        fig, axes = plt.subplots(2)

    ax1, ax2 = axes

    expected_dendrites, standard_deviation = get_probability_of_x_or_more(
        x=equal_or_more, connect_p=original_p, assembly_size=assembly_size, n_somas=n_somas
    )
    ax1.plot(
        X,
        gaussian_pdf(X, mean=expected_dendrites, std_dev=standard_deviation),
        label=f"theory a the moment ({original_p})",
        color="#0570b0",
    )
    expected_dendrites_shifted, standard_deviation_shifted = get_probability_of_x_or_more(
        x=equal_or_more, connect_p=new_p, assembly_size=assembly_size, n_somas=n_somas
    )
    (
        expected_dendrites_shifted_plus_one,
        standard_deviation_shifted_plus_one,
    ) = get_probability_of_x_or_more(
        x=equal_or_more + 1, connect_p=original_p, assembly_size=assembly_size, n_somas=n_somas
    )
    Y_for_higher_p = gaussian_pdf(
        X, mean=expected_dendrites_shifted, std_dev=standard_deviation_shifted
    )
    ax1.plot(
        X,
        Y_for_higher_p,
        label=f"theory for higher p ({new_p})",
        color="#fd8d3c",
    )

    bins = np.zeros(max_x)
    bins_original = np.zeros(max_x)

    all_silenced = np.zeros(max_x).astype(int)
    for xx in range(max_x):
        X = np.linspace(xx - 0.5, xx + 0.5, 100)
        probability_of_situation = np.sum(
            gaussian_pdf(X, mean=expected_dendrites_shifted, std_dev=standard_deviation_shifted)
        ) * (X[1] - X[0])

        probability_of_situation_plus_one = np.sum(
            gaussian_pdf(
                X,
                mean=expected_dendrites_shifted_plus_one,
                std_dev=standard_deviation_shifted_plus_one,
            )
        ) * (X[1] - X[0])

        bins_original[xx] = (
            probability_of_situation - probability_of_situation_plus_one
        ) * prob_for_equal_to_potentiate + probability_of_situation_plus_one

        if xx <= assembly_size:
            bins[xx] = probability_of_situation
        else:
            # we now remove the most useful number of dendrites
            # that is, we want on average to remove one from the first 21 (if 21) ...
            overflow = xx - assembly_size

            p = overflow / xx

            to_silence = n_somas * p
            to_silence_int = int(np.round(to_silence))
            all_silenced[xx] = to_silence_int

            max_number_in_first_segment = np.min([to_silence_int + 1, xx, n_somas])
            probs = (
                np.array(
                    [
                        probability_of_x_ones(n_somas, to_silence_int, m=xx, x=ll)
                        for ll in range(max_number_in_first_segment)
                    ][::-1]
                )
                * probability_of_situation
            )

            # print(xx, probs)

            positions = np.zeros_like(bins)
            for ii in range(max_number_in_first_segment):
                # print(xx, xx - ii)
                positions[xx - ii] = True

            bins[np.where(positions)[0]] += probs

    ax1.bar(range(max_x), bins_original, alpha=0.5, label="bins before SOM shuffle", color="#fd8d3c")
    ax1.bar(range(max_x), bins, label="bins after SOM shuffle", color="#238443")
    ax1.axvline(np.argmax(bins), color="#238443")
    x = np.argmax(bins)
    y = bins[x]
    ax1.annotate(
        f"{np.round(np.argmax(bins),1)}",
        xy=(x, y),
        xytext=(x * 1.01, y * 1.02),
        # arrowprops=dict(facecolor="black", shrink=0.05),
    )
    ax1.legend()

    ax2.scatter(range(max_x), all_silenced, s=5, color="#6a51a3", label="best to silence")
    # print(all_silenced)

    connection_probability = 0.038
    con = np.zeros(max_x)
    x = np.linspace(0, max_x - 1, max_x)
    for kk in range(1, max_x - 20):
        con[kk + 20] = expected_connected(kk, N_post=n_somas, connect_p=connection_probability)

    ax2.plot(x, con, label=f"SOM connects with p = {connection_probability}")
    ax2.legend()
    ax2.set(xlabel="Active neurons", ylabel="Dendrites to silence")
    ax1.set(xlabel="Dendrites with 4 or more highly active inputs", ylabel="prob")
    if show_plot:
        plt.show()

    return bins, bins_original


def expected_connected(kk, N_post, connect_p):
    return N_post * (1 - (1 - connect_p) ** kk)


def probability_of_x_ones(N, n, m, x):
    """
    Calculates the probability of having exactly x 1s in the first m positions
    of an array of size N after setting n elements to 1.
    """
    favorable = math.comb(m, x) * math.comb(N - m, n - x)
    total = math.comb(N, n)
    return favorable / total


def gaussian_pdf(x, mean, std_dev):
    exponent = -((x - mean) ** 2 / (2 * std_dev**2))
    return (1 / (std_dev * math.sqrt(2 * math.pi))) * np.exp(exponent)


def prob_k_inputs_from_subset(k, p, assembly_size):
    return math.comb(assembly_size, k) * p**k * (1 - p) ** (assembly_size - k)


def get_probability_of_x_or_more(x, connect_p, assembly_size, n_somas):
    prob_less_than_x = sum(
        prob_k_inputs_from_subset(k, p=connect_p, assembly_size=assembly_size) for k in range(x)
    )
    prob_at_least_x = 1 - prob_less_than_x
    expected_dendrites = prob_at_least_x * n_somas

    bernoulli_variance = prob_at_least_x * (1 - prob_at_least_x)
    total_variance = n_somas * bernoulli_variance
    standard_deviation = math.sqrt(total_variance)

    return expected_dendrites, standard_deviation


def try_some_things():
    n_somas = 400
    n_dends_each = 6

    # np.random.seed(6)
    selected_ids = np.random.choice(n_somas * n_dends_each, size=40, replace=False)

    # print(selected_ids // n_dends_each)

    selected_ids = selected_ids // n_dends_each

    idx_sort = np.argsort(selected_ids)

    # sorts records array so all unique elements are together
    sorted_records_array = selected_ids[idx_sort]

    # returns the unique values, the index of the first occurrence of a value, and the count for each element
    vals, idx_start, count = np.unique(sorted_records_array, return_counts=True, return_index=True)

    # print(count)
    # print(vals)
    # print(len(vals), len(selected_ids), np.max(count))

    all_to_silence = np.zeros(70)
    for ii in range(70):
        overflow = ii - 20

        if overflow <= 0:
            continue

        prob = overflow / ii

        to_silence = n_somas * n_dends_each * prob
        all_to_silence[ii] = to_silence

    fig, ax = plt.subplots()
    ax.scatter(range(70), all_to_silence)
    plt.show()

    # # splits the indices into separate arrays
    # res = np.split(idx_sort, idx_start[1:])

    # # filter them with respect to their size, keeping only items occurring more than once
    # vals = vals[count > 1]
    # res = filter(lambda x: x.size > 1, res)

    # print(res)


def try_some_more_things():
    n_somas = 400
    p = 0.05
    n_dends_each = 6
    # connect to neurons based on probability distribution

    possible_dends_that_potentiate = []
    overlap_on_neuron = []
    n_overlaps = []

    for kk in range(1000):
        connectivity = np.zeros((n_somas * n_dends_each, n_somas))

        for soma in range(n_somas):  # this is pre
            targets = np.where(np.random.rand(n_somas) * n_dends_each < p)[0]
            # print(len(targets))
            # targets = np.random.choice(n_somas, size=int(n_dends_each * n_somas * p), replace=False)

            # dend_id = np.random.choice(6, size=len(targets), replace=True)

            connectivity[targets * n_dends_each + dend_id, soma] = 1

        for ii in range(20):
            sources = np.random.choice(n_somas, size=20, replace=False)
            X = np.sum(connectivity[:, sources], axis=1)

            dends_bigger_threshold = X > 3

            dend_ids_bigger_threshold = np.where(dends_bigger_threshold == 1)[0]
            soma_ids_bigger_threshold = dend_ids_bigger_threshold // n_dends_each
            # print(soma_ids_bigger_threshold)
            vals, idx_start, count = np.unique(
                np.sort(soma_ids_bigger_threshold), return_counts=True, return_index=True
            )

            # print(np.max(count))

            possible_dends_that_potentiate.append(np.sum(dends_bigger_threshold))
            overlap_on_neuron.append(np.max(count) > 1)
            n_overlaps.append(np.sum(count > 1))

    # print(np.sum(overlap_on_neuron) / len(overlap_on_neuron))

    # print(np.unique(n_overlaps))
    fig, ax = plt.subplots()
    ax.hist(possible_dends_that_potentiate, bins=np.arange(20, 60))
    plt.show()


def try_som_stuff(p_to_rec=0.4, p_to_dend=0.2, n_rec=50, threshold_rec=1):
    n_somas = 400
    n_dends_each = 6

    connections_rec_soma = (np.random.rand(n_rec, n_somas) < p_to_rec).astype(int)
    connections_dend_rec = (np.random.rand(n_somas * n_dends_each, n_rec) < p_to_dend).astype(int)

    fig, ax = plt.subplots()
    ax.hist(np.sum(connections_rec_soma, axis=1))
    # plt.show()
    all_silenced = []
    all_active_rec = []
    test_active_cells = [ii for ii in range(60)]
    for kk in range(100):
        all_silenced.append([])
        all_active_rec.append([])
        for ii in test_active_cells:
            selected_somas = np.random.choice(n_somas, size=ii, replace=False)

            inputs_to_rec = np.sum(connections_rec_soma[:, selected_somas], axis=1)
            active_rec = np.where(inputs_to_rec >= threshold_rec)[0]

            all_active_rec[-1].append(len(active_rec))

            inputs_to_dends = np.sum(connections_dend_rec[:, active_rec], axis=1)

            silenced_dends = np.where(inputs_to_dends > 0)[0]

            all_silenced[-1].append(len(silenced_dends))

    fig, (ax, ax2) = plt.subplots(2)
    ax.plot(test_active_cells, np.mean(all_silenced, axis=0))
    ax.plot(test_active_cells, np.array(all_silenced).T, alpha=0.1, lw=0.4, color="k")

    ax2.plot(test_active_cells, np.mean(all_active_rec, axis=0))
    ax2.plot(test_active_cells, np.array(all_active_rec).T, alpha=0.1, lw=0.4, color="k")

    plt.show()


if __name__ == "__main__":
    show_dendrite_distributions(
        axes=None,
        equal_or_more=4,
        n_somas=400,
        n_dends_each=6,
        assembly_size=20,
        show_plot=True,
        original_p=0.07,
        new_p=0.081,
        max_x=70,
    )

    # try_som_stuff(p_to_rec=0.08, p_to_dend=0.1, n_rec=60, threshold_rec=3)
    # try_some_things()
    # try_some_more_things()
    main()
