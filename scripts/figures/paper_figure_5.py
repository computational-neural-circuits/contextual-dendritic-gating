from brian2.units import *

from src.network_task import NetworkTask
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
from torchvision import datasets, transforms
import hashlib
import json
from src.get_activity_metrics import get_activity_metrics_from_assembly_neurons
from os.path import abspath, dirname, join
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable

parameter_dict = {}


def get_path_to_save_file_name(folder_name, save_file_name):
    path = abspath(
        join(
            dirname(__file__),
            "..",
            "results",
            "figures",
            folder_name,
            save_file_name,
        )
    )

    # check if folder exist
    path_to_results_folder = join(
        dirname(__file__),
        "..",
        "results",
        "figures",
    )
    exists = os.path.exists(path_to_results_folder)
    if not exists:
        os.makedirs(path_to_results_folder)

    path_to_specific_results_folder = join(
        dirname(__file__), "..", "results", "figures", folder_name
    )
    exists = os.path.exists(path_to_specific_results_folder)
    if not exists:
        os.makedirs(path_to_specific_results_folder)

    return path


def add_colorbar_to_figure(fig, im):
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    # fig.colorbar(im, cax=cax, orientation="vertical")


def main(
    seed=5,
    n_recall=4,
    n_train=10,
    n_train_additional=0,
    create_random_patch_positions=True,
    setup_new_task=False,
    context_0_for_B=False,
):
    folder_name = f"task_with_emnist_seed_{seed}"

    np.random.seed(seed)
    target_labels = {"0": 0, "1": 1, "l": 47, "O": 24}  # 0, 1, l, O
    auditory_map_for_labels = {"0": 0, "1": 1, "l": 2, "O": 3}

    samples, patches = load_EMNIST_data_and_create_gabor_patches(
        target_labels, seed=seed, create_random_patch_positions=create_random_patch_positions
    )

    fig_gabor, ax_gabor = plt.subplots(20, 20, figsize=(22, 22))
    fig_gabor_positions, ax_gabor_positions = plt.subplots(figsize=(22, 22))
    all_patch_values = [patch for patch, (_, _) in patches]
    min_max = np.max(
        [
            np.abs(min([np.min(arr) for arr in all_patch_values])),
            np.abs(max([np.max(arr) for arr in all_patch_values])),
        ]
    )

    for ii, ax in enumerate(ax_gabor.flatten()):
        patch, (x, y) = patches[ii]
        to_show = np.zeros((28, 28))
        patch_x, patch_y = patch.shape
        print(x, y, patch_x, patch_y)

        try:
            to_show[x - patch_x // 2 : x + patch_x // 2, y - patch_y // 2 : y + patch_y // 2] = patch
            im = ax.imshow(to_show, cmap="bwr", vmin=-min_max, vmax=min_max)
        except ValueError:
            im = ax.imshow(patch, cmap="bwr", vmin=-min_max, vmax=min_max)

        ax.axis("off")

        ax_gabor_positions.scatter(x, y, color="k")

    ax_gabor_positions.set(xlim=[0, 27], ylim=[0, 27])
    add_colorbar_to_figure(fig_gabor, im)

    fig_gabor.savefig(get_path_to_save_file_name(folder_name, "gabor_patches.pdf"), dpi=600)
    fig_gabor_positions.savefig(
        get_path_to_save_file_name(folder_name, "gabor_patch_positions.pdf"), dpi=600
    )

    all_inputs = np.zeros((len(target_labels), n_train + n_recall + n_train_additional, 400))
    all_input_images = np.zeros(
        (len(target_labels), n_train + n_recall + n_train_additional, 28, 28)
    )

    row_col = int(np.ceil(np.sqrt(len(target_labels))))
    fig_all_filtered_inputs, ax = plt.subplots(row_col, row_col, figsize=(22, 22))
    fig_all_filtered_inputs_avg, ax_avg = plt.subplots(figsize=(22, 22))
    fig_images, ax_image = plt.subplots(
        len(target_labels), n_train + n_recall + n_train_additional, figsize=(22, 22)
    )
    for ll, label in enumerate(target_labels):
        for ii in range(n_train + n_recall + n_train_additional):
            print(ii)
            print(np.array(samples[label]).shape)
            orig = samples[label][ii]
            gabor = filter_image_with_patches(
                orig, patches=patches, create_random_patch_positions=create_random_patch_positions
            )
            max_response = 12
            x = process_gabor_filtered_array_for_input(gabor, n=20, max_response=12).flatten()

            if ll == 0 and ii == 0:
                sorted_indices = np.argsort(x)

            all_inputs[ll, ii, :] = x[sorted_indices]
            all_input_images[ll, ii, :, :] = orig.T
            # all_inputs[ll, ii, :] = x
            ax_image[ll, ii].imshow(orig.T, cmap="Greys")
            ax_image[ll, ii].axis("off")
            title = "Recall"
            if ii < n_train:
                title = "Train"
            if ii >= n_train + n_recall:
                title = "Train +"

            ax_image[ll, ii].set_title(title)

        this_ax = ax.flatten()[ll]
        im = this_ax.imshow(all_inputs[ll].T, vmin=0, vmax=max_response, cmap="Greys")
        this_ax.set(aspect=(n_train + n_recall + n_train_additional) / 400.0, title=label)
        this_ax.axvline(x=n_train - 0.5, color="r", linestyle="--")
        this_ax.axvline(x=n_train + n_recall - 0.5, color="r", linestyle="--")

        # Adding labels below the x-axis
        this_ax.text(
            n_train / 2 - 0.5, -0.1, "Training", ha="center", transform=this_ax.get_xaxis_transform()
        )
        this_ax.text(
            n_train + n_recall / 2 - 0.5,
            -0.1,
            "Recall",
            ha="center",
            transform=this_ax.get_xaxis_transform(),
        )
        this_ax.text(
            n_train + n_recall + (n_train_additional / 2) - 0.5,
            -0.1,
            "Additional Training",
            ha="center",
            transform=this_ax.get_xaxis_transform(),
        )
        this_ax.set(
            xticks=[ii for ii in range(n_train + n_recall + n_train_additional)], xticklabels=[]
        )

        add_colorbar_to_figure(fig_all_filtered_inputs, im)

    im = ax_avg.imshow(
        [np.mean(all_inputs[ll], axis=0) for ll in range(len(target_labels))],
        vmin=0,
        vmax=max_response,
        cmap="Greys",
    )
    ax_avg.set(aspect=400 / len(target_labels))
    ax_avg.set(
        yticks=[ii for ii in range(len(target_labels))],
        yticklabels=[label for label in target_labels],
    )
    add_colorbar_to_figure(fig_all_filtered_inputs_avg, im)

    fig_all_filtered_inputs.savefig(
        get_path_to_save_file_name(folder_name, "images_filtered.pdf"), dpi=600
    )
    fig_all_filtered_inputs_avg.savefig(
        get_path_to_save_file_name(folder_name, "images_filtered_avg.pdf"), dpi=600
    )
    fig_images.savefig(get_path_to_save_file_name(folder_name, "images.pdf"), dpi=600)
    # plt.show()

    all_assembly_inputs = []
    all_imprint_ids = []

    all_assembly_inputs_for_additional_training = []
    all_imprint_ids_for_additional_training = []

    all_recall_ids = []
    all_assembly_inputs_recall = []

    for ll, label in enumerate(target_labels):
        for ii in range(n_train):
            all_assembly_inputs.append(all_inputs[ll, ii].tolist())
            # context A, input for A, context B, input for B, context C
            context_B = 0
            context_C = 0
            if label in ["l", "O"]:
                if not context_0_for_B:
                    context_B = 1
                context_C = 1
            all_imprint_ids.append(
                [
                    0,
                    len(all_assembly_inputs) - 1,
                    context_B,
                    auditory_map_for_labels[label],
                    context_C,
                ]
            )

    for ii in range(n_recall):
        for ll, label in enumerate(target_labels):
            all_assembly_inputs_recall.append(all_inputs[ll, ii + n_train].tolist())

            context_B = 0
            context_C = 0
            if label in ["l", "O"]:
                if not context_0_for_B:
                    context_B = 1
                context_C = 1
            all_recall_ids.append(
                [
                    0,
                    0,
                    context_B,
                    -1,
                    context_C,
                ]
            )

    for ll, label in enumerate(target_labels):
        for ii in range(n_train_additional):
            all_assembly_inputs_for_additional_training.append(
                all_inputs[ll, ii + n_train + n_recall].tolist()
            )
            # context A, input for A, context B, input for B, context C
            context_B = 0
            context_C = 0
            if label in ["l", "O"]:
                if not context_0_for_B:
                    context_B = 1
                context_C = 1
            all_imprint_ids_for_additional_training.append(
                [
                    0,
                    len(all_assembly_inputs_for_additional_training) - 1,
                    context_B,
                    auditory_map_for_labels[label],
                    context_C,
                ]
            )

    np.random.seed(seed)
    np.random.shuffle(all_imprint_ids_for_additional_training)

    np.random.seed(seed)
    np.random.shuffle(all_imprint_ids)

    target_key_list = [key for (key, _) in target_labels.items()]
    all_last_imprint_ids = {}
    for ii, imprint_ids in enumerate(all_imprint_ids):
        all_last_imprint_ids[target_key_list[imprint_ids[1] // n_train]] = ii

    for ii, imprint_ids in enumerate(all_imprint_ids_for_additional_training):
        all_last_imprint_ids[target_key_list[imprint_ids[1] // n_train_additional]] = (
            ii + len(target_labels) * n_train
        )

    print(all_last_imprint_ids)

    print(all_imprint_ids)
    print(all_imprint_ids_for_additional_training)

    all_assembly_inputs_key = hashlib.sha1(
        json.dumps({"all_assembly_inputs": all_assembly_inputs}, sort_keys=True).encode()
    ).hexdigest()[:12]

    all_assembly_neuron_ids = [
        [ii for ii in range(20)],  # "One"
        [ii for ii in range(20, 40)],  # "El"
        [ii for ii in range(60, 80)],  # "oh"
        [ii for ii in range(80, 100)],  # "zero"
    ]

    if len(target_labels) > len(all_assembly_neuron_ids):
        all_assembly_neuron_ids = [
            [ii for ii in range(20)],  # "One"
            [ii for ii in range(20, 40)],  # "El"
            [ii for ii in range(40, 60)],  # "El"
            [ii for ii in range(60, 80)],  # "El"
            [ii for ii in range(80, 100)],  # "El"
            [ii for ii in range(100, 120)],  # "El"
            [ii for ii in range(120, 140)],  # "El"
            [ii for ii in range(160, 180)],  # "El"
            [ii for ii in range(180, 200)],  # "El"
            [ii for ii in range(200, 220)],  # "El"
            [ii for ii in range(60, 80)],  # "oh"
            [ii for ii in range(80, 100)],  # "zero"
        ]

    parameters_for_run = {
        "runtime_imprint": 6.0 * second,
        "runtime_baseline": 0.8 * second,
        "seed": seed,
        "all_assembly_neuron_ids": all_assembly_neuron_ids,
        "all_assembly_inputs_key": all_assembly_inputs_key,
        "all_imprint_ids": all_imprint_ids,
    }

    save_file_name = "task_eminst"

    net = NetworkTask(
        parameter_file_name="parameters",
        parameters_for_run=parameters_for_run,
        save_file_name=save_file_name,
        parameter_dict=parameter_dict,
        rerun=False,
        setup_new_task=setup_new_task,
    )

    # plt.show()
    net.run_imprint(report_style="text", all_assembly_inputs=all_assembly_inputs)

    if n_train_additional > 0:
        all_assembly_inputs_key_for_additional_training = hashlib.sha1(
            json.dumps(
                {
                    "all_assembly_inputs_for_additional_training": all_assembly_inputs_for_additional_training
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()[:12]

        net.parameters_for_run[
            "all_assembly_inputs_key"
        ] += all_assembly_inputs_key_for_additional_training
        net.parameters_for_run["all_imprint_ids"] = all_imprint_ids_for_additional_training

        filename_for_stored_network = net.save_dict["filename_for_stored_network"]
        if not type(filename_for_stored_network) is str:
            filename_for_stored_network = filename_for_stored_network.decode("utf-8")
        net.network.restore(
            filename=net.get_path_to_stored_networks(file_name=filename_for_stored_network)
        )
        net.run_imprint(
            report_style="text", all_assembly_inputs=all_assembly_inputs_for_additional_training
        )

    # net.show_spike_rasters(show_plot=True, sort_by_new_algorithm=True)
    # net.show_spike_rasters(show_plot=True, sort_by_new_algorithm=True)

    rtm = net.parameters_for_run["runtime_imprint"] / msecond
    bsl = net.parameters_for_run["runtime_baseline"] / msecond
    runtime_recall = 2 * second
    rtm_rec = runtime_recall / msecond

    # we need to get the last imprint ids

    sorted_ids_before_recall = []

    for context_id in range(2):
        sorted_ids_before_recall.append([])
        for area in net.all_areas:
            sorted_neuron_ids = net.sort_neurons_by_weights(area=area, context_id=context_id)
            sorted_ids_before_recall[-1].append(sorted_neuron_ids)

    assembly_neuron_ids_for_targets_and_all_areas = {}
    last_imprint_rate_for_targets_and_all_areas = {}
    last_imprint_n_active_for_targets_and_all_areas = {}

    first_last_imprint_start = 1e10
    for key, imprint_id in all_last_imprint_ids.items():
        assembly_neuron_ids_for_targets_and_all_areas[key] = []
        last_imprint_rate_for_targets_and_all_areas[key] = []
        last_imprint_n_active_for_targets_and_all_areas[key] = []
        start_time = (imprint_id + 1) * (bsl + rtm) - rtm_rec + bsl * int(n_train_additional > 0)
        end_time = start_time + rtm_rec

        if start_time < first_last_imprint_start:
            first_last_imprint_start = start_time

        sorted_last_imprint_ids = []

        actual_imprint_id = imprint_id - len(target_labels) * n_train * int(n_train_additional > 0)

        for area in net.all_areas:
            sorted_neuron_ids_for_imprint, _, assembly_neuron_ids = net.sort_neurons_by_firing_rate(
                area=area,
                sort_for_specific_imprint=actual_imprint_id,
                t_0=int(n_train_additional > 0) * (len(target_labels) * n_train * (bsl + rtm) + bsl),
            )
            sorted_last_imprint_ids.append(sorted_neuron_ids_for_imprint)
            assembly_neuron_ids_for_targets_and_all_areas[key].append(assembly_neuron_ids)

            (
                avg_firing_rate,
                n_active_neurons,
                avg_firing_rate_not_in_assembly,
                n_active_neurons_not_in_assembly,
            ) = get_activity_metrics_from_assembly_neurons(
                active_threshold=4,
                net=net,
                area=area,
                sorted_neuron_ids=sorted_neuron_ids,
                selected_ids=assembly_neuron_ids,
                start_time=start_time,
                end_time=end_time,
                remove_fraction_of_start=0,
                select_randomly_for_background=False,
                run_for_task=True,
            )

            print("##: ", avg_firing_rate, n_active_neurons)
            last_imprint_rate_for_targets_and_all_areas[key].append(avg_firing_rate)
            last_imprint_n_active_for_targets_and_all_areas[key].append(n_active_neurons)

        use_these_imprint_ids = all_imprint_ids
        if n_train_additional > 0:
            use_these_imprint_ids = all_imprint_ids_for_additional_training
        context_A, _, context_B, _, context_C = use_these_imprint_ids[actual_imprint_id]
        sorted_ids = [
            sorted_ids_before_recall[context_A][0],
            sorted_ids_before_recall[context_B][1],
            sorted_ids_before_recall[context_C][2],
        ]

        net.show_spike_rasters(
            sorted_ids=sorted_ids,
            show_plot=True,
            show_vertical_lines_at=[
                start_time,
                end_time,
            ],
            highlight_neuron_ids=[
                [ii, vals]
                for ii, vals in enumerate(assembly_neuron_ids_for_targets_and_all_areas[key])
            ],
            highlight_neuron_ids_between_specific_time_points=[start_time, end_time],
            save_file_name=get_path_to_save_file_name(
                folder_name, f"final_assemblies_spikes_for_{key}"
            ),
        )

    net.save_weight_matrix_for_matlab(
        show_plot=False,
        matlab_export_name=f"../../results/figures/task_with_emnist_seed_{seed}/WEIGHTS",
    )
    # n_recall, n_targets, assembly_id, areas, rate/n_active
    recall_results = np.zeros((n_recall, len(target_labels), len(target_labels), 3, 2)) * float(
        "nan"
    )
    for running_recall_number, (recall_inputs, recall_id) in enumerate(
        zip(all_assembly_inputs_recall, all_recall_ids)
    ):
        for run_recall_after_imprint in [True]:
            all_assembly_inputs_key_recall = hashlib.sha1(
                json.dumps({"all_assembly_inputs_recall": recall_inputs}, sort_keys=True).encode()
            ).hexdigest()[:12]
            net.parameters_for_run.update(
                {
                    "recall_id": recall_id,
                    "all_assembly_inputs_key_recall": all_assembly_inputs_key_recall,
                    "runtime_recall": 2 * second,
                    "run_recall_after_imprint": run_recall_after_imprint,
                    "runtime_baseline_recall": 0.3 * second,
                }
            )

            net.run_recall(report_style="text", recall_inputs=recall_inputs)

            print("#############")
            print(recall_id)

            ll = running_recall_number % n_recall
            ii = running_recall_number // n_recall

            start_time = (
                bsl
                + bsl * int(n_train_additional > 0)
                + (n_train + n_train_additional) * len(target_labels) * (bsl + rtm)
            )
            end_time = start_time + rtm_rec

            assembly_ids_for_recall_for_all_areas = []
            for area_id, area in enumerate(net.all_areas):
                for assembly_id, key in enumerate(target_key_list):
                    assembly_neuron_ids = assembly_neuron_ids_for_targets_and_all_areas[key][area_id]

                    if assembly_id == ll:
                        assembly_ids_for_recall_for_all_areas.append(assembly_neuron_ids)

                    (
                        avg_firing_rate,
                        n_active_neurons,
                        avg_firing_rate_not_in_assembly,
                        n_active_neurons_not_in_assembly,
                    ) = get_activity_metrics_from_assembly_neurons(
                        active_threshold=4,
                        net=net,
                        area=area,
                        sorted_neuron_ids=None,
                        selected_ids=assembly_neuron_ids,
                        start_time=start_time,
                        end_time=end_time,
                        remove_fraction_of_start=0,
                        select_randomly_for_background=False,
                        run_for_task=True,
                    )

                    print("HUHUHUH: ", ll, ii, assembly_id, area_id, avg_firing_rate)
                    # fig, ax = plt.subplots()
                    # ax.imshow(all_input_images[ll, ii + n_train], cmap="Greys")
                    # ax.axis("off")
                    # n_recall, n_targets, assembly_id, areas, rate/n_active
                    recall_results[ii, ll, assembly_id, area_id, 0] = avg_firing_rate
                    recall_results[ii, ll, assembly_id, area_id, 1] = n_active_neurons

            # fig, ax = plt.subplots()
            # ax.imshow(all_input_images[ll, ii + n_train], cmap="Greys")
            # ax.axis("off")

            sorted_ids = [
                sorted_ids_before_recall[recall_id[0]][0],
                sorted_ids_before_recall[recall_id[2]][1],
                sorted_ids_before_recall[recall_id[4]][2],
            ]
            net.show_spike_rasters(
                show_plot=True,
                sorted_ids=sorted_ids,
                show_vertical_lines_at=[
                    start_time,
                    end_time,
                ],
                highlight_neuron_ids=[
                    [ii, vals] for ii, vals in enumerate(assembly_ids_for_recall_for_all_areas)
                ],
                highlight_neuron_ids_between_specific_time_points=[start_time, end_time],
                save_file_name=get_path_to_save_file_name(
                    folder_name, f"recall_spikes_for_recall_image_{target_key_list[ll]}_{ii}"
                ),
                only_show_spikes_after=first_last_imprint_start,
            )

    fig, axes = plt.subplots(2, len(net.all_areas), figsize=(16, 12))
    # n_recall, n_targets, assembly_id, areas, rate/n_active

    xticklabels = []
    xticks = []

    color = "#2171b5"
    color_after_imprint = "#fe9929"
    for rate_n_active in [0, 1]:
        for area_id in range(len(net.all_areas)):
            ax = axes[rate_n_active, area_id]
            ax.set_title(f"Area {net.all_areas[area_id].name}")
            for target_id, target_name in enumerate(target_key_list):
                label_imprint = "last value after imprint"
                label_values = "recall values for recall 0"
                label_values_mean = "recall values mean"
                if rate_n_active != 0 or area_id != 0 or target_id != 0:
                    label_values = None
                    label_values_mean = None
                    label_imprint = None

                last_imprint = last_imprint_n_active_for_targets_and_all_areas
                if rate_n_active == 0:
                    last_imprint = last_imprint_rate_for_targets_and_all_areas
                ax.scatter(
                    target_id,
                    last_imprint[target_name][area_id],
                    color=color_after_imprint,
                    label=label_imprint,
                )
                last_imprint_n_active_for_targets_and_all_areas
                y = recall_results[:, target_id, :, area_id, rate_n_active]
                values_for_x = [
                    target_id - 0.25 + (0.5 * ff / (len(target_labels) - 1))
                    for ff in range(len(target_labels))
                ]
                ax.errorbar(
                    values_for_x,
                    np.nanmean(y, axis=0),
                    yerr=np.std(y, axis=0),
                    color=color,
                    label=label_values_mean,
                )
                xticks.append(target_id)
                xticklabels.append(target_name)
                recall_colors = ["#e41a1c", "#4daf4a", "#984ea3", "#ff7f00"]

                for recall_id in range(n_recall):
                    if label_values is not None:
                        label_values = label_values[:-1] + f"{recall_id}"
                    ax.plot(
                        values_for_x,
                        y[recall_id, :],
                        color=recall_colors[recall_id],
                        alpha=0.4,
                        label=label_values,
                    )
            ylabel = "avg firing rate of assembly"
            if rate_n_active == 1:
                ylabel = "n_active of assembly"
            ax.set(xticklabels=xticklabels, xticks=xticks, ylabel=ylabel)
            if area_id == 0 and rate_n_active == 0:
                ax.legend()

    fig.savefig(get_path_to_save_file_name(folder_name, "recall_results.pdf"), dpi=600)
    # net.show_end_of_imprint_vs_end_of_recall_in_spike_raster(show_plot=False)
    # net.save_weight_matrix_for_matlab()
    # net.show_spike_rasters(show_plot=True, ort_by_new_algorithm=True)
    # plt.show()


# Function to generate a Gabor patch
def gabor_patch(size, frequency, sigma, orientation, phase):
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    x, y = np.meshgrid(x, y)
    theta = np.deg2rad(orientation)
    x_prime = x * np.cos(theta) + y * np.sin(theta)
    y_prime = -x * np.sin(theta) + y * np.cos(theta)
    gaussian_envelope = np.exp(-(x_prime**2 + y_prime**2) / (2 * sigma**2))
    sinusoidal_grating = np.cos(2 * np.pi * frequency * x_prime + np.deg2rad(phase))
    gabor = gaussian_envelope * sinusoidal_grating
    return gabor


def process_gabor_filtered_array_for_input(arr, n, max_response=12):
    # Flatten the array and sort indices based on values
    flat_arr = arr.flatten()
    sorted_indices = np.argsort(flat_arr)

    # Select n lowest and n highest elements
    lowest_indices = sorted_indices[:n]
    highest_indices = sorted_indices[-n:]

    # Initialize a new array with all elements set to 0.1
    new_arr = np.full(flat_arr.shape, 0.1)

    # Set the n lowest elements to 0
    new_arr[lowest_indices] = 0

    # Scale the n highest elements to maintain ratios and yield 10 on average
    highest_values = flat_arr[highest_indices]
    # Ensure the maximum value does not exceed 14
    scaled_values = highest_values / highest_values.mean() * 10
    scaled_values = np.clip(scaled_values, None, max_response)

    # Recalculate to make sure the average is 10
    total = np.sum(scaled_values)
    count = len(scaled_values)
    correction_factor = 10 * count / total
    corrected_values = np.clip(scaled_values * correction_factor, None, max_response)

    print(np.mean(corrected_values), corrected_values)

    # Assign the scaled highest values to the new array
    new_arr[highest_indices] = corrected_values

    # Reshape back to the original shape
    return new_arr.reshape(arr.shape)


def filter_image_with_patches(image, patches, create_random_patch_positions):
    response_grid = np.zeros((20, 20))
    for idx, (patch, (x, y)) in enumerate(patches):
        filtered_image = convolve(image, patch)
        if not create_random_patch_positions:
            response_grid[idx // 20, idx % 20] = np.sum(filtered_image[x - 2 : x + 2, y - 2 : y + 2])
        else:
            response_grid[idx // 20, idx % 20] = np.sum(filtered_image[x - 1 : x + 1, y - 1 : y + 1])

    return response_grid


def load_EMNIST_data_and_create_gabor_patches(
    target_labels, seed, create_random_patch_positions=True
):
    np.random.seed(seed)

    # Load EMNIST dataset (letters and digits)
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: x.numpy().squeeze())]
    )

    emnist_train = datasets.EMNIST(
        root="./data", split="byclass", train=True, download=True, transform=transform
    )
    emnist_test = datasets.EMNIST(
        root="./data", split="byclass", train=False, download=True, transform=transform
    )

    # Collect all samples for each target label first
    all_samples = {label: [] for label in target_labels}
    for img, label in emnist_train:
        for key, val in target_labels.items():
            if label == val:
                all_samples[key].append(img)

    samples = {label: [] for label in target_labels}
    # Randomly select 50 samples for each target label
    for key in target_labels.keys():
        indices = np.random.choice(len(all_samples[key]), 50, replace=False)
        samples[key] = [all_samples[key][i] for i in indices]

    # samples = 0

    # # Parameters
    image_size = 28
    num_patches = 400
    max_patch_size = image_size  # // 2
    min_patch_size = image_size

    # Generate Gabor patches and their locations
    patches = [(np.zeros((2, 2)), (0, 0)) for _ in range(num_patches)]
    locations = []

    #     sigma_values = [1, 3, 5]
    # lambda_values =
    # theta_values = np.linspace(0, np.pi, 8)

    running_number = 0
    n_reps = 10
    all_orientations = np.linspace(0, 180, 4)
    all_phases = [0, 180]
    all_frequencies = [1, 1.5]
    all_sigma = [0.2, 0.4]

    if not create_random_patch_positions:
        all_orientations = np.linspace(0, 180, 5)[:4]
        n_reps = 12
    for ii in range(n_reps):
        for sigma in all_sigma:
            for orientation in all_orientations:  # , 90, 135]:
                for phase in all_phases:
                    for frequency in all_frequencies:
                        patch = gabor_patch(image_size, frequency, sigma, orientation, phase)

                        start_x, end_x = 5, image_size - 5
                        start_y, end_y = 5, image_size - 5
                        if (not create_random_patch_positions) and (ii < 9):
                            size = 6  # (28 - 10)//3

                            n_x = ii % 3
                            n_y = ii // 3

                            start_x = 5 + n_x * size
                            end_x = start_x + size

                            start_y = 5 + n_y * size
                            end_y = start_y + size

                        x = np.random.randint(start_x, end_x)
                        y = np.random.randint(start_y, end_y)

                        patches[running_number] = (patch, (x, y))
                        running_number += 1

    print(len(patches), running_number)
    # for _ in range(num_patches):
    #     patch_size = np.random.randint(min_patch_size, max_patch_size + 1)
    #     frequency = np.random.uniform(0.1, 1.3)

    #     orientation = np.random.uniform(0, 180)
    #     phase = np.random.uniform(0, 180)
    #     patch = gabor_patch(patch_size, frequency, sigma, orientation, phase)
    #     # x = np.random.randint(0, image_size)
    #     x = image_size // 2
    #     # y = np.random.randint(0, image_size)
    #     y = image_size // 2
    #     patches.append((patch, (x, y)))

    return samples, patches


if __name__ == "__main__":
    # main(seed=2, n_train_additional=0)
    # main(seed=2, n_train_additional=5)
    # main(seed=927, n_train_additional=5, create_random_patch_positions=False, context_0_for_B=True)
    main(seed=672, n_train_additional=5, create_random_patch_positions=False)
    # main(seed=672, n_train_additional=0, create_random_patch_positions=False)

    # main(seed=3892, n_train_additional=0, create_random_patch_positions=False, setup_new_task=True)
    # main(seed=3892, n_train_additional=5, create_random_patch_positions=False, setup_new_task=True)
