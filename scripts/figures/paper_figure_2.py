from brian2.units import *
import brian2 as br2

from src.single_neuron import SingleNeuron

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from multiprocessing import Pool

plt.style.use("../../plots_style.txt")
# br2.prefs.codegen.target = "numpy"

parameter_dict = {}

parameters_for_run = {
    "monitor_dt": 1 * ms,
    "monitor_dt_weights": 1 * ms,
    "seed": 0,
}

show_percent_of_positive_change = False

RERUN_SIM = False


# former figure_2.py


def paper_figure_2(make_nmda_spikes_linear=False):
    if make_nmda_spikes_linear:
        parameters_for_run["make_nmda_spikes_linear"] = True

    fig = plt.figure(figsize=(24, 16))
    gs = fig.add_gridspec(20, 13)
    ax_1 = fig.add_subplot(gs[7, 0:4])
    ax_2 = fig.add_subplot(gs[8:12, 0:4])
    ax_3 = fig.add_subplot(gs[12:13, 0:4])
    axes = [ax_1, ax_2, ax_3]

    n_active_pre = [6, 5, 1]
    inhibitory_rates = [40, 100, 350]

    if make_nmda_spikes_linear:
        n_active_pre = [11, 10, 1]
        inhibitory_rates = [8, 100, 350]

    voltage_trace_and_weight_changes_close_up(
        axes=axes,
        show_plot=False,
        n_active_pre=n_active_pre[0],
        inhibitory_rate=inhibitory_rates[0],
        xmin=3599,
        xmax=4400,
    )

    ax_2 = fig.add_subplot(gs[7:11, 5:9])
    ax_3 = fig.add_subplot(gs[11:13, 5:9])
    axes = [ax_2, ax_3]

    voltage_trace_and_weight_changes_for_different_n_of_active_inputs(
        axes=axes,
        show_plot=False,
        n_active_pre=n_active_pre,
        inhibitory_rate=inhibitory_rates[0],
        all_annotation_numbers=["a", "b", "c"],
        make_nmda_spikes_linear=make_nmda_spikes_linear,
    )

    ax_2 = fig.add_subplot(gs[7:11, 9:])
    ax_3 = fig.add_subplot(gs[11:13, 9:])
    axes = [ax_2, ax_3]

    voltage_traces_and_weight_changes_for_different_inhibitory_rates(
        axes=axes,
        show_plot=False,
        n_active_pre=n_active_pre[0],
        inhibitory_rates=inhibitory_rates,
        remove_y_ticks=False,
        all_annotation_numbers=["a", "b*", "c*"],
        make_nmda_spikes_linear=make_nmda_spikes_linear,
    )

    ax = fig.add_subplot(gs[15:, 5:])

    weight_changes_for_different_contexts(
        ax=ax, show_plot=False, inhibitory_rate=inhibitory_rates[0]
    )

    axes = fig.add_subplot(gs[0:6, 5:])
    weight_change_averages_for_n_active_and_inhibitory_rate(
        axes=axes,
        n_active_pre=n_active_pre,
        inhibitory_rates=inhibitory_rates,
        show_plot=False,
    )

    # plt.show()
    save_name = "../../results/figures/paper_fig_2.pdf"
    if show_percent_of_positive_change:
        save_name = "../../results/figures/paper_fig_2_percent_LTP.pdf"
    if "make_nmda_spikes_linear" in parameters_for_run:
        if parameters_for_run["make_nmda_spikes_linear"]:
            save_name = "../../results/figures/paper_fig_2_linear_nmda.pdf"
            if show_percent_of_positive_change:
                save_name = "../../results/figures/paper_fig_2_linear_nmda_percent_LTP.pdf"

    plt.savefig(save_name)


def voltage_trace_and_weight_changes_for_different_n_of_active_inputs(
    axes=None,
    show_plot=True,
    remove_y_ticks=False,
    n_active_pre=[6, 5, 1],
    inhibitory_rate=0,
    all_annotation_numbers=["a", "b", "c"],
    make_nmda_spikes_linear=False,
):
    if axes is None:
        fig = plt.figure()
        gs = fig.add_gridspec(6, 1)
        ax_2 = fig.add_subplot(gs[0:5, :])
        ax_3 = fig.add_subplot(gs[5:, :])
        axes = [ax_2, ax_3]

    new_pars_for_run = {}
    new_pars_for_run.update(parameters_for_run)

    new_pars_for_run["runtime"] = 6 * second
    new_pars_for_run["inhibitory_rate"] = inhibitory_rate

    new_pars_for_run["prevent_plasticity"] = True

    new_pars_for_run["add_silent_synapses"] = True
    new_pars_for_run["silent_synapse_starting_weight"] = 5.0
    new_pars_for_run["record_all"] = True

    if "make_nmda_spikes_linear" in parameters_for_run:
        new_pars_for_run["make_nmda_spikes_linear"] = True

    for mm, (n_active, color) in enumerate(
        zip(
            n_active_pre,
            ["#0570b0", "#dd3497", "#238b45"],
        )
    ):
        new_pars_for_run["n_active"] = n_active

        if mm == 0:
            print("##############")
            print(parameter_dict)
            print(new_pars_for_run)
        neuron = SingleNeuron(
            parameter_file_name="parameters",
            parameters_for_run=new_pars_for_run,
            save_file_name="paper_figure_2",
            parameter_dict=parameter_dict,
            rerun=RERUN_SIM,
        )
        neuron.run(report_style="text")
        plot_results(
            axes=axes,
            neuron=neuron,
            target_dendrite=0,
            label=f"{new_pars_for_run['n_active']} active inputs",
            color=color,
            remove_y_ticks=remove_y_ticks,
            annotation_number=all_annotation_numbers[mm],
            make_nmda_spikes_linear=make_nmda_spikes_linear,
        )

    if show_plot:
        plt.show()


def voltage_traces_and_weight_changes_for_different_inhibitory_rates(
    axes=None,
    show_plot=True,
    inhibitory_rates=[36, 120, 400],
    n_active_pre=6,
    remove_y_ticks=False,
    all_annotation_numbers=["a", "b", "c"],
    make_nmda_spikes_linear=True,
):
    if axes is None:
        fig = plt.figure()
        gs = fig.add_gridspec(6, 1)
        ax_2 = fig.add_subplot(gs[0:5, :])
        ax_3 = fig.add_subplot(gs[5:, :])
        axes = [ax_2, ax_3]

    new_pars_for_run = {}
    new_pars_for_run.update(parameters_for_run)

    new_pars_for_run["runtime"] = 6 * second
    new_pars_for_run["add_silent_synapses"] = True
    new_pars_for_run["silent_synapse_starting_weight"] = 5.0
    new_pars_for_run["prevent_plasticity"] = True
    new_pars_for_run["record_all"] = True

    new_pars_for_run["n_active"] = n_active_pre

    if "make_nmda_spikes_linear" in parameters_for_run:
        new_pars_for_run["make_nmda_spikes_linear"] = True

    for mm, (firing_rate, color) in enumerate(
        zip(
            inhibitory_rates,
            ["#0570b0", "#dd3497", "#238b45"],
        )
    ):
        new_pars_for_run["inhibitory_rate"] = firing_rate

        if mm == 0:
            print("##############")
            print(parameter_dict)
            print(new_pars_for_run)
        neuron = SingleNeuron(
            parameter_file_name="parameters",
            parameters_for_run=new_pars_for_run,
            save_file_name="paper_figure_2",
            parameter_dict=parameter_dict,
            rerun=RERUN_SIM,
        )
        neuron.run(report_style="text")
        plot_results(
            axes=axes,
            neuron=neuron,
            target_dendrite=0,
            label=f"inhibitory rate {int(new_pars_for_run['inhibitory_rate']/Hz)} Hz",
            color=color,
            annotation_number=all_annotation_numbers[mm],
            remove_y_ticks=remove_y_ticks,
            make_nmda_spikes_linear=make_nmda_spikes_linear,
        )

    if show_plot:
        plt.show()


def voltage_trace_and_weight_changes_close_up(
    axes=None,
    show_plot=True,
    n_active_pre=6,
    inhibitory_rate=30,
    xmin=499,
    xmax=1000,
):
    new_pars_for_run = {}
    new_pars_for_run.update(parameters_for_run)
    # new_pars_for_run["runtime"] = 6 * second

    new_pars_for_run["runtime"] = 6 * second
    new_pars_for_run["add_silent_synapses"] = False
    new_pars_for_run["prevent_plasticity"] = False
    new_pars_for_run["normalize"] = False
    new_pars_for_run["record_all"] = True
    new_pars_for_run["n_active"] = n_active_pre
    new_pars_for_run["inhibitory_rate"] = inhibitory_rate

    print(new_pars_for_run)

    neuron = SingleNeuron(
        parameter_file_name="parameters",
        parameters_for_run=new_pars_for_run,
        save_file_name="paper_figure_2",
        parameter_dict=parameter_dict,
        rerun=RERUN_SIM,
    )

    neuron.run(report_style="text")

    if axes is None:
        fig = plt.figure()
        gs = fig.add_gridspec(6, 1)
        ax_1 = fig.add_subplot(gs[0, :])
        ax_2 = fig.add_subplot(gs[1:5, :])
        ax_3 = fig.add_subplot(gs[5:, :])
        axes = [ax_1, ax_2, ax_3]

    plot_results(axes=axes, neuron=neuron, target_dendrite=0, xmin=xmin, xmax=xmax)

    if show_plot:
        plt.show()


def weight_change_averages_for_n_active_and_inhibitory_rate(
    axes=None,
    n_active_pre=[0, 0, 0],
    inhibitory_rates=[0, 0, 0],
    show_plot=True,
    linear_NMDA=False,
    extra_seed=None,
    all_active=None,
    just_run_the_simulations=False,
):
    new_pars_for_run = {}
    new_pars_for_run.update(parameters_for_run)

    new_pars_for_run["add_silent_synapses"] = True
    new_pars_for_run["monitor_dt"] = 10 * ms
    new_pars_for_run["monitor_dt_weights"] = 10 * ms
    new_pars_for_run["runtime"] = 10 * second
    new_pars_for_run["silent_synapse_starting_weight"] = 5.0
    new_pars_for_run["prevent_plasticity"] = True

    if axes is None and not just_run_the_simulations:
        fig, axes = plt.subplots()

    if all_active is None:
        all_active = [ii for ii in range(0, 16)]  # [ii for ii in range(0, 16)]
    all_rates = [ii for ii in range(400) if ii % 2 == 0]  # [ii for ii in range(400) if ii % 2 == 0]
    all_seeds = range(10)  # range(10)

    new_pars_for_run["all_inhibitory_rates"] = all_rates
    new_pars_for_run["all_n_active"] = all_active
    new_pars_for_run["target_dendrite"] = 0

    if ("make_nmda_spikes_linear" in parameters_for_run) or linear_NMDA:
        new_pars_for_run["make_nmda_spikes_linear"] = True

    all_weight_changes = np.zeros((len(all_seeds), 10, len(all_active), len(all_rates)))

    for seed in all_seeds:
        if extra_seed is not None:
            if seed != extra_seed:
                continue

        print("SEED: ", seed, all_active, all_rates)
        new_pars_for_run["seed"] = seed

        new_pars_for_run["inhibitory_rate"] = None  # I just need to set it

        neuron = SingleNeuron(
            parameter_file_name="parameters",
            parameters_for_run=new_pars_for_run,
            save_file_name="single_neuron_scan",
            save_parameters=False,
            parameter_dict=parameter_dict,
            rerun=RERUN_SIM,
        )
        neuron.run_scan(report_style="text")

        all_weight_changes[seed, :, :, :] = neuron.save_dict["all_weight_changes"]

    if just_run_the_simulations:
        return
    all_weight_changes = all_weight_changes.reshape(
        (
            all_weight_changes.shape[0] * all_weight_changes.shape[1],
            all_weight_changes.shape[2],
            all_weight_changes.shape[3],
        )
    )
    # print(all_weight_changes.shape)

    divnorm = colors.TwoSlopeNorm(
        vmin=np.min(all_weight_changes), vcenter=0.0, vmax=np.max(all_weight_changes)
    )

    if show_percent_of_positive_change:
        divnorm = None
        all_weight_changes = all_weight_changes >= 0

    half_x = (all_rates[1] - all_rates[0]) / 2.0
    half_y = (all_active[1] - all_active[0]) / 2.0
    im = axes.imshow(
        np.mean(all_weight_changes, axis=0),
        extent=[
            all_rates[0] - half_x,
            all_rates[-1] + half_x,
            all_active[0] - half_y,
            all_active[-1] + half_y,
        ],
        origin="lower",
        cmap="RdBu_r",
        norm=divnorm,
    )
    axes.set_aspect(0.85 * len(all_rates) / len(all_active))

    ticks = [-1, -2, -3, 0, 5, 10, 15]
    label = "avg weight change of 'silent' weights\nafter 10 seconds (10 synapses/6 seeds)"
    if "make_nmda_spikes_linear" in parameters_for_run:
        label = "percent of LTD synapses"
        cbar = plt.colorbar(
            im,
            label=label,
        )
    else:
        cbar = plt.colorbar(
            im,
            ticks=ticks,
            label=label,
        )

    axes.set(xlabel="inhibitory firing rate in Hz", ylabel="# of highly active inputs")

    for y_val, number in zip(n_active_pre, ["a", "b", "c"]):
        axes.annotate(
            f"{number}",
            xy=(inhibitory_rates[0], y_val),
            xytext=(0, 0),  # 4 points vertical offset.
            textcoords="offset points",
            ha="center",
            va="center",
        )

    for x_val, number in zip(inhibitory_rates[1:], ["b", "c"]):
        axes.annotate(
            f"{number}*",
            xy=(x_val, n_active_pre[0]),
            xytext=(0, 0),  # 4 points vertical offset.
            textcoords="offset points",
            ha="center",
            va="center",
        )

    if show_plot:
        plt.show()

    return


def weight_changes_for_different_contexts(ax=None, show_plot=True, inhibitory_rate=40):
    if ax is None:
        fig, ax = plt.subplots()

    new_pars_for_run = {}
    new_pars_for_run.update(parameters_for_run)

    new_pars_for_run["runtime"] = 4 * second
    new_pars_for_run["add_silent_synapses"] = True
    new_pars_for_run["prevent_plasticity"] = True
    new_pars_for_run["record_all"] = True
    new_pars_for_run["silent_synapse_starting_weight"] = 5.0
    new_pars_for_run["inhibitory_rate"] = inhibitory_rate
    new_pars_for_run["seed"] = 4
    new_pars_for_run["all_context_ids"] = [0, 1, 2, 0, 1, 2]

    avg_weights = []

    new_pars_for_run["n_active_inputs_per_dendrite"] = [1, 6, 4, 7]
    neuron = SingleNeuron(
        parameter_file_name="parameters",
        parameters_for_run=new_pars_for_run,
        save_file_name="paper_figure_2",
        rerun=RERUN_SIM,
    )
    neuron.run_different_contexts(report_style="text")

    res = neuron.save_dict

    avg_weight_changes = []
    all_n_active = []
    for dendrite_id, n_active in enumerate(new_pars_for_run["n_active_inputs_per_dendrite"]):
        weight_change = np.mean(
            res["silent_synapses_weight"][dendrite_id * 10 : (dendrite_id + 1) * 10, :],
            axis=0,
        )
        avg_weight_changes.append(weight_change)
        all_n_active.append(n_active)

    for context_id, (color, weights, n_active) in enumerate(
        zip(["#1f78b4", "#984ea3", "#ff7f00", "#4daf4a"], avg_weight_changes, all_n_active)
    ):
        ax.plot(
            res["voltage_dends_t"],
            weights,
            color=color,
            label=f"# active: {n_active} | context {context_id + 1}",
        )

    for ii, context_id in enumerate(new_pars_for_run["all_context_ids"]):
        x_val = ii * new_pars_for_run["runtime"] / ms
        ax.axvline(x=x_val, c="k", ls="--", lw=1.0, alpha=0.4)
        ax.annotate(
            f"context {context_id + 1}",
            xy=(x_val, ax.get_ylim()[1]),
            xytext=(4, -10),  # 6 points vertical offset.
            textcoords="offset points",
            ha="left",
            va="center",
        )

    ax.set(xlabel="Time in ms", ylabel="avg silent weight")
    ax.legend()

    if show_plot:
        plt.show()


def plot_results(
    axes=None,
    neuron=None,
    target_dendrite=0,
    xmin=499,
    xmax=-1,
    label=None,
    color="#0570b0",
    spike_offset_inh=0,
    spike_offset_exc=0,
    remove_y_ticks=False,
    annotation_number=None,
    make_nmda_spikes_linear=True,
):
    res = neuron.save_dict

    if len(axes) == 3:
        ax_1, ax_2, ax_3 = axes
    else:
        ax_2, ax_3 = axes

    synapses_to_look_at = res["ff_weights_dends_w"][: len(res["ff_weights_dends_w"]) // 2]

    rec_time = res["voltage_dends_t"][:]
    rec_time = rec_time - xmin
    end_time = rec_time[xmax]

    if len(axes) == 3:
        this_color = "k"
        if spike_offset_exc != 0:
            this_color = color

        label = "excitatory spikes"

        neuron_counter = 0
        for neuron_index in range(neuron.n_inputs_ff):
            spike_times_for_neuron = (
                res["spikes_inputs_t"][np.where(res["spikes_inputs_i"] == neuron_index)[0]] - xmin
            )

            if not np.any(
                np.logical_and(spike_times_for_neuron > 0, spike_times_for_neuron < end_time)
            ):
                continue

            ax_1.vlines(
                spike_times_for_neuron,
                ymin=spike_offset_exc + neuron_counter - 0.5,
                ymax=spike_offset_exc + neuron_counter + 0.5,
                colors=this_color,
                label=label,
            )
            label = None
            neuron_counter += 1

        inhibitory_neuron_ids_that_target_first_dendrite = np.where(
            neuron.area.input_synapses[0].j == 0
        )
        spike_times_for_neuron = (
            res["spikes_inhibition_t"][np.where(res["spikes_inhibition_i"] == 0)[0]] - xmin
        )

        this_color = "r"
        if spike_offset_inh != 0:
            this_color = color

        ax_1.vlines(
            spike_times_for_neuron,
            ymin=spike_offset_inh - 2 - 0.5,
            ymax=spike_offset_inh - 2 + 0.5,
            colors=this_color,
            label="inhibitory spikes",
            lw=2,
        )

        print([0, np.max(rec_time)])
        ax_1.fill_between(
            [-500, 900],
            y1=neuron.parameters_for_run["n_active"] - 0.5,
            y2=-0.5,
            label="highly active",
            color="#bdbdbd",
        )
        ax_1.legend()

    lw = 1.4
    alpha = 1

    if len(axes) == 3:
        ax_2.plot(rec_time, res["u_plus_dends"][0], ls="--", label="u+", c="#000000")
        ax_2.axhline(
            y=neuron.parameters["theta_plus"] / mV,
            lw=1,
            ls="-",
            label=f"theta + ({neuron.parameters['theta_plus']})",
            c="#000000",
        )
        ax_2.plot(rec_time, res["u_minus_dends"][0], ls="-.", label="u-", c="#bdbdbd")
        ax_2.axhline(
            y=neuron.parameters["theta_minus"] / mV,
            lw=1,
            ls="-",
            label=f"theta - ({neuron.parameters['theta_minus']})",
            c="#bdbdbd",
        )

    else:
        lw = 0.5
        alpha = 0.5

        def smooth(y, box_pts):
            box = np.ones(box_pts) / box_pts
            y_smooth = np.convolve(y, box, mode="same")
            return y_smooth

        # ax_2.plot(rec_time, smooth(res["voltage_dends_v"][0, :], box_pts=120), color=color, lw=1)
        ax_2.plot(rec_time, res["u_plus_dends"][0], color=color, lw=1.5, label="u+")

    ax_2.plot(
        rec_time,
        res["voltage_dends_v"][0, :],
        color=color,
        lw=lw,
        alpha=alpha,
        label="voltage trace",
    )

    start_time_id = np.argmax(rec_time >= 0)

    if label is None:
        end_time_id = np.argmax(rec_time > end_time)
        this_label = "all weights of highly active"
        for tt, trace in enumerate(
            synapses_to_look_at[: neuron.parameters_for_run["n_active"], start_time_id:end_time_id]
        ):
            if tt > 0:
                this_label = None
            ax_3.plot(
                rec_time[start_time_id:end_time_id],
                trace - trace[0],
                color="#bdbdbd",
                label=this_label,
            )

        label = "avg weight of highly active"

        ax_3.plot(
            rec_time[start_time_id:end_time_id],
            np.mean(
                synapses_to_look_at[
                    : neuron.parameters_for_run["n_active"], start_time_id:end_time_id
                ]
                - synapses_to_look_at[
                    : neuron.parameters_for_run["n_active"], start_time_id : start_time_id + 1
                ],
                axis=0,
            ),
            label=label,
            color=color,
        )

        ax_3.set(xlabel="Time in ms", ylabel="weight change", xlim=[0, end_time])
    else:
        print("###", res["silent_synapses_weight"].shape)
        if show_percent_of_positive_change:
            weight_change = (
                res["silent_synapses_weight"][:10, :]
                - res["silent_synapses_weight"][:10, start_time_id : start_time_id + 1]
            ).T
            y_val = weight_change[-1, 0]
            lw = 0.8
            alpha = 0.8
            label = None
        else:
            weight_change = np.mean(
                res["silent_synapses_weight"][:10, :]  # we show the first 10 here (first context)
                - res["silent_synapses_weight"][:10, start_time_id : start_time_id + 1],
                axis=0,
            )
            y_val = weight_change[-1]
            alpha = 1
            lw = 1.2

        ax_3.plot(
            rec_time,
            weight_change,
            label=label,
            color=color,
            lw=lw,
            alpha=alpha,
        )
        ax_3.annotate(
            f"{annotation_number}",
            xy=(rec_time[-1], y_val),
            xytext=(0, 6),  # 6 points vertical offset.
            textcoords="offset points",
            ha="center",
            va="center",
        )
        ax_3.set(
            xlabel="Time in ms",
            ylabel="avg weight change\nof 'silent' synapses",
            xlim=[0, end_time],
        )

    if len(axes) == 3:
        ax_1.set(xticklabels=[], ylabel="Input neuron", xlim=[0, end_time])
        ax_2.legend()
    else:
        ax_3.set(ylim=[-2, 2])
        if make_nmda_spikes_linear:
            ax_3.set(ylim=[-4, 1])

    ax_2.set(xticklabels=[], ylabel="Dendritic voltage in mV", xlim=[0, end_time])

    ax_3.legend()

    if remove_y_ticks:
        for ax in axes:
            ax.set(yticklabels=[], ylabel="")


def run_scan_on_server(cores=80):
    params = []
    for extra_seed in range(10):
        for linear_NMDA in [True, False]:
            for all_active in [
                [ii for ii in range(0, 4)],
                [ii for ii in range(4, 8)],
                [ii for ii in range(8, 12)],
                [ii for ii in range(12, 16)],
            ]:
                params.append([None, None, None, False, linear_NMDA, extra_seed, all_active, True])

    with Pool(cores) as pool:
        _ = pool.starmap(
            weight_change_averages_for_n_active_and_inhibitory_rate,
            params,
        )


if __name__ == "__main__":
    # run_scan_on_server(cores=8)

    paper_figure_2(make_nmda_spikes_linear=False)
    paper_figure_2(make_nmda_spikes_linear=True)
