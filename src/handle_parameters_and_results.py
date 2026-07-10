from brian2.units import *
import numpy as np
import h5py
import hashlib
import re
import json
import os
from os.path import abspath, dirname, join
import time
import pickle
import uuid


class HandleParametersAndResults:
    def __init__(
        self,
        parameter_file_name,
        save_file_name,
        parameter_dict={},
        parameters_for_run={},
        equation_file_name="equations",
        load_from_key=None,
        save_parameters=True,
        new_file_to_save_to=None,
        rerun=False,
        **kwargs,
    ):
        self.return_closest_keys = False
        self.equation_file_name = equation_file_name
        self.parameter_file_name = parameter_file_name

        self.rerun = rerun

        self.save_file_name = save_file_name
        self.new_file_to_save_to = new_file_to_save_to

        self.results_folder_name = "sim_files"
        self.parameters_folder_name = "model_specs"
        self.equations_folder_name = "model_specs"
        self.parameters = self.load_parameters()
        self.equations = self.load_equations()

        self.parameters.update(parameter_dict)

        final_pars_for_run = {}
        for key, val in parameters_for_run.items():
            if key in self.parameters:
                self.parameters[key] = val
            else:
                final_pars_for_run[key] = val

        self.parameters_for_run = final_pars_for_run

        self.save_dict = {}

        self.create_network = True
        self.only_load_results = False
        self.save_parameters = save_parameters

        self.load_from_key = load_from_key

        if "only_load_results" in kwargs:
            if kwargs["only_load_results"]:
                self.only_load_results = True
                self.check_for_results()
                if self.save_dict:
                    self.create_network = False

        self.figure_name = None
        if "figure_name" in kwargs:
            self.figure_name = kwargs["figure_name"]

    def show_all_saved_dictionaries(
        self, extra_key=None, show_only_closest_results=True, print_results=True
    ):
        if show_only_closest_results:
            closest_results_groups = self.show_all_saved_dictionaries(
                extra_key=None, show_only_closest_results=False, print_results=False
            )

            for group_name in closest_results_groups:
                self.show_all_saved_dictionaries(
                    extra_key=group_name,
                    print_results=True,
                    show_only_closest_results=False,
                )

            return

        save_file_path = self.get_path_to_save_file_name()

        current_paramter_dict = self.get_full_parameter_dict()
        current_dict = {
            **current_paramter_dict["network_parameters"],
            **current_paramter_dict["run_parameters"],
            **self.equations,
        }

        # n_deviations
        least_deviations = 100
        closest_results_groups = []

        if os.path.exists(save_file_path):
            with h5py.File(save_file_path, "r") as hf:
                for group_name in hf:
                    if extra_key is not None:
                        if extra_key != group_name:
                            continue
                    if print_results:
                        print(f"Comparing stored result group: {group_name}")
                        print("Parameter differences:")
                    n_deviations = 0
                    for key, val in hf[group_name].attrs.items():
                        try:
                            try:
                                if np.any(current_dict[key] != val):
                                    n_deviations += 1
                                    if print_results:
                                        print(
                                            f"Difference for {key}: {val}/{current_dict[key]}"
                                        )
                            except ValueError:
                                n_deviations += 1
                                if print_results:
                                    print(
                                        f"Value error for {key}: {val}/{current_dict[key]}"
                                    )

                        except KeyError:
                            n_deviations += 1
                            if print_results:
                                print(
                                    f"Stored parameter missing from current configuration: {key}"
                                )

                    for key in current_dict:
                        if key not in hf[group_name].attrs.keys():
                            n_deviations += 1
                            if print_results:
                                print(
                                    f"Current parameter missing from stored results: {key}"
                                )
                    if print_results:
                        print("Finished comparing stored result group.")

                    if n_deviations < least_deviations:
                        closest_results_groups = [group_name]
                        least_deviations = n_deviations
                    elif n_deviations == least_deviations:
                        closest_results_groups.append(group_name)

        return closest_results_groups

    def get_full_parameter_dict(self):
        return {
            "network_parameters": self.remove_units(self.parameters),
            "run_parameters": self.remove_units(self.parameters_for_run),
        }

    def get_path_to_save_file_name(self, save_to_new_file=False):
        save_file_name = self.save_file_name
        if self.new_file_to_save_to is not None and save_to_new_file:
            save_file_name = self.new_file_to_save_to

        path = abspath(
            join(
                dirname(__file__),
                "..",
                "results",
                self.results_folder_name,
                f"{save_file_name}.h5",
            )
        )

        # check if folder exist
        path_to_results_folder = join(dirname(__file__), "..", "results")
        exists = os.path.exists(path_to_results_folder)
        if not exists:
            os.makedirs(path_to_results_folder)

        path_to_specific_results_folder = join(
            dirname(__file__), "..", "results", self.results_folder_name
        )
        exists = os.path.exists(path_to_specific_results_folder)
        if not exists:
            os.makedirs(path_to_specific_results_folder)

        return path

    def check_for_results(
        self, key=None, addon=None, ignore_all_keys_with_keywords=None
    ):
        if self.rerun and not self.only_load_results:
            self.save_dict = {}
            return self.save_dict

        save_file_path = self.get_path_to_save_file_name()

        self.save_dict = {}
        if key is None:
            key = self.get_unique_paramter_and_equation_key(
                ignore_all_keys_with_keywords=ignore_all_keys_with_keywords
            )

            if addon is not None:
                # this is used for partial results
                key += f"_{addon}"

        if self.load_from_key is not None:
            key = self.load_from_key

        if os.path.exists(save_file_path):
            for mm in range(5):
                if mm == 99:
                    raise ValueError("REACHED FILE OPENING LIMIT")
                try:
                    with h5py.File(save_file_path, "a") as hf:
                        for group_name in hf:
                            if group_name == key:
                                for name in hf[group_name]:
                                    item = hf[group_name][name]
                                    self.save_dict[name] = item[()]

                    break
                except (
                    BlockingIOError,
                    PermissionError,
                    RuntimeError,
                    KeyError,
                    OSError,
                ) as e:
                    random_retry_time = 4 + 6 * np.random.rand()
                    print(
                        f"Result load retry after {e}; waiting {random_retry_time} seconds."
                    )
                    time.sleep(random_retry_time)

        if self.save_dict and self.new_file_to_save_to is not None:
            self.save_results(
                save_to_new_file=True,
                ignore_all_keys_with_keywords=ignore_all_keys_with_keywords,
            )

        if self.save_dict:
            print(f"Loaded result key {key} from {save_file_path}.")
        return self.save_dict

    def save_results(self, ignore_all_keys_with_keywords=None, save_to_new_file=False):
        group_name = self.get_unique_paramter_and_equation_key(
            ignore_all_keys_with_keywords=ignore_all_keys_with_keywords
        )
        par_dict = self.get_full_parameter_dict()

        for mm in range(50):
            # we do 50 tries to save the data
            try:
                with h5py.File(
                    self.get_path_to_save_file_name(save_to_new_file=save_to_new_file),
                    "a",
                ) as hf:
                    if group_name in hf:
                        # need to update
                        group = hf.get(group_name)
                    else:
                        # need to create
                        group = hf.create_group(group_name)

                    if self.save_parameters:
                        group.attrs.update(
                            {
                                **par_dict["network_parameters"],
                                **par_dict["run_parameters"],
                                **self.equations,
                            }
                        )
                    for result_name, res in self.save_dict.items():
                        if result_name in group:
                            del group[result_name]

                        _ = group.create_dataset(result_name, data=res)
                break
            except (
                BlockingIOError,
                PermissionError,
                RuntimeError,
                KeyError,
                OSError,
            ) as e:
                if mm == 49:
                    raise ValueError("REACHED FILE SAVING LIMIT")
                else:
                    random_retry_time = 4 + 6 * np.random.rand()
                    print(
                        f"Result save retry after {e}; waiting {random_retry_time} seconds."
                    )
                    time.sleep(random_retry_time)

    def get_unique_paramter_and_equation_key(
        self, key_length=8, ignore_all_keys_with_keywords=None
    ):
        dictionary = self.get_full_parameter_dict()

        comp_dict = {
            **dictionary["network_parameters"],
            **dictionary["run_parameters"],
            **self.equations,
        }

        if ignore_all_keys_with_keywords is not None:
            dictionary = {}
            for key in comp_dict:
                found_keyword = False
                for keyword in ignore_all_keys_with_keywords:
                    if keyword in key:
                        found_keyword = True
                        break
                if not found_keyword:
                    dictionary[key] = comp_dict[key]
        else:
            dictionary = comp_dict

        unique_dict_key = hashlib.sha1(
            json.dumps(dictionary, sort_keys=True).encode()
        ).hexdigest()[:key_length]
        return unique_dict_key

    def remove_units(self, dictionary):
        new_dict = {}
        for key, val in dictionary.items():
            try:
                base_unit = get_unit(val.dim)
                unitless_val = val / base_unit
            except AttributeError:
                unitless_val = val

            new_dict[key] = unitless_val

        return new_dict

    def get_path_to_stored_networks(self, file_name):
        if self.figure_name is None:
            path = abspath(
                join(dirname(__file__), "..", "stored_networks", f"{file_name}")
            )
        else:
            path = abspath(
                join(
                    dirname(__file__),
                    "..",
                    "stored_networks",
                    f"{self.figure_name}",
                    f"{file_name}",
                )
            )
        return path

    def store_network(self, filename):
        tmp = f"{filename}.tmp_{os.getpid()}_{uuid.uuid4().hex}"
        try:
            self.network.store(filename=tmp)
            os.replace(tmp, filename)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    def restore_network(self, filename):
        try:
            self.network.restore(filename=filename)
            return True
        except (EOFError, pickle.UnpicklingError, OSError) as e:
            print(f"Stored network {filename} is unreadable ({e}).")
            return False

    def get_path(self, folder_name, file_name, file_ending="txt"):
        path = abspath(
            join(dirname(__file__), folder_name, f"{file_name}.{file_ending}")
        )
        return path

    def load_parameters(self):
        skipchars = ["#"]
        with open(
            self.get_path(self.parameters_folder_name, self.parameter_file_name), "r"
        ) as f:
            params = dict()
            for line in f:
                line = line.strip()
                if not line or line[0] in skipchars:
                    continue

                for skipchar in skipchars:
                    line = line.split(skipchar)[0]

                p = line.split("=")
                params[p[0].strip()] = eval(p[1].strip())
        return params

    def load_equations(self):
        eqs = {}
        name = ""
        with open(
            self.get_path(self.equations_folder_name, self.equation_file_name), "r"
        ) as f:
            for line in f:
                line = line.split("#")[0].strip()

                if line:
                    match = re.match(r"\[\s*(.*?)\s*\]", line)
                    if match is not None:
                        name = match.groups(1)[0]
                        eqs[name] = ""
                    elif name:
                        eqs[name] += line + "\n"
                    else:
                        print(
                            "Omitting text before the first section in the equations file."
                        )
        return eqs
