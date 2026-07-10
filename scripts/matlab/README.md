# MATLAB figure scripts

This directory contains MATLAB code for selected
main and supplementary figure panels.

- `plot/` contains scripts that generate plots from exported MATLAB or text
  data.
- `simulate/` contains the standalone MATLAB simulations for Figures S4 and
  S5.

The corresponding input files are organized under `results/matlab/` by figure.
They are local data assets and remain excluded from Git.

The plotting scripts locate their inputs relative to their own file locations,
so they can be run from any MATLAB working directory. Plotting and simulation
scripts create figures invisibly, export vector PDFs to `results/figures/`,
and close figure handles after export.

Known non-path issues remain intentionally unchanged:

- `Fig_S5_A2A3A4_simulation.m` requires an undefined `nr_total_runs` value.
- `Fig_4E_plots.m` contains an existing loop-condition issue for one data
  branch.
