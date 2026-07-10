# Publication figures

This directory contains versioned final figure-pdfs and MATLAB-generated panel
PDFs. Python diagnostics and other intermediates belong in `results/Fig_*` and
remain local.

| PDF | Generating script |
| --- | --- |
| `Fig_2.pdf` | `scripts/Fig_2.py` |
| `Fig_3.pdf` | `scripts/Fig_3.py` |
| `Fig_3C_conn_matrices.pdf`, `Fig_3D_conn_matrix.pdf` | `scripts/matlab/plot/Fig_3CD_conn_matrix.m` |
| `Fig_4.pdf` | `scripts/Fig_4.py` |
| `Fig_4E_forgetting_fraction.pdf`, `Fig_4E_conn_matrices_same_context.pdf`, `Fig_4E_conn_matrices_diff_context.pdf` | `scripts/matlab/plot/Fig_4E_plots.m` |
| `Fig_5.pdf` | `scripts/Fig_5.py` |
| `Fig_5B_conn_matrices.pdf`, `Fig_5C_assembly_graph.pdf` | `scripts/matlab/plot/Fig_5BC_conn_matrix_and_graph.m` |
| `Fig_6B_concept_area_graph.pdf` | `scripts/matlab/plot/Fig_6B_graph.m` |
| `Fig_7.pdf` | `scripts/Fig_7.py` |
| `Fig_8.pdf`, `Fig_8_full.pdf` | `scripts/Fig_8.py` |
| `Fig_S1.pdf` | `scripts/Fig_S1.py` |
| `Fig_S2.pdf` | `scripts/Fig_S2.py` |
| `Fig_S2D_conn_matrix.pdf` | `scripts/matlab/plot/Fig_S2D_conn_matrix.m` |
| `Fig_S3.pdf` | `scripts/Fig_S3.py` |
| `Fig_S4_A1A3_1to1_connectivity.pdf`, `Fig_S4_B2B3_parameter_sweep.pdf`, `Fig_S4_B2B3_chosen_points.pdf`, `Fig_S4_B2B3_forgetting_vs_overlap.pdf`, `Fig_S4_C2C3_fixed_ItoD.pdf`, `Fig_S4_D2D3_fixed_CtoI.pdf` | `scripts/matlab/plot/Fig_S4_all_plots.m` |
| `Fig_S5_plasticity_rules_and_forgetting.pdf` | `scripts/matlab/plot/Fig_S5_all_plots.m` |
| `Fig_S7.pdf` | `scripts/Fig_8.py` |

To regenerate figures, download the optional data from
https://doi.org/10.5281/zenodo.21299329. Extract result data into the matching
`results/` subdirectories and stored-network ZIP archives into
`stored_networks/`, preserving figure subdirectories. Then run the
corresponding scripts from `scripts/`. MATLAB scripts write their descriptive
panel PDFs directly to this directory.

## Python figure scripts

- `Fig_2.py` — At the single-cell level, shows how inhibitory context gates
  dendrite-specific excitatory plasticity. A multi-compartment neuron with
  NMDA-nonlinear dendrites switches between LTP, LTD, and no plasticity
  depending on the balance of active excitatory inputs and inhibition, and
  context-specific inhibition gates plasticity 'on' or 'off' per dendrite.
- `Fig_S1.py` — Control for the NMDA nonlinearity. With linear NMDA synapses,
  switching between plasticity regimes requires much larger changes in
  excitatory and inhibitory input than with the nonlinear NMDA synapses.
- `Fig_3.py` — Formation of dendrite-specific assemblies in a recurrent network
  of 400 neurons within a single context. Multiple assemblies are imprinted
  sequentially from feedforward input, and the resulting assemblies support
  reliable recall and pattern completion from partial or weak inputs.
- `Fig_S2.py` — Shows that assemblies become driven by both feedforward and
  recurrent input after learning, and compares the full model to a reduced
  single-dendrite control that also forms assemblies, recalls, and completes
  patterns.
- `Fig_S3.py` — Illustrates how feedforward and recurrent inhibition, together
  with dendrite-specific normalization, keep assembly size roughly constant,
  analogous to a k-winner-take-all mechanism.
- `Fig_4.py` — Learning assemblies with overlapping inputs. Imprinting
  overlapping assemblies in the same context degrades the original assembly
  (forgetting), while switching contexts preserves it; forgetting is quantified
  as depression of within-assembly synapses.
- `Fig_5.py` — Multi-area computations. Dendrite-specific gating supports
  learning projections and associations across areas in different contexts,
  allowing overlapping assemblies to be combined without forgetting and to
  perform pattern completion.
- `Fig_6.py` — A visual-auditory association task separating ambiguous stimuli
  ("0"/"O", "1"/"l"). Using EMNIST images filtered by Gabor patches and paired
  auditory input, distinct contexts (letters vs. numbers) let a downstream
  concept area separate visually similar stimuli.
- `Fig_S6.py` — Task variants of Figure 6. Tests auditory-only presentation and
  shape-based contexts, showing that reliable separation of visually similar
  stimuli requires distinct contexts for those stimuli.
- `Fig_7.py` — Recall and pattern completion across a hierarchy of areas
  (X to Y to Z). Silencing part of the assembly in area Y still allows reliable
  recall and pattern completion in area Z, reflecting a chain of recurrent
  amplification.
- `Fig_8.py` — Compares three associative learning strategies: sequential
  projections, simultaneous association, and sequential association on top of
  existing assemblies. Each strategy is characterized by assembly-identity
  overlap, input-synapse distributions, and pattern completion. It also
  generates `Fig_S7.pdf`, which shows that learning an association on top of an
  existing projection preserves most neurons of the original assembly.

## MATLAB figure scripts

These scripts live in `scripts/matlab/` and export their panel PDFs directly to
this directory.

- `Fig_3CD_conn_matrix.m` — Connectivity matrices for Figure 3C and 3D, showing
  the within-assembly weight structure and the weights of multiple assemblies
  imprinted in a single context.
- `Fig_4E_plots.m` — Figure 4E: the fraction of forgetting as a function of the
  number of overlapping assemblies, and connectivity matrices for imprinting in
  the same versus a different context.
- `Fig_5BC_conn_matrix_and_graph.m` — Figure 5B and 5C: connectivity matrices
  and the assembly graph illustrating projections and associations across areas.
- `Fig_6B_graph.m` — Figure 6B: the concept-area assembly graph, showing how
  letters and numbers are separated into distinct assemblies.
- `Fig_S2D_conn_matrix.m` — Figure S2D: the single-dendrite weight matrix for
  the reduced control model.
- `Fig_S4_simulation.m` and `Fig_S4_all_plots.m` — Figure S4: a reduced
  statistical model that relaxes the one-to-one gating assumption, sweeping
  connectivity parameters to measure assembly size and forgetting.
- `Fig_S5_A2A3A4_simulation.m`, `Fig_S5_B2B3B4_simulation.m`, and
  `Fig_S5_all_plots.m` — Figure S5: winner-take-all inhibitory plasticity from
  context to inhibitory neurons and from inhibitory neurons to dendrites, which
  supports the emergence of the one-to-one gating structure.
