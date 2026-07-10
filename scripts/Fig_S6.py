from Fig_6 import Fig_6

if __name__ == "__main__":
    Fig_6(
        seed=927,
        n_train_additional=5,
        n_recall_auditory=1,
        n_recall_both=4,
        create_random_patch_positions=False,
        context_0_for_B=True,
        use_opposite_context=True,  # True for Fig_S6, False for Fig_6
        use_variance_in_auditory=False,
    )
