script_dir = fileparts(mfilename("fullpath"));
repo_root = fileparts(fileparts(fileparts(script_dir)));
data_dir = fullfile(repo_root, "results", "matlab", "Fig_S5");
output_dir = fullfile(repo_root, "results", "figures");
if ~exist(output_dir, "dir")
    mkdir(output_dir)
end

load(fullfile(data_dir, "data_Fig_S5_A2A3A4.mat"))

fig = figure('Visible', 'off');
subplot(2,3,1)
hold on
plot(0:0.1:5,([0:0.1:5]-theta_SST)./tau_w_SST,'k')
yline(0,'k--')
hold off
xlabel('r_Iv(Hz)')
ylabel('\Delta w_{IC} (a.u.)')
title('C-to-I plasticity rule')

subplot(2,3,2)
hold on
histogram(squeeze(sum(W_CtoI_0_store>0,2)),'FaceColor', 'y','Normalization','probability')
histogram(squeeze(sum(W_CtoI_final_store>0,2)),'FaceColor', 'k','Normalization','probability')
hold off
legend('before learning','after learning')
xlabel('Number of contexts per I')
ylabel('Fraction')

subplot(2,3,3)
hold on
plot(overlap_fraction,1-fliplr(mean(mean_plast_run_case2,1)),'k')
plot(overlap_fraction,1-fliplr(mean(mean_plast_run_case1,1)),'b')
hold off
legend('before learning','after learning')
xlabel('Fract. of stim. overlap')
ylabel('Fract. of forgetting')
ylim([-0.1,1])
axis("square")

load(fullfile(data_dir, "data_Fig_S5_B2B3B4.mat"))

subplot(2,3,4)
hold on
plot(0:0.1:15,(theta_D-[0:0.1:15])./tau_w_D,'k')
yline(0,'k--')
hold off
xlabel('r_I')
ylabel('\Delta w_{DI} (a.u.)')
title('I-to-D plasticity rule')

dummy_init=W_ItoD_mask0*W_CtoI;
selectivity_per_dendrite_init=sum(dummy_init>0,2);

dummy_final=W_ItoD_final*W_CtoI;
selectivity_per_dendrite_final=sum(dummy_final>0,2);

subplot(2,3,5)
hold on
histogram(selectivity_per_dendrite_init,'FaceColor', 'y','Normalization','probability')
histogram(selectivity_per_dendrite_final,'FaceColor', 'k','Normalization','probability')
hold off
legend('before learning','after learning')
xlabel('Number of contextual inputs per D')
ylabel('Fraction')

subplot(2,3,6)
hold on
plot(overlap_fraction,1-fliplr(mean(gating_selectivity_ratio_ACROSS_stimuli_case2,1,'omitmissing')),'k')
plot(overlap_fraction,1-fliplr(mean(gating_selectivity_ratio_ACROSS_stimuli_case1,1,'omitmissing')),'b')
hold off
legend('before learning','after learning')
xlabel('Fract. of input overlap')
ylabel('Fract. of forgetting')
ylim([-0.1,1])
axis("square")
exportgraphics(fig, fullfile(output_dir, "Fig_S5_plasticity_rules_and_forgetting.pdf"), 'ContentType', 'vector');
close(fig)
