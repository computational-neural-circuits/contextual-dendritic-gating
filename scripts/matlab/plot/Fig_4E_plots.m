% Code for Figure 4 panel E

script_dir = fileparts(mfilename("fullpath"));
repo_root = fileparts(fileparts(fileparts(script_dir)));
data_dir = fullfile(repo_root, "results", "matlab", "Fig_4", "data_Fig_4E");
output_dir = fullfile(repo_root, "results", "figures");
if ~exist(output_dir, "dir")
    mkdir(output_dir)
end
load_weight_matrix = @(filename) importdata(fullfile(data_dir, filename));

seeds=[24,32,34,52,63,78,89,321,485,612,625,673,733,789,932,995,2062,3523,7387];
assembly_size_per_seed=[23,20,18,22,20,29,22,22,25,19,23,24,22,22,17,23,21,20,16];

seed_12=[24,612];

%% calculate how many neurons loose their connections
strong_weights=2.5; %define the assembly weights (max weight is 5)
strong_LTD_thresh=2.5; % min reduction to consider strong LTD
nr_imprints=[1,2,4,8,12];

frac_of_interference_same=ones(length(seeds),length(nr_imprints)).*NaN;
frac_of_interference_diff=ones(length(seeds),length(nr_imprints)).*NaN;

for jj2=1:length(seeds)

size_assembly_seed=assembly_size_per_seed(jj2);

weights_of_imprint_0 = load_weight_matrix(['seed_' num2str(seeds(jj2)) '_0_weights_of_context_0_sorted_for_assemblies.txt']);
dummy_assembly_imprint0=weights_of_imprint_0(1:size_assembly_seed,1:size_assembly_seed);
idx_strong_weight=find(dummy_assembly_imprint0>=strong_weights);

    for jj1=1:length(nr_imprints)

        if ismember(seeds(jj2),seed_12)==1 && jj1==length(nr_imprints)
            weights_of_imprint_jj1_same = load_weight_matrix(['seed_' num2str(seeds(jj2)) '_' num2str(nr_imprints(jj1)) '_context_0_weights_of_context_0_sorted_for_assemblies.txt']);
            dummy_assembly_imprint_jj1_same=weights_of_imprint_jj1_same(1:size_assembly_seed,1:size_assembly_seed);
            idx_strong_LTD_imprint_jj1_same=find(abs(dummy_assembly_imprint_jj1_same(idx_strong_weight)-dummy_assembly_imprint0(idx_strong_weight))>=strong_LTD_thresh);
            frac_of_interference_same(jj2,jj1)=numel(idx_strong_LTD_imprint_jj1_same)/numel(idx_strong_weight);

            weights_of_imprint_jj1_diff = load_weight_matrix(['seed_' num2str(seeds(jj2)) '_' num2str(nr_imprints(jj1)) '_context_3_weights_of_context_0_sorted_for_assemblies.txt']);
            dummy_assembly_imprint_jj1_diff=weights_of_imprint_jj1_diff(1:size_assembly_seed,1:size_assembly_seed);
            idx_strong_LTD_imprint_jj1_diff=find(abs(dummy_assembly_imprint_jj1_diff(idx_strong_weight)-dummy_assembly_imprint0(idx_strong_weight))>=strong_LTD_thresh);
            frac_of_interference_diff(jj2,jj1)=numel(idx_strong_LTD_imprint_jj1_diff)/numel(idx_strong_weight);


        elseif jj1<length(nr_imprints)
           weights_of_imprint_jj1_same = load_weight_matrix(['seed_' num2str(seeds(jj2)) '_' num2str(nr_imprints(jj1)) '_context_0_weights_of_context_0_sorted_for_assemblies.txt']);
            dummy_assembly_imprint_jj1_same=weights_of_imprint_jj1_same(1:size_assembly_seed,1:size_assembly_seed);
            idx_strong_LTD_imprint_jj1_same=find(abs(dummy_assembly_imprint_jj1_same(idx_strong_weight)-dummy_assembly_imprint0(idx_strong_weight))>=strong_LTD_thresh);
            frac_of_interference_same(jj2,jj1)=numel(idx_strong_LTD_imprint_jj1_same)/numel(idx_strong_weight);

            if jj1<=3 %context 1
                weights_of_imprint_jj1_diff = load_weight_matrix(['seed_' num2str(seeds(jj2)) '_' num2str(nr_imprints(jj1)) '_context_1_weights_of_context_0_sorted_for_assemblies.txt']);
            elseif jj2==4
                weights_of_imprint_jj1_diff = load_weight_matrix(['seed_' num2str(seeds(jj2)) '_' num2str(nr_imprints(jj1)) '_context_2_weights_of_context_0_sorted_for_assemblies.txt']);
            end
            dummy_assembly_imprint_jj1_diff=weights_of_imprint_jj1_diff(1:size_assembly_seed,1:size_assembly_seed);
            idx_strong_LTD_imprint_jj1_diff=find(abs(dummy_assembly_imprint_jj1_diff(idx_strong_weight)-dummy_assembly_imprint0(idx_strong_weight))>=strong_LTD_thresh);
            frac_of_interference_diff(jj2,jj1)=numel(idx_strong_LTD_imprint_jj1_diff)/numel(idx_strong_weight);

        end
    end
end

%%
fig_4E_forgetting = figure('Visible', 'off');
hold on
plot(nr_imprints,mean(frac_of_interference_diff,'omitnan'),'k')
plot(nr_imprints,mean(frac_of_interference_same,'omitnan'),'r')
hold off
xlim([1,12])
ylim([0 1])
xlabel('# of overlaps')
ylabel('Fraction of forgetting')
exportgraphics(fig_4E_forgetting, fullfile(output_dir, "Fig_4E_forgetting_fraction.pdf"), 'ContentType', 'vector');
close(fig_4E_forgetting)

%% plot connectivity matrices
weights_of_imprint_0 = load_weight_matrix('seed_24_0_weights_of_context_0_sorted_for_assemblies.txt');
weights_of_imprint_1_same = load_weight_matrix('seed_24_1_context_0_weights_of_context_0_sorted_for_assemblies.txt');
weights_of_imprint_2_same = load_weight_matrix('seed_24_2_context_0_weights_of_context_0_sorted_for_assemblies.txt');
weights_of_imprint_4_same = load_weight_matrix('seed_24_4_context_0_weights_of_context_0_sorted_for_assemblies.txt');
weights_of_imprint_8_same = load_weight_matrix('seed_24_8_context_0_weights_of_context_0_sorted_for_assemblies.txt');
weights_of_imprint_12_same = load_weight_matrix('seed_24_12_context_0_weights_of_context_0_sorted_for_assemblies.txt');

weights_of_imprint_1_diff = load_weight_matrix('seed_24_1_context_1_weights_of_context_0_sorted_for_assemblies.txt');
weights_of_imprint_2_diff = load_weight_matrix('seed_24_2_context_1_weights_of_context_0_sorted_for_assemblies.txt');
weights_of_imprint_4_diff = load_weight_matrix('seed_24_4_context_1_weights_of_context_0_sorted_for_assemblies.txt');
weights_of_imprint_8_diff = load_weight_matrix('seed_24_8_context_2_weights_of_context_0_sorted_for_assemblies.txt');
weights_of_imprint_12_diff = load_weight_matrix('seed_24_12_context_3_weights_of_context_0_sorted_for_assemblies.txt');


nr_neurons_plot=30;

fig_4E_same_context = figure('Visible', 'off');
subplot(3,3,1)
imagesc(weights_of_imprint_0(1:nr_neurons_plot,1:nr_neurons_plot));
colormap(flipud(gray))
set(gca, 'YDir', 'normal');
colorbar
clim([0 5])
axis('square')

subplot(3,3,2)
imagesc(weights_of_imprint_1_same(1:nr_neurons_plot,1:nr_neurons_plot));
set(gca, 'YDir', 'normal');
colorbar
clim([0 5])
axis('square')

subplot(3,3,3)
imagesc(weights_of_imprint_2_same(1:nr_neurons_plot,1:nr_neurons_plot));
set(gca, 'YDir', 'normal');
colorbar
clim([0 5])
axis('square')

subplot(3,3,4)
imagesc(weights_of_imprint_4_same(1:nr_neurons_plot,1:nr_neurons_plot));
set(gca, 'YDir', 'normal');
colorbar
clim([0 5])
axis('square')

subplot(3,3,5)
imagesc(weights_of_imprint_8_same(1:nr_neurons_plot,1:nr_neurons_plot));
set(gca, 'YDir', 'normal');
colorbar
clim([0 5])
axis('square')

subplot(3,3,6)
imagesc(weights_of_imprint_12_same(1:nr_neurons_plot,1:nr_neurons_plot));
set(gca, 'YDir', 'normal');
colorbar
clim([0 5])
axis('square')
exportgraphics(fig_4E_same_context, fullfile(output_dir, "Fig_4E_conn_matrices_same_context.pdf"), 'ContentType', 'vector');
close(fig_4E_same_context)

fig_4E_diff_context = figure('Visible', 'off');
subplot(3,3,1)
imagesc(weights_of_imprint_0(1:nr_neurons_plot,1:nr_neurons_plot));
colormap(flipud(gray))
set(gca, 'YDir', 'normal');
colorbar
clim([0 5])
axis('square')

subplot(3,3,2)
imagesc(weights_of_imprint_1_diff(1:nr_neurons_plot,1:nr_neurons_plot));
set(gca, 'YDir', 'normal');
colorbar
clim([0 5])
axis('square')

subplot(3,3,3)
imagesc(weights_of_imprint_2_diff(1:nr_neurons_plot,1:nr_neurons_plot));
set(gca, 'YDir', 'normal');
colorbar
clim([0 5])
axis('square')

subplot(3,3,4)
imagesc(weights_of_imprint_4_diff(1:nr_neurons_plot,1:nr_neurons_plot));
set(gca, 'YDir', 'normal');
colorbar
clim([0 5])
axis('square')

subplot(3,3,5)
imagesc(weights_of_imprint_8_diff(1:nr_neurons_plot,1:nr_neurons_plot));
set(gca, 'YDir', 'normal');
colorbar
clim([0 5])
axis('square')

subplot(3,3,6)
imagesc(weights_of_imprint_12_diff(1:nr_neurons_plot,1:nr_neurons_plot));
set(gca, 'YDir', 'normal');
colorbar
clim([0 5])
axis('square')
exportgraphics(fig_4E_diff_context, fullfile(output_dir, "Fig_4E_conn_matrices_diff_context.pdf"), 'ContentType', 'vector');
close(fig_4E_diff_context)
