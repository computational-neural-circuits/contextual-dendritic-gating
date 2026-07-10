script_dir = fileparts(mfilename("fullpath"));
repo_root = fileparts(fileparts(fileparts(script_dir)));
data_dir = fullfile(repo_root, "results", "matlab", "Fig_3");
output_dir = fullfile(repo_root, "results", "figures");
if ~exist(output_dir, "dir")
    mkdir(output_dir)
end

% panel C
load(fullfile(data_dir, "data_Fig_3C.mat"))

max_w=max([max(weights_in_context_sorted),max(weights_sorted(1:5,13:24)),max(weights_sorted(1:60,1:400))]);

fig_3C = figure('Visible', 'off');
subplot(2,2,1)
imagesc(flipud(weights_in_context_sorted(1:50,1:50)))
colormap(flipud(gray))
clim([0 max_w])
yt = yticks;
yticklabels(flip(yt-10))
axis('square')
xlabel('RC pre neurons')
ylabel('RC post neurons')

subplot(2,2,[2,4])
imagesc(flipud(weights_sorted(1:50,1:200)'))
colormap(flipud(gray))
c=colorbar;
c.Label.String = 'w';
clim([0 max_w])
yt = yticks;
yticklabels(flip(yt-20))
xlabel('RC pre neurons')
ylabel('RC post dendrites')

subplot(2,2,3)
imagesc(flipud(weights_sorted(1:5,13:24)'))
colormap(flipud(gray))
set(gca, 'YTickLabel', []);
xlabel('RC pre neurons')
ylabel('RC post dendrites')
exportgraphics(fig_3C, fullfile(output_dir, "Fig_3C_conn_matrices.pdf"), 'ContentType', 'vector');
close(fig_3C)



% panel D
load(fullfile(data_dir, "data_Fig_3D.mat"))

fig_3D = figure('Visible', 'off');
imagesc(flipud(weights_of_context_0_sorted_for_assemblies'))
colormap(flipud(gray))
colorbar
yt = yticks;
yticklabels(flip(yt-50))
xlabel('RC pre neurons')
ylabel('RC post dendrites')
exportgraphics(fig_3D, fullfile(output_dir, "Fig_3D_conn_matrix.pdf"), 'ContentType', 'vector');
close(fig_3D)
