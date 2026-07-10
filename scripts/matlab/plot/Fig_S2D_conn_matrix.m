script_dir = fileparts(mfilename("fullpath"));
repo_root = fileparts(fileparts(fileparts(script_dir)));
data_dir = fullfile(repo_root, "results", "matlab", "Fig_S2");
output_dir = fullfile(repo_root, "results", "figures");
if ~exist(output_dir, "dir")
    mkdir(output_dir)
end

weights = load(fullfile(data_dir, "data_Fig_S2D.txt"));

fig = figure('Visible', 'off');
imagesc(flipud(weights'))
colormap(flipud(gray))
colorbar
yt = yticks;
yticklabels(flip(yt-50))
xlabel('RC pre neurons')
ylabel('RC post dendrites')
exportgraphics(fig, fullfile(output_dir, "Fig_S2D_conn_matrix.pdf"), 'ContentType', 'vector');
close(fig)
