script_dir = fileparts(mfilename("fullpath"));
repo_root = fileparts(fileparts(fileparts(script_dir)));
data_dir = fullfile(repo_root, "results", "matlab", "Fig_5");
output_dir = fullfile(repo_root, "results", "figures");
if ~exist(output_dir, "dir")
    mkdir(output_dir)
end

load(fullfile(data_dir, "data_Fig_5BC.mat"))

%% plot the dendrite-neuron matrix

only_strong_weights=all_weights_sorted_by_context_0;
only_strong_weights(only_strong_weights<1.5)=0;
only_strong_weights(only_strong_weights>1.5)=1;


neuron_neuron=flipud(weights_of_context_0_sorted_by_context_0(1:50,1:50));
neuron_dendrite=flipud(all_weights_sorted_by_context_0(1:400,1:50));

max_w=max([max(neuron_neuron),max(neuron_dendrite)]);

fig_5B = figure('Visible', 'off');
subplot(2,2,1)
imagesc(neuron_neuron)
colormap(flipud(gray))
clim([0 max_w])
yt = yticks;
yticklabels(flip(yt-10))
axis('square')
xlabel('Pre neurons')
ylabel('Post neurons')


subplot(2,2,[2,4])
imagesc(neuron_dendrite)
colormap(flipud(gray))
c=colorbar;
c.Label.String = 'w';
clim([0 max_w])
yt = yticks;
yticklabels(flip(yt-50))
xlabel('Pre neurons')
ylabel('Post dendrites')

subplot(2,2,3)
imagesc(flipud(only_strong_weights(13:24,5:15)))
colormap(flipud(gray))
set(gca, 'YTickLabel', []);
xticks(1:5:11)
xticklabels(5:5:15)
xlabel('Pre neurons')
ylabel('Post dendrites')
exportgraphics(fig_5B, fullfile(output_dir, "Fig_5B_conn_matrices.pdf"), 'ContentType', 'vector');
close(fig_5B)

%% make the graph
neuron_idx_0_proj= [10, 41, 42, 67, 98, 125, 135, 153, 181, 184, 186, 202, 226, 234, 251, 266, 286, 313, 320, 331, 338, 353, 354, 61, 208];
neuron_idx_0_asso=[5, 34, 43, 60, 77, 104, 105, 107, 152, 160, 163, 168, 171, 262, 281, 344, 360, 365];
neuron_idx_2_proj=[15, 19, 61, 67, 125, 150, 154, 159, 200, 202, 213, 221, 234, 249, 266, 290, 305, 320, 353, 331, 313];
neuron_idx_2_asso=[5, 34, 77, 105, 107, 152, 163, 168, 171, 207, 259, 262, 281, 344, 345, 360, 364, 365, 374, 60, 43];
neuron_idx_1_proj=[51, 79, 87, 91, 138, 158, 176, 190, 253, 262, 270, 330, 347, 354, 376, 377, 381, 394, 398, 177];
neuron_idx_1_asso= [16, 17, 18, 26, 49, 58, 84, 107, 152, 178, 210, 242, 254, 299, 308, 340];


weight_D1=all_weights_not_sorted(1:6:end,:);
weight_D2=all_weights_not_sorted(2:6:end,:);
weight_D3=all_weights_not_sorted(3:6:end,:);

W_0_proj=zeros(400,400); W_0_asso=zeros(400,400);
W_1_proj=zeros(400,400); W_1_asso=zeros(400,400);
W_2_proj=zeros(400,400); W_2_asso=zeros(400,400);

W_0_proj(neuron_idx_0_proj,neuron_idx_0_proj)=weight_D1(neuron_idx_0_proj,neuron_idx_0_proj); W_0_proj(W_0_proj>0)=0.5;
W_0_asso(neuron_idx_0_asso,neuron_idx_0_asso)=weight_D1(neuron_idx_0_asso,neuron_idx_0_asso); W_0_asso(W_0_asso>0)=1.1;
W_1_proj(neuron_idx_1_proj,neuron_idx_1_proj)=weight_D3(neuron_idx_1_proj,neuron_idx_1_proj); W_1_proj(W_1_proj>0)=1.5;
W_1_asso(neuron_idx_1_asso,neuron_idx_1_asso)=weight_D3(neuron_idx_1_asso,neuron_idx_1_asso); W_1_asso(W_1_asso>0)=2;
W_2_proj(neuron_idx_2_proj,neuron_idx_2_proj)=weight_D2(neuron_idx_2_proj,neuron_idx_2_proj); W_2_proj(W_2_proj>0)=2.4;
W_2_asso(neuron_idx_2_asso,neuron_idx_2_asso)=weight_D2(neuron_idx_2_asso,neuron_idx_2_asso); W_2_asso(W_2_asso>0)=3;

W=W_0_proj+W_0_asso+W_1_proj+W_1_asso+W_2_proj+W_2_asso;

fig_5C = figure('Visible', 'off');
N_all = digraph(W');
isolated_nodes = find(indegree(N_all) + outdegree(N_all) == 0);
N_sub = rmnode(N_all,isolated_nodes);

edges_sub=N_sub.Edges;

synapses_C0_proj=find(edges_sub{:,2}==0.5);
synapses_C0_proj_add=find(edges_sub{:,2}==2.9);
indx_C0_proj=edges_sub{[synapses_C0_proj;synapses_C0_proj_add],1};
synapses_C1_proj=find(edges_sub{:,2}==1.5);
indx_C1_proj=edges_sub{synapses_C1_proj,1};
synapses_C2_proj=find(edges_sub{:,2}==2.4);
synapses_C2_proj_add=find(edges_sub{:,2}==2.9);
indx_C2_proj=edges_sub{[synapses_C2_proj;synapses_C2_proj_add],1};
synapses_C0_asso=find(edges_sub{:,2}==1.1);
synapses_C0_asso_add=find(edges_sub{:,2}==4.1);
indx_C0_asso=edges_sub{[synapses_C0_asso;synapses_C0_asso_add],1};
synapses_C1_asso=find(edges_sub{:,2}==2);
synapses_C1_asso_add=find(edges_sub{:,2}==3.1);
indx_C1_asso=edges_sub{[synapses_C1_asso;synapses_C1_asso_add],1};
synapses_C2_asso=find(edges_sub{:,2}==3);
synapses_C2_asso_add=find(edges_sub{:,2}==4.1);
indx_C2_asso=edges_sub{[synapses_C2_asso;synapses_C2_asso_add],1};

line_width=0.5;

p=plot(N_sub,'NodeColor',[0 0 0],'Marker','^','NodeLabel',{},'ShowArrows','off');
highlight(p,indx_C0_proj(:,1),indx_C0_proj(:,2),'EdgeColor','#82c77f','LineWidth',line_width)
highlight(p,indx_C1_proj(:,1),indx_C1_proj(:,2),'EdgeColor','#beaed4','LineWidth',line_width)
highlight(p,indx_C2_proj(:,1),indx_C2_proj(:,2),'EdgeColor','#fcc085','LineWidth',line_width)
highlight(p,indx_C0_asso(:,1),indx_C0_asso(:,2),'EdgeColor','#056a38','LineWidth',line_width)
highlight(p,indx_C1_asso(:,1),indx_C1_asso(:,2),'EdgeColor','#67338f','LineWidth',line_width)
highlight(p,indx_C2_asso(:,1),indx_C2_asso(:,2),'EdgeColor','#b37129','LineWidth',line_width)

exportgraphics(fig_5C, fullfile(output_dir, "Fig_5C_assembly_graph.pdf"), 'ContentType', 'vector');
close(fig_5C)

%saveas(gcf,'Fig_4_graph2.pdf')
