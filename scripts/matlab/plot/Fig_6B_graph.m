script_dir = fileparts(mfilename("fullpath"));
repo_root = fileparts(fileparts(fileparts(script_dir)));
data_dir = fullfile(repo_root, "results", "matlab", "Fig_6");
output_dir = fullfile(repo_root, "results", "figures");
if ~exist(output_dir, "dir")
    mkdir(output_dir)
end

load(fullfile(data_dir, "data_Fig_6B.mat"))

% seed 672, concept area
neuron_idx_ZERO=[8, 31, 51, 74, 77, 82, 85, 124, 162, 178, 187, 192, 235, 288, 302, 337, 339, 392, 395, 212, 293];
neuron_idx_O=[27, 48, 106, 118, 152, 158, 161, 202, 228, 258, 280, 286, 312, 363, 380, 391, 308, 157];
neuron_idx_l=[23, 44, 53, 58, 71, 81, 136, 171, 172, 177, 232, 274, 351, 355, 293];
neuron_idx_ONE=[35, 128, 147, 150, 167, 172, 174, 179, 196, 197, 200, 220, 300, 353, 360, 386, 397, 91, 137, 99];

% different W used to to highlight connections in different colors
W_ZERO=zeros(400,400); W_ONE=zeros(400,400); W_O=zeros(400,400); W_l=zeros(400,400);

W_ZERO(neuron_idx_ZERO,neuron_idx_ZERO)=weights_area_C_of_context_0(neuron_idx_ZERO,neuron_idx_ZERO); W_ZERO(W_ZERO>0)=0.5;
W_ONE(neuron_idx_ONE,neuron_idx_ONE)=weights_area_C_of_context_0(neuron_idx_ONE,neuron_idx_ONE); W_ONE(W_ONE>0)=1;
W_O(neuron_idx_O,neuron_idx_O)=weights_area_C_of_context_1(neuron_idx_O,neuron_idx_O); W_O(W_O>0)=1.5;
W_l(neuron_idx_l,neuron_idx_l)=weights_area_C_of_context_1(neuron_idx_l,neuron_idx_l); W_l(W_l>0)=2;

W=W_ZERO+W_ONE+W_O+W_l;

% make graph with nodes that are part of an assembly
N_all=digraph(W');
isolated_nodes = find(indegree(N_all) + outdegree(N_all) == 0);
N_sub = rmnode(N_all,isolated_nodes);

edges_sub=N_sub.Edges;

synapses_ZERO=find(edges_sub{:,2}==0.5);
syn_indx_ZERO=edges_sub{synapses_ZERO,1};
synapses_ONE=find(edges_sub{:,2}==1);
syn_indx_ONE=edges_sub{synapses_ONE,1};
synapses_O=find(edges_sub{:,2}==1.5);
syn_indx_O=edges_sub{synapses_O,1};
synapses_l=find(edges_sub{:,2}==2);
syn_indx_l=edges_sub{synapses_l,1};

fig = figure('Visible', 'off');
subplot(3,1,[1,2])
p=plot(N_sub,'NodeColor',[0 0 0],'Marker','^','NodeLabel',{},'ShowArrows','off');
highlight(p,syn_indx_ZERO(:,1),syn_indx_ZERO(:,2),'EdgeColor','#82c77f')
highlight(p,syn_indx_ONE(:,1),syn_indx_ONE(:,2),'EdgeColor','#006838')
highlight(p,syn_indx_O(:,1),syn_indx_O(:,2),'EdgeColor','#beaed4')
highlight(p,syn_indx_l(:,1),syn_indx_l(:,2),'EdgeColor','#662d91')

%saveas(gcf,'Fig_6_graph3.pdf')
exportgraphics(fig, fullfile(output_dir, "Fig_6B_concept_area_graph.pdf"), 'ContentType', 'vector');
close(fig)
