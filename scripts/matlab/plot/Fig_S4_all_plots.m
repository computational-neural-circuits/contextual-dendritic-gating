script_dir = fileparts(mfilename("fullpath"));
repo_root = fileparts(fileparts(fileparts(script_dir)));
data_dir = fullfile(repo_root, "results", "matlab", "Fig_S4");
output_dir = fullfile(repo_root, "results", "figures");
if ~exist(output_dir, "dir")
    mkdir(output_dir)
end

%% panels A1, A3
load(fullfile(data_dir, "data_Fig_S4_A1A3.mat"))

max_color_asse=80;
min_color_asse=0;

limits=12;

fig = figure('Visible', 'off');
subplot(2,2,1)
hold on
plot([0:limits],[0:limits].*LTP_thresh+LTP_thresh,'r')
plot([0:limits],[0:limits].*LTD_thresh+LTD_thresh,'b')
hold off
xlim([0 3])
ylim([0 limits])
xlabel('# of I input')
ylabel('# of E inputs')

ax2=subplot(2,2,2);
hold on
imagesc(gaussianFilter2D(assembly_size_avg, kernelSize, sigma))
[C1_1,~] =contour(gaussianFilter2D(assembly_size_avg, kernelSize, sigma),[lower_bound_asse,lower_bound_asse],'k');
[C1_2,~] =contour(gaussianFilter2D(assembly_size_avg, kernelSize, sigma),[upper_bound_asse,upper_bound_asse],'k');
hold off
colorbar
colormap(ax2,flipud(pink))
set(gca,'YDir','normal')
axis square;
clim([min_color_asse,max_color_asse])
set(gca, 'XTick',5:5:length(N_D_vec), 'XTickLabel',N_D_vec(5:5:end))
set(gca, 'YTick',9:10:length(N_C_vec), 'YTickLabel',N_C_vec(9:10:end))
xlim(([0.5 0.5+length(N_D_vec)]))
ylim(([0.5 0.5+length(N_C_vec)]))
title('Assembly size')
xlabel('N_D')
ylabel('N_C')

% Extract from the contour matrix directly (doesn't work other way)
x1_1 = C1_1(1, 2:1+C1_1(2,1)); y1_1 = C1_1(2, 2:1+C1_1(2,1));
x1_2 = C1_2(1, 2:1+C1_2(2,1)); y1_2 = C1_2(2, 2:1+C1_2(2,1));

ax4=subplot(2,2,4);
hold on
imagesc(gaussianFilter2D(multi_gated_N_ratio, kernelSize, sigma))
plot(x1_1,y1_1,'k')
plot(x1_2,y1_2,'k')
hold off
colorbar
colormap(ax4,flipud(bone))
set(gca,'YDir','normal')
axis square;
set(gca, 'XTick',5:5:length(N_D_vec), 'XTickLabel',N_D_vec(5:5:end))
set(gca, 'YTick',9:10:length(N_C_vec), 'YTickLabel',N_C_vec(9:10:end))
xlim(([0.5 0.5+length(N_D_vec)]))
ylim(([0.5 0.5+length(N_C_vec)]))
title('Multi gated neurons')
xlabel('N_D')
ylabel('N_C')


sgtitle('Model with 1-to-1 connectivity')
exportgraphics(fig, fullfile(output_dir, "Fig_S4_A1A3_1to1_connectivity.pdf"), 'ContentType', 'vector');
close(fig)



%% panels B2, B3
load(fullfile(data_dir, "data_Fig_S4_B2B3.mat"))

fig = figure('Visible', 'off');

for gg1=1:3

ax1=subplot(3,3,gg1);
hold on
imagesc(gaussianFilter2D(squeeze(assembly_size_avg(gg1,:,:)), kernelSize, sigma))
[C1_1,~] =contour(gaussianFilter2D(squeeze(assembly_size_avg(gg1,:,:)), kernelSize, sigma),[lower_bound_asse,lower_bound_asse],'k');
[C1_2,~] =contour(gaussianFilter2D(squeeze(assembly_size_avg(gg1,:,:)), kernelSize, sigma),[upper_bound_asse,upper_bound_asse],'k');
hold off
colormap(ax1,flipud(pink));
colorbar
set(gca,'YDir','normal')
axis square;
clim([min_color_asse, max_color_asse])
set(gca, 'YTick',1:20:length(p_IC_vec), 'YTickLabel',p_IC_vec(1:20:end))
set(gca, 'XTick',1:5:length(p_DI_vec), 'XTickLabel',p_DI_vec(1:5:end))
title(['Assembly size, N_I=' num2str(N_I_vec(gg1))])
ylabel('p_{IC}')
xlabel('p_{DI}')

% Extract from the contour matrix directly (doesn't work other way)
x1_1{gg1} = C1_1(1, 2:1+C1_1(2,1)); y1_1{gg1} = C1_1(2, 2:1+C1_1(2,1));
x1_2{gg1} = C1_2(1, 2:1+C1_2(2,1)); y1_2{gg1} = C1_2(2, 2:1+C1_2(2,1));

ax2=subplot(3,3,3+gg1);
hold on
imagesc(gaussianFilter2D(squeeze(multi_gated_N_ratio(gg1,:,:)), kernelSize, sigma))
plot(x1_1{gg1},y1_1{gg1},'k')
plot(x1_2{gg1},y1_2{gg1},'k')
hold off
colormap(ax2,flipud(bone));
colorbar
set(gca,'YDir','normal')
axis square;
clim([min_color_gateN, max_color_gateN])
set(gca, 'YTick',1:20:length(p_IC_vec), 'YTickLabel',p_IC_vec(1:20:end))
set(gca, 'XTick',1:5:length(p_DI_vec), 'XTickLabel',p_DI_vec(1:5:end))
title(['Frac. multi gated neurons, N_I=' num2str(N_I_vec(gg1))])
xlim(([0.5 0.5+length(p_DI_vec)]))
ylim(([0.5 0.5+length(p_IC_vec)]))
ylabel('p_{IC}')
xlabel('p_{DI}')


ax3=subplot(3,3,6+gg1);
hold on
imagesc(gaussianFilter2D(squeeze(open_dendrites_avg(gg1,:,:)), kernelSize, sigma))
plot(x1_1{gg1},y1_1{gg1},'k')
plot(x1_2{gg1},y1_2{gg1},'k')
hold off
colormap(ax3,flipud(gray));
colorbar
set(gca,'YDir','normal')
axis square;
clim([min_color_alwgateD, max_color_alwgateD])
set(gca, 'YTick',1:20:length(p_IC_vec), 'YTickLabel',p_IC_vec(1:20:end))
set(gca, 'XTick',1:5:length(p_DI_vec), 'XTickLabel',p_DI_vec(1:5:end))
xlim(([0.5 0.5+length(p_DI_vec)]))
ylim(([0.5 0.5+length(p_IC_vec)]))
title(['Frac. "open" Ds, N_I=' num2str(N_I_vec(gg1))])
title('Fraction of always gated Ds')
ylabel('p_{IC}')
xlabel('p_{DI}')

end
exportgraphics(fig, fullfile(output_dir, "Fig_S4_B2B3_parameter_sweep.pdf"), 'ContentType', 'vector');
close(fig)

fig = figure('Visible', 'off');
for gg1=1:3
subplot(3,3,gg1)
hold on
for gg2=1:3
    scatter(p_DI_vec_part2(3*(gg1-1)+gg2),p_IC_vec_part2(3*(gg1-1)+gg2),20,'filled')
end
hold off
axis square;
xlim([min(p_DI_vec), max(p_DI_vec)])
ylim([min(p_IC_vec), max(p_IC_vec)])
title(['Frac. "open" Ds, N_I=' num2str(N_I_vec(gg1))])
title(['Chosen points, N_I=' num2str(N_I_vec(gg1))])
ylabel('p_{IC}')
xlabel('p_{DI}')
end
exportgraphics(fig, fullfile(output_dir, "Fig_S4_B2B3_chosen_points.pdf"), 'ContentType', 'vector');
close(fig)


fig = figure('Visible', 'off');
subplot(3,3,1)
hold on
for pp1=1:3
    plot(overlap_fraction,1-flipud(squeeze(gating_selectivity_ratio_ACROSS_stimuli_avg(pp1,pp1,pp1,:))))
end
yline(0,'k--')
hold off
legend('1','2','3','')
xlabel('Fract. of stim. overlap')
ylabel('Fract. of forgetting')
ylim([-0.05,1])
axis("square")

subplot(3,3,2)
hold on
for pp1=4:6
    plot(overlap_fraction,1-flipud(squeeze(gating_selectivity_ratio_ACROSS_stimuli_avg(pp1,pp1,pp1,:))))
end
yline(0,'k--')
hold off
xlabel('Fract. of stim. overlap')
ylabel('Fract. of forgetting')
ylim([-0.05,1])
axis("square")

subplot(3,3,3)
hold on
for pp1=7:9
    plot(overlap_fraction,1-flipud(squeeze(gating_selectivity_ratio_ACROSS_stimuli_avg(pp1,pp1,pp1,:))))
end
yline(0,'k--')
hold off
xlabel('Fract. of stim. overlap')
ylabel('Fract. of forgetting')
ylim([-0.05,1])
axis("square")
exportgraphics(fig, fullfile(output_dir, "Fig_S4_B2B3_forgetting_vs_overlap.pdf"), 'ContentType', 'vector');
close(fig)



%% panels C2, C3
load(fullfile(data_dir, "data_Fig_S4_C2C3.mat"))

multi_gated_N_ratio(isinf(multi_gated_N_ratio))=NaN;

fig = figure('Visible', 'off');
ax1=subplot(3,3,1);
hold on
imagesc(gaussianFilter2D(squeeze(assembly_size_avg(:,:)), kernelSize, sigma))
[C1_1,~] =contour(gaussianFilter2D(squeeze(assembly_size_avg(:,:)), kernelSize, sigma),[lower_bound_asse,lower_bound_asse],'k');
[C1_2,~] =contour(gaussianFilter2D(squeeze(assembly_size_avg(:,:)), kernelSize, sigma),[upper_bound_asse,upper_bound_asse],'k');
hold off
colorbar
colormap(ax1,flipud(pink))
set(gca,'YDir','normal')
axis square;
clim([min_color_asse, max_color_asse])
set(gca, 'YTick',1:10:length(N_I_vec), 'YTickLabel',N_I_vec(1:10:end))
set(gca, 'XTick',1:10:length(p_IC_vec), 'XTickLabel',p_IC_vec(1:10:end))
title(['Assembly size'])
ylabel('N_I')
xlabel('p_{IC}')

% Extract from the contour matrix directly (doesn't work other way)
x1_1 = C1_1(1, 2:1+C1_1(2,1)); y1_1 = C1_1(2, 2:1+C1_1(2,1));
x1_2 = C1_2(1, 2:1+C1_2(2,1)); y1_2 = C1_2(2, 2:1+C1_2(2,1));

ax4=subplot(3,3,4);
hold on
imagesc(gaussianFilter2D(squeeze(multi_gated_N_ratio(:,:)), kernelSize, sigma))
plot(x1_1,y1_1,'k')
plot(x1_2,y1_2,'k')
hold off
colormap(ax4,flipud(bone))
colorbar
set(gca,'YDir','normal')
axis square;
clim([min_color_gateN, max_color_gateN])
set(gca, 'YTick',1:10:length(N_I_vec), 'YTickLabel',N_I_vec(1:10:end))
set(gca, 'XTick',1:10:length(p_IC_vec), 'XTickLabel',p_IC_vec(1:10:end))
title(['Frac. multi gated neurons'])
xlim(([0.5 0.5+length(p_IC_vec)]))
ylim(([0.5 0.5+length(N_I_vec)]))
ylabel('N_I')
xlabel('p_{CI}')


ax7=subplot(3,3,7);
hold on
imagesc(gaussianFilter2D(squeeze(open_dendrites_avg(:,:)), kernelSize, sigma))
plot(x1_1,y1_1,'k')
plot(x1_2,y1_2,'k')
hold off
colorbar
colormap(ax7,flipud(gray))
set(gca,'YDir','normal')
axis square;
clim([min_color_alwgateD, max_color_alwgateD])
set(gca, 'YTick',1:10:length(N_I_vec), 'YTickLabel',N_I_vec(1:10:end))
set(gca, 'XTick',1:10:length(p_IC_vec), 'XTickLabel',p_IC_vec(1:10:end))
xlim(([0.5 0.5+length(p_IC_vec)]))
ylim(([0.5 0.5+length(N_I_vec)]))
title(['Frac. "open" Ds'])
title('Fraction of always gated Ds')
ylabel('N_I')
xlabel('p_{IC}')


subplot(3,3,2)
hold on
for gg2=1:3
    scatter(p_IC_vec_part2(gg2),N_I_vec_part2(gg2),20,'filled')
end
hold off
axis square;
xlim([min(p_IC_vec), max(p_IC_vec)])
ylim([min(N_I_vec), max(N_I_vec)])
title(['Frac. "open" Ds'])
title(['Chosen points'])
ylabel('N_I')
xlabel('p_{IC}')


subplot(3,3,3)
hold on
for pp1=1:3
    plot(overlap_fraction,1-flipud(squeeze(gating_selectivity_ratio_ACROSS_stimuli_avg(pp1,pp1,:))))
end
yline(0,'k--')
hold off
xlabel('Fract. of stim. overlap')
ylabel('Fract. of forgetting')
ylim([-0.05,1])
axis("square")

sgtitle('Fixed I-to-D, random C-to-I')
exportgraphics(fig, fullfile(output_dir, "Fig_S4_C2C3_fixed_ItoD.pdf"), 'ContentType', 'vector');
close(fig)


%% panels D2, D3
load(fullfile(data_dir, "data_Fig_S4_D2D3.mat"))

fig = figure('Visible', 'off');
ax1=subplot(3,3,1);
hold on
imagesc(gaussianFilter2D(squeeze(assembly_size_avg(:,:)), kernelSize, sigma))
[C1_1,~] =contour(gaussianFilter2D(squeeze(assembly_size_avg(:,:)), kernelSize, sigma),[lower_bound_asse,lower_bound_asse],'k');
[C1_2,~] =contour(gaussianFilter2D(squeeze(assembly_size_avg(:,:)), kernelSize, sigma),[upper_bound_asse,upper_bound_asse],'k');
hold off
colorbar
colormap(ax1,flipud(pink))
set(gca,'YDir','normal')
axis square;
clim([min_color_asse, max_color_asse])
set(gca, 'YTick',1:10:length(N_I_vec), 'YTickLabel',N_I_vec(1:10:end))
set(gca, 'XTick',1:10:length(p_DI_vec), 'XTickLabel',p_DI_vec(1:10:end))
title(['Assembly size'])
ylabel('N_I')
xlabel('p_{DI}')

% Extract from the contour matrix directly (doesn't work other way)
x1_1 = C1_1(1, 2:1+C1_1(2,1)); y1_1 = C1_1(2, 2:1+C1_1(2,1));
x1_2 = C1_2(1, 2:1+C1_2(2,1)); y1_2 = C1_2(2, 2:1+C1_2(2,1));

ax4=subplot(3,3,4);
hold on
imagesc(gaussianFilter2D(squeeze(multi_gated_N_ratio(:,:)), kernelSize, sigma))
plot(x1_1,y1_1,'k')
plot(x1_2,y1_2,'k')
hold off
colormap(ax4,flipud(bone))
colorbar
set(gca,'YDir','normal')
axis square;
clim([min_color_gateN, max_color_gateN])
set(gca, 'YTick',1:10:length(N_I_vec), 'YTickLabel',N_I_vec(1:10:end))
set(gca, 'XTick',1:10:length(p_DI_vec), 'XTickLabel',p_DI_vec(1:10:end))
title(['Frac. multi gated neurons'])
xlim(([0.5 0.5+length(p_DI_vec)]))
ylim(([0.5 0.5+length(N_I_vec)]))
ylabel('N_I')
xlabel('p_{DI}')


ax7=subplot(3,3,7);
hold on
imagesc(gaussianFilter2D(squeeze(open_dendrites_avg(:,:)), kernelSize, sigma))
plot(x1_1,y1_1,'k')
plot(x1_2,y1_2,'k')
hold off
colorbar
colormap(ax7,flipud(gray))
set(gca,'YDir','normal')
axis square;
clim([min_color_alwgateD, max_color_alwgateD])
set(gca, 'YTick',1:10:length(N_I_vec), 'YTickLabel',N_I_vec(1:10:end))
set(gca, 'XTick',1:10:length(p_DI_vec), 'XTickLabel',p_DI_vec(1:10:end))
xlim(([0.5 0.5+length(p_DI_vec)]))
ylim(([0.5 0.5+length(N_I_vec)]))
title(['Frac. "open" Ds'])
title('Fraction of always gated Ds')
ylabel('N_I')
xlabel('p_{DI}')


subplot(3,3,2)
hold on
for gg2=1:3
    scatter(p_DI_vec_part2(gg2),N_I_vec_part2(gg2),20,'filled')
end
hold off
axis square;
xlim([min(p_DI_vec), max(p_DI_vec)])
ylim([min(N_I_vec), max(N_I_vec)])
title(['Frac. "open" Ds'])
title(['Chosen points'])
ylabel('N_I')
xlabel('p_{DI}')


subplot(3,3,3)
hold on
for pp1=1:3
    plot(overlap_fraction,1-flipud(squeeze(gating_selectivity_ratio_ACROSS_stimuli_avg(pp1,pp1,:))))
end
yline(0,'k--')
hold off
xlabel('Fract. of stim. overlap')
ylabel('Fract. of forgetting')
ylim([-0.05,1])
axis("square")

sgtitle('Fixed C-to-I, random I-to-D')
exportgraphics(fig, fullfile(output_dir, "Fig_S4_D2D3_fixed_CtoI.pdf"), 'ContentType', 'vector');
close(fig)

%%
function output = gaussianFilter2D(input, kernelSize, sigma)
%Applies a 2D Gaussian filter to a matrix

    arguments
        input double
        kernelSize (1,1) {mustBePositive, mustBeInteger, mustBeOdd}
        sigma (1,1) {mustBePositive}
    end

    % Create coordinate grid centered at 0
    halfSize = floor(kernelSize / 2);
    [x, y] = meshgrid(-halfSize:halfSize, -halfSize:halfSize);

    % 2D Gaussian kernel
    G = exp(-(x.^2 + y.^2) / (2 * sigma^2));
    G = G / sum(G(:));  % Normalize

    % Validity mask: 1 where input is finite, 0 where NaN
    validMask = double(isfinite(input));

    % Replace NaNs with zero for convolution
    inputZero = input;
    inputZero(~isfinite(input)) = 0;

    % Convolve input and mask
    filteredValues = conv2(inputZero, G, 'same');
    normalization = conv2(validMask, G, 'same');

    % Renormalize: divide by available weights
    output = filteredValues ./ normalization;

    % ----- APPLY TRUST RULE (XX% threshold) -----
    % Gaussian kernel sums to 1 → normalization = fraction of valid weights.
    define_NaN_thresh=0.2;
    tooManyNaNs = normalization < define_NaN_thresh;

    % In areas with no valid data at all, normalization = 0
    noData = normalization == 0;

    % Assign NaN where too many NaNs OR no data
    output(tooManyNaNs | noData) = NaN;

end

% Helper to enforce odd kernel size
function mustBeOdd(n)
    if mod(n,2) == 0
        error('kernelSize must be an odd integer.');
    end
end
