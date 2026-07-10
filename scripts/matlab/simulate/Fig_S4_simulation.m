% code for Figure S4 of Onasch, Miehl et al., 2026 (Neuron)

clear all

repo_root = fileparts(fileparts(fileparts(fileparts(mfilename('fullpath')))));
results_dir = fullfile(repo_root, 'results', 'Fig_S4');
figures_dir = fullfile(repo_root, 'results', 'figures');
if ~exist(results_dir, 'dir')
    mkdir(results_dir)
end
if ~exist(figures_dir, 'dir')
    mkdir(figures_dir)
end

% choose here the connectivity case to simulate.
% ==1 means all random, ==2 means only I-to-D random, ==3 means only C-to-I random, ==4 is the 1:1 case
connectivity_cases=4;

% fixed parameters
N_RC=400; % number of recurrent neurons
N_FF=40; % number of feedforward input neurons. Have 40 feedforward inputs so we can either activate the same stimul (same 20 neurons) or two distinct stimuli (20 vs 20) or anything in between.
p_FF_E=0.081; % FF to RC neuron connection probability

r_I_target=10; % E input that pushes I to a target Hz
w_Ito_D_strength=2; % connection strength
w_CtoI_strength=r_I_target; % to switch I off if context is on
LTP_thresh=w_Ito_D_strength.*r_I_target.*4; % E-I>=LTP_thresh -> LTP
LTD_thresh=w_Ito_D_strength.*r_I_target.*3; % LTD_thresh>=E-I<LTP_thresh -> LTD

r_E_FF=w_Ito_D_strength.*r_I_target;

N_D=6; % number of dendrites
N_C=6; % number of contexts

lower_bound_asse=10; upper_bound_asse=30; % bounds for "reasonable" assembly size

N_repeats=20; % number of random connectivity draws

% for interference part
N_repeats_part2=40;
overlap_steps=2; % step size of overlap (in total 2x 20 neurons)
nr_overlap_sims=20/overlap_steps+1; % 20 is the stimulus size

% colorbar bounds
max_color_asse=80;
min_color_asse=0;
max_color_gateN=0.7;
min_color_gateN=0;
max_color_alwgateD=0.4;
min_color_alwgateD=0;

% for 2D Gaussian filtering
kernelSize=5;
sigma=5;

if connectivity_cases==1 % case where all connections are random


    N_I_vec=[40,70,100]; % number of inhibitory neurons
    p_IC_vec=0.1:0.005:0.8; % C-to-I connection probabilities
    p_DI_vec=0.05:0.005:0.2; % I-to-D connection probabilities

    % initialize measures
    assembly_size_avg=zeros(length(N_I_vec),length(p_IC_vec),length(p_DI_vec));
    multi_gated_N_ratio=zeros(length(N_I_vec),length(p_IC_vec),length(p_DI_vec));
    open_dendrites_avg=zeros(length(N_I_vec),length(p_IC_vec),length(p_DI_vec));

    for uu1=1:length(N_I_vec) % loop through parameters
        for uu2=1:length(p_IC_vec)
            for uu3=1:length(p_DI_vec)

                N_I=N_I_vec(uu1);
                p_IC=p_IC_vec(uu2);
                p_DI=p_DI_vec(uu3);

                assembly_size=zeros(N_C,N_repeats);
                multi_gated_N=zeros(N_C,N_repeats);
                open_dendrites=zeros(N_repeats,1);
                gating_selectivity_ratio=zeros(N_repeats,1);
                gating_selectivity_ratio_ACROSS_stimuli=zeros(N_repeats,1);

                for kk4=1:N_repeats

                    % random FF to D connectivity (for the two FF stimuli)
                    W_FFtoD=rand(N_RC*N_D,N_FF); W_FFtoD(W_FFtoD<=1-p_FF_E)=0; W_FFtoD(W_FFtoD>1-p_FF_E)=1;
                    Esyn_D_stim1=r_E_FF.*sum(W_FFtoD(:,1:20),2); % choose here stimulus 1 (first 20 FF inputs)

                    % random C to I connectivity
                    W_CtoI=rand(N_I,N_C); W_CtoI(W_CtoI<=1-p_IC)=0; W_CtoI(W_CtoI>1-p_IC)=w_CtoI_strength;

                    % random I to D connectivity
                    W_ItoD=rand(N_RC*N_D,N_I); W_ItoD(W_ItoD<=1-p_DI)=0; W_ItoD(W_ItoD>1-p_DI)=w_Ito_D_strength;

                    dendrite_LTP_gates_across_C_stim1=zeros(N_RC*N_D,1);

                    for kk3=1:N_C %loop through contexts

                        r_I=r_I_target-W_CtoI(:,kk3); % choose I cells that are active (hence 1-W) for given context (if there is one input, shut down I)

                        Isyn_D=W_ItoD*r_I;

                        delta_syn_D_stim1=Esyn_D_stim1-Isyn_D;
                        LTP_dend_stim1=delta_syn_D_stim1; LTP_dend_stim1(LTP_dend_stim1<LTP_thresh)=0; LTP_dend_stim1(LTP_dend_stim1>=LTP_thresh)=1;
                        LTD_dend_stim1=delta_syn_D_stim1; LTD_dend_stim1(LTD_dend_stim1<LTD_thresh)=0; LTD_dend_stim1(LTD_dend_stim1>=LTD_thresh)=1; LTD_dend_stim1(LTD_dend_stim1>=LTP_thresh)=0;

                        dendrite_LTP_gates_across_C_stim1=dendrite_LTP_gates_across_C_stim1+LTP_dend_stim1;

                        assembly_neuron_stim1=movsum(LTP_dend_stim1,N_D,"Endpoints","discard"); assembly_neuron_stim1=assembly_neuron_stim1(1:N_D:end);
                        for_multi=movsum(LTD_dend_stim1,N_D,"Endpoints","discard"); for_multi=for_multi(1:N_D:end);

                        assembly_neuron_dummy_stim1=assembly_neuron_stim1;
                        assembly_neuron_dummy_stim1(assembly_neuron_dummy_stim1>1)=1;
                        assembly_size(kk3,kk4)=sum(assembly_neuron_dummy_stim1);  % neuron is part of an assembly if one or more dendrites is LTP

                        multi_gated_N(kk3,kk4)=sum(for_multi>1)./assembly_size(kk3,kk4);

                    end
                    open_dendrites(kk4,1)=sum(dendrite_LTP_gates_across_C_stim1==N_C)/sum(dendrite_LTP_gates_across_C_stim1>0);
                end
                multi_gated_N_ratio(uu1,uu2,uu3)=mean(mean(multi_gated_N,'omitmissing'),'omitmissing'); % mean across contexts & repeats
                assembly_size_avg(uu1,uu2,uu3)=mean(mean(assembly_size,'omitmissing'),'omitmissing');
                open_dendrites_avg(uu1,uu2,uu3)=mean(open_dendrites,'omitmissing'); % fraction of dendrites that are never inhibited and undergo LTP out of the total number of LTP dendrites
            end
        end
    end

    % plot results
    multi_gated_N_ratio(isinf(multi_gated_N_ratio))=NaN;

    fig_parameter_sweep = figure('Visible', 'off');

    for gg1=1:3

    ax1=subplot(3,3,gg1);
    hold on
    imagesc(gaussianFilter2D(squeeze(assembly_size_avg(gg1,:,:)), kernelSize, sigma))
    [C1_1,~] =contour(gaussianFilter2D(squeeze(assembly_size_avg(gg1,:,:)), kernelSize, sigma),[lower_bound_asse,lower_bound_asse],'k');
    [C1_2,~] =contour(gaussianFilter2D(squeeze(assembly_size_avg(gg1,:,:)), kernelSize, sigma),[upper_bound_asse,upper_bound_asse],'k');
    hold off
    colorbar
    colormap(ax1,flipud(pink))
    set(gca,'YDir','normal')
    axis square;
    clim([min_color_asse, max_color_asse])
    set(gca, 'YTick',1:20:length(p_IC_vec), 'YTickLabel',p_IC_vec(1:20:end))
    set(gca, 'XTick',1:5:length(p_DI_vec), 'XTickLabel',p_DI_vec(1:5:end))
    title(['Assembly size, N_I=' num2str(N_I_vec(gg1))])
    ylabel('p_{IC}')
    xlabel('p_{DI}')

    x1_1{gg1} = C1_1(1, 2:1+C1_1(2,1)); y1_1{gg1} = C1_1(2, 2:1+C1_1(2,1));
    x1_2{gg1} = C1_2(1, 2:1+C1_2(2,1)); y1_2{gg1} = C1_2(2, 2:1+C1_2(2,1));

    ax2=subplot(3,3,3+gg1);

    hold on
    imagesc(gaussianFilter2D(squeeze(multi_gated_N_ratio(gg1,:,:)), 11, sigma))
    plot(x1_1{gg1},y1_1{gg1},'k')
    plot(x1_2{gg1},y1_2{gg1},'k')
    hold off
    colormap(ax2,flipud(bone))
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
    imagesc(gaussianFilter2D(squeeze(open_dendrites_avg(gg1,:,:)), 11, sigma))
    plot(x1_1{gg1},y1_1{gg1},'k')
    plot(x1_2{gg1},y1_2{gg1},'k')
    hold off
    colorbar
    colormap(ax3,flipud(gray))
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

    exportgraphics(fig_parameter_sweep, fullfile(figures_dir, 'Fig_S4_B2B3_parameter_sweep.pdf'), 'ContentType', 'vector');
    close(fig_parameter_sweep)

    fig_chosen_points = figure('Visible', 'off');

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

    exportgraphics(fig_chosen_points, fullfile(figures_dir, 'Fig_S4_B2B3_chosen_points.pdf'), 'ContentType', 'vector');
    close(fig_chosen_points)

    % now choose a subset of parameters and show forgetting as a function of stimulus overlap
    N_I_vec_part2=[N_I_vec(1),N_I_vec(1),N_I_vec(1),N_I_vec(2),N_I_vec(2),N_I_vec(2),N_I_vec(3),N_I_vec(3),N_I_vec(3)];
    p_IC_vec_part2=[0.2,0.45,0.6,0.3,0.6,0.75,0.5,0.7,0.75];
    p_DI_vec_part2=[0.09,0.125,0.19,0.06,0.11,0.175,0.06,0.1,0.125];

    gating_selectivity_ratio_ACROSS_stimuli_avg=zeros(length(N_I_vec_part2),length(p_IC_vec_part2),length(p_DI_vec_part2),nr_overlap_sims);

    for uu1=1:length(N_I_vec_part2)
        for uu2=1:length(p_IC_vec_part2)
            for uu3=1:length(p_DI_vec_part2)

                N_I=N_I_vec_part2(uu1);
                p_IC=p_IC_vec_part2(uu2);
                p_DI=p_DI_vec_part2(uu3);

                gating_selectivity_ratio_ACROSS_stimuli=zeros(N_repeats_part2,nr_overlap_sims);

                for kk4=1:N_repeats_part2
                    % random FF to D connectivity (for the two FF stimuli)
                    W_FFtoD=rand(N_RC*N_D,N_FF); W_FFtoD(W_FFtoD<=1-p_FF_E)=0; W_FFtoD(W_FFtoD>1-p_FF_E)=1;
                    Esyn_D_stim1=r_E_FF.*sum(W_FFtoD(:,1:20),2); % choose here stimulus 1 (first 20 FF inputs)
                    % random C to I connectivity
                    W_CtoI=rand(N_I,N_C); W_CtoI(W_CtoI<=1-p_IC)=0; W_CtoI(W_CtoI>1-p_IC)=w_CtoI_strength;

                    % random I to D connectivity
                    W_ItoD=rand(N_RC*N_D,N_I); W_ItoD(W_ItoD<=1-p_DI)=0; W_ItoD(W_ItoD>1-p_DI)=w_Ito_D_strength;

                    for kk5=1:nr_overlap_sims
                        Esyn_D_stim2=r_E_FF.*sum(W_FFtoD(:,1+overlap_steps*(kk5-1):20+overlap_steps*(kk5-1)),2); % choose here stimulus 2 (with potential overlap)

                        dendrite_LTDorLTP_gates_across_C_across_STIM=zeros(N_RC*N_D,2); % first colum is LTP in context 1, second is LTD or LTP in other contexts for stim 2
                        for kk3=1:N_C %loop through contexts

                            r_I=r_I_target-W_CtoI(:,kk3); % choose I cells that are active (hence 1-W) for given context (if there is one input, shut down I)

                            Isyn_D=W_ItoD*r_I;

                            delta_syn_D_stim1=Esyn_D_stim1-Isyn_D;
                            LTP_dend_stim1=delta_syn_D_stim1; LTP_dend_stim1(LTP_dend_stim1<LTP_thresh)=0; LTP_dend_stim1(LTP_dend_stim1>=LTP_thresh)=1;
                            LTD_dend_stim1=delta_syn_D_stim1; LTD_dend_stim1(LTD_dend_stim1<LTD_thresh)=0; LTD_dend_stim1(LTD_dend_stim1>=LTD_thresh)=1; LTD_dend_stim1(LTD_dend_stim1>=LTP_thresh)=0;

                            delta_syn_D_stim2=Esyn_D_stim2-Isyn_D;
                            LTP_dend_stim2=delta_syn_D_stim2; LTP_dend_stim2(LTP_dend_stim2<LTP_thresh)=0; LTP_dend_stim2(LTP_dend_stim2>=LTP_thresh)=1;
                            LTD_dend_stim2=delta_syn_D_stim2; LTD_dend_stim2(LTD_dend_stim2<LTD_thresh)=0; LTD_dend_stim2(LTD_dend_stim2>=LTD_thresh)=1; LTD_dend_stim2(LTD_dend_stim2>=LTP_thresh)=0;

                            if kk3==1 % have LTP D in context 1, and check in other contexts if another stimulus would lead to LTD or LTP
                                dendrite_LTDorLTP_gates_across_C_across_STIM(:,1)=dendrite_LTDorLTP_gates_across_C_across_STIM(:,1)+LTP_dend_stim1;
                            elseif kk3>1
                                dendrite_LTDorLTP_gates_across_C_across_STIM(:,2)=dendrite_LTDorLTP_gates_across_C_across_STIM(:,2)+LTP_dend_stim2+LTP_dend_stim2;
                            end

                        end

                        gating_selectivity_ratio_ACROSS_stimuli(kk4,kk5) = (sum(dendrite_LTDorLTP_gates_across_C_across_STIM(:,1)>0)-sum(dendrite_LTDorLTP_gates_across_C_across_STIM(:,1) ~= 0 & dendrite_LTDorLTP_gates_across_C_across_STIM(:,2) ~= 0))/sum(dendrite_LTDorLTP_gates_across_C_across_STIM(:,1)>0);
                    end
                end
                gating_selectivity_ratio_ACROSS_stimuli_avg(uu1,uu2,uu3,:)=mean(gating_selectivity_ratio_ACROSS_stimuli,1,'omitmissing'); % fraction of D that do NOT change (LTD or LTP) out of the D part of an assemblyif another stimulus is shown in a different context
            end
        end
    end


    overlap_fraction=[0,overlap_steps:overlap_steps:20]./20;

    fig_forgetting_vs_overlap = figure('Visible', 'off');
    subplot(3,3,1)
    hold on
    for pp1=1:3
        plot(overlap_fraction,1-flipud(squeeze(gating_selectivity_ratio_ACROSS_stimuli_avg(pp1,pp1,pp1,:))))
    end
    legend('1','2','3')
    hold off
    xlabel('Fract. of input overlap')
    ylabel('Fract. of forgetting')
    ylim([0,1])
    axis("square")

    subplot(3,3,2)
    hold on
    for pp1=4:6
        plot(overlap_fraction,1-flipud(squeeze(gating_selectivity_ratio_ACROSS_stimuli_avg(pp1,pp1,pp1,:))))
    end
    hold off
    xlabel('Fract. of input overlap')
    ylabel('Fract. of forgetting')
    ylim([0,1])
    axis("square")

    subplot(3,3,3)
    hold on
    for pp1=7:9
        plot(overlap_fraction,1-flipud(squeeze(gating_selectivity_ratio_ACROSS_stimuli_avg(pp1,pp1,pp1,:))))
    end
    hold off
    xlabel('Fract. of input overlap')
    ylabel('Fract. of forgetting')
    ylim([0,1])
    axis("square")

    exportgraphics(fig_forgetting_vs_overlap, fullfile(figures_dir, 'Fig_S4_B2B3_forgetting_vs_overlap.pdf'), 'ContentType', 'vector');
    close(fig_forgetting_vs_overlap)

elseif connectivity_cases==2 % ==2 means only I-to-D random

    N_I_vec=[50:2:140];
    p_DI_vec=0.015:0.0025:0.1;

    % initialize measures
    assembly_size_avg=zeros(length(N_I_vec),length(p_DI_vec));
    multi_gated_N_ratio=zeros(length(N_I_vec),length(p_DI_vec));
    open_dendrites_avg=zeros(length(N_I_vec),length(p_DI_vec));

    for uu1=1:length(N_I_vec)
        for uu3=1:length(p_DI_vec)

            N_I=N_I_vec(uu1);
            p_DI=p_DI_vec(uu3);

            assembly_size=zeros(N_C,N_repeats);
            multi_gated_N=zeros(N_C,N_repeats);
            open_dendrites=zeros(N_repeats,1);

            for kk4=1:N_repeats

                % random FF to D connectivity (for the two FF stimuli)
                W_FFtoD=rand(N_RC*N_D,N_FF); W_FFtoD(W_FFtoD<=1-p_FF_E)=0; W_FFtoD(W_FFtoD>1-p_FF_E)=1;
                Esyn_D_stim1=r_E_FF.*sum(W_FFtoD(:,1:20),2); % choose here stimulus 1 (first 20 FF inputs) [first values to get FRs]

                N_I_perC=floor(N_I/N_C); % arbitrary number of inhibitory neurons per context, just to make sure that the dendrite is always blocked if the context is not "on"

                % FIXED C to I connectivity (a distinct number of I receives input from the same C) [same for every repeat]
                W_CtoI=zeros(N_I,N_C);
                for pp1=1:N_C
                    W_CtoI(1+N_I_perC*(pp1-1):N_I_perC*pp1,pp1)=ones(N_I_perC,1)*w_CtoI_strength; % if one C is on, this shuts down SST completely
                end

                % random I to D connectivity
                W_ItoD=rand(N_RC*N_D,N_I); W_ItoD(W_ItoD<=1-p_DI)=0; W_ItoD(W_ItoD>1-p_DI)=w_Ito_D_strength;

                dendrite_LTP_gates_across_C_stim1=zeros(N_RC*N_D,1);

                for kk3=1:N_C %loop through contexts

                    r_I=r_I_target-W_CtoI(:,kk3); % choose I cells that are active (hence 1-W) for given context (if there is one input, shut down I)

                    Isyn_D=W_ItoD*r_I;

                    delta_syn_D_stim1=Esyn_D_stim1-Isyn_D;
                    LTP_dend_stim1=delta_syn_D_stim1; LTP_dend_stim1(LTP_dend_stim1<LTP_thresh)=0; LTP_dend_stim1(LTP_dend_stim1>=LTP_thresh)=1;
                    LTD_dend_stim1=delta_syn_D_stim1; LTD_dend_stim1(LTD_dend_stim1<LTD_thresh)=0; LTD_dend_stim1(LTD_dend_stim1>=LTD_thresh)=1; LTD_dend_stim1(LTD_dend_stim1>=LTP_thresh)=0;

                    dendrite_LTP_gates_across_C_stim1=dendrite_LTP_gates_across_C_stim1+LTP_dend_stim1;

                    assembly_neuron_stim1=movsum(LTP_dend_stim1,N_D,"Endpoints","discard"); assembly_neuron_stim1=assembly_neuron_stim1(1:N_D:end);
                    for_multi=movsum(LTD_dend_stim1,N_D,"Endpoints","discard"); for_multi=for_multi(1:N_D:end);

                    assembly_neuron_dummy_stim1=assembly_neuron_stim1;
                    assembly_neuron_dummy_stim1(assembly_neuron_dummy_stim1>1)=1;
                    assembly_size(kk3,kk4)=sum(assembly_neuron_dummy_stim1);  % neuron is part of an assembly if one or more dendrites is LTP

                    multi_gated_N(kk3,kk4)=sum(for_multi>1)./assembly_size(kk3,kk4);
                end
                    open_dendrites(kk4,1)=sum(dendrite_LTP_gates_across_C_stim1==N_C)/sum(dendrite_LTP_gates_across_C_stim1>0);
            end
            multi_gated_N_ratio(uu1,uu3)=mean(mean(multi_gated_N,'omitmissing'),'omitmissing'); % mean across contexts & repeats
            assembly_size_avg(uu1,uu3)=mean(mean(assembly_size,'omitmissing'),'omitmissing');
            open_dendrites_avg(uu1,uu3)=mean(open_dendrites,'omitmissing'); % fraction of dendrites that are never inhibited and undergo LTP out of the total number of LTP dendrites
        end
    end

    % now choose a subset of parameters and show forgetting as a function of stimulus overlap
    N_I_vec_part2=[120,75,60];
    p_DI_vec_part2=[0.03,0.05,0.065];

    gating_selectivity_ratio_ACROSS_stimuli_avg=zeros(length(N_I_vec_part2),length(p_DI_vec_part2),nr_overlap_sims);

    for uu1=1:length(N_I_vec_part2)
        for uu3=1:length(p_DI_vec_part2)

            N_I=N_I_vec_part2(uu1);
            p_DI=p_DI_vec_part2(uu3);

            gating_selectivity_ratio_ACROSS_stimuli=zeros(N_repeats_part2,nr_overlap_sims);

            for kk4=1:N_repeats_part2
                % random FF to D connectivity (for the two FF stimuli)
                W_FFtoD=rand(N_RC*N_D,N_FF); W_FFtoD(W_FFtoD<=1-p_FF_E)=0; W_FFtoD(W_FFtoD>1-p_FF_E)=1;
                Esyn_D_stim1=r_E_FF.*sum(W_FFtoD(:,1:20),2); % choose here stimulus 1 (first 20 FF inputs)

                N_I_perC=floor(N_I/N_C); % arbitrary number of inhibitory neurons per context, just to make sure that the dendrite is always blocked if the context is not "on"

                % FIXED C to I connectivity (a distinct number of I receives input from the same C) [same for every repeat]
                W_CtoI=zeros(N_I,N_C);
                for pp1=1:N_C
                    W_CtoI(1+N_I_perC*(pp1-1):N_I_perC*pp1,pp1)=ones(N_I_perC,1)*w_CtoI_strength;
                end

                % random I to D connectivity
                W_ItoD=rand(N_RC*N_D,N_I); W_ItoD(W_ItoD<=1-p_DI)=0; W_ItoD(W_ItoD>1-p_DI)=w_Ito_D_strength;

                for kk5=1:nr_overlap_sims
                    Esyn_D_stim2=r_E_FF.*sum(W_FFtoD(:,1+overlap_steps*(kk5-1):20+overlap_steps*(kk5-1)),2); % choose here stimulus 2 (with potential overlap)

                    dendrite_LTDorLTP_gates_across_C_across_STIM=zeros(N_RC*N_D,2); % first colum is LTP in context 1, second is LTD or LTP in other contexts for stim 2
                    for kk3=1:N_C %loop through contexts

                        r_I=r_I_target-W_CtoI(:,kk3); % choose I cells that are active (hence 1-W) for given context (if there is one input, shut down I)

                        Isyn_D=W_ItoD*r_I;

                        delta_syn_D_stim1=Esyn_D_stim1-Isyn_D;
                        LTP_dend_stim1=delta_syn_D_stim1; LTP_dend_stim1(LTP_dend_stim1<LTP_thresh)=0; LTP_dend_stim1(LTP_dend_stim1>=LTP_thresh)=1;
                        LTD_dend_stim1=delta_syn_D_stim1; LTD_dend_stim1(LTD_dend_stim1<LTD_thresh)=0; LTD_dend_stim1(LTD_dend_stim1>=LTD_thresh)=1; LTD_dend_stim1(LTD_dend_stim1>=LTP_thresh)=0;

                        delta_syn_D_stim2=Esyn_D_stim2-Isyn_D;
                        LTP_dend_stim2=delta_syn_D_stim2; LTP_dend_stim2(LTP_dend_stim2<LTP_thresh)=0; LTP_dend_stim2(LTP_dend_stim2>=LTP_thresh)=1;
                        LTD_dend_stim2=delta_syn_D_stim2; LTD_dend_stim2(LTD_dend_stim2<LTD_thresh)=0; LTD_dend_stim2(LTD_dend_stim2>=LTD_thresh)=1; LTD_dend_stim2(LTD_dend_stim2>=LTP_thresh)=0;

                        if kk3==1 % have LTP D in context 1, and check in other contexts if another stimulus would lead to LTD or LTP
                            dendrite_LTDorLTP_gates_across_C_across_STIM(:,1)=dendrite_LTDorLTP_gates_across_C_across_STIM(:,1)+LTP_dend_stim1;
                        elseif kk3>1
                            dendrite_LTDorLTP_gates_across_C_across_STIM(:,2)=dendrite_LTDorLTP_gates_across_C_across_STIM(:,2)+LTP_dend_stim2+LTP_dend_stim2;
                        end

                    end

                    gating_selectivity_ratio_ACROSS_stimuli(kk4,kk5) = (sum(dendrite_LTDorLTP_gates_across_C_across_STIM(:,1)>0)-sum(dendrite_LTDorLTP_gates_across_C_across_STIM(:,1) ~= 0 & dendrite_LTDorLTP_gates_across_C_across_STIM(:,2) ~= 0))/sum(dendrite_LTDorLTP_gates_across_C_across_STIM(:,1)>0);
                end
            end
            gating_selectivity_ratio_ACROSS_stimuli_avg(uu1,uu3,:)=mean(gating_selectivity_ratio_ACROSS_stimuli,1,'omitmissing'); % fraction of D that do NOT change (LTD or LTP) out of the D part of an assemblyif another stimulus is shown in a different context
        end
    end


    % plot results
    multi_gated_N_ratio(isinf(multi_gated_N_ratio))=NaN;

    fig_fixed_CtoI = figure('Visible', 'off');
    ax1 = subplot(3,3,1);
    hold on
    imagesc(gaussianFilter2D(squeeze(assembly_size_avg(:,:)), kernelSize, sigma))
    [C1_1,~] =contour(gaussianFilter2D(squeeze(assembly_size_avg(:,:)), kernelSize, sigma),[lower_bound_asse,lower_bound_asse],'k');
    [C1_2,~] =contour(gaussianFilter2D(squeeze(assembly_size_avg(:,:)), kernelSize, sigma),[upper_bound_asse,upper_bound_asse],'k');
    hold off
    colormap(ax1,flipud(pink))
    colorbar(ax1)
    set(gca,'YDir','normal')
    axis square;
    clim([min_color_asse, max_color_asse])
    set(gca, 'YTick',1:10:length(N_I_vec), 'YTickLabel',N_I_vec(1:10:end))
    set(gca, 'XTick',1:10:length(p_DI_vec), 'XTickLabel',p_DI_vec(1:10:end))
    title(['Assembly size'])
    ylabel('N_I')
    xlabel('p_{DI}')

    x1_1 = C1_1(1, 2:1+C1_1(2,1)); y1_1 = C1_1(2, 2:1+C1_1(2,1));
    x1_2 = C1_2(1, 2:1+C1_2(2,1)); y1_2 = C1_2(2, 2:1+C1_2(2,1));

    ax2 = subplot(3,3,4);
    hold on
    imagesc(gaussianFilter2D(squeeze(multi_gated_N_ratio(:,:)), kernelSize, sigma))
    plot(x1_1,y1_1,'k')
    plot(x1_2,y1_2,'k')
    hold off
    colormap(ax2, flipud(bone))
    colorbar(ax2)
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


    ax3 = subplot(3,3,7);
    hold on
    imagesc(gaussianFilter2D(squeeze(open_dendrites_avg(:,:)), kernelSize, sigma))
    plot(x1_1,y1_1,'k')
    plot(x1_2,y1_2,'k')
    hold off
    colormap(ax3, flipud(gray))
    colorbar(ax3)
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

    for gg1=1:3
        subplot(3,3,2+3*(gg1-1))
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
    end

    overlap_fraction=[0,overlap_steps:overlap_steps:20]./20;

    subplot(3,3,3)
    hold on
    for pp1=1:3
        plot(overlap_fraction,1-flipud(squeeze(gating_selectivity_ratio_ACROSS_stimuli_avg(pp1,pp1,:))))
    end
    legend('1','2','3')
    hold off
    xlabel('Fract. of input overlap')
    ylabel('Fract. of forgetting')
    ylim([0,1])
    axis("square")

    sgtitle('Fixed C-to-I, random I-to-D')

    exportgraphics(fig_fixed_CtoI, fullfile(figures_dir, 'Fig_S4_D2D3_fixed_CtoI.pdf'), 'ContentType', 'vector');
    close(fig_fixed_CtoI)

elseif connectivity_cases==3 % ==3 means only C-to-I random

    N_I_vec=[50:2:140];
    p_IC_vec=0.51:0.02:0.9;

    % initialize measures
    assembly_size_avg=zeros(length(N_I_vec),length(p_IC_vec));
    multi_gated_N_ratio=zeros(length(N_I_vec),length(p_IC_vec));
    open_dendrites_avg=zeros(length(N_I_vec),length(p_IC_vec));

    for uu1=1:length(N_I_vec)
        for uu3=1:length(p_IC_vec)

            N_I=N_I_vec(uu1);
            p_IC=p_IC_vec(uu3);

            assembly_size=zeros(N_C,N_repeats);
            multi_gated_N=zeros(N_C,N_repeats);
            open_dendrites=zeros(N_repeats,1);

            for kk4=1:N_repeats

                % random FF to D connectivity (for the two FF stimuli)
                W_FFtoD=rand(N_RC*N_D,N_FF); W_FFtoD(W_FFtoD<=1-p_FF_E)=0; W_FFtoD(W_FFtoD>1-p_FF_E)=1;
                Esyn_D_stim1=r_E_FF.*sum(W_FFtoD(:,1:20),2); % choose here stimulus 1 (first 20 FF inputs)

                % random C to I connectivity
                W_CtoI=rand(N_I,N_C); W_CtoI(W_CtoI<=1-p_IC)=0; W_CtoI(W_CtoI>1-p_IC)=w_CtoI_strength;

                N_I_perC=floor(N_I/N_C); % arbitrary number of inhibitory neurons per context, just to make sure that the dendrite is always blocked if the context is not "on"

                % FIXED I to D connectivity (each D receives input from exactly one random I subset)
                W_ItoD=zeros(N_RC*N_D,N_I);
                rand_C_per_D=randi(N_C,N_D*N_RC,1);

                for pp2=1:N_D*N_RC
                    W_ItoD(pp2,1+(rand_C_per_D(pp2)-1)*N_I_perC:rand_C_per_D(pp2)*N_I_perC)=ones(1,N_I_perC).*w_Ito_D_strength;
                end

                dendrite_LTP_gates_across_C_stim1=zeros(N_RC*N_D,1);
                for kk3=1:N_C %loop through contexts

                    r_I=r_I_target-W_CtoI(:,kk3); % choose I cells that are active (hence 1-W) for given context (if there is one input, shut down I)

                    Isyn_D=W_ItoD*r_I;

                    delta_syn_D_stim1=Esyn_D_stim1-Isyn_D;
                    LTP_dend_stim1=delta_syn_D_stim1; LTP_dend_stim1(LTP_dend_stim1<LTP_thresh)=0; LTP_dend_stim1(LTP_dend_stim1>=LTP_thresh)=1;
                    LTD_dend_stim1=delta_syn_D_stim1; LTD_dend_stim1(LTD_dend_stim1<LTD_thresh)=0; LTD_dend_stim1(LTD_dend_stim1>=LTD_thresh)=1; LTD_dend_stim1(LTD_dend_stim1>=LTP_thresh)=0;

                    dendrite_LTP_gates_across_C_stim1=dendrite_LTP_gates_across_C_stim1+LTP_dend_stim1;

                    assembly_neuron_stim1=movsum(LTP_dend_stim1,N_D,"Endpoints","discard"); assembly_neuron_stim1=assembly_neuron_stim1(1:N_D:end);
                    for_multi=movsum(LTD_dend_stim1,N_D,"Endpoints","discard"); for_multi=for_multi(1:N_D:end);
                    %multi_gated_N(kk3,kk4)=sum(assembly_neuron_stim1>1)/sum(assembly_neuron_stim1>0);


                    assembly_neuron_dummy_stim1=assembly_neuron_stim1;
                    assembly_neuron_dummy_stim1(assembly_neuron_dummy_stim1>1)=1;
                    assembly_size(kk3,kk4)=sum(assembly_neuron_dummy_stim1);  % neuron is part of an assembly if one or more dendrites is LTP

                    multi_gated_N(kk3,kk4)=sum(for_multi>1)./assembly_size(kk3,kk4);
                end
                open_dendrites(kk4,1)=sum(dendrite_LTP_gates_across_C_stim1==N_C)/sum(dendrite_LTP_gates_across_C_stim1>0);
            end
            multi_gated_N_ratio(uu1,uu3)=mean(mean(multi_gated_N,'omitmissing'),'omitmissing'); % mean across contexts & repeats
            assembly_size_avg(uu1,uu3)=mean(mean(assembly_size,'omitmissing'),'omitmissing');
            open_dendrites_avg(uu1,uu3)=mean(open_dendrites,'omitmissing'); % fraction of dendrites that are never inhibited and undergo LTP out of the total number of LTP dendrites
        end
    end

    % now choose a subset of parameters and show forgetting as a function of stimulus overlap
    N_I_vec_part2=[120,75,60];
    p_IC_vec_part2=[0.85,0.75,0.7];

    gating_selectivity_ratio_ACROSS_stimuli_avg=zeros(length(N_I_vec_part2),length(p_IC_vec_part2),nr_overlap_sims);

    for uu1=1:length(N_I_vec_part2)
        for uu3=1:length(p_IC_vec_part2)

            N_I=N_I_vec_part2(uu1);
            p_IC=p_IC_vec_part2(uu3);

            gating_selectivity_ratio_ACROSS_stimuli=zeros(N_repeats_part2,nr_overlap_sims);

            for kk4=1:N_repeats_part2
                % random FF to D connectivity (for the two FF stimuli)
                W_FFtoD=rand(N_RC*N_D,N_FF); W_FFtoD(W_FFtoD<=1-p_FF_E)=0; W_FFtoD(W_FFtoD>1-p_FF_E)=1;
                Esyn_D_stim1=r_E_FF.*sum(W_FFtoD(:,1:20),2); % choose here stimulus 1 (first 20 FF inputs)

                % random C to I connectivity
                W_CtoI=rand(N_I,N_C); W_CtoI(W_CtoI<=1-p_IC)=0; W_CtoI(W_CtoI>1-p_IC)=w_CtoI_strength;

                N_I_perC=floor(N_I/N_C); % arbitrary number of inhibitory neurons per context, just to make sure that the dendrite is always blocked if the context is not "on"

                % FIXED I to D connectivity (each D receives input from exactly one random I subset)
                W_ItoD=zeros(N_RC*N_D,N_I);
                rand_C_per_D=randi(N_C,N_D*N_RC,1);

                for pp2=1:N_D*N_RC
                    W_ItoD(pp2,1+(rand_C_per_D(pp2)-1)*N_I_perC:rand_C_per_D(pp2)*N_I_perC)=ones(1,N_I_perC).*w_Ito_D_strength;
                end

                for kk5=1:nr_overlap_sims
                    Esyn_D_stim2=r_E_FF.*sum(W_FFtoD(:,1+overlap_steps*(kk5-1):20+overlap_steps*(kk5-1)),2); % choose here stimulus 2 (with potential overlap)

                    dendrite_LTDorLTP_gates_across_C_across_STIM=zeros(N_RC*N_D,2); % first colum is LTP in context 1, second is LTD or LTP in other contexts for stim 2
                    for kk3=1:N_C %loop through contexts

                        r_I=r_I_target-W_CtoI(:,kk3); % choose I cells that are active (hence 1-W) for given context (if there is one input, shut down I)

                        Isyn_D=W_ItoD*r_I;

                        delta_syn_D_stim1=Esyn_D_stim1-Isyn_D;
                        LTP_dend_stim1=delta_syn_D_stim1; LTP_dend_stim1(LTP_dend_stim1<LTP_thresh)=0; LTP_dend_stim1(LTP_dend_stim1>=LTP_thresh)=1;
                        LTD_dend_stim1=delta_syn_D_stim1; LTD_dend_stim1(LTD_dend_stim1<LTD_thresh)=0; LTD_dend_stim1(LTD_dend_stim1>=LTD_thresh)=1; LTD_dend_stim1(LTD_dend_stim1>=LTP_thresh)=0;

                        delta_syn_D_stim2=Esyn_D_stim2-Isyn_D;
                        LTP_dend_stim2=delta_syn_D_stim2; LTP_dend_stim2(LTP_dend_stim2<LTP_thresh)=0; LTP_dend_stim2(LTP_dend_stim2>=LTP_thresh)=1;
                        LTD_dend_stim2=delta_syn_D_stim2; LTD_dend_stim2(LTD_dend_stim2<LTD_thresh)=0; LTD_dend_stim2(LTD_dend_stim2>=LTD_thresh)=1; LTD_dend_stim2(LTD_dend_stim2>=LTP_thresh)=0;

                        if kk3==1 % have LTP D in context 1, and check in other contexts if another stimulus would lead to LTD or LTP
                            dendrite_LTDorLTP_gates_across_C_across_STIM(:,1)=dendrite_LTDorLTP_gates_across_C_across_STIM(:,1)+LTP_dend_stim1;
                        elseif kk3>1
                            dendrite_LTDorLTP_gates_across_C_across_STIM(:,2)=dendrite_LTDorLTP_gates_across_C_across_STIM(:,2)+LTP_dend_stim2+LTP_dend_stim2;
                        end

                    end

                    gating_selectivity_ratio_ACROSS_stimuli(kk4,kk5) = (sum(dendrite_LTDorLTP_gates_across_C_across_STIM(:,1)>0)-sum(dendrite_LTDorLTP_gates_across_C_across_STIM(:,1) ~= 0 & dendrite_LTDorLTP_gates_across_C_across_STIM(:,2) ~= 0))/sum(dendrite_LTDorLTP_gates_across_C_across_STIM(:,1)>0);
                end
            end
            gating_selectivity_ratio_ACROSS_stimuli_avg(uu1,uu3,:)=mean(gating_selectivity_ratio_ACROSS_stimuli,1,'omitmissing'); % fraction of D that do NOT change (LTD or LTP) out of the D part of an assemblyif another stimulus is shown in a different context
        end
    end


    % plot results
    multi_gated_N_ratio(isinf(multi_gated_N_ratio))=NaN;

    fig_fixed_ItoD = figure('Visible', 'off');
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

    x1_1 = C1_1(1, 2:1+C1_1(2,1)); y1_1 = C1_1(2, 2:1+C1_1(2,1));
    x1_2 = C1_2(1, 2:1+C1_2(2,1)); y1_2 = C1_2(2, 2:1+C1_2(2,1));

    ax2=subplot(3,3,4);
    hold on
    imagesc(gaussianFilter2D(squeeze(multi_gated_N_ratio(:,:)), kernelSize, sigma))
    plot(x1_1,y1_1,'k')
    plot(x1_2,y1_2,'k')
    hold off
    colormap(ax2,flipud(bone))
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


    ax3=subplot(3,3,7);
    hold on
    imagesc(gaussianFilter2D(squeeze(open_dendrites_avg(:,:)), kernelSize, sigma))
    plot(x1_1,y1_1,'k')
    plot(x1_2,y1_2,'k')
    hold off
    colorbar
    colormap(ax3,flipud(gray))
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

    for gg1=1:3
        subplot(3,3,2+3*(gg1-1))
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
    end

    overlap_fraction=[0,overlap_steps:overlap_steps:20]./20;

    subplot(3,3,3)
    hold on
    for pp1=1:3
        plot(overlap_fraction,1-flipud(squeeze(gating_selectivity_ratio_ACROSS_stimuli_avg(pp1,pp1,:))))
    end
    legend('1','2','3')
    hold off
    xlabel('Fract. of input overlap')
    ylabel('Fract. of forgetting')
    ylim([0,1])
    axis("square")

    sgtitle('Fixed I-to-D, random C-to-I')

    exportgraphics(fig_fixed_ItoD, fullfile(figures_dir, 'Fig_S4_C2C3_fixed_ItoD.pdf'), 'ContentType', 'vector');
    close(fig_fixed_ItoD)

    elseif connectivity_cases==4

    N_I_perC=10; % arbitrary number of inhibitory neurons per context, just to make sure that the dendrite is always blocked if the context is not "on"
    N_C_vec=2:1:40;
    N_D_vec=1:1:20;

    % initialize measures
    assembly_size_avg=zeros(length(N_C_vec),length(N_D_vec));
    multi_gated_N_ratio=zeros(length(N_C_vec),length(N_D_vec));


    for kk1=1:length(N_C_vec)
        for kk2=1:length(N_D_vec)

            N_C=N_C_vec(kk1);
            N_D=N_D_vec(kk2);

            % C to I connectivity (a distinct number of I receives input from the same C) [same for every repeat]
            W_CtoI=zeros(N_I_perC*N_C,N_C);
            for pp1=1:N_C
                W_CtoI(1+N_I_perC*(pp1-1):N_I_perC*pp1,pp1)=ones(N_I_perC,1).*w_CtoI_strength;
            end

            multi_gated_N=zeros(N_C,N_repeats);
            assembly_size=zeros(N_C,N_repeats);
            for kk4=1:N_repeats

                % random FF to D connectivity (for the two FF stimuli)
                W_FFtoD=rand(N_RC*N_D,N_FF); W_FFtoD(W_FFtoD<=1-p_FF_E)=0; W_FFtoD(W_FFtoD>1-p_FF_E)=1;
                Esyn_D_stim1=r_E_FF.*sum(W_FFtoD(:,1:20),2); % choose here stimulus 1 (first 20 FF inputs)

                % I to D connectivity (each D receives input from exactly one random I subset)
                W_ItoD=zeros(N_RC*N_D,N_I_perC*N_C);
                rand_C_per_D=randi(N_C,N_D*N_RC,1);

                for pp2=1:N_D*N_RC
                    W_ItoD(pp2,1+(rand_C_per_D(pp2)-1)*N_I_perC:rand_C_per_D(pp2)*N_I_perC)=ones(1,N_I_perC).*w_Ito_D_strength;
                end

                dendrite_LTP_gates_across_C_stim1=zeros(N_RC*N_D,1);
                for kk3=1:N_C %loop through contexts

                    r_I=r_I_target-W_CtoI(:,kk3); % choose I cells that are active (hence 1-W) for given context (if there is one input, shut down I)

                    Isyn_D=W_ItoD*r_I;

                    delta_syn_D_stim1=Esyn_D_stim1-Isyn_D;
                    LTP_dend_stim1=delta_syn_D_stim1; LTP_dend_stim1(LTP_dend_stim1<LTP_thresh)=0; LTP_dend_stim1(LTP_dend_stim1>=LTP_thresh)=1;
                    LTD_dend_stim1=delta_syn_D_stim1; LTD_dend_stim1(LTD_dend_stim1<LTD_thresh)=0; LTD_dend_stim1(LTD_dend_stim1>=LTD_thresh)=1; LTD_dend_stim1(LTD_dend_stim1>=LTP_thresh)=0;

                    dendrite_LTP_gates_across_C_stim1=dendrite_LTP_gates_across_C_stim1+LTP_dend_stim1;

                    assembly_neuron_stim1=movsum(LTP_dend_stim1,N_D,"Endpoints","discard"); assembly_neuron_stim1=assembly_neuron_stim1(1:N_D:end);
                    for_multi=movsum(LTD_dend_stim1,N_D,"Endpoints","discard"); for_multi=for_multi(1:N_D:end);

                    assembly_neuron_dummy_stim1=assembly_neuron_stim1;
                    assembly_neuron_dummy_stim1(assembly_neuron_dummy_stim1>1)=1;
                    assembly_size(kk3,kk4)=sum(assembly_neuron_dummy_stim1);  % neuron is part of an assembly if one or more dendrites is LTP

                    multi_gated_N(kk3,kk4)=sum(for_multi>1)./assembly_size(kk3,kk4);

                end
            end
            multi_gated_N_ratio(kk1,kk2)=mean(mean(multi_gated_N)); % mean across contexts & repeats
            assembly_size_avg(kk1,kk2)=mean(mean(assembly_size));
        end
    end

    % plot results

    multi_gated_N_ratio(isinf(multi_gated_N_ratio))=NaN;

    max_color_asse=80;
    min_color_asse=0;

    fig_1to1_connectivity = figure('Visible', 'off');
    ax1=subplot(2,2,1);
    hold on
    imagesc(gaussianFilter2D(assembly_size_avg, kernelSize, sigma))
    [C1_1,~] =contour(gaussianFilter2D(assembly_size_avg, kernelSize, sigma),[lower_bound_asse,lower_bound_asse],'k');
    [C1_2,~] =contour(gaussianFilter2D(assembly_size_avg, kernelSize, sigma),[upper_bound_asse,upper_bound_asse],'k');
    hold off
    colorbar
    colormap(ax1,flipud(pink))
    set(gca,'YDir','normal')
    axis square;
    clim([min_color_asse,max_color_asse])
    set(gca, 'XTick',5:5:length(N_D_vec), 'XTickLabel',N_D_vec(5:5:end))
    set(gca, 'YTick',9:10:length(N_C_vec), 'YTickLabel',N_C_vec(9:10:end))
    xlim(([0.5 0.5+length(N_D_vec)]))
    ylim(([0.5 0.5+length(N_C_vec)]))
    title('Assembly size')
    xlabel('# of Dendrites')
    ylabel('# of Contexts')

    x1_1 = C1_1(1, 2:1+C1_1(2,1)); y1_1 = C1_1(2, 2:1+C1_1(2,1));
    x1_2 = C1_2(1, 2:1+C1_2(2,1)); y1_2 = C1_2(2, 2:1+C1_2(2,1));

    ax2=subplot(2,2,2);
    hold on
    imagesc(gaussianFilter2D(multi_gated_N_ratio, kernelSize, sigma))
    plot(x1_1,y1_1,'k')
    plot(x1_2,y1_2,'k')
    hold off
    colorbar
    colormap(ax2,flipud(bone))
    set(gca,'YDir','normal')
    axis square;
    set(gca, 'XTick',5:5:length(N_D_vec), 'XTickLabel',N_D_vec(5:5:end))
    set(gca, 'YTick',9:10:length(N_C_vec), 'YTickLabel',N_C_vec(9:10:end))
    xlim(([0.5 0.5+length(N_D_vec)]))
    ylim(([0.5 0.5+length(N_C_vec)]))
    title('Multi gated neurons')
    xlabel('# of Dendrites')
    ylabel('# of Contexts')

    limits=12;

    subplot(2,2,3)
    hold on
    plot([0:limits],[0:limits].*LTP_thresh+LTP_thresh,'r')
    plot([0:limits],[0:limits].*LTD_thresh+LTD_thresh,'b')
    hold off
    xlim([0 3])
    ylim([0 limits])
    xlabel('# of I input')
    ylabel('# of E inputs')

    sgtitle('Model with 1-to-1 connectivity')

    exportgraphics(fig_1to1_connectivity, fullfile(figures_dir, 'Fig_S4_A1A3_1to1_connectivity.pdf'), 'ContentType', 'vector');
    close(fig_1to1_connectivity)


end

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
