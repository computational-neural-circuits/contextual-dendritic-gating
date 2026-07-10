% code for Figure S5A2-A4 of Onasch, Miehl et al., 2026 (Neuron)

clear all

repo_root = fileparts(fileparts(fileparts(fileparts(mfilename('fullpath')))));
results_dir = fullfile(repo_root, 'results', 'Fig_S5');
figures_dir = fullfile(repo_root, 'results', 'figures');
if ~exist(results_dir, 'dir')
    mkdir(results_dir)
end
if ~exist(figures_dir, 'dir')
    mkdir(figures_dir)
end

% parameters
N_C=6; % number of contexts
N_I=60; % number of inhibitory neurons

dt=0.1; % integration time step

tau_r_I=100; % in timesteps
end_time=10000000; % total number of time steps

r_I_target=10; % spontaneous firing rate of I cells
r_I=r_I_target.*ones(N_I,1);

p_IC=0.7; % C to I connection probability

% plasticity parameters
theta_IC=0.5; % target firing rate if context is ON
tau_w_I=10; % weight time scale

% heterosynaptic competition parameters
w_I_tot=r_I_target-theta_IC;
w_C_tot=(r_I_target-theta_IC)*N_I/N_C; % to ensure there is no bias of only one context over the others. (pre normalization)
tau_norm_I=10; % I normalization time scale
tau_norm_C=100; % C normalization time scale

times_switches=500; % context switches in time steps
r_C=zeros(N_C,1);
open_context=randi(N_C,1,1); % initial context that is "on"

W_CtoI_final_store=zeros(N_I,N_C,nr_total_runs);
W_CtoI_0_store=zeros(N_I,N_C,nr_total_runs);

for yy1=1:nr_total_runs
    yy1

    % random C to I connectivity
    W_CtoI_mask0=rand(N_I,N_C); W_CtoI_mask0(W_CtoI_mask0<=1-p_IC)=0; W_CtoI_mask0(W_CtoI_mask0>1-p_IC)=1;
    W_CtoI_mask=W_CtoI_mask0;

    W_CtoI_0=normrnd(8,4,N_I,N_C).*W_CtoI_mask0./sum(W_CtoI_mask0,2); %if I want to add heterogeneity to the weights
    W_CtoI_0(isnan(W_CtoI_0))=0; % NaN means there are no C to I connections at that neuron
    W_CtoI_0(W_CtoI_0<0)=0;
    W_CtoI=W_CtoI_0;


    counter=0;
    for tt=dt:dt:end_time
        counter=counter+1;

        if mod(round(tt/dt),round(times_switches/dt))==0
            r_C=zeros(N_C,1);
            open_context=randi(N_C,1,1);
            r_C(open_context)=1;
        end

        r_I=r_I+(-r_I+r_I_target-W_CtoI*r_C)/tau_r_I*dt;
        r_I(r_I<0)=0;

        % homeostatic plasticity
        W_CtoI=W_CtoI+W_CtoI_mask.*(repmat(r_C',N_I,1).*(repmat(r_I,1,N_C)-theta_IC))./tau_w_I*dt;

        % heterosynaptic competition
        W_CtoI=W_CtoI+W_CtoI_mask.*(w_I_tot-repmat(sum(W_CtoI,2),1,N_C))./tau_norm_I*dt;

        % presynaptic competition
        W_CtoI=W_CtoI+W_CtoI_mask.*(w_C_tot-repmat(sum(W_CtoI,1),N_I,1))./tau_norm_C*dt;


        W_CtoI(W_CtoI<0)=0; % lower bound

        update_mask=find(W_CtoI<0.1 & W_CtoI>0);

        if isempty(update_mask)==0
            W_CtoI(update_mask)=0;
            W_CtoI_mask(update_mask)=0;
        end
    end

    W_CtoI_final=W_CtoI;
    W_CtoI_final_store(:,:,yy1)=W_CtoI_final;
    W_CtoI_0_store(:,:,yy1)=W_CtoI_0;


    %%  next test forgetting before vs after plasticity

    N_RC=400;
    N_FF=40; % number of FF cells
    p_FF_E=0.081;
    N_D=6;

    % plasticity thresholds
    LTP_thresh=4; % E-I>=LTP_thresh -> LTP
    LTD_thresh=3; % LTD_thresh>=E-I<LTP_thresh -> LTD

    N_repeats_part2=20;
    overlap_steps=2; % step size of overlap (in total 2x 20 neurons)
    nr_overlap_sims=20/overlap_steps+1; % 20 is the stimulus size

    % case 1 (learned weights)
    gating_selectivity_ratio_ACROSS_stimuli_case1=zeros(N_repeats_part2,nr_overlap_sims);

    for kk4=1:N_repeats_part2
        % random FF to D connectivity (for the two FF stimuli)
        W_FFtoD=rand(N_RC*N_D,N_FF); W_FFtoD(W_FFtoD<=1-p_FF_E)=0; W_FFtoD(W_FFtoD>1-p_FF_E)=1;
        Esyn_D_stim1=sum(W_FFtoD(:,1:20),2); % choose here stimulus 1 (first 20 FF inputs)


        N_I_perC=floor(N_I/N_C); % arbitrary number of inhibitory neurons per context, just to make sure that the dendrite is always blocked if the context is not "on"

        % FIXED I to D connectivity (each D receives input from exactly one random I subset)
        W_ItoD=zeros(N_RC*N_D,N_I);
        rand_C_per_D=randi(N_C,N_D*N_RC,1);

        for pp3=1:N_C % get me inhibitory neurons for each context (so they can be matched to the dendrites)
            indx_per_C{pp3}=find(W_CtoI_final(:,pp3)>0);
        end

        for pp2=1:N_D*N_RC
            W_ItoD(pp2,indx_per_C{rand_C_per_D(pp2)})=ones(1,length(indx_per_C{rand_C_per_D(pp2)}));

        end

        for kk5=1:nr_overlap_sims
            Esyn_D_stim2=sum(W_FFtoD(:,1+overlap_steps*(kk5-1):20+overlap_steps*(kk5-1)),2); % choose here stimulus 2 (with potential overlap)

            dendrite_LTDorLTP_gates_across_C_across_STIM=zeros(N_RC*N_D,2); % first colum is LTP in context 1, second is LTD or LTP in other contexts for stim 2

            for kk3=1:N_C %loop through contexts

                W_CtoI_dummy=W_CtoI_final;
                W_CtoI_dummy(W_CtoI_dummy>0)=1;

                r_I=1-W_CtoI_dummy(:,kk3); % choose I cells that are active (hence 1-W) for given context (if there is one input, shut down I)

                Isyn_D(:,kk3)=W_ItoD*r_I;

                delta_syn_D_stim1(:,kk3)=Esyn_D_stim1-Isyn_D(:,kk3);
                LTP_dend_stim1=delta_syn_D_stim1(:,kk3); LTP_dend_stim1(LTP_dend_stim1<LTP_thresh)=0; LTP_dend_stim1(LTP_dend_stim1>=LTP_thresh)=1;
                LTD_dend_stim1=delta_syn_D_stim1(:,kk3); LTD_dend_stim1(LTD_dend_stim1<LTD_thresh)=0; LTD_dend_stim1(LTD_dend_stim1>=LTD_thresh)=1; LTD_dend_stim1(LTD_dend_stim1>=LTP_thresh)=0;

                delta_syn_D_stim2(:,kk3)=Esyn_D_stim2-Isyn_D(:,kk3);
                LTP_dend_stim2=delta_syn_D_stim2(:,kk3); LTP_dend_stim2(LTP_dend_stim2<LTP_thresh)=0; LTP_dend_stim2(LTP_dend_stim2>=LTP_thresh)=1;
                LTD_dend_stim2=delta_syn_D_stim2(:,kk3); LTD_dend_stim2(LTD_dend_stim2<LTD_thresh)=0; LTD_dend_stim2(LTD_dend_stim2>=LTD_thresh)=1; LTD_dend_stim2(LTD_dend_stim2>=LTP_thresh)=0;

                if kk3==1 % have LTP/D in context 1, and check in other contexts if another stimulus would lead to LTD or LTP
                    dendrite_LTDorLTP_gates_across_C_across_STIM(:,1)=dendrite_LTDorLTP_gates_across_C_across_STIM(:,1)+LTP_dend_stim1;
                elseif kk3>1
                    dendrite_LTDorLTP_gates_across_C_across_STIM(:,2)=dendrite_LTDorLTP_gates_across_C_across_STIM(:,2)+LTD_dend_stim2;
                end

            end

            gating_selectivity_ratio_ACROSS_stimuli_case1(kk4,kk5) = (sum(dendrite_LTDorLTP_gates_across_C_across_STIM(:,1)>0)-sum(dendrite_LTDorLTP_gates_across_C_across_STIM(:,1) ~= 0 & dendrite_LTDorLTP_gates_across_C_across_STIM(:,2) ~= 0))/sum(dendrite_LTDorLTP_gates_across_C_across_STIM(:,1)>0);
        end
    end

    % case 2 (initial weights)
    gating_selectivity_ratio_ACROSS_stimuli_case2=zeros(N_repeats_part2,nr_overlap_sims);

    for kk4=1:N_repeats_part2
        % random FF to D connectivity (for the two FF stimuli)
        W_FFtoD=rand(N_RC*N_D,N_FF); W_FFtoD(W_FFtoD<=1-p_FF_E)=0; W_FFtoD(W_FFtoD>1-p_FF_E)=1;
        Esyn_D_stim1=sum(W_FFtoD(:,1:20),2); % choose here stimulus 1 (first 20 FF inputs)

        N_I_perC=floor(N_I/N_C); % arbitrary number of inhibitory neurons per context, just to make sure that the dendrite is always blocked if the context is not "on"

        % FIXED I to D connectivity (each D receives input from exactly one random I subset)
        W_ItoD=zeros(N_RC*N_D,N_I);
        rand_C_per_D=randi(N_C,N_D*N_RC,1);

        for pp2=1:N_D*N_RC
            W_ItoD(pp2,1+(rand_C_per_D(pp2)-1)*N_I_perC:rand_C_per_D(pp2)*N_I_perC)=ones(1,N_I_perC);
        end

        for kk5=1:nr_overlap_sims
            Esyn_D_stim2=sum(W_FFtoD(:,1+overlap_steps*(kk5-1):20+overlap_steps*(kk5-1)),2); % choose here stimulus 2 (with potential overlap)

            dendrite_LTDorLTP_gates_across_C_across_STIM=zeros(N_RC*N_D,2); % first colum is LTP in context 1, second is LTD or LTP in other contexts for stim 2

            for kk3=1:N_C %loop through contexts

                W_CtoI_dummy=W_CtoI_0;
                W_CtoI_dummy(W_CtoI_dummy>0)=1;

                r_I=1-W_CtoI_dummy(:,kk3); % choose I cells that are active (hence 1-W) for given context (if there is one input, shut down I)

                Isyn_D(:,kk3)=W_ItoD*r_I;

                delta_syn_D_stim1(:,kk3)=Esyn_D_stim1-Isyn_D(:,kk3);
                LTP_dend_stim1=delta_syn_D_stim1(:,kk3); LTP_dend_stim1(LTP_dend_stim1<LTP_thresh)=0; LTP_dend_stim1(LTP_dend_stim1>=LTP_thresh)=1;
                LTD_dend_stim1=delta_syn_D_stim1(:,kk3); LTD_dend_stim1(LTD_dend_stim1<LTD_thresh)=0; LTD_dend_stim1(LTD_dend_stim1>=LTD_thresh)=1; LTD_dend_stim1(LTD_dend_stim1>=LTP_thresh)=0;

                delta_syn_D_stim2(:,kk3)=Esyn_D_stim2-Isyn_D(:,kk3);
                LTP_dend_stim2=delta_syn_D_stim2(:,kk3); LTP_dend_stim2(LTP_dend_stim2<LTP_thresh)=0; LTP_dend_stim2(LTP_dend_stim2>=LTP_thresh)=1;
                LTD_dend_stim2=delta_syn_D_stim2(:,kk3); LTD_dend_stim2(LTD_dend_stim2<LTD_thresh)=0; LTD_dend_stim2(LTD_dend_stim2>=LTD_thresh)=1; LTD_dend_stim2(LTD_dend_stim2>=LTP_thresh)=0;

                if kk3==1 % have LTP/D in context 1, and check in other contexts if another stimulus would lead to LTD or LTP
                    dendrite_LTDorLTP_gates_across_C_across_STIM(:,1)=dendrite_LTDorLTP_gates_across_C_across_STIM(:,1)+LTP_dend_stim1;
                elseif kk3>1
                    dendrite_LTDorLTP_gates_across_C_across_STIM(:,2)=dendrite_LTDorLTP_gates_across_C_across_STIM(:,2)+LTD_dend_stim2;%+LTP_dend_stim2;
                end

            end

            gating_selectivity_ratio_ACROSS_stimuli_case2(kk4,kk5) = (sum(dendrite_LTDorLTP_gates_across_C_across_STIM(:,1)>0)-sum(dendrite_LTDorLTP_gates_across_C_across_STIM(:,1) ~= 0 & dendrite_LTDorLTP_gates_across_C_across_STIM(:,2) ~= 0))/sum(dendrite_LTDorLTP_gates_across_C_across_STIM(:,1)>0);
        end
    end

    mean_plast_run_case1(yy1,:)=mean(gating_selectivity_ratio_ACROSS_stimuli_case1,1,'omitmissing');
    mean_plast_run_case2(yy1,:)=mean(gating_selectivity_ratio_ACROSS_stimuli_case2,1,'omitmissing');

end

%% plot results
overlap_fraction=[0,overlap_steps:overlap_steps:20]./20;

fig_S5_A2A3A4 = figure('Visible', 'off');
subplot(2,3,1)
hold on
plot(0:0.1:5,([0:0.1:5]-theta_IC)./tau_w_I,'k')
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

exportgraphics(fig_S5_A2A3A4, fullfile(figures_dir, 'Fig_S5_A2A3A4.pdf'), 'ContentType', 'vector');
close(fig_S5_A2A3A4)
