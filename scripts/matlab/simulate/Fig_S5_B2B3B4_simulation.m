% code for Figure S5B2-B4 of Onasch, Miehl et al., 2026 (Neuron)

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
N_RC=400; % number of recurrent neurons
N_C=6; % number of contexts
N_D=6; % number of dendrites
N_I=60; % number of I cells
tau_r_I=100; % timescale of I in timesteps

r_I_target=10; % spontaneous firing rate of I cells
r_I=ones(N_I,1)*r_I_target;

dt=0.1; % integration time step
end_time=100000; % total number of time steps

tau_r_D=dt; % timescale of D in timesteps (instantaneous)
r_D=zeros(N_RC*N_D,1);

p_DI=0.065; % I to D connection probability

% fixed C-to-I connectivity
N_I_perC=floor(N_I/N_C);
W_CtoI=zeros(N_I,N_C);
for pp1=1:N_C
    W_CtoI(1+N_I_perC*(pp1-1):N_I_perC*pp1,pp1)=ones(N_I_perC,1)*r_I_target; % if a C is on, then this shuts I off completely
end

% random I to D connectivity
w_Ito_D_strength=2;
W_ItoD_mask0=rand(N_RC*N_D,N_I); W_ItoD_mask0(W_ItoD_mask0<=1-p_DI)=0; W_ItoD_mask0(W_ItoD_mask0>1-p_DI)=1;
W_ItoD_mask=W_ItoD_mask0;
W_ItoD_0=W_ItoD_mask.*(0.1*(rand(N_RC*N_D,N_I)-0.5)+w_Ito_D_strength);
W_ItoD=W_ItoD_0;

w_max_ItoD=ceil(mean(sum(W_ItoD,2)));


% plasticity parameters. Use anti-Hebbian: r_post*(theta-r_pre)
theta_D=r_I_target; % LTD/LTP split for firing of I cells
tau_w_D=500; % plasticity timescale

% heterosynaptic competition parameters
w_D_tot=w_max_ItoD; % this is the value that the plasticity rule will reach in the current setup
tau_norm_D=10; % normalization timescale

% context switches
times_switches=500;
r_C=zeros(N_C,1);
open_context=randi(N_C,1,1); % initial context that is "on"

% FF excitation
p_FF_E=0.081; % connection probability
N_FF=20; % number of FF inputs
W_FFtoD=rand(N_RC*N_D,N_RC); W_FFtoD(W_FFtoD<=1-p_FF_E)=0; W_FFtoD(W_FFtoD>1-p_FF_E)=1;

rand_stim=randi(N_RC,N_FF,1); % choose random subset of neurons being active
Esyn_D=sum(W_FFtoD(:,rand_stim),2);

stim_switches=2000; % time steps at which the stimulus switches


for tt=dt:dt:end_time


    if mod(round(tt/dt),round(times_switches/dt))==0 % choose new context that is "on"
        r_C=zeros(N_C,1);
        open_context=randi(N_C,1,1);
        r_C(open_context)=1;
    end

    if mod(round(tt/dt),round(stim_switches/dt))==0 %  choose new stimulus
        rand_stim=randi(N_RC,N_FF,1); % choose random subset of neurons being active
        Esyn_D=sum(W_FFtoD(:,rand_stim),2);
    end


    r_I=r_I+(-r_I+r_I_target-W_CtoI*r_C)/tau_r_I*dt;
    r_I(r_I<0)=0; r_I(r_I>10)=10; % maximum inhibitory rate

    r_D=r_D+(-r_D+w_Ito_D_strength.*r_I_target.*Esyn_D-W_ItoD*r_I)/tau_r_D*dt;
    r_D(r_D<0)=0;

    % homeostatic plasticity
    W_ItoD=W_ItoD+W_ItoD_mask.*(repmat(r_D,1,N_I).*(theta_D-repmat(r_I',N_RC*N_D,1)))./tau_w_D*dt;

    % heterosynaptic competition
    W_ItoD=W_ItoD+W_ItoD_mask.*(w_D_tot-repmat(sum(W_ItoD,2),1,N_I))./tau_norm_D*dt;

    W_ItoD(W_ItoD<0)=0; % lower bound

    update_mask=find(W_ItoD<0.1 & W_ItoD>0);

    if isempty(update_mask)==0
        W_ItoD(update_mask)=0;
        W_ItoD_mask(update_mask)=0;
    end
end

W_ItoD_final=W_ItoD;

%%  next test forgetting before vs after plasticity

N_FF=40; % number of FF cells

% plasticity thresholds
LTP_thresh=w_Ito_D_strength.*r_I_target.*4; % E-I>=LTP_thresh -> LTP
LTD_thresh=w_Ito_D_strength.*r_I_target.*3; % LTD_thresh>=E-I<LTP_thresh -> LTD

N_repeats_part2=20;
overlap_steps=2; % step size of overlap (in total 2x 20 neurons)
nr_overlap_sims=20/overlap_steps+1; % 20 is the stimulus size

% case 1 (learned weights)
gating_selectivity_ratio_ACROSS_stimuli_case1=zeros(N_repeats_part2,nr_overlap_sims);

for kk4=1:N_repeats_part2

    % random FF to D connectivity (for the two FF stimuli)
    W_FFtoD=rand(N_RC*N_D,N_FF); W_FFtoD(W_FFtoD<=1-p_FF_E)=0; W_FFtoD(W_FFtoD>1-p_FF_E)=1;
    Esyn_D_stim1=sum(W_FFtoD(:,1:20),2); % choose here stimulus 1 (first 20 FF inputs)

    for kk5=1:nr_overlap_sims
        Esyn_D_stim2=sum(W_FFtoD(:,1+overlap_steps*(kk5-1):20+overlap_steps*(kk5-1)),2); % choose here stimulus 2 (with potential overlap)

        dendrite_LTDorLTP_gates_across_C_across_STIM=zeros(N_RC*N_D,2); % first colum is LTP in context 1, second is LTD or LTP in other contexts for stim 2

        for kk3=1:N_C %loop through contexts

            r_I=r_I_target-W_CtoI(:,kk3);

            Isyn_D(:,kk3)=W_ItoD_final*r_I;

            delta_syn_D_stim1(:,kk3)=w_Ito_D_strength.*r_I_target.*Esyn_D_stim1-Isyn_D(:,kk3);
            LTP_dend_stim1=delta_syn_D_stim1(:,kk3); LTP_dend_stim1(LTP_dend_stim1<LTP_thresh)=0; LTP_dend_stim1(LTP_dend_stim1>=LTP_thresh)=1;
            LTD_dend_stim1=delta_syn_D_stim1(:,kk3); LTD_dend_stim1(LTD_dend_stim1<LTD_thresh)=0; LTD_dend_stim1(LTD_dend_stim1>=LTD_thresh)=1; LTD_dend_stim1(LTD_dend_stim1>=LTP_thresh)=0;

            delta_syn_D_stim2(:,kk3)=w_Ito_D_strength.*r_I_target.*Esyn_D_stim2-Isyn_D(:,kk3);
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


    for kk5=1:nr_overlap_sims
        Esyn_D_stim2=sum(W_FFtoD(:,1+overlap_steps*(kk5-1):20+overlap_steps*(kk5-1)),2); % choose here stimulus 2 (with potential overlap)

        dendrite_LTDorLTP_gates_across_C_across_STIM=zeros(N_RC*N_D,2); % first colum is LTP in context 1, second is LTD or LTP in other contexts for stim 2

        for kk3=1:N_C %loop through contexts

            r_I=r_I_target-W_CtoI(:,kk3);

            Isyn_D(:,kk3)=W_ItoD_mask0*r_I;

            delta_syn_D_stim1(:,kk3)=w_Ito_D_strength.*r_I_target.*Esyn_D_stim1-Isyn_D(:,kk3);
            LTP_dend_stim1=delta_syn_D_stim1(:,kk3); LTP_dend_stim1(LTP_dend_stim1<LTP_thresh)=0; LTP_dend_stim1(LTP_dend_stim1>=LTP_thresh)=1;
            LTD_dend_stim1=delta_syn_D_stim1(:,kk3); LTD_dend_stim1(LTD_dend_stim1<LTD_thresh)=0; LTD_dend_stim1(LTD_dend_stim1>=LTD_thresh)=1; LTD_dend_stim1(LTD_dend_stim1>=LTP_thresh)=0;

            delta_syn_D_stim2(:,kk3)=w_Ito_D_strength.*r_I_target.*Esyn_D_stim2-Isyn_D(:,kk3);
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


overlap_fraction=[0,overlap_steps:overlap_steps:20]./20;


%% plot results
fig_S5_B2B3B4 = figure('Visible', 'off');
subplot(2,3,1)
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

subplot(2,3,2)
hold on
histogram(selectivity_per_dendrite_init,'FaceColor', 'y','Normalization','probability')
histogram(selectivity_per_dendrite_final,'FaceColor', 'k','Normalization','probability')
hold off
legend('before learning','after learning')
xlabel('Number of contextual inputs per D')
ylabel('Fraction')

subplot(2,3,3)
hold on
plot(overlap_fraction,1-fliplr(mean(gating_selectivity_ratio_ACROSS_stimuli_case2,1,'omitmissing')),'k')
plot(overlap_fraction,1-fliplr(mean(gating_selectivity_ratio_ACROSS_stimuli_case1,1,'omitmissing')),'b')
hold off
legend('before learning','after learning')
xlabel('Fract. of input overlap')
ylabel('Fract. of forgetting')
ylim([-0.1,1])
axis("square")

exportgraphics(fig_S5_B2B3B4, fullfile(figures_dir, 'Fig_S5_B2B3B4.pdf'), 'ContentType', 'vector');
close(fig_S5_B2B3B4)
