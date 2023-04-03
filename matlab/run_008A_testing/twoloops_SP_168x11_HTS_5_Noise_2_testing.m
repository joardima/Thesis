% build a Reduced Order Model ROM based on sensor data for C-Town
% It builds a Sparse Identification of Nonlinear Dynamics (SINDy)

clc;
clear;
close all
fclose all;
format short g;

load('wntr_sp_1848_dt_5_Nu_2.mat');

H_cv = H';
D_cv = D(:,1:6)*1000;
D_cv = D_cv';
Qp_cv = Q*1000;
Qp_cv = Qp_cv';
T = double(T);
intervals = (length(T) - 1)/11;

lambda = 20;
node = 2;
link = 2; %7

%%Read EPANET report

[resout, TT, node_ids, link_ids] = read_EPANET_rpt('wdn_reports/Twoloops_SP_168_HTS_5_Noise_2.txt',168,5);
H = resout(1:end,8:14);
H = H';
D = resout(1:end,1:6)/3.6; 
D = D';
Q = resout(1:end,22:29)/3.6;
Q = Q';
T = TT';

clear [resout,ans,TT];


%% Build the matrices
T24  = mod(T(:,1:end),24*3600);

%H    = curr_states{2}; 
node_sensors = find((max(H(:,1:end),[],2)-min(H(:,1:end),[],2)) > 0.1); %1:size(H,1);
Hx   = H(node_sensors, 1:end-1); 
dHx  = H(node_sensors, 2:end); 
[H_names] = create_namesall_2('H',size(H,1)); 
H_names = H_names(1,node_sensors); 

%D    = curr_states{2}; 
node_sensors = find((max(D(:,1:end),[],2)-min(D(:,1:end),[],2)) >= 0.001); %1:size(D,1);
Dx   = D(node_sensors, 1:end-1); 
dDx  = D(node_sensors, 2:end); 
[D_names] = create_namesall_2('D',size(D,1)); 
D_names = D_names(1,node_sensors);

%Q   = curr_states{5}; 
link_sensors = find((max(abs(Q(:,1:end)),[],2) - min(abs(Q(:,1:end)),[],2)) > 0.001); % 1:size(Q,1); %[1,2,4,6:8,10,12:15]; %1:15;
Qpx  = Q(link_sensors, 1:end-1); 
dQpx = Q(link_sensors, 2:end); 
Qp_names = create_namesall_2('Q',size(Q,1)); 
Qp_names = Qp_names(1,link_sensors); 

% second set based on cuadratic terms of flows abs(Q)*Q
%Qpx2  = abs(Qpx).*Qpx;
%dQpx2 = abs(dQpx).*dQpx;
%Qp2_names = create_namesall_2('Q^2',size(Q,1)); 
%Qp2_names = Qp2_names(1,link_sensors); 

%% Create SYSTEM MATRICES
X     = [ Hx;  Dx;  Qpx]';
Xdot  = [dHx; dDx; dQpx]';

nVars = size(X,2);
varNames = {H_names{:}, D_names{:}, Qp_names{:}};

%% Pool Data  (i.e., build library of nonlinear time series)
tic;
polyorder = 1;
Theta = poolData_2(X,nVars,polyorder); % corrected combinations generator
m     = size(Theta,2);
toc

%% Compute Sparse regression: sequential least squares and then the estimate
% Xdot_hat
tic;
%lambda = 0.01;      % lambda is our sparsification knob.
nreg = 20;
Xi     = sparsifyDynamics(Theta,Xdot,lambda,nreg);
toc

%% Make the estimate (hat) of Xdot
tic;
Xdot_hat = Theta*Xi;
toc

err = (Xdot - Xdot_hat);
err2 = sqrt(sum(err.*err)/(length(T)-1));

%% Show how good or bad is the adjustment for each variable

ERR2_cv = zeros(0)
ERR2_cv = vertcat(ERR2_cv,err2);
for i=1:10
    
    Hx_cv = H_cv(1:length(H_names),intervals*i+1:intervals*(i+1)+1);
    Hx_cv = Hx_cv(:,1:end-1);   
    dHx_cv = H_cv(1:length(H_names),intervals*i+1:intervals*(i+1)+1);
    dHx_cv = dHx_cv(:,2:end);

    Dx_cv = D_cv(1:length(D_names),intervals*i+1:intervals*(i+1)+1);
    Dx_cv = Dx_cv(:,1:end-1);  
    dDx_cv = D_cv(1:length(D_names),intervals*i+1:intervals*(i+1)+1);
    dDx_cv = dDx_cv(:,2:end);
    
    Qpx_cv = Qp_cv(1:length(Qp_names),intervals*i+1:intervals*(i+1)+1);
    Qpx_cv = Qpx_cv(:,1:end-1);
    dQpx_cv = Qp_cv(1:length(Qp_names),intervals*i+1:intervals*(i+1)+1);
    dQpx_cv = dQpx_cv(:,2:end);
    
    X_cv     = [ Hx_cv;  Dx_cv;  Qpx_cv]';
    Xdot_cv  = [dHx_cv; dDx_cv; dQpx_cv]';
    
    nVars = size(X_cv,2);
    varNames = {H_names{:}, D_names{:}, Qp_names{:}};

    polyorder = 1;
    Theta_cv = poolData_2(X_cv,nVars,polyorder);

    Xdot_hat_cv = Theta_cv*Xi;

    err_cv = (Xdot_cv - Xdot_hat_cv);
    err2_cv = sqrt(sum(err_cv.*err_cv)/(length(T)-1));
    
    ERR2_cv = vertcat(ERR2_cv,err2_cv);
end

%% plot results of error estimation
H_index = [1 length(H_names)];
D_index = [H_index(2)+1 H_index(2)+length(D_names)];
Q_index = [D_index(2)+1 D_index(2)+length(Qp_names)];

make_it_tight = true;
%              h = subtightplot(m, n, p,  [gap_ver, gap_hor],[marg_vert_down, marg_vert_up],[marg_hor_L, marg_hor_R], varargin)
subplot = @(m,n,p) subtightplot (m, n, p, [0.15 0.15], [0.120, 0.010], [0.05 0.010]);
if not(make_it_tight)
  clear subplot;
end

% Plotting options
text_size    = 11;
text_weight  = 'Bold';
% figopt.FontName   = 'Lucida Sans';
% figopt.FontName   = 'Palatino Linotype';
% figopt.FontName   = 'Verdana';
figopt.FontName   = 'Cambria';
figopt.FontSize   = text_size;
figopt.FontWeight = text_weight;
%figopt.Xlim       = [0 168];
%figopt.XTick      = 0:12:168;
%figopt.XTicklabel = {0:12:168};
  
figure(1)
set(gcf,'Color',[1 1 1])
set(gcf,'Position',[70 300 1070 400]);
subplot(2,1,1)
bar(err2(:,H_index(1):H_index(2)));
%bar(sort(err2(:,1:377),'descend'));
set(gca,figopt);
xlabel('(A) Nodes (Pressure [m])');
ylabel('RMSE');
grid on;

subplot(2,1,2)
bar(err2(:,Q_index(1):Q_index(2))); 
%bar(sort(err2(:,378:744),'descend'));
set(gca,figopt);
xlabel('(B) Links (Flows [l/s])');
ylabel('RMSE');
grid on;

%% Display results
%xlist  = poolDataLIST_2(varNames,Xi,nVars,polyorder); % corrected combinations generator

%% Plot the results for specific variable

make_it_tight = true;
%              h = subtightplot(m, n, p,  [gap_ver, gap_hor],[marg_vert_down, marg_vert_up],[marg_hor_L, marg_hor_R], varargin)
subplot = @(m,n,p) subtightplot (m, n, p, [0.07 0.05], [0.120, 0.060], [0.04 0.011]);
if not(make_it_tight)
  clear subplot;
end

flag_T24 = 1;

my_xtick = 0:48:T(end)/3600;

% Level tanks = 371:377
% 244 is a good result
% 616 is a good result
% 390 is a good result
sel_var = node; %616 %300 %244; %73 % 735 %371; %73; %232 %616; %390; %616; %300; % % 816 % 4 is a tank

xmin = min([min(Xdot(:,sel_var)),min(Xdot_hat(:,sel_var))]);
xmax = max([max(Xdot(:,sel_var)),max(Xdot_hat(:,sel_var))]);

curr_err = err(:,sel_var);
xmax_err = max(abs([min(curr_err) max(curr_err)]));

figure(2)
set(gcf,'Color',[1 1 1])
set(gcf,'Position',[70 300 1070 400]);


subplot(2,4,1:3)
if (flag_T24 == 0)
  plot(T(2:end)/3600,Xdot(:,sel_var),'-','linewidth',3); hold on;
  plot(T(2:end)/3600,Xdot_hat(:,sel_var),'--','linewidth',3); hold off;
else
  plot(mod(T(2:end)/3600,24),Xdot(:,sel_var),'s','linewidth',1); hold on;
  plot(mod(T(2:end)/3600,24),Xdot_hat(:,sel_var),'.','linewidth',1); hold off;
end
set(gca,figopt);

if flag_T24 == 0
  set(gca,'xlim',[-5 T(end)/3600+6]);
  set(gca,'xtick',my_xtick);
else
  set(gca,'xlim',[-0.5 24.5]);
  set(gca,'xtick',0:2:24);
end
xlabel('Time [hr]');
legend('Observed','Simulated','location','NE');
title(['Results for ',varNames{sel_var}]);
grid on

subplot(2,4,5:7)
if flag_T24 == 0
  stem(T(2:end)/3600,err(:,sel_var),'-','linewidth',1,'marker','None');
else
  stem(mod(T(2:end)/3600,24),err(:,sel_var),'.','linewidth',1,'marker','None');
end
set(gca,figopt);
if flag_T24 == 0
  set(gca,'xlim',[-5 T(end)/3600+6]);
  set(gca,'xtick',my_xtick);
else
  set(gca,'xlim',[-0.5 24.5]);
  set(gca,'xtick',0:2:24);
end
set(gca,'ylim',[-xmax_err xmax_err]);
xlabel('Time [hr]');
legend('error','location','SE');
%title(['Obtained Model Reduction for ',varNames{sel_var}]);
grid on

subplot(2,4,4)
plot(Xdot(:,sel_var),Xdot_hat(:,sel_var),'.'); hold on;
set(gca,figopt);
ylabel('Simulated')
xlabel('Observed')
plot([xmin xmax],[xmin xmax],'k--'); hold off
axis([xmin xmax xmin xmax],'square');
grid on

subplot(2,4,8)
[nn,nedges] = histcounts(err(:,sel_var),10); %'BinMethod','Sturges','Normalization','probability');
%barh(nn); %

xhist = linspace(-xmax_err-eps,xmax_err+eps,100);
barh(xhist,hist(curr_err,xhist));
set(gca,figopt);
set(gca,'ylim',[-xmax_err xmax_err]);
ylabel('error')
xlabel('Frequency')
%axis([xmin xmax xmin xmax],'square');
grid on