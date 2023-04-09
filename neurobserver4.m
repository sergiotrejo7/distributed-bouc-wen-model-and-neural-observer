%       system parameters
tau=1e-3; % NP 1e-4 % P 1e-2  %0.2e-2  %1e-3 para test  %1e-2 para el training del hg, 
%tau=1e-4; %1e-4 para el training del invmodel
B=1/tau;       
A=-1/tau;  
%       controller parameters
kappa1 = 1e4; % NP 1e4  % P 120 %5e3 %4e3 %4e4;
k1 = 25;
k2 = 25;
%       observer parameters 
theta = 100; % NP 100  % P 450  %450 for f10;  %100 for f1
%C_bar=[1;0];            %3 & 5 states, aqui controlas  la variable a ajustar
C_bar=[1,0];%;0,1];
S=[ 1/theta , B/theta^2 ; B/theta^2 , (2*B^2)/theta^3 ];
Gamma=[ 0 , -B ; 0 , 0 ];
G= S\C_bar';


%       training parameters
C_tilde = [1,0;0,1]; 
G_c = S\C_tilde;   %no restriction for neuroobserver G in R^(n x my), where z in R^n and y in R^my, n = 2, my=1
Gamma_c_tilde = Gamma - G_c*C_tilde; %5 gt 2 state  
eig(Gamma_c_tilde) %if both negatives is Hurwitz

% Gamma_c_bar
G_c = S\C_bar';   %no restriction for neuroobserver G in R^(n x my), where z in R^n and y in R^my, n = 2, my=1
Gamma_c_bar = Gamma - G_c*C_bar; %5 gt 2 state 
eig(Gamma_c_bar) %if both negatives is Hurwitz

%%
% Load datafile in to the workspace

% dataf30x1 = load('train_f30_x1.mat');
% dataf30x2 = load('train_f30_x2.mat');
% dataf30x3 = load('train_f30_x3.mat');
% dataf30x4 = load('train_f30_x4.mat');
% dataf30x5 = load('train_f30_x5.mat');
% dataf30x6 = load('train_f30_x6.mat');
% dataf10x1 = load('train_f10_x1.mat');
% dataf10x2 = load('train_f10_x2.mat');
% dataf10x3 = load('train_f10_x3.mat');
% dataf10x4 = load('train_f10_x4.mat');
% dataf10x5 = load('train_f10_x5.mat');
% dataf10x6 = load('train_f10_x6.mat');
% dataf1x1 = load('train_f1_x1.mat');
% dataf1x2 = load('train_f1_x2.mat');
% dataf1x3 = load('train_f1_x3.mat');
% dataf1x4 = load('train_f1_x4.mat');
% dataf1x5 = load('train_f1_x5.mat');
% dataf1x6 = load('train_f1_x6.mat');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% dataf30x1 = load('train_hg_f30_L5.mat');
% dataf30x2 = load('train_hg_f30_L75.mat');
% dataf30x3 = load('train_hg_f30_L10.mat'); 
% dataf30x4 = load('train_hg_f30_L125.mat');
% dataf30x5 = load('train_hg_f30_L15.mat');
% dataf30x6 = load('train_hg_f30_L175.mat'); 
% 
% dataf10x1 = load('train_hg_f10_L5.mat');
% dataf10x2 = load('train_hg_f10_L75.mat');
% dataf10x3 = load('train_hg_f10_L10.mat');
% dataf10x4 = load('train_hg_f10_L125.mat');
% dataf10x5 = load('train_hg_f10_L15.mat');
% dataf10x6 = load('train_hg_f10_L175.mat');
% 
% dataf1x1 = load('train_hg_f1_L5.mat');
% dataf1x2 = load('train_hg_f1_L75.mat');
% dataf1x3 = load('train_hg_f1_L10.mat');
% dataf1x4 = load('train_hg_f1_L125.mat');
% dataf1x5 = load('train_hg_f1_L15.mat');
% dataf1x6 = load('train_hg_f1_L175.mat');
% 
% trainf30 = cat(2,dataf30x1.train,dataf30x2.train,dataf30x3.train,dataf30x4.train,dataf30x5.train,dataf30x6.train);
% trainf10 = cat(2,dataf10x1.train,dataf10x2.train,dataf10x3.train,dataf10x4.train,dataf10x5.train,dataf10x6.train);
% %trainf1d = cat(2,dataf1x1.train(:,5001)*(L/5000)^2,dataf1x2.train(:,5001)*(L/7500)^2,dataf1x3.train(:,5001)*(L/10000)^2,dataf1x4.train(:,5001)*(L/12500)^2,dataf1x5.train(:,5001)*(L/15000)^2,dataf1x6.train(:,5001)*(L/17500)^2);
% trainf1 = cat(2,dataf1x1.train(:,1:10001),dataf1x2.train(:,1:10001),dataf1x3.train(:,1:10001),dataf1x4.train(:,1:10001),dataf1x5.train(:,1:10001),dataf1x6.train(:,1:10001));
% train = cat(2,trainf30,trainf10,trainf1);
% %train = cat(2,trainf30);
% tm = train(1,:);
% nt = length(tm);
% tm = 0.0001:0.0001:(nt*0.0001); 
% hist=zeros(2,1000);
% 
% v = train(2:3, :);      %v=[x;u]
% z = train(4:5, :);      %z=[y;h]
% z_d = z;        %z=[y_x;h_d]

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

dataf30x1 = load('train2_invmod_f30_x1.mat');
dataf30x2 = load('train2_invmod_f30_x2.mat');
dataf30x3 = load('train2_invmod_f30_x3.mat'); 
dataf30x4 = load('train2_invmod_f30_x4.mat');
dataf30x5 = load('train2_invmod_f30_x5.mat');
dataf30x6 = load('train2_invmod_f30_x6.mat'); 

dataf10x1 = load('train2_invmod_f10_x1.mat');
dataf10x2 = load('train2_invmod_f10_x2.mat');
dataf10x3 = load('train2_invmod_f10_x3.mat');
dataf10x4 = load('train2_invmod_f10_x4.mat');
dataf10x5 = load('train2_invmod_f10_x5.mat');
dataf10x6 = load('train2_invmod_f10_x6.mat');

dataf1x1 = load('train2_invmod_f1_x1.mat');
dataf1x2 = load('train2_invmod_f1_x2.mat');
dataf1x3 = load('train2_invmod_f1_x3.mat');
dataf1x4 = load('train2_invmod_f1_x4.mat');
dataf1x5 = load('train2_invmod_f1_x5.mat');
dataf1x6 = load('train2_invmod_f1_x6.mat');

%trainf30 = cat(2,dataf30x1.train,dataf30x2.train,dataf30x3.train,dataf30x4.train,dataf30x5.train,dataf30x6.train);  % x1= 5000, x2=7500, x3=10000, x4=12500, x5=15000, x6=17500
trainf30 = cat(2,dataf30x2.train,dataf30x5.train);  % x2=7500, x5+15000
% trainf10 = cat(2,dataf10x1.train,dataf10x2.train,dataf10x3.train,dataf10x4.train,dataf10x5.train,dataf10x6.train);  % x1= 5000, x2=7500, x3=10000, x4=12500, x5=15000, x6=17500
trainf10 = cat(2,dataf10x2.train,dataf10x5.train);  % x2=7500, x5+15000
% trainf1 = cat(2,dataf1x1.train,dataf1x2.train,dataf1x3.train,dataf1x4.train,dataf1x5.train,dataf1x6.train);  % x1= 5000, x2=7500, x3=10000, x4=12500, x5=15000, x6=17500
trainf1 = cat(2,dataf1x2.train,dataf1x5.train);  % x2=7500, x5+15000
train = cat(2,trainf30,trainf10,trainf1);
% train = cat(2,trainf10);
tm = train(1,:);
nt = length(tm);
tm = 0.0001:0.0001:(nt*0.0001); 
hist=zeros(2,1000);

v = train(2:3, :);      %v=[x;u]
y = train(4:5, :);      %y=[y_L;y_d]
h = train(6:7, :);      %h=[h;h_d]
z_L = [y(1,:);h(1,:)];        %z=[y_L;h]
z_d = [y(2,:);h(2,:)];        %z=[y_x;h_d]

clearvars data*
clearvars train*

dt = tm(2) - tm(1);

% Initialize Observer States
hz = zeros(2,nt);       %z_hat=[hy;hh]
hdz = zeros(2,nt);      %z_hat_dot=[hdy;hdh]
%%
% Initialize DRNN Weight Matrices
n=3;   %number of neurons
gt=3;   %groundtruth size

W = randn(2, n);       % (state size,number neurons)
V = randn(n, gt);       % (number neurons,groundtruth size)

epoch=0;
loss_sum=0;
loss_sum_ant=0;

loss1=[1;1];
loss_vec=[0;0];
loss_ant=[0;0];

loss_dot=[0;0];
loss_dot_ant=[0;0];

loss_ddot=[0;0];

%%                              TRAINING
% learning rates
n=2000;          %2
eta1=10*n;        %1*n               %learning rate 1 
rho1=eta1*0.001; %eta1*0.01*n       % small positive number, damping factor 1
eta2=10*n;       %25*n              %learning rate 2
rho2=eta2*0.001; %eta2*0.01*n      % small positive number, damping factor 2
count=0;

% parameters to learn two states
% l1= eta1*(Gamma_c_tilde'\C_tilde'*C_tilde);   %optimizes over 2 states
% l2= eta2*(Gamma_c_tilde'\C_tilde'*C_tilde);   %optimizes over 2 states
% K1=abs(l2)/2;
% rho1=(K1^2/norm(C_tilde));
% rho2=(1/norm(C_tilde));

% l1= eta1*C_tilde/Gamma_c_tilde';       %\C_tilde'*);   %optimizes over 2 states
% l2= eta2*C_tilde/Gamma_c_tilde';       %\C_tilde'*C_tilde);   %optimizes over 2 states

% parameters to learn one state
l1 = eta1*(Gamma_c_bar')\C_bar'*C_bar;   %(Gamma_c-T)*C_1*(C_bar*z) %5 gt 2 states
l2 = eta2*(Gamma_c_bar')\C_bar'*C_bar;   %optimizes over 1 state
% K1=abs(l2)/2;
% rho1=(K1^2/norm(C_bar));
% rho2=(1/norm(C_bar));

% l1= eta1*C_bar'*C_bar/Gamma_c_bar';     
% l2= eta2*C_bar'*C_bar/Gamma_c_bar';


%   =======================
%   ||   Training loop   ||
%   =======================
for idata = 1:5000 %loop for state vector-2    

    % Set Initial Observer States
    hz(:, 1) = z_d(:, 1);     % z_hat = z


    for i = 1:nt-1                                                          %loop for all the data along the time
        %===============
        % DRNN Observer|
        %===============
        z_tild = z_d(:, i) - hz(:, i);        %gt                          % state vector: z_tilde = z-z_hat
        hZ = [v(:,i);z_L(1,i)];%dv(2,i)];        %3 input measurable states     %x,u,y_L
        sigma = sig_fnc(V*hZ);                                              % (8x1) = (8x8)*(8x1)
        hg = W*sigma;                               %5 gt 2 states
        hdz(:, i+1) = Gamma*hz(:, i) + hg + G*(C_bar*z_tild);              %G_c*y_tild;        %             % (2x1) = (2x2)*(2x1) + (2x1) + (2x2)*(2x1)   ---> eq 4          
        hz(:, i+1) = hz(:, i) + dt*hdz(:, i);                               % x_hat(t+1) = hat_x(t) + dt*x_hat_dot(t)
%                                                                           x = v(1,i); u = v(2,i);y=C_tilde*z(:, i);d_d = (x/L)*(x/L)*d_p;hg = [A*y+B*d_d*u;W*sigma];
        %=====================                                               n=8, y=1, mu=8 m=neurons n=labels size
        % Update the weights |                                               z in R^n, y in R^m, mu in R^s=8, W in R^(sxm)=(8x8), V in R^(mxn)=(8x8) 
        %=====================                                               (8x2) = (8x2) + (1x1).*((8x8)*(8x1)*(2x1)') - (1x1)*(1x1)*(8x8)*(8x2));                                                                                %W = W + dt*(s*y_tild*sigma' - k*norm(y_tild)*s*W);     
%                                                                                S = -nu*Gamma_c'\C_tilde'*C_tilde;%         F = -nu*Gamma_c'\C_tilde'*C_tilde;%         W = W + dt*(S*z_tild*sigma' - k*norm(z_tild)*S*W);%         V = V + dt*((W*sigma*(1-sigma)')'*F*z_tild*sign(hZ)' - k*norm(z_tild)*V);        
        sigmoid=diag(sigma.^2);           
        I = eye(length(sigma));
        W = W + dt*(-(z_tild'*l1')'*sigma' - rho1*norm(C_bar*z_tild)*W);                    %3 gt 2 states
        V = V + dt*(-(z_tild'*l2'*W*(I-sigmoid))'*sign(hZ)' - rho2*norm(C_bar*z_tild)*V);   %3 gt 2 states
        %loss_tilde = loss_tilde + norm(C_tilde'*z_tild);     
        %loss_bar = loss_bar + norm(C_bar'*z_tild); 
        loss_vec = loss_vec + abs(C_tilde*z_tild);
    end

    epoch=epoch+1;
    idata
    epoch
    %loss_tilde=loss_tilde/(nt-1) % menor a 5.8928e-06 es bueno
    %loss_bar=loss_bar/(nt-1);
    %loss_tilde=loss_tilde/(nt-1);
    loss_vec = loss_vec/(nt-1);
    loss_sum = sum(loss_vec,'all')
    hist(1,idata) = loss_vec(1);
    hist(2,idata) = loss_vec(2);

% %=======================
% %     Validation loop
% %=======================    
%     for i = 1:nt-1
% %         ===============
% %         DRNN Observer
% %         ===============
% 
%         z_tild = z(:, i) - hz(:, i);                                        % state vector: z_tilde = z-z_hat
%         y_tild = C_tilde*z_tild;                                              % output vector: y_tilde = C'x(t)-C'x_hat(t)  
%         %hZ = [v(:, i);z(:, i);dv(:, i);dz(:, i)];                           % (8x1)
%         %hZ = [v(:,i);z(1,i);dv(2,i);dz(1, i)];
%         hZ = [v(:,i);z(1,i);dv(2,i)];
%         sigma = sig_fn(V*hZ);                                              % (8x1) = (8x8)*(8x1)
%         hg = W*sigma;                                                       % hg = (2x1) = (2x8)*(8x1)
%         hdz(:, i+1) = Gamma*hz(:, i) + hg + G*(C_bar*z_tild); %G_c*y_tild;                     % (2x1) = (2x2)*(2x1) + (2x1) + (2x2)*(2x1)   ---> eq 4          
%         hz(:, i+1) = hz(:, i) + dt*hdz(:, i);                               % x_hat(t+1) = hat_x(t) + dt*x_hat_dot(t)
%         
% %         ==============================
% %         No weight updation performed
% %         ==============================
%     end
% 
%     figure(1)
%     plot(tm, v(2,:))
%     title('Voltage', 'Interpreter', 'Latex')
%     xlabel('time s', 'Interpreter', 'Latex')
%     ylabel('$$u$$ [volts]', 'Interpreter', 'Latex')
%     grid minor
% 
%     %Plot states
%     figure(2)
%     %plot(tm, z(1, :), tm(1, 1:10:end), hz(1, 1:10:end), '.')
%     plot(tm, z(1, :), tm, hz(1, :))
%     title('Trajectory of State $$y$$, $$\hat{y}$$', 'Interpreter', 'Latex')
%     xlabel('time s', 'Interpreter', 'Latex')
%     ylabel('$$y$$, $$\hat{y}$$ [$$\mu$$m]', 'Interpreter', 'Latex')
%     grid minor
%     
%     figure(3)
%     %plot(tm, z(2, :), tm(1, 1:10:end), hz(2, 1:10:end), '.')
%     plot(tm, z(2, :), tm, hz(2, :))
%     %plot(tm, z(2, :), tm(1, :), hdz(2, :),'.')
%     title('Trajectory of State $$h_d$$, $$\hat{h}_d$$', 'Interpreter', 'Latex')
%     xlabel('time s', 'Interpreter', 'Latex')
%     ylabel('$$h_d$$, $$\hat{h}_d$$ [$$\mu$$m]', 'Interpreter', 'Latex')
%     grid minor
%     
    
%     if loss1 > loss_bar   %2e-06 para 8, 
%         loss1 = loss_bar;
%         W1=W;
%         V1=V; 
%     end
    loss_dot = loss_vec-loss_ant;%distingue si disminuye o aumenta
    loss_ddot = loss_dot-loss_dot_ant;%si es positivo tiende a dejar de entrenar, si negativo incrementa entrenamiento
    if all(loss_dot<0) % si loss sigue disminuyendo
        if all(loss_ddot < 0)
            fprintf('Disminuyen pero frenan ');
        elseif all(loss_ddot <= 0)
            fprintf('Disminuyen y aceleran ');
        else 
            fprintf('Disminuyen pero uno frena y otro acelera \n');
            loss_vec
        end      
        loss1 = loss_vec;
        W1=W;
        V1=V;
        count=0;
    elseif all(loss_dot>=0) % si loss empieza a aumentar                                
        if all(loss_ddot >= 0)
            count = count+1
            fprintf('Aumentan y aceleran \n');
        elseif all(loss_ddot < 0)
            count=count-1
            fprintf('Aumentan pero frenan \n');
        else
            fprintf('Aumentan pero uno frena y otro acelera \n');
        end
        if count > 50   
            break
        elseif count < -150   
            break    
        end
    else
        fprintf('Uno aumenta otro disminuye: \n');
        %if loss_sum<loss_sum_ant
        count = 0;
        loss_vec
    end
    loss_dot_ant=loss_dot;
    loss_ant=loss_vec;

end

fprintf('Training complete\n');
%%      LOADS THE BEST TRAINING
W=W1;
V=V1;
%%                              VALIDATION
nt = length(tm);
dt = tm(2) - tm(1);
%   =======================
%   ||  Validation loop
%   =======================
for vdata = 1:1   
    % Initialize Observer States
    hz_v = zeros(2,nt);       %x_hat=[hy;hh]
    hdz_v = zeros(2,nt);      %x_hat_dot=[hdy;hdh]

    % Set Initial Observer States
    hz_v(:, 1) = z_d(:, 1);     % y_hat = y
    
    for e = 1:nt-1
        z_tild_v = z_d(:, e) - hz_v(:, e);                                     % state vector: z_tilde = z-z_hat
        %y_tild_v = C_tilde*z_tild_v;                                              % output vector: y_tilde = C'x(t)-C'x_hat(t)  
        hZ = [v(:,e);z_L(1,e)];%dv(2,i)];        %3 measurable states     %x,u,y
        sigma = sig_fnc(V*hZ);                                              % (8x1) = (8x8)*(8x1)
        hg = W*sigma;                              %5 states                                 % hg = (2x1) = (2x8)*(8x1)      
        hdz_v(:, e+1) = Gamma*hz_v(:, e) + hg + G*(C_bar*z_tild_v);        %G_c*y_tild;%G*C_bar*z_tild;                       % (2x1) = (2x2)*(2x1) + (2x1) + (2x2)*(2x1)   ---> eq 4          
        hz_v(:, e+1) = hz_v(:, e) + dt*hdz_v(:, e);                               % x_hat(t+1) = hat_x(t) + dt*x_hat_dot(t)        
    end
end

fprintf('Validation complete\n');

figure(1)
loglog(0:1:(idata-2),hist(1,1:(idata-1)),0:1:(idata-2),hist(2,1:(idata-1)));
title('loss', 'Interpreter', 'Latex')
xlabel('epoch', 'Interpreter', 'Latex')
ylabel('$$loss$$', 'Interpreter', 'Latex')
grid minor

figure(4)
plot(tm, v(2,:))
title('Voltage', 'Interpreter', 'Latex')
xlabel('time s', 'Interpreter', 'Latex')
ylabel('$$u$$ [volts]', 'Interpreter', 'Latex')
grid minor

figure(5)
plot(tm, z_L(1,:))
title('Displacement $y_L$ at the tip', 'Interpreter', 'Latex')
xlabel('time s', 'Interpreter', 'Latex')
ylabel('$$y_L$$ [meters]', 'Interpreter', 'Latex')
grid minor

% Plot states/torques
figure(2)
%plot(tm, z(1, :), tm(1, 1:10:end), hz(1, 1:10:end), '.')
plot(tm, z_d(1, :), tm, hz_v(1, :))
title('Trajectory of State $$y$$, $$\hat{y}$$', 'Interpreter', 'Latex')
xlabel('time s', 'Interpreter', 'Latex')
%ylabel('$$y$$, $$\hat{y}$$ [$$\mu$$m]', 'Interpreter', 'Latex')
grid minor

figure(3)
%plot(tm, z(2, :), tm(1, 1:10:end), hz(2, 1:10:end), '.')
plot(tm, z_d(2, :), tm, hz_v(2, :))
%plot(tm, z(2, :), tm(1, :), hdz(2, :),'.')
title('Trajectory of State $$h_d$$, $$\hat{h}_d$$', 'Interpreter', 'Latex')
xlabel('time s', 'Interpreter', 'Latex')
%ylabel('$$h_d$$, $$\hat{h}_d$$ [$$\mu$$m]', 'Interpreter', 'Latex')
grid minor
%% save best learning
save('VWf10theta200x2x5kapp120k25tau1e-3.mat','V1','W1','n','eta1','rho1','eta2','rho2','loss1');
fprintf('best learning saved done');
%% recover best learning
training = matfile('VWf-all.mat'); %VWf30 chido
V1 = training.V1;
W1 = training.W1;
%% save incomplete learning    learning_paramsf-30.mat
save('learning_paramsf-all-invmod7.mat','n','eta1','rho1','eta2','rho2','V','W','V1','W1','hist','loss_vec','loss1')
fprintf('complete learning saved done');
%% recover incomplete learning
learning = matfile('learning_paramsf-all-invmod7.mat');
V = learning.V;
W = learning.W;
V1 = learning.V1;
W1 = learning.W1;
n = learning.n;
eta1 = learning.eta1;
rho1 = learning.rho1;
eta2 = learning.eta2;
rho2 = learning.rho2;
%% codigo para encontrar si una matriz es hurwitz
A = [1 1;1 2]; % Example matrix
%A = [-12 -20 -21; 2 1 3 ; 5 10 9]; % Example matrix
eig_H = eig(H);
flag = 0;
for i = 1:rank(H)
  if eig_H(i) <= 0 
  flag = 1;
  end
end
if flag == 1
  disp('the matrix is not positive definite, is hurwitz')
  else
  disp('the matrix is positive definite')
end
%% codigo para encontrar si una matriz es hurwitz
A = [1 1;1 2]; % Example matrix
%A = [-12 -20 -21; 2 1 3 ; 5 10 9]; % Example matrix
eig_H = eig(Gamma);
flag = 0;
for i = 1:rank(Gamma)
  if eig_H(i) <= 0 
  flag = 1;
  end
end
if flag == 1
  disp('the matrix is not positive definite, is hurwitz')
  else
  disp('the matrix is positive definite')
end
%% observability
O=[C_tilde;C_tilde*Gamma_c]
rank(O)
%%                                  SIMULATION PARAMETERS
theta= 450;  %100 for f1
%C_bar=[1;0];
C_bar=[1,0;0,1];
S=[1/theta,B/theta^2;B/theta^2,(2*B^2)/theta^3];

%       controller parameters
k1=7;  %3 para f1 %7 para f10 %40; para f1 %0.6; para f30 %                %20;%1.0;     este cambia la amplitud de la oscilacion de u_dot     
k2=7; %5 para f1 %5 para f10 %30; para f1 %1; para f30 %                % 2e9;    %15;%1.0;          
kappa1 =120;%1.5e3; %9e3 para f1 %1e3 para f10 %1e4 para f30           %    %500; arriba de 100 es casi perfecta la señal u, generando una u mas suave, arriba de 5e3 oscila mucho 'y'
%% epsilon
figure('PaperSize',[5.1 2.1],'DefaultAxesFontSize',12)
%subplot(1,2,1,'FontSize',11); 
hold on
% plot(out.epsilon_y_nn.Time,out.epsilon_y_nn.Data,'k:','LineWidth',1.5);
% plot(out.epsilon_h_nn.Time,out.epsilon_h_nn.Data,'r:','LineWidth',1);
% plot(out.epsilon_y_nn1.Time,out.epsilon_y_nn1.Data,'g--','LineWidth',1);
% plot(out.epsilon_h_nn1.Time,out.epsilon_h_nn1.Data,'b--','LineWidth',1);
plot(out.epsilon_y_nn3.Time,out.epsilon_y_nn3.Data,'k-','LineWidth',1.5);
plot(out.epsilon_h_nn3.Time,out.epsilon_h_nn3.Data,'r-','LineWidth',1.5);
plot(out.epsilon_y_nn4.Time,out.epsilon_y_nn4.Data,'g--','LineWidth',1.5);
plot(out.epsilon_h_nn4.Time,out.epsilon_h_nn4.Data,'b--','LineWidth',1.5);
% plot(out.epsilon_y_nn4.Time,out.epsilon_y_nn4.Data,'k:','LineWidth',1);
% plot(out.epsilon_h_nn4.Time,out.epsilon_h_nn4.Data,'r:','LineWidth',1);
% plot(out.epsilon_y_nn5.Time,out.epsilon_y_nn5.Data,'g--','LineWidth',1);
% plot(out.epsilon_h_nn5.Time,out.epsilon_h_nn5.Data,'b--','LineWidth',1);
hold off
ylabel('$\epsilon$','interpreter', 'latex');  
xlabel('time[sec]','interpreter', 'latex'); 
grid on;   
h = legend('$\epsilon_{y} = \hat{y} - y$ for $x=15$', '$\epsilon_{h} = \hat{h} - h$ for $x=15$','$\epsilon_{y} = \hat{y} - y$ for $x=7.5$', '$\epsilon_{h} = \hat{h} - h$ for $x=7.5$');
set(h, 'Interpreter', 'latex');
%% epsilon
figure('PaperSize',[5.1 2.1],'DefaultAxesFontSize',12)
%subplot(1,2,1,'FontSize',11); 
hold on
plot(out.epsilon_y_nn.Time,out.epsilon_y_nn.Data,'k:','LineWidth',1.5);
plot(out.epsilon_h_nn.Time,out.epsilon_h_nn.Data,'r:','LineWidth',1.5);
plot(out.epsilon_y_nn1.Time,out.epsilon_y_nn1.Data,'g--','LineWidth',1.5);
plot(out.epsilon_h_nn1.Time,out.epsilon_h_nn1.Data,'b--','LineWidth',1.5);
hold off
ylabel('$\epsilon$','interpreter', 'latex');  
xlabel('time[sec]','interpreter', 'latex'); 
grid on;   
h = legend('$\epsilon_{y} = \hat{y} - y$ for $x=15$', '$\epsilon_{h} = \hat{h} - h$ for $x=15$','$\epsilon_{y} = \hat{y} - y$ for $x=7.5$', '$\epsilon_{h} = \hat{h} - h$ for $x=7.5$');
set(h, 'Interpreter', 'latex');

%% x-x_d
figure('PaperSize',[5.1 2.1],'DefaultAxesFontSize',12)
%subplot(1,2,2,'FontSize',11); 
hold on
% plot(out.y_ref_nn.Time,out.y_ref_nn.Data,'k:','LineWidth',1.2); 
% plot(out.y_nn.Time,out.y_nn.Data,'g--','LineWidth',1.2);
% plot(out.y_ref_nn1.Time,out.y_ref_nn1.Data,'r:','LineWidth',1.2); 
% plot(out.y_nn1.Time,out.y_nn1.Data,'b--','LineWidth',1);
plot(out.y_ref_nn3.Time,out.y_ref_nn3.Data,'k:','LineWidth',1.5); 
plot(out.y_d_nn3.Time,out.y_d_nn3.Data(1,:),'g--','LineWidth',1.5);
plot(out.y_ref_nn4.Time,out.y_ref_nn4.Data,'r:','LineWidth',1.5); 
plot(out.y_d_nn4.Time,out.y_d_nn4.Data(1,:),'b--','LineWidth',1.5);
% plot(out.y_ref_nn4.Time,out.y_ref_nn4.Data,'k:','LineWidth',1.2); 
% plot(out.y_nn4.Time,out.y_nn4.Data,'g--','LineWidth',1.2);
% plot(out.y_ref_nn5.Time,out.y_ref_nn5.Data,'r:','LineWidth',1.2); 
% plot(out.y_nn5.Time,out.y_nn5.Data,'b--','LineWidth',1);
hold off
ylabel('$y, y_r$','interpreter', 'latex');  
xlabel('time[sec]','interpreter', 'latex'); 
grid on; 
r = legend('$y_{r}$ for $x=15$', '$y$ for $x=15$','$y_{r}$ for $x=7.5$', '$y$ for $x=7.5$');
set(r, 'Interpreter', 'latex');

%% h vs hd  ***********************
figure('PaperSize',[5.1 2.1],'DefaultAxesFontSize',12);
%subplot(1,2,2,'FontSize',11); 
%plot(out.h.Time,[out.h.Data,out.h_ref.Data],'k','LineWidth',1.5);
hold on
% plot(out.h_nn.Time,out.h_nn.Data,'k:','LineWidth',1.5);
% plot(out.h_ref_nn.Time,out.h_ref_nn.Data,'g-.','LineWidth',1.5);
% plot(out.h_nn1.Time,out.h_nn1.Data,'r-','LineWidth',1.5);
% plot(out.h_ref_nn1.Time,out.h_ref_nn1.Data,'b--','LineWidth',1.5);

plot(out.h_nn3.Time,out.h_nn3.Data,'k:','LineWidth',1.5);
plot(out.h_ref_nn3.Time,out.h_ref_nn3.Data,'g-.','LineWidth',1.5);
plot(out.h_nn4.Time,out.h_nn4.Data,'r-','LineWidth',1.5);
plot(out.h_ref_nn4.Time,out.h_ref_nn4.Data,'b--','LineWidth',1.5);

% plot(out.h_nn4.Time,out.h_nn4.Data,'k:','LineWidth',1.5);
% plot(out.h_ref_nn4.Time,out.h_ref_nn4.Data,'g-.','LineWidth',1.5);
% plot(out.h_nn5.Time,out.h_nn5.Data,'r-','LineWidth',1.5);
% plot(out.h_ref_nn5.Time,out.h_ref_nn5.Data,'b--','LineWidth',1.5);
hold off
ylabel('$h, h_r$','interpreter', 'latex');  
xlabel('time[sec]','interpreter', 'latex'); 
grid on; 
r = legend('$h$ for $x=7.5$', '$h_r$ for $x=7.5$','$h$ for $x=15$', '$h_r$ for $x=15$');
set(r, 'Interpreter', 'latex');

%% udot, u vs time *********************
figure('PaperSize',[5.1 4.2],'DefaultAxesFontSize',12)
subplot(2,1,1,'FontSize',11); 
hold on
% plot(out.u_dot_nn.Time,out.u_dot_nn.Data,'b-','LineWidth',1.5); 
% plot(out.u_dot_nn1.Time,out.u_dot_nn1.Data,'r-.','LineWidth',1.5); 
plot(out.u_dot_nn3.Time,out.u_dot_nn3.Data,'b-','LineWidth',1.5); 
plot(out.u_dot_nn4.Time,out.u_dot_nn4.Data,'r-.','LineWidth',1.5); 
% plot(out.u_dot_nn4.Time,out.u_dot_nn4.Data,'b-','LineWidth',1.5); 
% plot(out.u_dot_nn5.Time,out.u_dot_nn5.Data,'r-.','LineWidth',1.5); 
hold off
ylabel('$\dot{u}$','interpreter', 'latex');  
xlabel('time[sec]','interpreter', 'latex'); 
grid on; 
r = legend('$\dot{u}$ for $x=7.5$','$\dot{u}$ for $x=15$');
set(r, 'Interpreter', 'latex');

subplot(2,1,2,'FontSize',11); 
hold on
% plot(out.u_nn.Time,out.u_nn.Data,'b-','LineWidth',1.5); 
% plot(out.u_nn1.Time,out.u_nn1.Data,'r-.','LineWidth',1.5); 
plot(out.u_nn2.Time,out.u_nn2.Data,'b-','LineWidth',1.5); 
plot(out.u_nn3.Time,out.u_nn3.Data,'r-.','LineWidth',1.5); 
% plot(out.u_nn4.Time,out.u_nn4.Data,'b-','LineWidth',1.5); 
% plot(out.u_nn5.Time,out.u_nn5.Data,'r-.','LineWidth',1.5); 
hold off
ylabel('$u$','interpreter', 'latex');  
xlabel('time[sec]','interpreter', 'latex'); 
grid on; 
r = legend('$u$ for $x=7.5$','$u$ for $x=15$');
set(r, 'Interpreter', 'latex');
%% (y_d,upsilon) OPEN-CLOSE LOOP ******************
figure('PaperSize',[5.2 4.2],'DefaultAxesFontSize',12)
subplot(2,1,1,'FontSize',11);
title('Open-loop response maps');
hold on
plot(out.u_nn3.Data,out.y_d_nn3.Data(1,:), 'b--','LineWidth',1.5);
plot(out.u_nn4.Data,out.y_d_nn4.Data(1,:), 'r-','LineWidth',1);
ylabel('$y$','interpreter', 'latex');  
xlabel('$u$','interpreter', 'latex'); 
grid on;   
hold off
h = legend('$(u,y)$ for $x=7.5$','$(u,y)$ for $x=15$');
set(h, 'Location','northwest','Interpreter', 'latex');

subplot(2,1,2,'FontSize',11);
title('Closed-loop response maps');
hold on
plot(out.y_ref_nn3.Data,out.y_d_nn3.Data(1,:), 'b--','LineWidth',1.5);
plot(out.y_ref_nn4.Data,out.y_d_nn4.Data(1,:), 'r-','LineWidth',1);
ylabel('$y$','interpreter', 'latex');  
xlabel('$y_r$','interpreter', 'latex'); 
grid on;   
hold off
h = legend('$(y_r,y)$ for $x=7.5$','$(y_r,y)$ for $x=15$');
set(h, 'Location','northwest','Interpreter', 'latex');
%% (u,y=Cx) CLOSED LOOP
figure('DefaultAxesFontSize',12)
plot(out.u_numeric.signals.values,out.x_numeric.signals.values, 'b','LineWidth',1.5);
ylabel('$y$','interpreter', 'latex');  
xlabel('$u$','interpreter', 'latex'); 
grid on;   
h = legend('Closed-loop response map $(u,y)$');
set(h, 'Interpreter', 'latex');
%% (udot,y=Cx) CLOSED LOOP
figure('DefaultAxesFontSize',12)
plot(out.udot_numeric.signals.values,out.x_numeric.signals.values, 'b','LineWidth',1.5);
ylabel('$y$','interpreter', 'latex');  
xlabel('$\dot{u}$','interpreter', 'latex'); 
grid on;   
h = legend('Closed-loop response map $(\dot{u},y)$');
set(h, 'Interpreter', 'latex');


%% (y_d,upsilon) OPEN LOOP
figure('DefaultAxesFontSize',12)
plot(out.u.Data,out.upsilon.Data, 'b','LineWidth',1.5);
ylabel('$\upsilon$','interpreter', 'latex');  
xlabel('$y_d$','interpreter', 'latex'); 
grid on;   
h = legend('Open-loop response map $(y(L),\upsilon)$');
set(h, 'Interpreter', 'latex');
%% (y_d,y=Cx) OPEN LOOP Vs  (y_d,y) CLOSED LOOP
figure('DefaultAxesFontSize',12)
subplot(2,1,1,'FontSize',11); 
plot(out.xref_numeric.signals.values,out.x_numeric_open.signals.values, 'b','LineWidth',1.5);
ylabel('$y$','interpreter', 'latex');  
xlabel('$y_d$','interpreter', 'latex'); 
grid on;   
h = legend('Open loop response map $(y_d,y)$');
set(h, 'Interpreter', 'latex');

subplot(2,1,2,'FontSize',11); 
plot(out.x_d_numeric.signals.values, out.x_numeric.signals.values, 'b','LineWidth',1.5);
ylabel('$y$','interpreter', 'latex');  
xlabel('$y_d$','interpreter', 'latex'); 
grid on;   
h = legend('Closed-loop response map $(y_d,y)$');
set(h, 'Interpreter', 'latex');

%% (y_d,y) CLOSE LOOP all freqs******************
figure('PaperSize',[5.5 3.5],'DefaultAxesFontSize',12)
title('Closed-loop response maps');
hold on
plot(out.y_ref_nn.Data,out.y_nn.Data, 'b-','LineWidth',1);
plot(out.y_ref_nn1.Data,out.y_nn1.Data, 'b:','LineWidth',1);
plot(out.y_ref_nn2.Data,out.y_nn2.Data, 'r-','LineWidth',1);
plot(out.y_ref_nn3.Data,out.y_nn3.Data, 'r:','LineWidth',1);
plot(out.y_ref_nn4.Data,out.y_nn4.Data, 'k-','LineWidth',1);
plot(out.y_ref_nn5.Data,out.y_nn5.Data, 'k:','LineWidth',1);
ylabel('$y$','interpreter', 'latex');  
xlabel('$y_r$','interpreter', 'latex'); 
grid on;   
hold off
h = legend('$(y_r,y)$ for $x=7.5$, $f=1Hz$','$(y_r,y)$ for $x=15$, $f=1Hz$','$(y_r,y)$ for $x=7.5$, $f=10Hz$','$(y_r,y)$ for $x=15$, $f=10Hz$','$(y_r,y)$ for $x=7.5$, $f=30Hz$','$(y_r,y)$ for $x=15$, $f=30Hz$');
set(h, 'Location','northwest','Interpreter', 'latex');

%%
function [sigvec] = sig_fnc(x_vec)
    sigvec = zeros(length(x_vec), 1);
    for i = 1:length(x_vec)
        sigvec(i, 1) = 2/(1 + exp(-2*x_vec(i, 1))) - 1;
    end
end