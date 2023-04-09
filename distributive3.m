%       nonlinear hysteresis parameters
alpha=0.9e-6;
beta=0.008;
gamma=0.008; 
L=20000;
d_p=1.6e-6;

%      linear system parameters     
tau=1e-4;       %0.001;  entre mas pequeño u_dot oscila más pero u es mejor
A=-1/tau;
B=1/tau;      
C = 1;       
C_bar=[1,0];
% C_bar=[1,0]; 

%       controller parameters
k1=7;          %0.25 for f1  %for f30 k1=1;%for f10 k1=2;  %3 menos ruido                %1;      %20;%1.0;     este cambia la amplitud de la oscilacion de u_dot     
k2=7;          %1 for f1   %for f30 k2=1;%for f10 k2=3; %2 menos ruido                %2e6 2e9;    %15;%1.0;          
kappa1 =4e4;    %4e4 para f30 %for f10  4e3; for f1 4e4    %500; arriba de 100 es casi perfecta la señal u, generando una u mas suave, arriba de 5e3 oscila mucho 'y'

%       observer parameters   k1=2 k2=2 kappa1=4e4 el mas general con overshoot, k1=3 k2=3 kappa1=3e3 
theta = 100;             %100;   
S = [ 1/theta , B/theta^2 ; B/theta^2 , (2*B^2)/theta^3 ];
G = S\C_bar';
Gamma = [0 , -B ; 0 , 0 ];

%       input parameters
amp=80e-6;    
fase=2*pi/10;   % 
angvel=(2*pi)*1; %k2=1e6 -> 0.1Hz, k2=2e9->10Hz, k2=2e9->30Hz

%       initial conditions
y_i=-3e-6;
h_i=2e-6;

%       experiments
e=[5000,7500,10000,12500,15000,17500];
x1=e(2);
x2=e(5);
%   Gamma/S = Gamma*S^-1
%   S \ Gamma = S^-1 * Gamma

eig(Gamma - G*C_bar)
%%
A = Gamma - G*C_bar; % Example matrix
eig_A = eig(A);
flag = 0;
for i = 1:rank(A)
  if eig_A(i) <= 0 
  flag = 1;
  end
end
if flag == 1
  disp('the matrix is negative definite, ie. Hurwitz')
  else
  disp('the matrix is positive definite')
end
%%
A2=[-1/tau,0;0,-1/tau]; 
B2=[1/tau,0;0,1/tau];      
C2 = [1,0,0];        %Y=[y(x);y(L)] 
G2=[[0;0],-B2;0 0 0];
O2=[C2;C2*G2;C2*G2*G2];
%%
%       nonlinear hysteresis parameters
alpha=0.9e-6;   
beta=0.008;   
gamma=0.008;  
L=20000;  
d_p=1.6e-6;

%      linear system parameters     
tau=1e-4;   %0.001;  entre mas pequeño u_dot oscila más pero u es mejor
%A=[-1/tau,0;0,-1/tau]; 
a11=-1/tau;
a22=-1/tau;
A=[a11,0;0,a22];
%B=[1/tau,0;0,1/tau];  
b11=1/tau;
b22=1/tau;
B=[b11,0;0,b22];
C = [1,0];        %Y=[y(x);y(L)] 
C_bar=[1,0,0];  

%       initial conditions
y_i=-3e-6;
h_i=2e-6;

%       controller parameters
k1=7;%0.25 for f1  %for f30 k1=1;%for f10 k1=2;  %3 menos ruido                %1;      %20;%1.0;     este cambia la amplitud de la oscilacion de u_dot     
k2=7;%1 for f1   %for f30 k2=1;%for f10 k2=3; %2 menos ruido                %2e6 2e9;    %15;%1.0;          
kappa1 =4e3; %4e4 para f30 %for f10  4e3; for f1 4e4    %500; arriba de 100 es casi perfecta la señal u, generando una u mas suave, arriba de 5e3 oscila mucho 'y'

%       observer parameters   k1=2 k2=2 kappa1=4e4 el mas general con overshoot, k1=3 k2=3 kappa1=3e3 
theta= 100;             %100;   

s11 = 1/theta;
s21 = b11/theta^2;
s31 = (b11*b22)/theta^3;
s12 = b11/theta^2;
s22 = (2*b11^2)/theta^3;
s32 = (3*b11^2*b22)/theta^4;
s13 = (b11*b22)/theta^3;
s23 = (3*b11^2*b22)/theta^4;
s33 = (6*b11^2*b22^2)/theta^5;

S=[s11,s12,s13;s21,s22,s23;s31,s32,s33];

G= S\C_bar';
Gamma=[[0;0],-B;0 0 0];%[0 , -B ; 0 , 0 ];

%       input parameters
amp=80e-6;    
fase=2*pi/10;   % 
angvel=(2*pi)*30; %k2=1e6 -> 0.1Hz, k2=2e9->10Hz, k2=2e9->30Hz


%       experiments
e=[5000,7500,10000,12500,15000,17500];
x1=e(2);
x2=e(5);
%   Gamma/S = Gamma*S^-1
%   S \ Gamma = S^-1 * Gamma
%%   theta*S + Gamma^T*S + S*Gamma + C^T*C=0        2x2
syms s11 s12 s13 s21 s22 s23 s31 s32 s33 theta b11 b12 b21 b22
%B=[b11,b12;b21,b22];
B=[b11,0;0,b22];
assume(B,'positive')
S=[s11,s12,s13;s21,s22,s23;s31,s32,s33];
Gamma=[[0;0],-B;0 0 0];
C_bar=[1,0,0]; 

eqn = theta*S + Gamma'*S + S*Gamma - C_bar'*C_bar == 0;
s = solve(eqn,S)
%%   theta*S + Gamma^T*S + S*Gamma + C^T*C=0        3x3
syms s11 s12 s21 s22 s31 s32 theta B
assume(B,'positive')
S=[s11,s12;s21,s22];
Gamma=[0,-B;0 0];
C_bar=[1,0]; 

eqn = theta*S + Gamma'*S + S*Gamma - C_bar'*C_bar == 0;
s = solve(eqn,S)
%% epsilon
figure('PaperSize',[5.1 2.1],'DefaultAxesFontSize',12)
%subplot(1,2,1,'FontSize',11); 
hold on
% plot(out.epsilon_y.Time,out.epsilon_y.Data,'k:','LineWidth',1.5);
% plot(out.epsilon_h.Time,out.epsilon_h.Data,'r:','LineWidth',1);
% plot(out.epsilon_y1.Time,out.epsilon_y1.Data,'g--','LineWidth',1);
% plot(out.epsilon_h1.Time,out.epsilon_h1.Data,'b--','LineWidth',1);
plot(out.epsilon_y2.Time,out.epsilon_y2.Data,'k-','LineWidth',1.5);
plot(out.epsilon_h2.Time,out.epsilon_h2.Data,'r-','LineWidth',1.5);
plot(out.epsilon_y3.Time,out.epsilon_y3.Data,'g--','LineWidth',1.5);
plot(out.epsilon_h3.Time,out.epsilon_h3.Data,'b--','LineWidth',1.5);
% plot(out.epsilon_y4.Time,out.epsilon_y4.Data,'k:','LineWidth',1);
% plot(out.epsilon_h4.Time,out.epsilon_h4.Data,'r:','LineWidth',1);
% plot(out.epsilon_y5.Time,out.epsilon_y5.Data,'g--','LineWidth',1);
% plot(out.epsilon_h5.Time,out.epsilon_h5.Data,'b--','LineWidth',1);
hold off
ylabel('$\epsilon$','interpreter', 'latex');  
xlabel('time[sec]','interpreter', 'latex'); 
grid on;   
h = legend('$\epsilon_{y} = \hat{y} - y$ for $x=15$', '$\epsilon_{h} = \hat{h} - h$ for $x=15$','$\epsilon_{y} = \hat{y} - y$ for $x=7.5$', '$\epsilon_{h} = \hat{h} - h$ for $x=7.5$');
set(h, 'Interpreter', 'latex');
%% epsilon all freqs
figure('PaperSize',[5.1 2.1],'DefaultAxesFontSize',12)
%subplot(1,2,1,'FontSize',11); 
hold on
plot(out.epsilon_y.Time,out.epsilon_y.Data,'k:','LineWidth',1.5);
plot(out.epsilon_h.Time,out.epsilon_h.Data,'r:','LineWidth',1.5);
plot(out.epsilon_y1.Time,out.epsilon_y1.Data,'g--','LineWidth',1.5);
plot(out.epsilon_h1.Time,out.epsilon_h1.Data,'b--','LineWidth',1.5);
plot(out.epsilon_y2.Time,out.epsilon_y2.Data,'k:','LineWidth',1.5);
plot(out.epsilon_h2.Time,out.epsilon_h2.Data,'r:','LineWidth',1.5);
plot(out.epsilon_y3.Time,out.epsilon_y3.Data,'g--','LineWidth',1.5);
plot(out.epsilon_h3.Time,out.epsilon_h3.Data,'b--','LineWidth',1.5);
plot(out.epsilon_y4.Time,out.epsilon_y4.Data,'k:','LineWidth',1.5);
plot(out.epsilon_h4.Time,out.epsilon_h4.Data,'r:','LineWidth',1.5);
plot(out.epsilon_y5.Time,out.epsilon_y5.Data,'g--','LineWidth',1.5);
plot(out.epsilon_h5.Time,out.epsilon_h5.Data,'b--','LineWidth',1.5);
hold off
ylabel('$\epsilon$','interpreter', 'latex');  
xlabel('time[sec]','interpreter', 'latex'); 
grid on;   
h = legend('$\epsilon_{y} = \hat{y} - y$ for $x=15$', ...
            '$\epsilon_{h} = \hat{h} - h$ for $x=15$');
%             '$\epsilon_{y} = \hat{y} - y$ for $x=7.5$', ...
%             '$\epsilon_{h} = \hat{h} - h$ for $x=7.5$', ...
%             '$\epsilon_{y} = \hat{y} - y$ for $x=15$', ...
%             '$\epsilon_{h} = \hat{h} - h$ for $x=15$', ...
%             '$\epsilon_{y} = \hat{y} - y$ for $x=7.5$', ...
%             '$\epsilon_{h} = \hat{h} - h$ for $x=7.5$',...
%             '$\epsilon_{y} = \hat{y} - y$ for $x=15$', ...
%             '$\epsilon_{h} = \hat{h} - h$ for $x=15$', ...
%             '$\epsilon_{y} = \hat{y} - y$ for $x=7.5$', ...
%             '$\epsilon_{h} = \hat{h} - h$ for $x=7.5$');
set(h, 'Interpreter', 'latex');
%% x-x_d
figure('PaperSize',[5.2 2.2],'DefaultAxesFontSize',12)
%subplot(1,2,2,'FontSize',11); 
hold on
% plot(out.y_ref.Time,out.y_ref.Data,'k:','LineWidth',1.5); 
% plot(out.y.Time,out.y.Data,'g--','LineWidth',1.2);
% plot(out.y_ref1.Time,out.y_ref1.Data,'r:','LineWidth',1.2); 
% plot(out.y1.Time,out.y1.Data,'b--','LineWidth',1);
plot(out.y_ref2.Time,out.y_ref2.Data,'k:','LineWidth',1.5); 
plot(out.y2.Time,out.y2.Data,'g--','LineWidth',1.5);
plot(out.y_ref3.Time,out.y_ref3.Data,'r:','LineWidth',1.5); 
plot(out.y3.Time,out.y3.Data,'b--','LineWidth',1.5);
% plot(out.y_ref4.Time,out.y_ref4.Data,'k:','LineWidth',1.5); 
% plot(out.y4.Time,out.y4.Data,'g--','LineWidth',1.2);
% plot(out.y_ref5.Time,out.y_ref5.Data,'r:','LineWidth',1.2); 
% plot(out.y5.Time,out.y5.Data,'b--','LineWidth',1);
hold off
ylabel('$y, y_r$','interpreter', 'latex');  
xlabel('time[sec]','interpreter', 'latex'); 
grid on; 
r = legend('$y_{r}$ for $x=15$', '$y$ for $x=15$','$y_{r}$ for $x=7.5$', '$y$ for $x=7.5$');
set(r, 'Interpreter', 'latex');

%% h vs hd  ***********************
figure('PaperSize',[5.2 2.2],'DefaultAxesFontSize',12);
%subplot(1,2,2,'FontSize',11); 
%plot(out.h.Time,[out.h.Data,out.h_ref.Data],'k','LineWidth',1.5);
hold on
% plot(out.h.Time,out.h.Data,'k:','LineWidth',1.5);
% plot(out.h_ref.Time,out.h_ref.Data,'g-.','LineWidth',1.5);
% plot(out.h1.Time,out.h1.Data,'r-','LineWidth',1.5);
% plot(out.h_ref1.Time,out.h_ref1.Data,'b--','LineWidth',1.5);
plot(out.h2.Time,out.h2.Data,'k:','LineWidth',1.5);
plot(out.h_ref2.Time,out.h_ref2.Data,'g-.','LineWidth',1.5);
plot(out.h3.Time,out.h3.Data,'r-','LineWidth',1.5);
plot(out.h_ref3.Time,out.h_ref3.Data,'b--','LineWidth',1.5);
% plot(out.h4.Time,out.h4.Data,'k:','LineWidth',1.5);
% plot(out.h_ref4.Time,out.h_ref4.Data,'g-.','LineWidth',1.5);
% plot(out.h5.Time,out.h5.Data,'r-','LineWidth',1.5);
% plot(out.h_ref5.Time,out.h_ref5.Data,'b--','LineWidth',1.5);
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
% plot(out.u_dot.Time,out.u_dot.Data,'b-','LineWidth',1.5); 
% plot(out.u_dot1.Time,out.u_dot1.Data,'r-.','LineWidth',1.5); 
plot(out.u_dot2.Time,out.u_dot2.Data,'b-','LineWidth',1.5); 
plot(out.u_dot3.Time,out.u_dot3.Data,'r-.','LineWidth',1.5); 
% plot(out.u_dot4.Time,out.u_dot4.Data,'b-','LineWidth',1.5); 
% plot(out.u_dot5.Time,out.u_dot5.Data,'r-.','LineWidth',1.5); 
hold off
ylabel('$\dot{u}$','interpreter', 'latex');  
xlabel('time[sec]','interpreter', 'latex'); 
grid on; 
r = legend('$\dot{u}$ for $x=7.5$','$\dot{u}$ for $x=15$');
set(r, 'Interpreter', 'latex');

subplot(2,1,2,'FontSize',11); 
hold on
% plot(out.u.Time,out.u.Data,'b-','LineWidth',1.5); 
% plot(out.u1.Time,out.u1.Data,'r-.','LineWidth',1.5); 
plot(out.u2.Time,out.u2.Data,'b-','LineWidth',1.5); 
plot(out.u3.Time,out.u3.Data,'r-.','LineWidth',1.5); 
% plot(out.u4.Time,out.u4.Data,'b-','LineWidth',1.5); 
% plot(out.u5.Time,out.u5.Data,'r-.','LineWidth',1.5); 
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
plot(out.u2.Data,out.y2.Data, 'b--','LineWidth',1);
plot(out.u3.Data,out.y3.Data, 'r-','LineWidth',1);
ylabel('$y$','interpreter', 'latex');  
xlabel('$u$','interpreter', 'latex'); 
grid on;   
hold off
h = legend('$(u,y)$ for $x=7.5$','$(u,y)$ for $x=15$');
set(h, 'Location','northwest','Interpreter', 'latex');

subplot(2,1,2,'FontSize',11);
title('Closed-loop response maps');
hold on
plot(out.y_ref2.Data,out.y2.Data, 'b--','LineWidth',1);
plot(out.y_ref3.Data,out.y3.Data, 'r-','LineWidth',1);
ylabel('$y$','interpreter', 'latex');  
xlabel('$y_r$','interpreter', 'latex'); 
grid on;   
hold off
h = legend('$(y_r,y)$ for $x=7.5$','$(y_r,y)$ for $x=15$');
set(h, 'Location','northwest','Interpreter', 'latex');
%% (y_d,y) CLOSE LOOP all freqs******************
figure('PaperSize',[5.5 3.5],'DefaultAxesFontSize',12)
title('Closed-loop response maps');
hold on
plot(out.y_ref.Data,out.y.Data, 'b-','LineWidth',1);
plot(out.y_ref1.Data,out.y1.Data, 'b:','LineWidth',1);
plot(out.y_ref2.Data,out.y2.Data, 'r-','LineWidth',1);
plot(out.y_ref3.Data,out.y3.Data, 'r:','LineWidth',1);
plot(out.y_ref4.Data,out.y4.Data, 'k-','LineWidth',1);
plot(out.y_ref5.Data,out.y5.Data, 'k:','LineWidth',1);
ylabel('$y$','interpreter', 'latex');  
xlabel('$y_r$','interpreter', 'latex'); 
grid on;   
hold off
h = legend('$(y_r,y)$ for $x=7.5$, $f=1Hz$','$(y_r,y)$ for $x=15$, $f=1Hz$','$(y_r,y)$ for $x=7.5$, $f=10Hz$','$(y_r,y)$ for $x=15$, $f=10Hz$','$(y_r,y)$ for $x=7.5$, $f=30Hz$','$(y_r,y)$ for $x=15$, $f=30Hz$');
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
%% tase
% alpha=0.9e-6;
% beta=0.008;
% gamma=0.008;
% L=15000;            % 
% d_p=1.6e-6; %16;    
% C = 1;         
% C_bar=[1,0];   
% tau=1e-3;%0.001;  entre mas pequeño u_dot oscila más pero u es mejor
% B=1/tau;       
% A=-1/tau;  
% theta=100;
% k1= 20;
% k2=15;
% amp= 80e-6;
% fase=0;
% alfa=1;
% angvel=(2*pi)*10; %k2=1e6 -> 0.1Hz, k2=2e9->10Hz, k2=2e9->30Hz
% freq=2*pi*10;
% 
% % initial conditions
% Y=0;%-3e-6;
% h=0;%2e-6;
% kappa1 =4e2;%4e3;    %500; arriba de 100 es casi perfecta la señal u, generando una u mas suave
% 
% S=[1/theta,B/theta^2;
%    B/theta^2,(2*B^2)/theta^3];
% 
% Gamma=[0, -B;
%        0,  0];
%%
q1 = theta*S+Gamma'*S+S*Gamma-C_bar'*C_bar
q2 = ((alfa-k1)/k2)^2
%%
h_d_hat=2;
h_d_ref=1;
syms c1 c2 k1 k2 alfa betta rho sigma
e=h_d_hat-h_d_ref;
%assume(c1>=abs(h_d_hat));
%assume(c2*abs(e)>=abs(h_d_ref_dot));
%assume(sigma,'positive');
%assume(betta,'positive');
%assume(rho,'positive');

eqn1 = c1 >= abs(h_d_hat);
eqn2 = c2*abs(e)>=abs(h_d_ef_dot);
eqn3 = sigma >=0;
eqn4 = betta >=0;
eqn5 = rho >=0;
eqn6 = k1-c1*rho==sigma;
eqn7 = k2-c2==betta;
eqn8 = (alfa/abs(alpha))*(abs(gamma)+abs(beta))==rho;

eqn9 = abs(e)<=min(((alfa-k1)/k2)*((alfa-k1)/k2),((betta*sqrt(4*rho*sigma+betta*betta)+2*sigma*rho)+betta*betta)/2*rho*rho);

sol = vpasolve([eqn1,eqn2,eqn3,eqn4,eqn5,eqn6,eqn7,eqn8],[c1,c2,k1,k2,alfa,betta,rho,sigma,e]);
u_ref_dot=(1/alpha)*(-k1*sign(e)*sqrt(abs(e))-k2*e);
