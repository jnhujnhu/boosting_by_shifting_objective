mex_all;
clear;
%% Load Dataset
load 'a9a.mat';

%% Add Bias
X = [ones(size(X, 1), 1) X];
[N, Dim] = size(X);
X = full(X');

%% Normalize Data
sum1 = 1./sqrt(sum(X.^2, 1));
if abs(sum1(1) - 1) > 10^(-10)
    X = X.*repmat(sum1, Dim, 1);
end
clear sum1;

%% Set Params
passes = 420;
model = 'logistic'; 
regularizer = 'L2'; 
init_weight = repmat(0, Dim, 1);
mu = 5 * 10^(-8);
L = (0.25 * max(sum(X.^2, 1)) + mu);
fprintf('Model: %s-%s\n', regularizer, model);

%% Run Algorithms

% SAGA
algorithm = 'SAGA';
loop = int64((passes - 1) * N); % One Extra Pass for initializing SAGA gradient table.
step_size = 1 / (2 * (mu * N + L));
fprintf('Algorithm: %s\n', algorithm);
tic;
[time1, hist1] = Interface(X, y, algorithm, model, regularizer, init_weight, mu, L...
    , step_size, loop);
time = toc;
fprintf('Time: %f seconds \n', time);
X_SAGA = [0 1 2:3:passes - 2]';
time1 = [time1, hist1];
hist1 = [X_SAGA, hist1];


% epoch length
m = 2 * N;
X_SVRG = [0:3:passes]';

% Katyusha
algorithm = 'Katyusha';
% step_size = 0.3;
tau_1 = 0.5;
if(sqrt(mu * m / (3 * L)) < 0.5)
    tau_1 = sqrt(mu * m / (3 * L));
end
loop = int64(passes / 3); % 3 passes per outer loop
fprintf('Algorithm: %s\n', algorithm);
tic;
[time2, hist2] = Interface(X, y, algorithm, model, regularizer, init_weight, mu, L...
    , tau_1, loop);
time = toc;
fprintf('Time: %f seconds \n', time);
time2 = [time2, hist2];
hist2 = [X_SVRG, hist2];

% BS_SVRG
algorithm = 'BS_SVRG';
alpha = sqrt((2.0 + sqrt(3)) * m * L * mu) - mu;
loop = int64(passes / 3); % 3 passes per loop
fprintf('Algorithm: %s\n', algorithm);
tic;
[time3, hist3] = Interface(X, y, algorithm, model, regularizer, init_weight, mu...
    , L, alpha, loop, 0);
time = toc;
fprintf('Time: %f seconds \n', time);
time3 = [time3, hist3];
hist3 = [X_SVRG, hist3];

% BS_SVRG_N
algorithm = 'BS_SVRG';
kappa = L / mu;
% Solving the optimal choice of alpha 
myfun = @(x,m,kappa) (1 + 1 ./ x).^(2*m)*(kappa-1)-x-kappa; 
fun = @(x) myfun(x,m,kappa);                                
x = fzero(fun,sqrt(4*m*kappa));
alpha = x * mu;
loop = int64(passes / 3); % 3 passes per loop
fprintf('Algorithm: %s\n', algorithm);
choice = -1; % Using numerical choice (different tau_x)
tic;
[time4, hist4] = Interface(X, y, algorithm, model, regularizer, init_weight, mu...
    , L, alpha, loop, choice);
time = toc;
fprintf('Time: %f seconds \n', time);
time4 = [time4, hist4];
hist4 = [X_SVRG, hist4];


%% Plot
aa1 = min(hist1(:, 2));
aa2 = min(hist2(:, 2));
aa3 = min(hist3(:, 2));
aa4 = min(hist4(:, 2));

% opt is found by running Katyusha for 1000 passes.
aa = max(max([hist1(:, 2)])) - opt; 
b = 2;

figure(101);
set(gcf,'position',[520,500,436,269]);
semilogy(hist1(1:b:end,1), abs(hist1(1:b:end,2) - opt),'b--o','linewidth',1.6,'markersize',4);
hold on,semilogy(hist2(1:b:end,1), abs(hist2(1:b:end,2) - opt),'r--^','linewidth',1.6,'markersize',4);
hold on,semilogy(hist3(1:b:end,1), abs(hist3(1:b:end,2) - opt),'k-.+','linewidth',1.6,'markersize',4);
hold on,semilogy(hist4(1:b:end,1), abs(hist4(1:b:end,2) - opt),'m-.d','linewidth',1.6,'markersize',4);
hold off;
xlabel('Number of effective passes','Interpreter','latex');
ylabel('Objective minus best','Interpreter','latex');
axis([0 ,passes, 1E-12,aa]);
legend('SAGA', 'Katyusha', 'BS-SVRG', 'BS-SVRG-N');
