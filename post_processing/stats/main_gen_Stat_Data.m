%% if this works... yep, it does!!! XDDD

clc
close all
clear

%%% create interpolate mesh first, using:
%%% python3 writer_int_pos.py

% pyversion('/usr/bin/python3');
pyenv(Version="/home/yuninw/apps/mpi_drl/miniforge3/envs/matlab/bin/python3.10");
%%
% Add parameters
Reb = 2900;
ys = 0;
amp = 0;

%fileName=sprintf('../result_data/LC_Reb%d_bdfd_y%d_amp%d.mat',Reb,ys,amp);
fileName=sprintf('./results_data/MC_Reb%d.mat',Reb);
fileName
nu=1/(Reb);
nx=100;
ny=1000;
path_int_data='int_fld';

dbProfs=pyrunfile('main_pHill_PP.py','dbProfs',nu=nu,path_int_data=path_int_data);

x = reshape(double(py.array.array('d',py.numpy.nditer(dbProfs{'x'}))),[ny,nx]);
yy = reshape(double(py.array.array('d',py.numpy.nditer(dbProfs{'y'}))),[ny,nx]);
U = reshape(double(py.array.array('d',py.numpy.nditer(dbProfs{'U'}))),[ny,nx]);
dUdy = reshape(double(py.array.array('d',py.numpy.nditer(dbProfs{'dUdy'}))),[ny,nx]);
uu = reshape(double(py.array.array('d',py.numpy.nditer(dbProfs{'uu'}))),[ny,nx]);
vv = reshape(double(py.array.array('d',py.numpy.nditer(dbProfs{'vv'}))),[ny,nx]);
ww = reshape(double(py.array.array('d',py.numpy.nditer(dbProfs{'ww'}))),[ny,nx]);
uv = reshape(double(py.array.array('d',py.numpy.nditer(dbProfs{'uv'}))),[ny,nx]);
%% Pressure 
p = reshape(double(py.array.array('d',py.numpy.nditer(dbProfs{'P'}))),[ny,nx]);
pp = reshape(double(py.array.array('d',py.numpy.nditer(dbProfs{'pp'}))),[ny,nx]);
%(pvp = reshape(double(py.array.array('d',py.numpy.nditer(dbProfs{'pvp'}))),[ny,nx]); *)
vvv = reshape(double(py.array.array('d',py.numpy.nditer(dbProfs{'vvv'}))),[ny,nx]);

data.x      = x 
data.yy     = yy

data.Ur     = U 
data.dUdyr  = dUdy 
data.uur    = uu 
data.vvr    = vv 
data.wwr    = ww 
data.uvr    = uv 

figure
scatter(x,yy)
figure
plot(yy(:,1),U(:,1)); hold on
plot(yy(:,42),U(:,42),'x')
plot(yy(:,100),U(:,100),'o')
plot(yy(:,100),mean(U,2),'^')

disp('done')

%%

data.nu=nu;
data.rho=1;
% data.y=yy(:,1)+1;
data.y=yy(:,1);
data.U=mean(U,2);
data.dUdy=mean(dUdy,2);
data.uu=mean(uu,2);
data.vv=mean(vv,2);
data.ww=mean(ww,2);
data.uv=mean(uv,2);
data.Pr     = p; 
data.pp    = pp; 
data.prms=mean(sqrt(data.pp),2);
data.urms = mean(sqrt(uu),2)

disp('Wall dUdy:')
data.dUdy(1)

data.twall=data.rho*data.nu*(data.dUdy(1));
data.utau=sqrt(data.twall/data.rho);
data.Ret=data.utau/data.nu;
data.lstar=data.nu/data.utau;

disp(['Utau:', num2str(data.utau),'  Ret:', num2str(data.Ret),...
    '  lstar:', num2str(data.lstar), '  tau_w:', num2str(data.twall)])

data.yp=data.y(1:ny)/data.lstar;
data.Up=(data.U(1:ny))/data.utau;
data.uup=(data.uu(1:ny))/data.utau^2;
data.vvp=(data.vv(1:ny))/data.utau^2;
data.wwp=(data.ww(1:ny))/data.utau^2;
data.uvp=(data.uv(1:ny))/data.utau^2;
data.urmsp=(data.urms(1:ny))/data.utau;
data.prmsp=(data.prms(1:ny))/(data.utau^2);
disp("Re_tau:")
disp(data.Ret)


data.Uinf=max(data.U)
data.theta= trapz([0; data.y],([0; data.U]/data.Uinf).*(1-[0; data.U]/data.Uinf));
data.deltas= trapz([0; data.y],(1.0-[0; data.U]/data.Uinf));
data.Reth=data.Uinf*data.theta/data.nu;
data.H12 = data.deltas / data.theta;

disp("Retheta:"); disp(data.Reth)
disp("H12:"); disp(data.H12); 
disp("Re_tau:")
disp(data.Ret)

figure
semilogx(data.yp,data.Up); hold on
xlim([1,data.Ret*2]);
xlabel('y^+');ylabel('U^+');
print('Figs/U+_prof.jpg', '-djpeg');

figure
semilogx(data.yp,data.prmsp); hold on
xlim([0,data.Ret*2]);
xlabel('y^+');ylabel('p^+_{\rm rms}');
print('Figs/prms+_prof.jpg', '-djpeg');

% save('../results/benchmark.mat','data','-mat');
% save('../results/oppo.mat','data','-mat');
save(fileName,'data','-mat');

disp('done')




