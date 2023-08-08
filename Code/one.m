clc;
clear
close all;
%Load data%
load('system_data (1).mat');
%-------------------genarate input signal--------------------%
%x=normrnd(0,1,1,100);
%load x
%plot(x)
%--------------------------output ---------------------------%
plot(y)

%------------------genarate dynamic input -------------------%
% u1=[zeros(1,1) u(1:100-1)]'; %x(t-1)
% u2=[zeros(1,2) u(1:100-2)]'; %x(t-2)
% u3=[zeros(1,3) u(1:100-3)]'; %x(t-3)
% u4=[zeros(1,4) u(1:100-4)]'; %x(t-4)
% 
% y1=[zeros(1,1) y(1:100-1)']'; %y(t-1)
% y2=[zeros(1,2) y(1:100-2)']'; %y(t-2)
% y3=[zeros(1,3) y(1:100-3)']'; %y(t-3)
% y4=[zeros(1,4) y(1:100-4)']'; %y(t-4)
% y5=[zeros(1,5) y(1:100-5)']'; %y(t-5)

%----------------- design & Train Neural Network ---------------%
x=x';
input=[x u1 u2 u3 u4 y1 y2 y3 y4 y5];
p=input';
j=0;
for i=75:75
    j=j+1;
net = newff(minmax(p),[3 1],{'tansig'  'purelin'});
net.biasConnect=[0;0]; % neurons without noise
net.trainParam.lr=0.01; % learning rate
net.trainParam.epochs = i;
[net,tr]=train(net,p(:,1:80),y(1:80)');
c2 = sim(net,p(:,1:80));
e=c2-y(1:80)';
msetrain(j)=mse(e)
c1 = net(p(:,81:100));
e1=c1-y(81:100)';
msetest(j)=mse(e1)
end
%-------------------NN2TF --------------------%
M=net.IW{1};
N=net.LW{2,1};
a=zeros(10,1);
for i=1:10
    for j=1:3
    a(i)=N(1,j)*M(j,i)+a(i);
    end
end
a=0.5*a;
a1=a(6);
a2=a(7);
a3=a(8);
a4=a(9);
a5=a(10);
poles=roots([1 -a1 -a2 -a3 -a4 -a5]);
abs(poles)
for j=1:5
    while(abs(poles(j)))>1
        poles(j)=0.9*poles(j);
    end
end
abs(poles)
poly(poles)
%-----------------plot Output,step,impulse,bode------------------%
z=tf('z');
h=(0.2341*z^5+0.0215*z^4-1.0039*z^3-0.9417*z^2-0.3744*z)/(z^5+0.2969*z^4+0.5506*z^3-0.2446*z^2+0.1297*z-0.0818);
d=poly(poles);
h_hat=(a(1)*z^5+a(2)*z^4+a(3)*z^3+a(4)*z^2+a(5)*z)/(z^5+d(2)*z^4+d(3)*z^3+d(4)*z^2+d(5)*z+d(6));

figure
step(h,'r')
hold on
step(h_hat,'b*')

figure
impulse(h,'r')
hold on
impulse(h_hat,'b*')

figure
bode(h,'r')
hold on
bode(h_hat,'b*')

yhat(1:5,1)=zeros(5,1);
for i=6:100
    yhat(i)=-[d(2) d(3) d(4) d(5) d(6)]*...
    [yhat(i-1) yhat(i-2) yhat(i-3) yhat(i-4) yhat(i-5)]' +...
       [a(1) a(2) a(3) a(4) a(5)]*...
       [x(i) x(i-1) x(i-2) x(i-3) x(i-4)]';
end
figure
plot(y,'r')
hold on
plot(yhat,'b*')