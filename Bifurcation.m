%Developed by: Navid Mohammad Mirzaei
%Bifurcation and Lyapunov exponents for the Cancer equation 
%dC/dt
%=(\lambda_C+\lambda_{CIL_6}[IL_6]+\lambda_{CA}[A])(1-[C]/C_0)[C]-(\delta_{CT_c}[T_c]+\delta_{CI_{\gamma}}[I_{\gamma}}+\delta_C)[C]

clc
clear

%parameter names
parameters = ["\lambda_{C}","\lambda_{CIL_6}","\lambda_{CA}","C_0","\delta_{CT_c}","\delta_{CI_{\gamma}}","\delta_{C}"]

%initial condition for each cluster
init = [0.017227359,0.020674446,0.036479484,0.083426907,0.056258913];

%Ask user to pick the bifurcation parameter
prompt = 'Which parameter do you want to bifurcate for?(lambda_C=1,lambda_CIL6=2,lambda_CA=3,C_0=4,delta_CTc=5,delta_CIg=6,delta_C=7)';
index1 = input(prompt);


%Cluster  parameter values. Order follows the parameters array above.
C_pars = [0.05756377,0.000434473,0.000235655,9.428,0.004399439,0.002740763,0.044916885;...
          0.042504105,0.000849434,0.001243396,7.619081152,0.005275728,0.000741518,0.032726367;...
          0.056693365,0.0003381,0.000285856,8.797812771,0.00493649,0.001653573,0.044212307;...
          0.034270978,0.005530259,0.001807167,8.923021064,0.008051837,0.002831723,0.026061805;...
          0.006762032,0.002981655,0.030370037,8.597468687,0.006762032,0.002981655,0.030370037];
C_SS =  [90318.70483,111760.0386,96786.42019,95428.30814,99041.80342];  %Steady state


%parameter interval and iteration constants
a0 = 0; a1 = 1; N = 2000; L = 2000;
a = linspace(a0,a1,N);
[r,c] = meshgrid(1:L,a); % associated cooridate data 

%create color scheme
str1 = '#3F9B0B';
color1 = sscanf(str1(2:end),'%2x%2x%2x',[1 3])/255;
str2 = '#FF796C';
color2 = sscanf(str2(2:end),'%2x%2x%2x',[1 3])/255;
str3 = '#0343DF';
color3 = sscanf(str3(2:end),'%2x%2x%2x',[1 3])/255;
str4 = '#000000';
color4 = sscanf(str4(2:end),'%2x%2x%2x',[1 3])/255;
str5 = '#D5B60A';
color5 = sscanf(str5(2:end),'%2x%2x%2x',[1 3])/255;
dcolor = [color1;color2;color3;color4;color5]; % Marker color setting: blue           

%Bifurcation
subplot(3,1,1);
for i=5:5
    mat1 = bif(@cancer,@lya_exp,init(i),C_pars(i,:),index1,a0,a1,N,L);
    hold on
    z1=surf(r,c,mat1*C_SS(i),'Marker','*','MarkerSize',2,'FaceColor','None','MarkerEdgeColor', dcolor(i,:),'EdgeColor','None');
    zlabel('[C]')
    ylabel(parameters(index1))
    view([90,0,0]) % change camera direction
    ylim([a0,a1])
end
set(gca,'Box','on');
hold off

%Lyapunov Exponent
subplot(3,1,2); 
hold on
for i=1:5
    x = init(i)*ones(1,N);
    lyapunov = zeros(1,N);
    for j=0:L
        x_temp = cancer(a,x,C_pars(i,:),index1);
        x = x_temp;
        lyapunov =lyapunov+lya_exp(a,x,C_pars(i,:),index1);
    end
    plot(a,lyapunov/L,'Color', dcolor(i,:),'LineWidth',3.5)
end
legend('Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster 5')
ylabel('Lyapunov Exponent')
xlabel(parameters(index1))
set(gca,'Box','on');
hold off



%Corresponding functions

function mat1 = bif(fun_1,fun_2,x0,pars,idx,a0,a1,N,L,p_siz)

% default settings
if ~exist('p_siz','var')
    p_siz = 1;
end
% initialization
mat1 = zeros(N,L);
mat2 = zeros(N,L);
a = linspace(a0,a1,N);
lyap=zeros(1,N);
% main loop
format long
for i = 1:N
    ca = a(i); % pick one parameter value at each time
    for j = 1:L % generate a sequence with length L
        if j == 1 
            pre = x0; % assign initial value
            for k = 1:500 % throw out bad data
               nxt = fun_1(ca,pre,pars,idx); 
               pre = nxt;
            end
        end
        nxt = fun_1(ca,pre,pars,idx); % generate sequence
        mat1(i,j) = nxt; % store in mat
        pre = nxt; % use latest value as the initial value for the next round        
    end
end
end



function c = cancer(r,x,pars,idx)
    if idx==1
        c = (r+pars(2)+pars(3)).*(1-(x./pars(4))).*x-(pars(5)+pars(6)+pars(7)).*x+x;
    end
    if idx==2
        c = (pars(1)+r+pars(3)).*(1-(x./pars(4))).*x-(pars(5)+pars(6)+pars(7)).*x+x;
    end
    if idx==3
        c = (pars(1)+pars(2)+r).*(1-(x./pars(4))).*x-(pars(5)+pars(6)+pars(7)).*x+x;
    end
    if idx==4
        c = (pars(1)+pars(2)+pars(3)).*(1-(x./r)).*x-(pars(5)+pars(6)+pars(7)).*x+x;
    end
    if idx==5
        c = (pars(1)+pars(2)+pars(3)).*(1-(x./pars(4))).*x-(r+pars(6)+pars(7)).*x+x;
    end
    if idx==6
        c = (pars(1)+pars(2)+pars(3)).*(1-(x./pars(4))).*x-(pars(5)+r+pars(6)).*x+x;
    end
    if idx==7
        c = (pars(1)+pars(2)+pars(3)).*(1-(x./pars(4))).*x-(pars(5)+pars(6)+r).*x+x;
    end
end

function l = lya_exp(r,x,pars,idx)
    if idx==1
        l = log(abs(1+(r+pars(2)+pars(3)-pars(5)-pars(6)-pars(7))-2.*x.*(r+pars(2)+pars(3))./pars(4)));
    end
    if idx==2
        l =  log(abs(1+(pars(1)+r+pars(3)-pars(5)-pars(6)-pars(7))-2.*x.*(pars(1)+r+pars(3))./pars(4)));
    end
    if idx==3
        l =  log(abs(1+(pars(1)+pars(2)+r-pars(5)-pars(6)-pars(7))-2.*x.*(pars(1)+pars(2)+r)./pars(4)));
    end
    if idx==4
        l =  log(abs(1+(pars(1)+pars(2)+pars(3)-pars(5)-pars(6)-pars(7))-2.*x.*(pars(1)+pars(2)+pars(3))./r));
    end
    if idx==5
        l =  log(abs(1+(pars(1)+pars(2)+pars(3)-r-pars(6)-pars(7))-2.*x.*(pars(1)+pars(2)+pars(3))./pars(4)));
    end
    if idx==6
        l =  log(abs(1+(pars(1)+pars(2)+pars(3)-pars(5)-r-pars(7))-2.*x.*(pars(1)+pars(2)+pars(3))./pars(4)));
    end
    if idx==7
        l =  log(abs(1+(pars(1)+pars(2)+pars(3)-pars(5)-pars(6)-r)-2.*x.*(pars(1)+pars(2)+pars(3))./pars(4)));
    end
end


    

