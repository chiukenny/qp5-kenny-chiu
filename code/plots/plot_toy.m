randn('seed',1);
rand('seed',1);

my_color = [0.466 0.674 0.188;   % green
            0 0.447 0.741;       % blue
            0.929 0.694 0.125;   % orange
            0.85 0.325 0.098;    % red
            0.494 0.184 0.556];  % purple

% Obtain approximate samples from the variational distribution 
Eps0 = zeros(T,param.dim_noise); 
zAppr = zeros(T,dim_z); 
Tr_epsilon_all = zeros(T,dim_z); 
for t=1:T
   if strcmp(vardist.peps.pdf,'standard_normal')
       Eps0(t,:) = randn(1, vardist.peps.dim_noise);
   elseif strcmp(vardist.peps.pdf,'uniform')
       Eps0(t,:) = rand(1, vardist.peps.dim_noise);
   end
   net = netforward(vardist.net, Eps0(t,:));
   Tr_epsilon = net{1}.Z; 
   zAppr(t,:) = Tr_epsilon  + vardist.sigma.*randn(1,dim_z); 
   Tr_epsilon_all(t,:) = Tr_epsilon;
end

% Obtain contour of the true distribution
if(strcmp(pxz.model, 'banana'))
    x = linspace(-3,3);
    y = linspace(-8,1);
elseif(strcmp(pxz.model, 'banana3D'))
    x = linspace(-3,3);
    y = linspace(-8,1);
    z = linspace(-8,1);
end

plt_size = 5;

if(strcmp(pxz.model, 'banana3D'))
    [X,Y,Z] = meshgrid(x,y,z);
    for i=1:length(x)
        for j=1:length(y)
            for k=1:length(z)
                P(i,j,k) = pxz.logdensity([x(i) y(j) z(k)], pxz.inargs{:});
            end
        end
    end
    % XY
    figure;
    [cs, h] = contour(X(:,:,1),Y(:,:,1),squeeze(sum(exp(P),3))','Color',my_color(3,:), 'Linewidth',0.8);
    box off;
    name = [param.outdir pxz.dataName '_' param.method '_ContourSamples_XY'];
    hold on;
    plot(zAppr(1:10:end,1),zAppr(1:10:end,2),'.','Color',my_color(2,:),'MarkerSize',7);
    xlabel('z_1')
    ylabel('z_2')
    figurepdf(plt_size,plt_size);
    print('-dpdf', [name '.pdf']);

    % XZ
    figure;
    [cs, h] = contour(squeeze(X(:,:,1)),squeeze(Z(1,:,:))',squeeze(sum(exp(P),2))','Color',my_color(3,:), 'Linewidth',0.8);
    box off;
    name = [param.outdir pxz.dataName '_' param.method '_ContourSamples_XZ'];
    hold on;
    plot(zAppr(1:10:end,1),zAppr(1:10:end,3),'.','Color',my_color(2,:),'MarkerSize',7);
    xlabel('z_1')
    ylabel('z_3')
    figurepdf(plt_size,plt_size);
    print('-dpdf', [name '.pdf']);

    % YZ
    figure;
    [cs, h] = contour(squeeze(Y(:,:,1)),squeeze(Z(1,:,:)),squeeze(sum(exp(P),1))','Color',my_color(3,:), 'Linewidth',0.8);
    box off;
    name = [param.outdir pxz.dataName '_' param.method '_ContourSamples_YZ'];
    hold on;
    plot(zAppr(1:10:end,2),zAppr(1:10:end,3),'.','Color',my_color(2,:),'MarkerSize',7);
    xlabel('z_2')
    ylabel('z_3')
    figurepdf(plt_size,plt_size);
    print('-dpdf', [name '.pdf']);
else
    [X,Y] = meshgrid(x,y);
    for i=1:length(x)
        for j=1:length(y)
            Z(i,j) = pxz.logdensity([x(i) y(j)], pxz.inargs{:});
        end
    end

    % Plot
    figure;
    [cs, h] = contour(X,Y,exp(Z)','Color',my_color(3,:), 'Linewidth',0.8);
    box off;
    name = [param.outdir pxz.dataName '_' param.method '_ContourSamples'];
    hold on;
    plot(zAppr(1:10:end,1),zAppr(1:10:end,2),'.','Color',my_color(2,:),'MarkerSize',7);
    if(strcmp(pxz.model, 'banana'))
        title('banana');
    else
        error(['Unknown model: ' pxz.model]);
    end
    figurepdf(plt_size,plt_size);
    print('-dpdf', [name '.pdf']);
end
