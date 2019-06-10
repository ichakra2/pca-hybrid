clear all;
close all;
N=6;
C = repmat(linspace(0.9,0.1,N).',1,3);
D(:,1) = linspace(0.9,0.5,N)';
D(:,2) = linspace(0.6,0.3,N)';
D(:,3) = linspace(0.4,0.1,N)';
count=0;
thr=0.99;
dev = [];
del = 2; %criteria
for i=0:17
    
    data = [];
    filename5 = ['./PCA_files_resnet20_plain/PCA_files_',num2str(i),'.out'];
    data(:,2) = load(filename5);

    if i<=5
        fignum=1;
    elseif i>11
        fignum=3;
    else 
        fignum=2;
    end

    if rem(i+1,N)==0
        count=0;
    end

    figure(fignum+N);
    hold on;
    plot(1:size(data(:,2)),data(:,2),'color',D(count+1,:),'Linewidth',2)
    filenamesave = ['./PCA_files/layer_',num2str(i+2),'.png'];
    legend(['Layer',num2str(i+1)]);
    ylim([0 1]);
    pause(1)
    count=count+1;
    datanew = data(:,2);
    filter_gt_99(i+1) = size(datanew,1)-size(datanew(datanew>thr),1);
    filter_gt_pc(i+1) = (filter_gt_99(i+1)/size(datanew,1))*100;
    if i>0
        if filter_gt_99(i+1)-filter_gt_99(i)>=del %Identifying significant layers
            disp(['Layer ',num2str(i+2)])
        end
    end
end
 figure;
 plot(filter_gt_99);
