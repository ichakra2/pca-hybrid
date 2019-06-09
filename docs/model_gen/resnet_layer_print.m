filename = ['layer.txt'];
fileID = fopen(filename,'w');
N =31; %Number of conv layers
n=10;  %Number of conv layers every basic block
arch = "bin"; %(Or "Plain")
count1=1;
countconv=1;
num_comp_all=[];
num_mem_all=[];
index1=[];

countbn=1;
i=1;
inflate = 1;
im_size = 32;
inp_channels = [3 16*ones(1,n+1) 32*ones(1,n) 64*ones(1,n-1)];
op_channels = [16 16*ones(1,n) 32*ones(1,n) 64*ones(1,n)];
stride = [1 ones(1,n) sqrt(2) ones(1,n-1) sqrt(2) ones(1,n-1)];
im_size = [32 32*ones(1,n+1) 16*ones(1,n) 8*ones(1,n-1)];

fprintf(fileID,'\t%8s\r\n',['self.conv',num2str(i),'=nn.Conv2d(',num2str(inp_channels(i)),',int(',num2str(op_channels(i)),'*self.inflate), kernel_size=3, stride=1, padding=1)']);
countconv = countconv+1;

fprintf(fileID,'\t%8s\r\n',['self.bn',num2str(i),'= nn.BatchNorm2d(int(',num2str(op_channels(i)),'*self.inflate))']);

countbn = countbn+1;
fprintf(fileID,'\t%8s\r\n',['self.relu',num2str(i),'=nn.ReLU(inplace=True)']);
from_clone=1;
from_clone2=1;
countmp=1;
if arch == "plain" 
    k_c = 100;
else
    k_c = 1;
    hyb_layers = [12,13,22,23,24];
end
if N==19
    downsample = [8,14];
    strides = [8,14];
else
    downsample = [12,22];
strides = [12,22];
end
from_clone2=1;
layers = [2,12,22];
for i = 2:N


    if ismember(i,strides)
        convstride = 2;
    else
        convstride=1;
    end
    if arch == "hyb" && ismember(i,hyb_layers)
        fprintf(fileID,'\t%8s\r\n',['self.conv',num2str(i),'=BinConv2d2(int(',num2str(inp_channels(i)),'*self.inflate), int(',num2str(op_channels(i)),'*self.inflate), kernel_size=3, stride=',num2str(convstride), ', padding=1)']);
        countconv = countconv+1;
    else
        fprintf(fileID,'\t%8s\r\n',['self.conv',num2str(i),'=BinConv2d(int(',num2str(inp_channels(i)),'*self.inflate), int(',num2str(op_channels(i)),'*self.inflate), kernel_size=3, stride=',num2str(convstride), ', padding=1)']);
        countconv = countconv+1;
    end
    if i>=layers(2)
        k=k_c;
    else
        k=k_c;
    end
    if rem(from_clone,k)==0 
        fprintf(fileID,'\t%8s\r\n',['self.bn',num2str(countbn),'= nn.BatchNorm2d(int(',num2str(op_channels(i)),'*self.inflate))']);
            countbn = countbn+1;
        if  ismember(i,downsample) 
            if arch == "hyb"
                fprintf(fileID,'\t%8s\r\n',['self.resconv',num2str(count1),'=nn.Sequential(BinConv2d2(int(',num2str(inp_channels(i-1)),'*self.inflate), int(',num2str(op_channels(i)),'*self.inflate), kernel_size=1, stride=2, padding=0),']);
                fprintf(fileID,'\t%8s\r\n',['nn.BatchNorm2d(int(',num2str(op_channels(i)),'*self.inflate)),']);
                fprintf(fileID,'\t%8s\r\n',['nn.ReLU(inplace=True),)']);
                index1(count1) = countconv;
                countconv = countconv+1;
                count1=count1+1;
            else
                fprintf(fileID,'\t%8s\r\n',['self.resconv',num2str(count1),'=nn.Sequential(BinConv2d(int(',num2str(inp_channels(i-1)),'*self.inflate), int(',num2str(op_channels(i)),'*self.inflate), kernel_size=1, stride=2, padding=0),']);
                fprintf(fileID,'\t%8s\r\n',['nn.BatchNorm2d(int(',num2str(op_channels(i)),'*self.inflate)),']);
                fprintf(fileID,'\t%8s\r\n',['nn.ReLU(inplace=True),)']);
                index1(count1) = countconv;
                countconv = countconv+1;
                count1=count1+1;
            end
            
        else
        end
        
        fprintf(fileID,'\t%8s\r\n',['self.relu',num2str(i),'=nn.ReLU(inplace=True)']);
        from_clone = 1;
        fprintf(fileID,'\t%8s\r\n\n',['#######################################################']);
    else
        fprintf(fileID,'\t%8s\r\n',['self.relu',num2str(i),'=nn.ReLU(inplace=True)']);
        from_clone=from_clone+1;        
        
    end
    
    if rem(i-1,n) == 0
        fprintf(fileID,'\t%8s \r\n',['#########Layer################']);
    end
     
end

num_classes = 100;
fprintf(fileID,'\t%8s\r\n',['self.avgpool=nn.AvgPool2d(8)']);
fprintf(fileID,'\t%8s\r\n',['self.bn',num2str(countbn),'= nn.BatchNorm1d(int(',num2str(op_channels(i)),'*self.inflate','))']);
countbn =countbn+1;
fprintf(fileID,'\t%8s\r\n',['self.fc=nn.Linear(int(',num2str(op_channels(i)),'*self.inflate),','num_classes)']);
fprintf(fileID,'\t%8s\r\n',['self.bn',num2str(countbn),'= nn.BatchNorm1d(',num2str(num_classes),')']);


fprintf(fileID,'\t%8s\r\n',['self.logsoftmax=nn.LogSoftmax()']);
