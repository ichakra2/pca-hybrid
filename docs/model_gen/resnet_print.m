filename = ['model.txt'];
fileID = fopen(filename,'w');
N =31; %Number of conv layers
n=10;  %Number of conv layers every basic block
ARCH = "hyb"; %(Or "Plain")
count1=1;
countbn=1;
i=1;
index1=[];
countconv=0;
fprintf(fileID,'\t%8s\r\n',['x = self.conv',num2str(i),'(x)']);
countconv = countconv+1;
fprintf(fileID,'\t%8s\r\n',['x = self.bn',num2str(i),'(x)']);
countbn = countbn+1;
fprintf(fileID,'\t%8s\r\n',['x = self.relu',num2str(i),'(x)']);
fprintf(fileID,'\t%8s \r\n',['residual = x.clone()']);
fprintf(fileID,'\t%8s \r\n',['out = x.clone()']);
countmp=1;
from_clone=1;
k=1;
if N==19
    downsample = [8,14];
    strides = [8,14];
else
    downsample = [12,22];
strides = [12,22];
end
from_clone2=1;
if ARCH == "plain" 
    k_c = 100;
else
    k_c = 1;
end

layers = [2,12,22];
for i = 2:N
    fprintf(fileID,'\t%8s\r\n',['out = self.conv',num2str(i),'(out)']);
    %fprintf(fileID,'\t%8s\r\n',['y.append(out.clone())']);
    countconv = countconv+1;
    if i>=layers(2)
        k=k_c;
    else
        k=k_c;
    end
    %
        
    if rem(from_clone,k)==0
        fprintf(fileID,'\t%8s\r\n',['out = self.bn',num2str(countbn),'(out)']);
        countbn = countbn+1;
        if  ismember(i,downsample) %|| im_size == im_size_save
            fprintf(fileID,'\t%8s\r\n',['residual1 = self.resconv',num2str(count1),'(residual1)']);
            countconv = countconv+1;
            index1(count1) = countconv;
            count1=count1+1;
            
        else
            %fprintf(fileID,'\t%8s\r\n',['self.resconv',num2str(count1),'=nn.Sequential(nn.Conv2d(',num2str(inp_channels),'*self.inflate,',num2str(op_channels),'*self.inflate, kernel_size=1, stride=1, padding=0),']);
            %fprintf(fileID,'\t%8s\r\n',['nn.BatchNorm2d(',num2str(op_channels),'*self.inflate),',')']);
        end
  
        
        %fprintf(fileID,'\t%8s\r\n',['out = self.relu',num2str(i),'(out)']);
        if 1==1%i>=layers(2) %i>=layers(2)%i>=layers(2)
            fprintf(fileID,'\t%8s\r\n',['out = F.relu','(out)']);
            fprintf(fileID,'\t%4s\r\n','out+=residual1');
        else
            fprintf(fileID,'\t%4s\r\n','out+=residual1');
            fprintf(fileID,'\t%8s\r\n',['out = F.relu','(out)']);
        end  
        
        
%         if i==maxpool(countmp)
%            fprintf(fileID,'\t%8s\r\n\n',['x = self.maxpool',num2str(countmp),'(x)']);
%            countmp=countmp+1;
%            im_size = im_size/2;
%         end
        fprintf(fileID,'\t%8s \r\n',['residual1 = out.clone()']);
        from_clone = 1;
        from_clone2 = from_clone2+1;
        fprintf(fileID,'\t%8s \r\n',['###################################']);
    else
        fprintf(fileID,'\t%8s\r\n',['out = self.relu',num2str(i),'(out)']);
        %fprintf(fileID,'\t%8s\r\n',['out = F.relu','(out)']);
        
        from_clone=from_clone+1;
        from_clone2 = from_clone2+1;
    end 
    if rem(i-1,n) == 0
        fprintf(fileID,'\t%8s \r\n',['#########Layer################']);
    end

        
end
fprintf(fileID,'\t%8s \r\n',['x=out']);
fprintf(fileID,'\t%8s\r\n\n',['x = self.avgpool(x)']);
fprintf(fileID,'\t%8s\r\n\n',['x = x.view(x.size(0), -1)']);
fprintf(fileID,'\t%8s\r\n\n',['x = self.bn',num2str(countbn),'(x)']);
countbn = countbn+1;
fprintf(fileID,'\t%8s\r\n\n',['x = self.fc(x)']);
fprintf(fileID,'\t%8s\r\n\n',['x = self.bn',num2str(countbn),'(x)']);
fprintf(fileID,'\t%8s\r\n\n',['x = self.logsoftmax(x)']);
fprintf(fileID,'\t%8s\r\n\n',['return x']);
fclose(fileID);
for i=1:length(index1)
    fprintf([num2str(index1(i)-1),','])
end
