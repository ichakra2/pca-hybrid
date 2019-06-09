Model Generation

This folder contains the model generator codes in Matlab. You have to paste this in resnethybunrolled.py or resnetplainunrolled.py in the forward function in ResNet class and __init__ function of ResNet_cifar100 class. 

Example output of resnet_print.m

```
	x = self.conv1(x)
	x = self.bn1(x)
	x = self.relu1(x)
	residual = x.clone() 
	out = x.clone() 
	out = self.conv2(out)
	out = self.bn2(out)
	out = F.relu(out)
	out+=residual1
	residual1 = out.clone() 
	################################### 
	out = self.conv3(out)
	out = self.bn3(out)
	out = F.relu(out)
	out+=residual1
	residual1 = out.clone() 
  
  ........
  out = self.conv31(out)
	out = self.bn31(out)
	out = F.relu(out)
	out+=residual1
	residual1 = out.clone() 
	################################### 
	#########Layer################ 
	x=out 
	x = self.avgpool(x)

	x = x.view(x.size(0), -1)

	x = self.bn32(x)

	x = self.fc(x)

	x = self.bn33(x)

	x = self.logsoftmax(x)

	return x
  ```
  
  Example output of resnet_layer_print.m
  
  ```
  self.conv1=nn.Conv2d(3,int(16*self.inflate), kernel_size=3, stride=1, padding=1)
	self.bn1= nn.BatchNorm2d(int(16*self.inflate))
	self.relu1=nn.ReLU(inplace=True)
	self.conv2=BinConv2d(int(16*self.inflate), int(16*self.inflate), kernel_size=3, stride=1, padding=1)
	self.bn2= nn.BatchNorm2d(int(16*self.inflate))
	self.relu2=nn.ReLU(inplace=True)
	#######################################################

	self.conv3=BinConv2d(int(16*self.inflate), int(16*self.inflate), kernel_size=3, stride=1, padding=1)
	self.bn3= nn.BatchNorm2d(int(16*self.inflate))
	self.relu3=nn.ReLU(inplace=True)
	#######################################################

	self.conv4=BinConv2d(int(16*self.inflate), int(16*self.inflate), kernel_size=3, stride=1, padding=1)
	self.bn4= nn.BatchNorm2d(int(16*self.inflate))
	self.relu4=nn.ReLU(inplace=True)
  
  ...................
  
  self.conv30=BinConv2d(int(64*self.inflate), int(64*self.inflate), kernel_size=3, stride=1, padding=1)
	self.bn30= nn.BatchNorm2d(int(64*self.inflate))
	self.relu30=nn.ReLU(inplace=True)
	#######################################################

	self.conv31=BinConv2d(int(64*self.inflate), int(64*self.inflate), kernel_size=3, stride=1, padding=1)
	self.bn31= nn.BatchNorm2d(int(64*self.inflate))
	self.relu31=nn.ReLU(inplace=True)
	#######################################################

	#########Layer################ 
	self.avgpool=nn.AvgPool2d(8)
	self.bn32= nn.BatchNorm1d(int(64*self.inflate))
	self.fc=nn.Linear(int(64*self.inflate),num_classes)
	self.bn33= nn.BatchNorm1d(100)
	self.logsoftmax=nn.LogSoftmax()
