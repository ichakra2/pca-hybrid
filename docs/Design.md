Steps to design Hybrid-Net

1. Train a XNOR-Net (For ResNet, train a plain network)

```
python main.py --arch resnetplainunrolled --model resnetplainunrolled --save test
```

2. Run evaluate using main_evaluate.py. This will generate multiple .out files named PCA_files_*.out. 

```
python main_evaluate.py --arch resnetplainunrolled --model resnetplainunrolled --pretrained savedfilename
```

3. Run PCAplotresnet.m on MATLAB to plot PCA analysis results of layer-wise significant components. Identify the significant layers using del parameter in code.

4. Design Hybrid-Net by assigning BinConv2d2 to significant layers and BinConv2d to binary layers. This can be done very easily by using the Matlab codes provided in /model_gen/resnet_print.m and /models/resnet_layer_print.m. Define model in resnethybunrolled.py. Example  of layer definitions given below:

```
#Binary Layer definition
  self.conv21=BinConv2d(int(32*self.inflate), int(32*self.inflate), kernel_size=3, stride=1, padding=1)
	self.bn21= nn.BatchNorm2d(int(32*self.inflate))
	self.relu21=nn.ReLU(inplace=True)
  
#k-bit layer definition

	self.conv23=BinConv2d2(int(64*self.inflate), int(64*self.inflate), kernel_size=3, stride=1, padding=1)
	self.bn23= nn.BatchNorm2d(int(64*self.inflate))
	self.relu23=nn.ReLU(inplace=True)
```

5. Run Hybrid network. 

```
python main.py --arch resnethybunrolled --model resnethybunrolled --save testhybrid
```
