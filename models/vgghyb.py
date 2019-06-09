import torch.nn as nn
import torch
import torch.nn.functional as F
__all__ = ['vgghyb']
class BinActive(torch.autograd.Function):
    '''
    Binarize the input activations and calculate the mean across channel dimension.
    '''
    def forward(self, input):
        self.save_for_backward(input)
        size = input.size()
        mean = torch.mean(input.abs(), 1, keepdim=True)
        input = input.sign()
        return input, mean

    def backward(self, grad_output, grad_output_mean):
        input, = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0
        return grad_input
class BinActive2(torch.autograd.Function):
    '''
    Binarize the input activations and calculate the mean across channel dimension.
    '''
    def forward(self, input):
        self.save_for_backward(input)
        size = input.size()
        mean = torch.mean(input.abs(), 1, keepdim=True)
	x = input
	#print('x before', x[0][0])
	xmax = x.abs().max()
	num_bits=2
	v0 = 1
	v1 = 2
	v2 = -0.5
	#x = x.div(xmax)
        y = 2.**num_bits - 1.
	x = x.add(v0).div(v1)
	#print('x', x[0][0])
	x = x.mul(y).round_()
        #print('x', x[0][0])
	x = x.div(y)
        #print('x', x[0][0])
	x = x.add(v2)
        #print('x', x[0][0])
	x = x.mul(v1)
	#print('x after', x[0][0])
	#raw_input()
	input = x
        #input = input.sign()
        return input, mean

    def backward(self, grad_output, grad_output_mean):
        input, = self.saved_tensors
	#x = torch.tanh(2*input)
	#y = x**2
	#z = y.mul(-1).add(1)
	#z = z.mul(2)
	#z = torch.exp(-torch.abs(4*input))
	#grad_input = grad_output.mul(z)
        grad_input = grad_output.clone()
        #grad_input[input.ge(1)] = 0
        #grad_input[input.le(-1)] = 0
        return grad_input
class BinConv2d(nn.Module): # change the name of BinConv2d
    def __init__(self, input_channels, output_channels,
            kernel_size=-1, stride=-1, padding=-1, groups=1, dropout=0,
            Linear=False):
        super(BinConv2d, self).__init__()
        self.layer_type = 'BinConv2d'
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dropout_ratio = dropout

        if dropout!=0:
            self.dropout = nn.Dropout(dropout)
        self.Linear = Linear
        if not self.Linear:
            self.bn = nn.BatchNorm2d(input_channels, eps=1e-4, momentum=0.1, affine=True)
            self.conv = nn.Conv2d(input_channels, output_channels,
                    kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
        else:
            self.bn = nn.BatchNorm1d(input_channels, eps=1e-4, momentum=0.1, affine=True)
            self.linear = nn.Linear(input_channels, output_channels)
       # self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
 
        x = self.bn(x)
        x, mean = BinActive()(x)
        if self.dropout_ratio!=0:
            x = self.dropout(x)
	if not self.Linear:
            x = self.conv(x)
        else:
            x = self.linear(x)
       
        return x

class BinConv2d2(nn.Module):
    def __init__(self, input_channels, output_channels,
            kernel_size=-1, stride=-1, padding=-1, dropout=0):
        super(BinConv2d2, self).__init__()
        self.layer_type = 'BinConv2d'
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dropout_ratio = dropout

        self.bn = nn.BatchNorm2d(input_channels, eps=1e-4, momentum=0.1, affine=True)
        if dropout!=0:
            self.dropout = nn.Dropout(dropout)
        self.conv = nn.Conv2d(input_channels, output_channels,
                kernel_size=kernel_size, stride=stride, padding=padding)
       # self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.bn(x)
        x, mean = BinActive2()(x)
        if self.dropout_ratio!=0:
            x = self.dropout(x)

        x = self.conv(x)


        #x = self.relu(x)
        return x

class vgg16(nn.Module):

    def __init__(self):
        super(vgg16, self).__init__()


    def _make_layer(self, block, planes, blocks, stride=1):
        for i in range(1, blocks-1):
            layers.append(block(self.inplanes, planes))
        layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
	x = self.conv1(x)
	x = self.bn1(x)
	x = self.relu1(x)
	residual = x.clone() 
	out = x.clone() 
	out = self.conv2(out)
	out = self.bn2(out)
	out = self.relu2(out)
	out = self.maxpool1(out)

	#########Layer################ 
	out = self.conv3(out)
	out = self.relu3(out)
	out = self.conv4(out)
	out = self.relu4(out)
	out = self.maxpool2(out)

	#########Layer################ 
	out = self.conv5(out)
	out = self.relu5(out)
	out = self.conv6(out)
	out = self.relu6(out)
	out = self.conv7(out)
	out = self.relu7(out)
	out = self.maxpool3(out)

	#########Layer################ 
	out = self.conv8(out)
	out = self.relu8(out)
	out = self.conv9(out)
	out = self.relu9(out)
	out = self.conv10(out)
	out = self.relu10(out)
	out = self.maxpool4(out)

	#########Layer################ 
	out = self.conv11(out)
	out = self.relu11(out)
	out = self.conv12(out)
	out = self.relu12(out)
	out = self.conv13(out)
	out = self.relu13(out)
	out = self.maxpool5(out)

	#########Layer################ 
	x = out

	x = x.view(x.size(0), -1)

	x = self.fc1(x)

	x = self.relu14(x)

	x = self.bn3(x)

	x = self.fc3(x)

	x = self.logsoftmax(x)

	return x



class VGG_cifar100(vgg16):

    def __init__(self, num_classes=100,depth=18):
        super(VGG_cifar100, self).__init__()
        self.inflate = 1
	self.conv1=nn.Conv2d(3,64*self.inflate, kernel_size=3, stride=1, padding=1)
	self.bn1= nn.BatchNorm2d(64*self.inflate)
	self.relu1=nn.ReLU(inplace=True)
	self.conv2=BinConv2d(64*self.inflate,64*self.inflate, kernel_size=3, stride=1, padding=1)
	self.bn2= nn.BatchNorm2d(64*self.inflate)
	self.relu2=nn.ReLU(inplace=True)
	#######################################################

	self.maxpool1=nn.MaxPool2d(kernel_size=2,stride=2)

	#######################################################

	#########Layer################ 
	self.conv3=BinConv2d2(64*self.inflate,128*self.inflate, kernel_size=3, stride=1, padding=1)
	self.relu3=nn.ReLU(inplace=True)
	self.conv4=BinConv2d(128*self.inflate,128*self.inflate, kernel_size=3, stride=1, padding=1)
	self.relu4=nn.ReLU(inplace=True)
	self.maxpool2=nn.MaxPool2d(kernel_size=2,stride=2)

	#######################################################

	#########Layer################ 
	self.conv5=BinConv2d2(128*self.inflate,256*self.inflate, kernel_size=3, stride=1, padding=1)
	self.relu5=nn.ReLU(inplace=True)
	self.conv6=BinConv2d(256*self.inflate,256*self.inflate, kernel_size=3, stride=1, padding=1)
	self.relu6=nn.ReLU(inplace=True)
	self.conv7=BinConv2d(256*self.inflate,256*self.inflate, kernel_size=3, stride=1, padding=1)
	self.relu7=nn.ReLU(inplace=True)
	self.maxpool3=nn.MaxPool2d(kernel_size=2,stride=2)

	#######################################################

	#########Layer################ 
	self.conv8=BinConv2d2(256*self.inflate,512*self.inflate, kernel_size=3, stride=1, padding=1)
	self.relu8=nn.ReLU(inplace=True)
	self.conv9=BinConv2d(512*self.inflate,512*self.inflate, kernel_size=3, stride=1, padding=1)
	self.relu9=nn.ReLU(inplace=True)
	self.conv10=BinConv2d(512*self.inflate,512*self.inflate, kernel_size=3, stride=1, padding=1)
	self.relu10=nn.ReLU(inplace=True)
	self.maxpool4=nn.MaxPool2d(kernel_size=2,stride=2)

	#######################################################

	#########Layer################ 
	self.conv11=BinConv2d2(512*self.inflate,512*self.inflate, kernel_size=3, stride=1, padding=1)
	self.relu11=nn.ReLU(inplace=True)
	self.conv12=BinConv2d2(512*self.inflate,512*self.inflate, kernel_size=3, stride=1, padding=1)
	self.relu12=nn.ReLU(inplace=True)
	self.conv13=BinConv2d(512*self.inflate,512*self.inflate, kernel_size=3, stride=1, padding=1)
	self.relu13=nn.ReLU(inplace=True)
	self.maxpool5=nn.MaxPool2d(kernel_size=2,stride=2)

	#######################################################

	#########Layer################ 
	self.fc1=BinConv2d(512*self.inflate,1024, Linear=True)
	self.relu14=nn.ReLU(inplace=True)
	self.bn3= nn.BatchNorm1d(1024)
	self.fc3=nn.Linear(1024, num_classes)
	self.logsoftmax=nn.LogSoftmax()

    	
	        #init_model(self)
        #self.regime = {
        #    0: {'optimizer': 'SGD', 'lr': 1e-1,
        #        'weight_decay': 1e-4, 'momentum': 0.9},
        #    81: {'lr': 1e-4},
        #    122: {'lr': 1e-5, 'weight_decay': 0},
        #    164: {'lr': 1e-6}
        #}
        self.regime = {
            0: {'optimizer': 'SGD', 'lr': 5e-3},
            101: {'lr': 1e-3},
            142: {'lr': 5e-4},
            184: {'lr': 1e-4},
            220: {'lr': 1e-5}
        }

def vgghyb(**kwargs):
    num_classes, depth, dataset = map(
        kwargs.get, ['num_classes', 'depth', 'dataset'])
    if dataset == 'cifar100':
        num_classes = num_classes or 100
        depth = depth or 18
        return VGG_cifar100(num_classes=num_classes, depth=depth)
