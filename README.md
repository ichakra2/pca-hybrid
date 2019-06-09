# pca-hybrid
Codes for PCA-driven Hybrid networks

This framework helps build hybrid neural networks using PCA-driven methodology (https://arxiv.org/abs/1906.01493) on PyTorch. 

---------------
Requirements:
---------------

- Python 2.7 

- NVIDIA GPU 1080Ti or higher (ImageNet simulations on ResNet-18 roughly take 3-4 days to complete, CIFAR-100 takes 2-3 hours)

-------------
Dependencies:
-------------

- Numpy

- matplotlib

- sklearn

- PyTorch 0.3.1

---------------
Documentation
---------------

Design steps are documented at [docs] (/docs/)

Code of XNOR-Net training algorithm has been derived from: https://github.com/jiecaoyu/XNOR-Net-PyTorch and also included in this code. 

---------------
How To Run
---------------

CIFAR - 100 (Specify model filename to save inside main.py save_state function)

```
python main.py --save test_cifar100 
```
ImageNet (Specify model filename to save inside main.py save_state function)

```
python main_imnet.py --save test_imagenet
```

---------------
Example Model Code
---------------

As the codes themselves are reasonably big, I will provide snippets to help understand the flow:

```
# Making the activations k-bit. 
class BinActive2(torch.autograd.Function):
    '''
    Make the input activations k-bit
    '''
    def forward(self, input):
        self.save_for_backward(input)
        size = input.size()
        mean = torch.mean(input.abs(), 1, keepdim=True)
	      x = input
        xmax = x.abs().max()
        num_bits=2
        v0 = 1
        v1 = 2
        v2 = -0.5
        y = 2.**num_bits - 1.
        x = x.add(v0).div(v1)
        x = x.mul(y).round_()
        x = x.div(y)
        x = x.add(v2)
        x = x.mul(v1)
        input = x
        return input, mean

    def backward(self, grad_output, grad_output_mean):
        input, = self.saved_tensors
        grad_input = grad_output.clone()
        return grad_input
 # Sample Convolution Block with k-bit activations       
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
    
    def forward(self, x):
        x = self.bn(x)
        x, mean = BinActive2()(x)
        if self.dropout_ratio!=0:
            x = self.dropout(x)
        x = self.conv(x)
        return x  
        
#Definition of a sample conv block
self.conv=BinConv2d2(int(512*self.inflate), int(512*self.inflate), kernel_size=3, stride=1, padding=1)
self.bn= nn.BatchNorm2d(int(512*self.inflate))
self.relu=nn.ReLU(inplace=True)

#Full Code can be found in [models]  (/models/)      
```
