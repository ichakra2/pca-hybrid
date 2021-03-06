# pca-hybrid

--------------
License Information
---------------

Copyright (C) 2019  Indranil Chakraborty

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

Codes for PCA-driven Hybrid networks

This framework helps build hybrid neural networks using PCA-driven methodology (https://arxiv.org/abs/1906.01493) on PyTorch. 

---------------
Requirements:
---------------

- Software -  Python 2.7 on RedHat or any other Linux distributions

- Hardware -  NVIDIA GPU 1080Ti or higher (ImageNet simulations on ResNet-18 roughly take 3-4 days to complete, CIFAR-100 takes 2-3 hours)

-------------
Dependencies:
-------------

- Numpy

- matplotlib

- sklearn

- PyTorch 0.3.1

----------------
Installation of Software
----------------
Installation instructions of above softwares are publicly available: https://pytorch.org/get-started/locally/

Install time is negligible. 

---------------
Documentation
---------------

Design steps are documented at [docs](/docs/design.md). This contains design steps of hybrid networks as well as codes to generate models based on specified significant layers. 

Code of XNOR-Net training algorithm has been derived from: https://github.com/jiecaoyu/XNOR-Net-PyTorch and also included in this code. 

---------------
How To Run a Hybrid-Network after Design
---------------

CIFAR - 100 (Specify model filename to save inside main.py save_state function)
Run time - 2-3 hours on NVIDIA TitanXP
```
python main.py --save test_cifar100 
```
ImageNet (Specify model filename to save inside main.py save_state function)
Run time - 3-4 days on NVIDIA 1080Ti
```
python main_imnet.py --save test_imagenet
```
---------------
Reproduction Instructions
---------------
All the codes use seeded random variables. However variations in seeds can lead to slightly different results. Exact reproducibility in deep neural networks is always difficult especially if number of workers are more. However, if seed is same for all cases, similar trends for different comparison networks should be observable.

---------------
Example Codes
---------------

Example PCA code in main_evaluate.py to get PCA plot

```
def run_PCA(activations_collect,key_idx, components, threshold=0.999):
        """threshold for minimal loss in performance=0.999
        activations_collect  function gathers activations over enough mini batches.
        components=number of filters in the layer you are compressing
        This is for a layer, you need to run this for multiple layers and store optimal_num_filters into a vector
        This vector is the significant dimensionality of all layers"""
        
        print('number of components are',components)
        activations=activations_collect[key_idx]#.replace('.weight','')]
	activations = (activations.data).cpu().numpy()
        print('shape of activations are:',activations.shape)
        a=activations.swapaxes(1,2).swapaxes(2,3)
        a_shape=a.shape
        print('reshaped ativations are of shape',a.shape)
	raw_input()
        pca = PCA(n_components=components) #number of components should be equal to the number of filters
        pca.fit(a.reshape(a_shape[0]*a_shape[1]*a_shape[2],a_shape[3])) #this should be N*H*W,M
        a_trans=pca.transform(a.reshape(a_shape[0]*a_shape[1]*a_shape[2],a_shape[3]))
        print('explained variance ratio is:',pca.explained_variance_ratio_)
	raw_input()
        plt.plot(numpy.cumsum(pca.explained_variance_ratio_))
	numpy.savetxt('./PCA_files_'+str(key_idx)+'.out',numpy.cumsum(pca.explained_variance_ratio_))
	raw_input()
        optimal_num_filters=numpy.sum(numpy.cumsum(pca.explained_variance_ratio_)<threshold) 
        print('we want to retain this percentage of explained variance',threshold)
        print('number of filters required to explain that much variance is',optimal_num_filters)
	return optimal_num_filters,pca.components_
```
This generates multiple files which can be put in a folder and plotted with function PCAplotresnet.m.

As the codes themselves are quite big, I will provide snippets to help understand the flow. This is a snippet to define both binary and k-bit layers in main.py.

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

#Full Code can be found in [models](./models/)      
```

