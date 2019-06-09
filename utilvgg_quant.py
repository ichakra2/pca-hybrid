
import torch.nn as nn
import numpy

class BinOp():
    def __init__(self, model):
         # count the number of Conv2d or linear
	count_targets = 0
        for m in model.modules():
	    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                count_targets = count_targets + 1
        print(count_targets)
	raw_input()
        start_range = 1
        end_range = count_targets-2
        if start_range==end_range:
           self.bin_range = numpy.linspace(start_range,
                end_range, end_range-start_range)\
                        .astype('int').tolist()
        else:

           self.bin_range = numpy.linspace(start_range,
                end_range, end_range-start_range+1)\
                        .astype('int').tolist()
	print(self.bin_range)
	#raw_input()
	res_conn = numpy.array([])
	kbit_conn = numpy.array([2,4,7,10,11]) #layers whose weights need to be made k-bit
	#kbit_conn = numpy.array([])
	#kbit_conn = numpy.array([3,4,5,6,11,12,13,22,23,24,25])
	res_conn = res_conn.astype('int').tolist()
	kbit_conn = kbit_conn.astype('int').tolist()

	print(kbit_conn)
	raw_input()
	self.bin_range = (list(set(self.bin_range) - set(res_conn)))
	print(self.bin_range)
	raw_input()
        self.num_of_params = len(self.bin_range)
        self.saved_params = []
        self.target_params = []
        self.target_modules = []
	self.onlybin_range = (list(set(self.bin_range) - set(kbit_conn)))
        index = -1
	print(self.num_of_params)
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
	        print(m)
		raw_input()
                index = index + 1
                if index in self.onlybin_range:
		    print('Binarizing')
		    raw_input()
                    tmp = m.weight.data.clone()
                    self.saved_params.append(tmp)
                    self.target_modules.append(m.weight)
		elif index in kbit_conn:
		     print('Making k-bit') #Know which layers weights are being made k-bit
		     raw_input()
                     tmp = m.weight.data.clone()
                     self.saved_params.append(tmp)
                     self.target_modules.append(m.weight)


    def binarization(self):
        
        self.meancenterConvParams()
        self.clampConvParams()
        self.save_params()
        self.binarizeConvParams()
	

    def meancenterConvParams(self):
        for index in range(self.num_of_params):
            s = self.target_modules[index].data.size()
            negMean = self.target_modules[index].data.mean(1, keepdim=True).\
                    mul(-1).expand_as(self.target_modules[index].data)
            self.target_modules[index].data = self.target_modules[index].data.add(negMean)

    def clampConvParams(self):
        for index in range(self.num_of_params):
            self.target_modules[index].data = \
                    self.target_modules[index].data.clamp(-1.0, 1.0)

    def save_params(self):
        for index in range(self.num_of_params):
            self.saved_params[index].copy_(self.target_modules[index].data)

    def binarizeConvParams(self):
    	kbit_conn = numpy.array([1,3,6,9,10]) #list of conv layers whose weights should be k-bit (starts from 0, hence -1 from other list)
	#kbit_conn = numpy.array([])
	#kbit_conn = numpy.array([2,3,4,5,10,11,12,21,22,23,24])
	kbit_conn = kbit_conn.astype('int').tolist()

        for index in range(self.num_of_params):
            n = self.target_modules[index].data[0].nelement()
            s = self.target_modules[index].data.size()
	    if len(s) == 4:
                m = self.target_modules[index].data.norm(1, 3, keepdim=True)\
                        .sum(2, keepdim=True).sum(1, keepdim=True).div(n)
            elif len(s) == 2:
                m = self.target_modules[index].data.norm(1, 1, keepdim=True).div(n)
	    
	    if index in kbit_conn:
	    	 #k-bit weights
	   	 x = self.target_modules[index].data
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
	   	 self.target_modules[index].data = x.mul(m.expand(s))
	    else:
	    	 #Binary weights
		 self.target_modules[index].data = \
                    self.target_modules[index].data.sign().mul(m.expand(s))

    def restore(self):
        for index in range(self.num_of_params):
            self.target_modules[index].data.copy_(self.saved_params[index])

    def updateBinaryGradWeight(self):
        for index in range(self.num_of_params):
            weight = self.target_modules[index].data
            n = weight[0].nelement()
            s = weight.size()
	    if len(s) == 4:
                m = weight.norm(1, 3, keepdim=True)\
                        .sum(2, keepdim=True).sum(1, keepdim=True).div(n).expand(s)
            elif len(s) == 2:
                m = weight.norm(1, 1, keepdim=True).div(n).expand(s)
            m[weight.lt(-1.0)] = 0 
            m[weight.gt(1.0)] = 0
            # m = m.add(1.0/n).mul(1.0-1.0/s[1]).mul(n)
            # self.target_modules[index].grad.data = \
            #         self.target_modules[index].grad.data.mul(m)
            m = m.mul(self.target_modules[index].grad.data)
            m_add = weight.sign().mul(self.target_modules[index].grad.data)
	    if len(s) == 4:
                m_add = m_add.sum(3, keepdim=True)\
                        .sum(2, keepdim=True).sum(1, keepdim=True).div(n).expand(s)
            elif len(s) == 2:
                m_add = m_add.sum(1, keepdim=True).div(n).expand(s)

            m_add = m_add.mul(weight.sign())
            self.target_modules[index].grad.data = m.add(m_add).mul(1.0-1.0/s[1]).mul(n)

