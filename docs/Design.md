Steps to design Hybrid-Net

1. Train a XNOR-Net (For ResNet, train a plain network)

```
python main.py --arch resnet
```

2. Run evaluate using main_evaluate.py. This will generate multiple .out files named PCA_files_*.out. 

3. Run PCAplotresnet20.m to plot PCA analysis results of layer-wise significant components. Identify the significant layers according to algorithm in paper. 

4. Design Hybrid-Net
