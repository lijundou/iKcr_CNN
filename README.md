1. Description  
It is a novel bioinformatics tool, named iKcr_CNN, to predict human non-histone Kcr sites based on the convolutional neuron network architecture, and the focal loss is implemented to solve the imbalance problem of data distribution. The software can be freely downloaded from the Git-Hub platform:  https://github.com/lijundou/iKcr_CNN/. 
2. Requirements 
Before prediction, please make sure the following packages are installed in the Python environment: 
python==3.6.13
keras==2.2.4
tensorflow==1.12.0
numpy==1.16.0
pandas==1.1.5
scikit_learn=0.24.2
Here, we provided both CPU and GPU-based models. For fast prediction, it is recommended to install the tensorflow package of GPU version if GPU is available.
3. Running
1) Prepare a Fasta file to record the protein sequence to be predicted;
2) Run the following command to perform prediction:
    If CPU: python iKcr_CNN_cpu.py -i  example.fasta -o results.csv
    If GPU: python iKcr_CNN_gpu.py -i  example.fasta -o results.csv
3) The prediction results are summarized in the file results file, including four columns of 'Model name', 'Protein ID', 'Sequence', 'Score' and 'Label'. 
