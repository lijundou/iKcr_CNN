#1. Description  <br>
It is a novel bioinformatics tool, named iKcr_CNN, to predict human non-histone Kcr sites based on the convolutional neuron network architecture, and the focal loss is implemented to solve the imbalance problem of data distribution. The software can be freely downloaded from the Git-Hub platform:  https://github.com/lijundou/iKcr_CNN/. <br>
#2. Requirements <br>
>Before prediction, please make sure the following packages are installed in the Python environment: <br>
python==3.6.13 <br>
keras==2.2.4 /<br>
tensorflow==1.12.0 <br>
numpy==1.16.0 <br>
pandas==1.1.5 <br>
scikit_learn=0.24.2 <br>
Here, we provided both CPU and GPU-based models. For fast prediction, it is recommended to install the tensorflow package of GPU version if GPU is available.<br>
3. Running <br>
* Prepare a Fasta file to record the protein sequence to be predicted; <br>
* Run the following command to perform prediction: <br>
    If CPU: python iKcr_CNN_cpu.py -i  example.fasta -o results.csv <br>
    If GPU: python iKcr_CNN_gpu.py -i  example.fasta -o results.csv <br>
* The prediction results are summarized in the file results file, including four columns of 'Model name', 'Protein ID', 'Sequence', 'Score' and 'Label'. 
