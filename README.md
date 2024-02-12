# It's DeepPacketAttack project for classification over encrypted traffic. 

# Installation and preparation:

1. Download and install cuda 11.7 if you want to use GPU:
https://developer.nvidia.com/cuda-11-7-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exe_local

2. Add %CUDA_PATH% env.var.: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7

3. Update %PATH% env.var.: %PATH% + C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7\bin + C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7\libnvvp

4. Downdload and install miniconda: https://docs.anaconda.com/free/miniconda/


```bash
conda env create -f env_linux_cuda117.yaml
```

```bash
python -m pip uninstall torch
```

```bash
python -m pip cache purge
```

```bash
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
```

```bash
python
```

```python
import torch
```

```python
torch.cuda.is_available()
```

```python
exit()
```

```bash
pip install click jupyterlab matplotlib datasets pandas plotly pyspark pytorch-lightning scapy[complete]==2.5.0rc1 scikit-learn seaborn tensorboard
```

# Unexpected dependencies:

Check guide https://kontext.tech/article/377/latest-hadoop-321-installation-on-windows-10-step-by-step-guide (not everything needed, but something is necessary)

hadoop-3.2.1: https://github.com/apache/hadoop/tree/release-3.2.1-RC0?ysclid=lsdd29z0h3791672047 and https://github.com/cdarlint/winutils/tree/master/hadoop-3.2.1/bin

maven-3.9.6: https://maven.apache.org/download.cgi 

# CLEAR DATASET PIPELINE
1. python preprocessing.py -s clear_pcaps -t clear_processed_data
2. python create_train_test_set.py -s clear_processed_data -t train_test_data
4. python train_cnn.py -p train_test_data/application_classification/train.parquet -r train_test_data/traffic_classification/train.parquet -a model/application_classification.cnn.model -t model/traffic_classification.cnn.model -v both
5. python eval_cnn.py --ct app --gpu True
6. python eval_cnn.py --ct traffic --gpu True
---------------------------------------------------------
# OBFUSCATED DATASET PIPELINE
1. python preprocessing.py -s obfuscated_pcaps -t obfuscated_processed_data
2. python create_test_set.py -s obfuscated_processed_data -t only_test_data
3. python eval_cnn.py --actdp only_test_data/app_classification/test.parquet --ct app --gpu True
4. python eval_cnn.py --tctdp only_test_data/traffic_classification/test.parquet --ct traffic --gpu True
