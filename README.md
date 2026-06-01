# KDD2026-SpectraMB
code of Dynamic Spectral Denoising with Global-Context Attention for Multi-Behavior Recommendation
## Requirements
python==3.9.23
torch==2.3.1+cu121
numba==0.60.0
numpy==1.26.3
pandas==2.3.2
## An example to run SpectraMB
### Taobao
python main.py --data_name='taobao' --content_coeff=1.25 --processed_global_coeff=0 
### ml
python main.py --data_name='ml' --content_coeff=1 --processed_global_coeff=1 
### Tmall
python main.py --data_name='tmall' --content_coeff=1 --processed_global_coeff=0
