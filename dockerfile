FROM determinedai/environments:cuda-11.3-pytorch-1.10-tf-2.8-gpu-9119094
RUN pip install colossalai==0.1.10+torch1.10cu11.3 -f https://release.colossalai.org
RUN pip install titans transformers
