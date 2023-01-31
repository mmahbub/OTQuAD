import matplotlib.pyplot as plt
import numpy as np

path         = '/net/kdinxidk03/opt/NFS/75y/data/qa/output/REVISION/REVIEWER-2/emrqa/medication/epoch-500/'

main_path = path

loss_path   = path+'loss_epochs.txt'

with open(loss_path, 'r') as f:
    loss = f.readlines()

squad_f1_all  = list(map(float, loss[13][1:-2].split(',')))

ite_ls = []
vv = 0
for i, v in enumerate(squad_f1_all):
  if vv <= v:
    vv = v
    if i >=50:
      ite_ls.append(i)
      print(i , v)
      
for ite in [107, 137, 141]:
  print(f'ITERATION: {ite}')
  configs.trained_model_name = f"model_epoch_{ite}.pt"
  print(configs.trained_model_name)
  
  !python3 train.py