# OCDB
Open Causal Discovery Benchmark

### Running Environment
```
python >= 3.8
torch >= 1.12.1
numpy >= 1.23.3
networkx >= 2.8.4
gcastle >= 1.0.3
cdt >= 0.6.0
```
### Usage Example
```python
from DCGL.data import DataLoader
from DCGL.model import DAG_GNN
import torch

# data pre-processing
columns, feature, ground_trues = DataLoader(name="NetSim").data()

# device = torch.device("cpu")
device = torch.device("cuda:0")

# model initialization
model = DAG_GNN(device, num_variable=len(columns), x_dim=1, hidden_dim_encoder=64, 
                hidden_dim_decoder=64, output_dim=len(columns), batch_size=1000)
# building the causal structure from data
model.fit(feature)
# evaluation
model.eval(ground_trues)
```

### Reproduce
To replicate the results from the paper, please run <strong><em>train_and_eval.ipynb</em></strong>.
