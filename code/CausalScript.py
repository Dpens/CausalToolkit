from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel  
from dowhy import CausalModel
import networkx as nx
import json
from castle.algorithms import ICALiNGAM, DirectLiNGAM, GraNDAG, GOLEM, DAG_GNN, Notears, NotearsNonlinear 
from OCDB.model import *
import torch

app = FastAPI()  

class PayLoad(BaseModel):  
    data: list
    columns: list
    method: str
    keyword: str

@app.post("/discover_causal_relationship")  
async def discover_causal_relationship(payload: PayLoad):  
    def causal_knowledge_base(keyword):
        with open("./causal_knowledge_base.json") as f:
            knowledge_dict = json.load(f)
        if keyword not in knowledge_dict.keys():
            return {"causal_information": "None"}
        else:
            return {"causal_information": knowledge_dict[keyword]}
    
    def causal_tools(data, columns, method):
        device = f'cuda:0' if torch.cuda.is_available() else 'cpu'
        if method == "ICALiNGAM":
            model = ICALiNGAM(random_state=42, max_iter=20, thresh=0.1)
            model.learn(data)
            causal_matrix = model.causal_matrix
        elif method == "DirectLiNGAM":
            model = DirectLiNGAM(thresh=0.5)
            model.learn(data)
            causal_matrix = model.causal_matrix
        elif method == "Notears":
            model = Notears(w_threshold=0.7)
            model.learn(data)
            causal_matrix = model.causal_matrix
        elif method == "NotearsNonlinear":
            model = NotearsNonlinear(max_iter=5, w_threshold=0.5, device_type="gpu", device_ids=0, rho_max=1e5)
            model.learn(data)
            causal_matrix = model.causal_matrix
        elif method == "DAG_GNN":
            model = DAG_GNN(encoder_hidden=128, decoder_hidden=128, lr=0.001, epochs=100, k_max_iter=20, 
                            encoder_dropout=0.0, decoder_dropout=0.0,
                            encoder_type="mlp", decoder_type="mlp", device_type="gpu", device_ids=0)
            model.learn(data)
            causal_matrix = model.causal_matrix
        elif method == "GraNDAG":
            model = GraNDAG(input_dim=data.shape[1], hidden_num=1, hidden_dim=50, batch_size=64)
            model.learn(data)
            causal_matrix = model.causal_matrix
        elif method == "GOLEM":
            model = GOLEM(lambda_1=0.02, lambda_2=6, graph_thres=0.3, device_type="gpu", device_ids=0, num_iter=20000)
            model.learn(data)
            causal_matrix = model.causal_matrix
        elif method == "TCDF":
            model = TCDF(device, len(columns), len(columns), kernel_size=128, hidden_layers=2, threshold=1)
            model.fit(data, epochs=50, significance=0.5, lr=0.001)
            causal_matrix = model.get_DAG()
        elif method == "GVAR":
            model = model = GVAR(device, len(columns), num_hidden_layers=1, hidden_layer_size=64, order=6)
            model.fit(data, Q=20, alpha=0.5, epochs=100)
            causal_matrix = model.get_DAG()
        elif method == "NTiCD":
            model = NTiCD(device, len(columns), input_size=1, output_size=1, hidden_dim=64, n_layers=3)
            model.fit(data, epochs=10, batch_size=64, lr=0.001, window_size=1)
            causal_matrix = model.get_DAG()
        elif method == "best-static":
            model = GraNDAG(input_dim=data.shape[1], hidden_num=1, hidden_dim=50, batch_size=64)
            model.learn(data)
            causal_matrix = model.causal_matrix
        elif method == "best-time-series":
            model = TCDF(device, len(columns), len(columns), kernel_size=128, hidden_layers=2, threshold=1)
            model.fit(data, epochs=50, significance=0.5, lr=0.001)
            causal_matrix = model.get_DAG()
        else:
            raise ValueError(f"method {method} not exists!")

        causal_information = []
        edge_list = []
        for p in range(len(columns)):
            for q in range(len(columns)):
                if causal_matrix[p, q] != 0:
                    edge_list.append((columns[p], columns[q]))

        causal_graph = nx.DiGraph()
        for k in range(5):
            causal_graph.add_node(f"V{k}")
        causal_graph.add_edges_from(edge_list)
        for node_i in columns:
            for node_j in columns:
                if node_i == node_j:
                    continue
                model= CausalModel(
                    data=data,
                    treatment=node_i,
                    outcome=node_j,
                    graph=causal_graph)
                identified_estimand = model.identify_effect()
                causal_estimate = model.estimate_effect(identified_estimand,
                        method_name="backdoor.linear_regression")
                if causal_matrix[int(node_i[-1]), int(node_j[-1])] == 1:
                    causal_information.append(f"在因果图中存在一条从{node_i}(原因)指向{node_j}(结果)的边，干预{node_i}对{node_j}的平均处理效应(ATE)是{causal_estimate.value}。")
                else:
                    causal_information.append(f"在因果图中不存在一条从{node_i}(原因)指向{node_j}(结果)的边，干预{node_i}对{node_j}的平均处理效应(ATE)是{causal_estimate.value}。")
    
        return {"causal_information": ",".join(causal_information)}
    
    try:
        data = payload.data
        columns = payload.columns
        method = payload.method
        keyword = payload.keyword
        print(keyword)
        if keyword != None:
            return causal_knowledge_base(keyword)
        else:
            return causal_tools(data, columns, method)
        
    except Exception as e:  
        import traceback  
        error_message = traceback.format_exc()  
        raise HTTPException(status_code=400, detail=error_message)  
    

if __name__ == "__main__":  
    import uvicorn  
    uvicorn.run(app, host="127.0.0.1", port=8001, reload=True)  