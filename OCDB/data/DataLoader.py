import numpy as np
import pandas as pd
import os
import wget
import zipfile

'''

'''
class DataLoader():
    def __init__(self, name, root="./DCGL/data/dataset") -> None:
        self.name = name
        self.root = root
    
    def data(self):
        # data_url = {"MemeTracker": "https://drive.google.com/u/0/uc?id=1u9ELnd3FtmGkjOX2VjIlTr31IHKReeog&export=download&confirm=t",
        #             }
        
        length = {"Finance": 9,
                  "fMRI": 28}

        if not os.path.exists(self.root):
            assert f"self.root not find"
        else:
            # if not os.path.exists(f"{self.root}/{self.name}"):
            #     os.makedirs(f"{self.root}/{self.name}")
            #     print(f"Downloading dataset {self.name}.")
            #     wget.download(data_url[self.name], f"{self.root}/{self.name}/{self.name}.zip", bar=wget.bar_adaptive)
            #     print(f"Unzipping begins.")
            #     file=zipfile.ZipFile(f"{self.root}/{self.name}/{self.name}.zip")
            #     file.extractall(f"{self.root}/{self.name}")
            #     file.close()
            #     os.remove(f"{self.root}/{self.name}/{self.name}.zip")
            #     print(f"Unzipping is complete and the zip is removed.")
            if self.name in {"sachs", "DWD", "CCS_Data", "Cad", "Abalone", "Auto_mpg", "Ozone"}:
                return self.load_static_data()
            # times series data
            elif self.name in {"NetSim", "Finance", "fMRI"}:
                if self.name == "NetSim":
                    return self.load_static_data()
                else:
                    return self.load_time_series_data(length[self.name])
            # event series data
            elif self.name in {"Wireless", "Microwave24V", "Microwave25V"}:
                return self.load_event_series_data()
            else:
                assert f"{self.name} not exists"

    def load_static_data(self):
        feature_pd = pd.read_csv(f"{self.root}/{self.name}/{self.name}.csv", encoding="gbk", index_col=0)
        ground_truth_pd = pd.read_csv(f"{self.root}/{self.name}/{self.name}_ground_truth.csv", encoding="gbk", index_col=0)
        columns = list(feature_pd.columns)
        feature = feature_pd.to_numpy()
        ground_truth = ground_truth_pd.to_numpy()
        return columns, feature, ground_truth
    
    def load_time_series_data(self, length):
        columns = []
        feature = []
        ground_truth = []
        for i in range(length):
            feature_pd = pd.read_csv(f"{self.root}/{self.name}/{self.name}{i+1}.csv", encoding="gbk", index_col=0)
            ground_truth_pd = pd.read_csv(f"{self.root}/{self.name}/{self.name}{i+1}_ground_truth.csv", encoding="gbk", index_col=0)
            columns.append(list(feature_pd.columns))
            feature.append(feature_pd.to_numpy())
            ground_truth.append(ground_truth_pd.to_numpy())
        return columns, feature, ground_truth

    def load_event_series_data(self):
        event_table = pd.read_csv(f"{self.root}/{self.name}/{self.name}.csv", encoding="gbk", index_col=0)
        ground_truth_pd = pd.read_csv(f"{self.root}/{self.name}/{self.name}_ground_truth.csv", encoding="gbk", index_col=0)
        columns = list(ground_truth_pd.columns)
        ground_truth = ground_truth_pd.to_numpy()
        return columns, event_table, ground_truth