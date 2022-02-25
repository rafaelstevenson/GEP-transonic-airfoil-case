import numpy as np
import pandas as pd
from GEP import GeneExpressionProgramming


df = pd.read_csv('../testing_datasets/transonic_airfoil_data_set.csv')
print(df.head())

class MinMaxScaler:
    def fit_transform(self, arr):
        self.max = max(arr)
        self.min = min(arr)
        arr = arr-self.min/(self.max-self.min)
        return arr
    def transform(self, arr):
        arr = arr-self.min/(self.max-self.min)
        return arr

#scaler_x = MinMaxScaler()
#scaler_y = MinMaxScaler()
#df['AoA'] = scaler_x.fit_transform(df['AoA'])
#df['Cd'] = scaler_y.fit_transform(df['Cd'])


func_set = ['+','-','*','/', '(sqrt)','(exp)', '(ln)']

term_set = ['a']
operator_probabilities = {
    "Mutation":0.2, "Inversion":0.1, "IS Transposition":0.1,
    "RIS Transposition":0.1, "One-point Recombination":0.3,
    "Two-point Recombination":0.3
}

head_length = 7
population_size = 300
generations = 20
fitness_func = 'r2'

GEPProcess = GeneExpressionProgramming(head_length,func_set,term_set,operator_probabilities)
GEPProcess.RunGEP(df['AoA'],df['Cd'],population_size,generations,fitness_func)
GEPProcess.VisualizeResults()