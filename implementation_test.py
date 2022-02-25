import pandas as pd
from GEP import GeneExpressionProgramming


df = pd.read_csv('../testing_datasets/transonic_airfoil_data_set.csv')
print(df.head())

func_set = ['+','-','*','/', '(sqrt)']

term_set = ['a']
operator_probabilities = {
    "Mutation":0.2, "Inversion":0.1, "IS Transposition":0.1,
    "RIS Transposition":0.1, "One-point Recombination":0.3,
    "Two-point Recombination":0.3
}

head_length = 7
population_size = 100
generations = 5
fitness_func = 'mse'

GEPProcess = GeneExpressionProgramming(head_length,func_set,term_set,operator_probabilities)
GEPProcess.RunGEP(df['AoA'],df['Cd'],population_size,generations,fitness_func)
GEPProcess.VisualizeResults()