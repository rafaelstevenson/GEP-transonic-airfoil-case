import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.metrics import mean_squared_error, r2_score

one_arity_func = ['(sqrt)', '(sin)', '(exp)', '(ln)', '(inv)', '(gau)', '(X2)']
two_arity_func = ['+', '-', '*', '/']

def ChromToET(chromosome):
    '''Take a string of chromosome change it to
    a dictionary of expression tree{row:element_on_row}'''
    expr_tree = {0: [chromosome[0]]}

    i = 1
    start_counter = 1
    while True:

        take_next = 0
        terminal_row = True

        for element in expr_tree[i - 1]:
            if element in two_arity_func:
                terminal_row = False
                take_next += 2
            elif element in one_arity_func:
                terminal_row = False
                take_next += 1

        if terminal_row==True:
            break

        last_counter = start_counter + take_next
        next_elem = chromosome[start_counter:last_counter]
        expr_tree[i] = next_elem
        start_counter = last_counter

        i += 1

    return expr_tree


def EvaluateET(chromosome, variable_dict, const_list):
    '''Take chromosome and terminal variable dictionary{'symbol':value}
    and perform calculation from the chromosome->ET->calculation->prediction'''

    # Change string to list in each row of ET and change variables to sample value
    expr_tree = ChromToET(chromosome)
    for i in range(len(expr_tree)):  # iterate rows
        el = 0
        el_dc = 0
        for element in expr_tree[i]:  # iterate elements in a row
            if element in variable_dict.keys():
                expr_tree[i][el] = str(variable_dict[element])
            elif element =='?':
                expr_tree[i][el] = str(const_list[el_dc])
            el += 1


    def operate_two_arity(representation, a, b):
        a = float(a)
        b = float(b)

        if representation=='+':
            result = a + b
        elif representation=='-':
            result = a - b
        elif representation=='*':
            result = a * b
        elif representation=='/':
            if b==0:
                b = 1e-6
            result = a / b

        return str(result)

    def operate_one_arity(representation, a):
        a = float(a)

        if representation=='(sqrt)':
            if a >= 0:
                result = a ** 0.5
            else:
                result = (abs(a)) ** 0.5
            # result = math.sqrt(a)

        elif representation=='(sin)':
            result = math.sin(a)
        elif representation=='(exp)':
            try:
                result = math.exp(a)
            except:
                result = 1e6
        elif representation=='(ln)':
            if a==0:
                a = 1e-6
            elif a < 0:
                a = abs(a)
            result = math.log(a, math.e)
        elif representation=='(inv)':
            if a==0:
                a = 1e-6
            result = 1 / a
        elif representation=='(gau)':
            result = np.random.normal(1)
        elif representation=='(X2)':
            result = a ** 2

        return str(result)

    for row in range(len(expr_tree) - 2, -1, -1):  # iterate rows from second last row to root
        i = 0
        for element in expr_tree[row]:
            if element in two_arity_func:
                a = expr_tree[row + 1][0]
                b = expr_tree[row + 1][1]

                result = operate_two_arity(element, a, b)
                # buang 2 elemen pertama di row+1 dan replace elemen pertama di row
                expr_tree[row + 1] = expr_tree[row + 1][2:]
                expr_tree[row][i] = result

            elif element in one_arity_func:
                a = expr_tree[row + 1][0]
                result = operate_one_arity(element, a)
                expr_tree[row + 1] = expr_tree[row + 1][1:]
                expr_tree[row][i] = result

            i += 1

    prediction = float(expr_tree[0][0])
    return prediction


# load dataset to get ground truth
df = pd.read_csv('../testing_datasets/transonic_airfoil_data_set.csv')
y_true = df['Cd']
x = df['AoA']

# declare the chromosome and the variable dictionary
chromosome = ['/', '(X2)', '/', 'a', '(exp)', '*', 'a', 'a', '?', '?', 'a', 'a', '?', '?', 'a', '6', '0', '4', '6', '3', '2', '3', '2']
term_set = ['a']
const_list = [0.01100726, 0.01682561, 0.01555461, 0.01705792, 0.00568456, 0.01645038,0.0129643,  0.01609162]

y_true = df['Cd']

# Perform Prediction
y_pred = []
for i in range(len(pd.DataFrame(x))):
    variable_dict = {}
    nth_input = 0
    for term in term_set:
        variable_dict[term] = pd.DataFrame(x).iloc[i, nth_input]
        nth_input += 1
    prediction = EvaluateET(chromosome, variable_dict, const_list)
    y_pred.append(prediction)

y_pred = np.array(y_pred)

# get metric of predictions
print(f'Prediction Mean Squared Error (MSE): {mean_squared_error(y_true, y_pred)}')
print(f'Prediction R-squared Score (R2): {r2_score(y_true, y_pred)}')

plt.scatter(x, y_pred ,label='Prediction')
plt.scatter(x, y_true, label='Ground Truth')
plt.legend()
plt.show()
