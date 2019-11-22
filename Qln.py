import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
dados = pd.read_excel('Dados_Rf.xlsx', index = False)
labels = np.array(dados['Cl'])
dados = dados.drop('Cl',axis = 1)
dados = np.array(dados)
# Split the data into training and testing sets
train_dados, test_dados, train_labels, test_labels = train_test_split(dados, labels, test_size = 0.30, random_state = 42)
#Instantiate model with 800 decision trees
rf = RandomForestRegressor(n_estimators = 800, random_state = 42)
rf.fit(train_dados, train_labels)
# Use the forest's predict method on the test data
predictions = rf.predict(test_dados)
# Calculate the absolute errors
errors = abs(predictions - test_labels)
# Print out the mean absolute error (mae)
print('Erro medio absoluto:', round(np.mean(errors), 2), '[Cl].')
# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / test_labels)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')
