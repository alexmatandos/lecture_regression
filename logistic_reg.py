from sklearn import linear_model
import pandas
#import statmodel to do regressions in one line in python

dataset = pandas.read_csv("dataset.csv")
#print(dataset)

#'iloc' locates elements within dataframe
column_1 = dataset.iloc[:, 1]
#print(column_1)

#remembering that the last number of the colon is not inclusive, thus, to include the last element add 1 to its position
data = dataset.iloc[:, 3:9]
print(data)

#always convert dataframe to matrix by adding the extension '.values'

dataset_matrix = dataset.values
data_matrix = data.values

machine = linear_model.LogisticRegression()
#print(machine)
#the 'fit' is the machine learning the relationships between 'y1' and other six independent variables
machine.fit(data, column_1)
#print(machine)

#adding data
new_data = [[0.99, .123, .764, .256, .420, 0]]

#now that the machine has already learned the relationships, you can forecast the value for 'y1' for other values of the independent variables
new_column = machine.predict(new_data)
print(new_column)