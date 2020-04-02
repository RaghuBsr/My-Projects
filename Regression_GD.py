import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#Load the data files
root = 'C:/New folder/'
df=pd.read_excel(root+'Adops & Data Scientist Sample Data.xlsx',sheet_name="Q2 Regression")
df.columns = ['A','B','C']

A=df['A']
B=df['B']
c=df['C']

#Visualization to check for outliers
sns.set(style="whitegrid")
ax = sns.boxplot(x=df['C'])
ax = sns.boxplot(x=df['A'])
ax = sns.boxplot(x=df['B'])

#Dropping the outlier values to improve the prediction(Totally 6 rows dropped)
df['C'].min()
# -9999.0 is the outlier value
df = df[df.C != -9999.0]
df = df[df.A <= 15]

#Stanadardize the data
df=(df-df.mean())/df.std()

#Assign length of the dataframe to M and also add a new identity vector to feature dataframe
m=len(df)
xO=np.ones(m)

#reassign feature names
A=df['A']
B=df['B']
c=df['C']
X = np.array([xO, A, B]).T

y=np.array(c)

B = np.array([0, 0,0])
B.shape
alpha = 0.015

#Cost function
def cost_function(X, y, B):
    m = len(y)
    J = np.sum((X.dot(B) - y) ** 2)/(2 * m)
    return J

inital_cost = cost_function(X, y, B)
print("Initial Cost")
print(inital_cost)

#Gradient descent function
def gradient_descent(X, Y, B, alpha, iterations):
    cost_history = [0] * iterations
    m = len(Y)
    
    for iteration in range(iterations):
        h = X.dot(B)
        loss = h - Y
        gradient = X.T.dot(loss) / m
        B = B - alpha * gradient
        cost = cost_function(X, Y, B)
        cost_history[iteration] = cost
        
    return B, cost_history

newB, cost_history = gradient_descent(X, y, B, alpha, 1000)


print("New Coefficients")
print(newB)

print("Cost")
print(cost_history[-1])

#Function to calculate RMSE
def rmse(y, y_pred):
    rmse = np.sqrt(sum((y - y_pred) ** 2) / len(y))
    return rmse

#Function to calculate Rsquare value
def r2_score(y, y_pred):
    mean_y = np.mean(y)
    ss_tot = sum((y - mean_y) ** 2)
    ss_res = sum((y - y_pred) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2

y_pred = X.dot(newB)

print("RMSE:")
print(rmse(y, y_pred))
print("R2 Score:")
print(r2_score(y, y_pred))

#plot
fig, ax = plt.subplots()
ax.scatter(y, y_pred)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()