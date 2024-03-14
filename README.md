# ARIMA_30
Arima model 
``` python
#import the necessary libraries 
import pandas as pd
from sklearn.model_selection import train_test_split
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt 
import warnings
warnings.filterwarnings('ignore')

# Read the Excel file 'monthly-champagne-sales.xlsx' into a DataFrame called df 
df = pd.read_excel('monthly-champagne-sales.xlsx') 

# Convert the 'Month' column in the DataFrame to datetime format using the pd.to_datetime() function
df['Month'] = pd.to_datetime(df['Month']) 

df.head() 


# Create a variable X to store the 'Month' column, which will be used as the independent variable
X = df['Month']

# Create a variable y to store the 'Sales' column, which will be used as the dependent variable
y = df['Sales'] 

plt.figure(figsize=(10, 6)) #Setting the size of the plot 

# Plot the data using X (Months) as the x-axis and y (Sales) as the y-axis. Add circular markers at data points.
plt.plot(X, y, marker='o')
plt.title('Monthly Champagne Sales') # Set the title of the plot 
plt.xlabel('Month') # Label the x-axis with 'Month' 
plt.ylabel('Sales') # Label the y-axis with 'Sales' 
plt.grid(True) # Turn on the grid lines in the plot 
plt.show() # Display the plot 

# Extract the month from the 'Month' column to create a new 'Month' column with month values
df['Month'] = df['Month'].dt.month 

# Create a new figure for the plot with a specific size 
plt.figure(figsize=(10, 6))

# Create a box plot for the 'Sales' column, grouped by the 'Month' column, without showing outliers
df.boxplot(column='Sales', by='Month', showfliers=False)


plt.title('Monthly Champagne Sales Box Plot (Month-wise)') # Set the title of the plot 
plt.xlabel('Month') # Label the x-axis with 'Month' 
plt.ylabel('Sales') # Label the y-axis with 'Sales' 
plt.grid(True) # Turn on the grid lines in the plot 
plt.show() # Display the plot 


#Check for trends, seasonality, and p-value to determine if time series is stationary.
#Use Augmented Dickey Fuller Test (ADF Test) to test if time series is stationary or not.
#Perform differencing if necessary to make time series stationary.


df_decompose = pd.read_excel('monthly-champagne-sales.xlsx') 

# Convert the 'Month' column in the DataFrame to datetime format using the pd.to_datetime() function
df_decompose['Month'] = pd.to_datetime(df_decompose['Month']) 
df_decompose 

df_decompose.set_index('Month',inplace=True) 
df_decompose.head() 

from statsmodels.tsa.seasonal import seasonal_decompose

decomposition = seasonal_decompose(df_decompose) 

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.figure(figsize=(16, 7))
plt.subplot(411)
plt.plot(df_decompose['Sales'], label='Original')
plt.legend(loc='best')

plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')

plt.subplot(413)
plt.plot(seasonal,label='Seasonality')
plt.legend(loc='best')

plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='best')
plt.tight_layout()

#Observations: There is an increasing trend.

from statsmodels.tsa.stattools import adfuller 

def stationarity_test(data_sales): 
    # ADF 
    dftest = adfuller(data_sales, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)

stationarity_test(df_decompose['Sales'])

#Observation:The p-value is 0.36 which is greater than 0.05, therefore the data is non-stationary.

# take first difference 
first_diff = df_decompose['Sales'] - df_decompose['Sales'].shift(1)
first_diff = first_diff.fillna(0) 
first_diff = pd.DataFrame(first_diff)

# checking if data is stationary
stationarity_test(first_diff) 

plt.plot(first_diff.index, first_diff.Sales)
plt.show()

#Observation:The p-value is less than 0.05, therefore the data is stationary.

from statsmodels.tsa.stattools import acf, pacf
import numpy as np 

# nlags - number of lags 
lag_acf = acf(first_diff, nlags=15)
lag_pacf = pacf(first_diff, nlags=15, method='ols')
plt.figure(figsize=(16, 7))

#Plotting ACF plot:
plt.subplot(121)
plt.plot(lag_acf, marker="o")
plt.axhline(y=0,linestyle='--',color='gray')

# confidence interval 
plt.axhline(y=-1.96/np.sqrt(len(first_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(first_diff)),linestyle='--',color='gray')
plt.title('Autocorrelation Function')

#Plotting PACF Plot:
plt.subplot(122)
plt.plot(lag_pacf, marker="o")
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(first_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(first_diff)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')


# Split the data into training and testing sets using a test size of 20% and preserving the order (shuffle=False)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False) 

from sklearn.metrics import mean_squared_error as MSE

p, d, q = 8, 1, 8  # Example values, you can change these

# Create an ARIMA model instance with the specified order (p, d, q)
arima_model = ARIMA(y_train, order=(p, d, q))

# Fit the ARIMA model to the training data
arima_fit = arima_model.fit()

# Predict your test data
y_pred = arima_fit.predict(start=len(y_train), end=len(y_train) + len(y_test) - 1, typ='levels') 

# Calculate the squared error
squared_error = (y_pred - y_test)**2
mse = squared_error.mean()

print("Mean Squared Error:", mse) 


print("Mean Squared Error:", MSE(y_test,y_pred)) # using sklearn to calculate MSE 

```
# Link to access data 
https://vocproxy-1-8.us-west-2.vocareum.com/files/home/labsuser/monthly-champagne-sales.xlsx?_xsrf=2%7C299393b7%7C0ec33d3124d8ae352e5f3aa26077ecdd%7C1709593587
```
