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















```
# Link to access data 
https://vocproxy-1-8.us-west-2.vocareum.com/files/home/labsuser/monthly-champagne-sales.xlsx?_xsrf=2%7C299393b7%7C0ec33d3124d8ae352e5f3aa26077ecdd%7C1709593587
```
