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


















```
# Link to access data 
https://vocproxy-1-8.us-west-2.vocareum.com/files/home/labsuser/monthly-champagne-sales.xlsx?_xsrf=2%7C299393b7%7C0ec33d3124d8ae352e5f3aa26077ecdd%7C1709593587
```
