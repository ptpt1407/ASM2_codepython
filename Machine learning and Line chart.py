import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
file_path = 'data_BSP.csv'
data = pd.read_csv(file_path)

# Convert the sale_date column to datetime format
data['sale_date'] = pd.to_datetime(data['sale_date'], errors='coerce')

# Line chart: Change in total sales amount over time
daily_sales = data.groupby('sale_date')['total_amount'].sum()
plt.figure(figsize=(12, 6))
daily_sales.plot(kind='line', marker='o', linestyle='-', color='b')
plt.title('Total Sales Amount Over Time')
plt.xlabel('Date')
plt.ylabel('Total Amount')
plt.grid(True)
plt.show()