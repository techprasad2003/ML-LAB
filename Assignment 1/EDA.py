import pandas as pd
from ydata_profiling import ProfileReport

# Load the dataset
file_path = 'D:/Desktop/EDA/usa_housing.csv'  
data = pd.read_csv(file_path)

# Generate the EDA report
profile = ProfileReport(data, title=" USA Housing EDA Report")

# Save the report to an HTML file
output_path = 'usa_housing_eda_report.html'
profile.to_file(output_path)

print(f"EDA report generated and saved to {output_path}")