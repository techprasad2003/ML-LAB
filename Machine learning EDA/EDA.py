import pandas as pd
from ydata_profiling import ProfileReport

# Load the dataset
file_path = "D:\Desktop\Machine learning EDA\placementdata.csv"
data = pd.read_csv(file_path)

# Generate the EDA report
profile = ProfileReport(data, title=" Placement Data Analysis")

# Save the report to an HTML file
output_path = 'placament_data_analysis.html'
profile.to_file(output_path)

print(f"EDA report generated and saved to {output_path}")