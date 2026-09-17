import pandas as pd
from analysis.data_quality import analyze_data_quality


data = {
    "age": [20, 21, 22, None, 22],
    "salary": [25000, 30000, 35000, 40000, 35000]
}

df = pd.DataFrame(data)

result = analyze_data_quality(df)

print("Data Quality Report")
print("-------------------")
print(result)
