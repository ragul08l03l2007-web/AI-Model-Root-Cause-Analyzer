import pandas as pd

from root_cause import analyze_root_cause
from report import generate_report


data = {
    "age": [20, 21, 22, 23, None, 25, 26, 27, 27, 29],
    "salary": [
        25000, 30000, 35000, 40000, 45000,
        50000, 55000, 60000, 60000, 70000
    ],
    "result": [
        "Pass", "Pass", "Pass", "Pass", "Pass",
        "Pass", "Pass", "Fail", "Fail", "Pass"
    ]
}

df = pd.DataFrame(data)

result = analyze_root_cause(df, "result")

generate_report(result)