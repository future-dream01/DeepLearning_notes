import pandas as pd
from scipy.stats import chi2_contingency

file_path = 'your_file_path_here.xlsx'
data = pd.read_excel(file_path)

data['是否购买净水器'] = data["20.您目前没有购买净水器的主要原因？"].apply(lambda x: '是' if x == '(跳过)' else '否')

def chi_square_and_contingency(data, column):
    contingency_table = pd.crosstab(data[column], data['是否购买净水器'])
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    result = {
        "contingency_table": contingency_table,
        "chi2": chi2,
        "p_value": p,
        "degrees_of_freedom": dof,
        "expected_frequencies": expected
    }
    return result

factors = ["2.您的性别：", "1.您的年龄段：", "3.您目前从事的职业：", "4.您的月收入："]
analysis_results = {factor: chi_square_and_contingency(data, factor) for factor in factors}

for factor, result in analysis_results.items():
    print(f"Analysis for {factor}:")
    print("Contingency Table:")
    print(result["contingency_table"])
    print(f"Chi-Square Statistic: {result['chi2']}")
    print(f"P-Value: {result['p_value']}")
    print(f"Degrees of Freedom: {result['degrees_of_freedom']}")
    print("Expected Frequencies:")
    print(result["expected_frequencies"])
    print("\n")