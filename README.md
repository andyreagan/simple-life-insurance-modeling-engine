# Simple Life Insurance Modeling Engine (SLIME)

Very simple tools for modeling/illustrating life insurance policies.

## Installation

Clone the repo or just download the python file.
Put it next to the script you want to model from.

## Running a simulation

```
import pandas as pd

from insuranceBlockAnalyzer import InsuranceBlockAnalyzer

# Create sample insured population data
insured_population = pd.DataFrame({
    'Five-Year Age Groups Code': [
        '50-54', '50-54', 
    ],
    'Issue_Age': [
        50, 50,
    ],
    'Gender': [
        'F', 'M', 
    ],
    'Population': [
        1000, 1000,  # 50-54
    ],
    'Annual_Premium': [
        10000.0, 10000.0,  # 50-54
    ],
    'Face_Amount': [
        500000, 500000,  # 50-54
    ]
})

mortality_table = pd.read_parquet(
    "https://github.com/andyreagan/vbt-processing/raw/refs/heads/main/vbt.parquet"
).loc[
    (mortality_table.smoker == 'PRE') & (mortality_table.age_method == 'ALB'), 
    :
]

analyzer = InsuranceBlockAnalyzer(insured_population, mortality_table)
results = analyzer.analyze_block()
print(results)

flows = analyzer.calculate_cohort_cashflows(insured_population.iloc[0, :])
print(flows)

flows_df = pd.DataFrame({k: [np.insert(v, 0, 0),v][k=='lives'] for k, v in flows.items()})
print(flows_df)
```

## Roadmap

- [x] Mortality
- [x] Lapse
- [x] Block of policies
- [ ] Unit & regression tests
- [ ] Dividends
- [ ] Reinsurance
