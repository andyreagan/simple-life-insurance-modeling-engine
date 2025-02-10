import pandas as pd
import numpy as np


class InsuranceBlockAnalyzer:
    def __init__(self, insured_population, mortality_table, discount_rate=0.05, lapse_table=None, lapse_rate=0.001):
        '''
        lapse_rate is a fallback if no lapse_table is provided.
        '''
        self.population = insured_population.copy()
        self.mortality = mortality_table.copy()
        self.discount_rate = discount_rate
        self.projection_years = 10
        self.lapse = lapse_table.copy() if lapse_table is not None else None
        self.lapse_rate = lapse_rate
    
    def get_mortality_rate(self, gender, issue_age, duration):
        """Get mortality rate for a specific gender, issue age, and duration"""
        mask = (
            (self.mortality['gender'] == gender) &
            (self.mortality['issue_age'] == issue_age) &
            (self.mortality['duration'] == duration)
        )
        rate = self.mortality.loc[mask, 'q'].iloc[0] if mask.any() else None
        return rate if rate is not None else 0.0
    
    def get_lapse_rate(self, gender, issue_age, duration):
        """Get lapse rate for a specific gender, issue age, and duration"""
        if self.lapse is None:
            return self.lapse_rate
        mask = (
            (self.mortality['gender'] == gender) &
            (self.mortality['issue_age'] == issue_age) &
            (self.mortality['duration'] == duration)
        )
        rate = self.lapse.loc[mask, 'q'].iloc[0] if mask.any() else None
        return rate if rate is not None else 0.0
    
    def calculate_cohort_cashflows(self, row):
        """Calculate cashflows for a specific cohort"""
        cashflows = {
            'premiums': np.zeros(self.projection_years),
            'claims': np.zeros(self.projection_years),
            'lives': np.zeros(self.projection_years + 1),
            'deaths': np.zeros(self.projection_years),
            'lapses': np.zeros(self.projection_years)
        }
        
        # Initialize starting lives
        cashflows['lives'][0] = row['Population']
        
        for t in range(self.projection_years):
            # Get mortality rate for this duration
            q = self.get_mortality_rate(
                row['Gender'],
                row['Issue_Age'],
                t + 1
            )
            q_lapse = self.get_lapse_rate(
                row['Gender'],
                row['Issue_Age'],
                t + 1
            )
            
            # Calculate deaths and surviving lives
            start_lives = cashflows['lives'][t]
            deaths = start_lives * q
            lapses = start_lives * q_lapse
            cashflows['lives'][t + 1] = start_lives - deaths - lapses
            cashflows['lapses'][t] = lapses
            cashflows['deaths'][t] = deaths
            
            # Calculate cashflows
            cashflows['premiums'][t] = start_lives * row['Annual_Premium']
            cashflows['claims'][t] = - deaths * row['Face_Amount']
        
        return cashflows
    
    def calculate_present_values(self, cashflows):
        """Calculate present values of cashflows"""
        discount_factors = 1 / (1 + self.discount_rate) ** np.arange(self.projection_years)
        
        return {
            'pv_premiums': np.sum(cashflows['premiums'] * discount_factors),
            'pv_claims': np.sum(cashflows['claims'] * discount_factors),
            'undiscounted_premiums': np.sum(cashflows['premiums']),
            'undiscounted_claims': np.sum(cashflows['claims']),
            'total_deaths': np.sum(cashflows['deaths'])
        }
    
    def analyze_block(self):
        """Analyze entire block of business"""
        results = []
        
        for _, row in self.population.iterrows():
            # Calculate cashflows for this cohort
            cohort_flows = self.calculate_cohort_cashflows(row)
            
            # Calculate present values
            present_values = self.calculate_present_values(cohort_flows)
            
            # Store results
            results.append({
                'Age_Group': row['Five-Year Age Groups Code'],
                'Gender': row['Gender'],
                'Initial_Population': row['Population'],
                'PV_Premiums': present_values['pv_premiums'],
                'PV_Claims': present_values['pv_claims'],
                'NPV': present_values['pv_premiums'] - present_values['pv_claims'],
                'Undiscounted_Premiums': present_values['undiscounted_premiums'],
                'Undiscounted_Claims': present_values['undiscounted_claims'],
                'Total_Deaths': present_values['total_deaths']
            })
        
        results_df = pd.DataFrame(results)
        
        # Add totals row
        totals = results_df.select_dtypes(include=[np.number]).sum()
        totals_row = pd.DataFrame([{
            'Age_Group': 'Total',
            'Gender': 'All',
            'Initial_Population': totals['Initial_Population'],
            'PV_Premiums': totals['PV_Premiums'],
            'PV_Claims': totals['PV_Claims'],
            'NPV': totals['NPV'],
            'Undiscounted_Premiums': totals['Undiscounted_Premiums'],
            'Undiscounted_Claims': totals['Undiscounted_Claims'],
            'Total_Deaths': totals['Total_Deaths']
        }])
        
        return pd.concat([results_df, totals_row], ignore_index=True)