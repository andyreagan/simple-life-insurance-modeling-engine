import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class SimulationParams:
    """Parameters for stochastic simulation"""
    n_simulations: int
    investment_return_mean: float
    investment_return_vol: float
    interest_rate_mean: float
    interest_rate_vol: float
    interest_rate_mean_reversion: float
    lapse_rate_base: float
    lapse_rate_interest_sensitivity: float
    
    
class WholeLifePolicy:
    def __init__(
        self,
        age: int,
        gender: str,
        death_benefit: float,
        annual_premium: float,
        mortality_table: pd.DataFrame,
        policy_params: Dict,
        sim_params: SimulationParams
    ):
        """
        Initialize a whole life insurance policy valuation model with stochastic simulation.
        
        Parameters:
        - age: Current age of policyholder
        - gender: 'M' or 'F' for gender-specific mortality rates
        - death_benefit: Face value of the policy
        - annual_premium: Annual premium payment
        - mortality_table: DataFrame with mortality rates by age and gender
        - policy_params: Dictionary containing key assumptions:
            - investment_return: Base expected return on invested premiums
            - dividend_rate: Base dividend payment rate
            - expense_ratio: Administrative and operational costs
            - cash_reserve_ratio: Required cash holdings ratio
            - reinsurance_rate: Percentage of risk ceded to reinsurer
            - reinsurance_cost: Cost of reinsurance as percentage of ceded premium
        - sim_params: SimulationParams object with simulation settings
        """
        self.age = age
        self.gender = gender.upper()
        self.death_benefit = death_benefit
        self.annual_premium = annual_premium
        self.mortality_table = mortality_table
        self.sim_params = sim_params
        
        # Extract parameters with default values
        self.base_investment_return = policy_params.get('investment_return', 0.04)
        self.base_dividend_rate = policy_params.get('dividend_rate', 0.02)
        self.expense_ratio = policy_params.get('expense_ratio', 0.03)
        self.cash_reserve_ratio = policy_params.get('cash_reserve_ratio', 0.10)
        self.reinsurance_rate = policy_params.get('reinsurance_rate', 0.90)
        self.reinsurance_cost = policy_params.get('reinsurance_cost', 0.01)
        
        # Maximum age in mortality table
        self.max_age = mortality_table.index.max()

    def simulate_market_factors(self, n_years: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate investment returns and interest rates using a mean-reverting process.
        Returns arrays of shape (n_simulations, n_years)
        """
        # Simulate interest rates using Vasicek model
        dt = 1.0
        n_steps = n_years
        rates = np.zeros((self.sim_params.n_simulations, n_steps))
        returns = np.zeros((self.sim_params.n_simulations, n_steps))
        
        for sim in range(self.sim_params.n_simulations):
            # Interest rates
            rates[sim, 0] = self.sim_params.interest_rate_mean
            for t in range(1, n_steps):
                dr = self.sim_params.interest_rate_mean_reversion * \
                     (self.sim_params.interest_rate_mean - rates[sim, t-1]) * dt + \
                     self.sim_params.interest_rate_vol * np.random.normal(0, np.sqrt(dt))
                rates[sim, t] = rates[sim, t-1] + dr
            
            # Investment returns with correlation to interest rates
            correlation = 0.3  # Correlation between returns and rates
            returns[sim] = self.sim_params.investment_return_mean + \
                          self.sim_params.investment_return_vol * \
                          (correlation * (rates[sim] - self.sim_params.interest_rate_mean) / 
                           self.sim_params.interest_rate_vol + 
                           np.sqrt(1 - correlation**2) * np.random.normal(0, 1, n_steps))
        
        return returns, rates

    def simulate_lapse_rates(self, interest_rates: np.ndarray) -> np.ndarray:
        """
        Simulate policy lapse rates based on interest rate environment
        """
        base_rate = self.sim_params.lapse_rate_base
        sensitivity = self.sim_params.lapse_rate_interest_sensitivity
        
        # Lapse rates increase when market rates are higher than policy returns
        rate_differential = np.maximum(0, interest_rates - self.base_investment_return)
        lapse_rates = base_rate + sensitivity * rate_differential
        
        # Ensure rates stay within reasonable bounds
        return np.clip(lapse_rates, 0, 0.15)

    def get_mortality_rate(self, age: int) -> float:
        """Get mortality rate for given age and gender"""
        if age > self.max_age:
            return 1.0
        return self.mortality_table.loc[age, f'mortality_rate_{self.gender}']

    def project_cash_flows_stochastic(self) -> pd.DataFrame:
        """
        Project cash flows using Monte Carlo simulation with multiple risk factors.
        Returns DataFrame with simulation statistics.
        """
        projection_years = self.max_age - self.age
        
        # Simulate market factors
        investment_returns, interest_rates = self.simulate_market_factors(projection_years)
        lapse_rates = self.simulate_lapse_rates(interest_rates)
        
        # Arrays to store results
        policy_values = np.zeros(self.sim_params.n_simulations)
        
        for sim in range(self.sim_params.n_simulations):
            survival_prob = 1.0
            lapse_prob = 1.0
            accumulated_value = 0
            present_value = 0
            
            for year in range(projection_years):
                current_age = self.age + year
                
                # Get rates for this period
                mortality_rate = self.get_mortality_rate(current_age)
                investment_return = investment_returns[sim, year]
                lapse_rate = lapse_rates[sim, year]
                
                # Update survival and lapse probabilities
                survival_prob *= (1 - mortality_rate)
                lapse_prob *= (1 - lapse_rate)
                
                # Calculate cash flows
                if survival_prob * lapse_prob < 0.0001:  # Policy effectively terminated
                    break
                    
                premium = self.annual_premium * lapse_prob
                investment_income = accumulated_value * investment_return
                death_benefit = -self.death_benefit * mortality_rate * survival_prob
                
                # Reinsurance effects
                ceded_premium = premium * self.reinsurance_rate
                ceded_death_benefit = death_benefit * self.reinsurance_rate
                reinsurance_cost = -ceded_premium * self.reinsurance_cost
                
                # Net cash flow
                net_cash_flow = (
                    premium +
                    investment_income +
                    death_benefit -
                    ceded_premium +
                    ceded_death_benefit +
                    reinsurance_cost
                ) * (1 - self.expense_ratio)
                
                # Discount to present value
                discount_rate = np.mean(interest_rates[sim, :year+1])
                present_value += net_cash_flow / (1 + discount_rate) ** year
                
                # Update accumulated value
                accumulated_value += net_cash_flow
            
            policy_values[sim] = present_value
        
        # Calculate statistics
        results = {
            'mean_value': np.mean(policy_values),
            'median_value': np.median(policy_values),
            'std_dev': np.std(policy_values),
            'var_95': np.percentile(policy_values, 5),
            'var_99': np.percentile(policy_values, 1),
            'max_value': np.max(policy_values),
            'min_value': np.min(policy_values)
        }
        
        return pd.DataFrame([results])
    
    
# Example usage
if __name__ == "__main__":
    # Sample mortality table with gender-specific rates
    ages = range(0, 121)
    sample_mortality_table = pd.DataFrame({
        'age': ages,
        'mortality_rate_M': [0.001 * (1.12 ** (age/10)) for age in ages],
        'mortality_rate_F': [0.001 * (1.10 ** (age/10)) for age in ages]
    }).set_index('age')
    
    # Policy parameters
    policy_params = {
        'investment_return': 0.04,
        'dividend_rate': 0.02,
        'expense_ratio': 0.03,
        'cash_reserve_ratio': 0.10,
        'reinsurance_rate': 0.90,
        'reinsurance_cost': 0.01
    }
    
    # Simulation parameters
    sim_params = SimulationParams(
        n_simulations=1000,
        investment_return_mean=0.04,
        investment_return_vol=0.08,
        interest_rate_mean=0.03,
        interest_rate_vol=0.015,
        interest_rate_mean_reversion=0.15,
        lapse_rate_base=0.02,
        lapse_rate_interest_sensitivity=0.3
    )
    
    # Create policy instance
    policy = WholeLifePolicy(
        age=35,
        gender='M',
        death_benefit=500000,
        annual_premium=40000,
        mortality_table=sample_mortality_table,
        policy_params=policy_params,
        sim_params=sim_params
    )
    
    # Run simulation
    results = policy.project_cash_flows_stochastic()
    print("\nSimulation Results:")
    print(results)