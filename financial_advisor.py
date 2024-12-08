import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
import logging
import json
import os
from dataclasses import dataclass, asdict
import pickle

@dataclass
class UserProfile:
    """User profile with tax and residency information"""
    tax_residency: str
    origin_country: str
    current_country: str
    has_foreign_income: bool
    tax_treaty_benefits: bool
    preferred_currency: str
    
    def to_dict(self):
        return asdict(self)

class PortfolioHistoryTracker:
    """Tracks historical portfolio performance"""
    def __init__(self, history_file='portfolio_history.pkl'):
        self.history_file = history_file
        self.history = self._load_history()
    
    def _load_history(self) -> Dict:
        """Load historical data from file"""
        if os.path.exists(self.history_file):
            with open(self.history_file, 'rb') as f:
                return pickle.load(f)
        return {'snapshots': [], 'performance': {}}
    
    def save_snapshot(self, portfolio_data: Dict):
        """Save current portfolio snapshot"""
        snapshot = {
            'date': datetime.now(),
            'data': portfolio_data
        }
        self.history['snapshots'].append(snapshot)
        self._save_history()
    
    def _save_history(self):
        """Save history to file"""
        with open(self.history_file, 'wb') as f:
            pickle.dump(self.history, f)
    
    def get_performance_metrics(self) -> Dict:
        """Calculate performance metrics from history"""
        if not self.history['snapshots']:
            return {}
        
        snapshots = self.history['snapshots']
        latest = snapshots[-1]['data']
        oldest = snapshots[0]['data']
        
        return {
            'total_return': (latest['total_value'] - oldest['total_value']) / oldest['total_value'] * 100,
            'period': (snapshots[-1]['date'] - snapshots[0]['date']).days
        }

class CurrencyConverter:
    """Handles multi-currency conversions"""
    def __init__(self):
        # Sample exchange rates (in production, would fetch from API)
        self.exchange_rates = {
            'USD': 1.0,
            'INR': 0.012,
            'EUR': 1.09,
            'GBP': 1.27
        }
    
    def convert(self, amount: float, from_currency: str, to_currency: str) -> float:
        """Convert amount between currencies"""
        if from_currency == to_currency:
            return amount
        
        usd_amount = amount / self.exchange_rates[from_currency]
        return usd_amount * self.exchange_rates[to_currency]

class TaxCalculator:
    """Advanced tax calculations with treaty considerations"""
    def __init__(self, user_profile: UserProfile):
        self.user_profile = user_profile
        self.tax_treaties = {
            'India': {
                'withholding_rate': 0.30,
                'capital_gains': {
                    'short_term': 0.30,
                    'long_term': 0.20
                },
                'treaty_benefits': ['reduced_withholding', 'foreign_tax_credit']
            }
        }
    
    def calculate_tax_obligation(self, gain: float, holding_period_days: int) -> Dict:
        """Calculate tax obligation with treaty considerations"""
        base_tax_rate = 0.30  # Non-resident base rate
        
        # Apply treaty benefits if applicable
        if (self.user_profile.tax_treaty_benefits and 
            self.user_profile.origin_country in self.tax_treaties):
            treaty = self.tax_treaties[self.user_profile.origin_country]
            
            # Long-term capital gains treatment
            if holding_period_days > 365:
                tax_rate = treaty['capital_gains']['long_term']
            else:
                tax_rate = treaty['capital_gains']['short_term']
        else:
            tax_rate = base_tax_rate
        
        tax_amount = gain * tax_rate
        
        return {
            'tax_rate': tax_rate,
            'tax_amount': tax_amount,
            'treaty_applied': self.user_profile.tax_treaty_benefits,
            'type': 'Long Term' if holding_period_days > 365 else 'Short Term'
        }

class PortfolioManager:
    """Enhanced portfolio management with all features"""
    def __init__(self, user_profile: UserProfile):
        self.portfolio_data = None
        self.last_update = None
        self.user_profile = user_profile
        self.history_tracker = PortfolioHistoryTracker()
        self.currency_converter = CurrencyConverter()
        self.tax_calculator = TaxCalculator(user_profile)
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('portfolio.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def import_portfolio(self, file_path: str) -> Dict:
        """Import and process portfolio data"""
        try:
            # Read file
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_path)
            else:
                return {"error": "Unsupported file format"}
            
            # Validate columns
            required_columns = [
                'Symbol', 'Quantity', 'Average Cost', 'Current Price', 
                'Type', 'Purchase Date'  # Added purchase date for tax calculations
            ]
            if not all(col in df.columns for col in required_columns):
                return {"error": f"Missing columns: {required_columns}"}
            
            # Process portfolio
            portfolio = {}
            for _, row in df.iterrows():
                symbol = row['Symbol']
                quantity = float(row['Quantity'])
                avg_cost = float(row['Average Cost'])
                current_price = float(row['Current Price'])
                purchase_date = pd.to_datetime(row['Purchase Date'])
                
                # Convert all monetary values to user's preferred currency
                avg_cost_converted = self.currency_converter.convert(
                    avg_cost, 'USD', self.user_profile.preferred_currency
                )
                current_price_converted = self.currency_converter.convert(
                    current_price, 'USD', self.user_profile.preferred_currency
                )
                
                market_value = quantity * current_price_converted
                gain_loss = (current_price_converted - avg_cost_converted) * quantity
                holding_period = (datetime.now() - purchase_date).days
                
                # Calculate tax implications
                tax_info = self.tax_calculator.calculate_tax_obligation(
                    gain_loss, holding_period
                )
                
                portfolio[symbol] = {
                    'type': row['Type'].lower(),
                    'quantity': quantity,
                    'average_cost': avg_cost_converted,
                    'current_price': current_price_converted,
                    'market_value': market_value,
                    'gain_loss': gain_loss,
                    'purchase_date': purchase_date,
                    'holding_period_days': holding_period,
                    'tax_implications': tax_info
                }
            
            self.portfolio_data = portfolio
            self.last_update = datetime.now()
            
            # Save snapshot for historical tracking
            self.history_tracker.save_snapshot({
                'holdings': portfolio,
                'total_value': sum(h['market_value'] for h in portfolio.values()),
                'total_gain_loss': sum(h['gain_loss'] for h in portfolio.values())
            })
            
            return {
                'status': 'success',
                'holdings': portfolio,
                'total_value': sum(h['market_value'] for h in portfolio.values()),
                'total_gain_loss': sum(h['gain_loss'] for h in portfolio.values())
            }
            
        except Exception as e:
            self.logger.error(f"Error importing portfolio: {e}")
            return {"error": str(e)}

class FinancialAdvisorAgent:
    """Complete Financial Advisor Agent with all features"""
    def __init__(self, profile_path='user_profile.json'):
        # Load or create user profile
        self.profile_path = profile_path
        self.user_profile = self._load_user_profile()
        self.portfolio_manager = PortfolioManager(self.user_profile)
    
    def _load_user_profile(self) -> UserProfile:
        """Load user profile from file or create default"""
        if os.path.exists(self.profile_path):
            with open(self.profile_path, 'r') as f:
                data = json.load(f)
                return UserProfile(**data)
        
        # Default profile (your specific case)
        return UserProfile(
            tax_residency='US_nonresident',
            origin_country='India',
            current_country='US',
            has_foreign_income=False,
            tax_treaty_benefits=True,
            preferred_currency='USD'
        )
    
    def save_user_profile(self):
        """Save user profile to file"""
        with open(self.profile_path, 'w') as f:
            json.dump(self.user_profile.to_dict(), f, indent=2)
    
    def update_user_profile(self, **kwargs):
        """Update user profile with new information"""
        for key, value in kwargs.items():
            if hasattr(self.user_profile, key):
                setattr(self.user_profile, key, value)
        self.save_user_profile()
    
    def import_portfolio_data(self, file_path: str) -> str:
        """Import portfolio data"""
        result = self.portfolio_manager.import_portfolio(file_path)
        if 'error' in result:
            return f"Error importing portfolio: {result['error']}"
        return f"Successfully imported portfolio with {len(result['holdings'])} positions"
    
    def get_portfolio_analysis(self) -> Dict:
        """Get comprehensive portfolio analysis"""
        if not self.portfolio_manager.portfolio_data:
            return {"error": "No portfolio data imported"}
        
        portfolio = self.portfolio_manager.portfolio_data
        total_value = sum(p['market_value'] for p in portfolio.values())
        
        # Get historical performance
        historical_performance = self.portfolio_manager.history_tracker.get_performance_metrics()
        
        # Calculate diversification
        diversification = {
            'stocks': sum(p['market_value'] for p in portfolio.values() if p['type'] == 'stock'),
            'crypto': sum(p['market_value'] for p in portfolio.values() if p['type'] == 'crypto')
        }
        
        return {
            'total_value': total_value,
            'total_gain_loss': sum(p['gain_loss'] for p in portfolio.values()),
            'tax_implications': {
                symbol: position['tax_implications'] 
                for symbol, position in portfolio.items()
            },
            'diversification': {
                k: v/total_value*100 for k, v in diversification.items()
            },
            'historical_performance': historical_performance,
            'recommendations': self._generate_portfolio_recommendations(portfolio, total_value),
            'currency': self.user_profile.preferred_currency
        }
    
    def _generate_portfolio_recommendations(self, portfolio: Dict, total_value: float) -> List[str]:
        """Generate enhanced portfolio recommendations"""
        recommendations = []
        
        # Concentration checks
        for symbol, position in portfolio.items():
            position_pct = (position['market_value'] / total_value) * 100
            if position_pct > 20:
                recommendations.append(
                    f"High concentration in {symbol} ({position_pct:.1f}%). Consider diversifying."
                )
        
        # Crypto exposure check
        crypto_value = sum(p['market_value'] for p in portfolio.values() if p['type'] == 'crypto')
        crypto_pct = (crypto_value / total_value) * 100
        if crypto_pct > 15:
            recommendations.append(
                f"High crypto exposure ({crypto_pct:.1f}%). Consider reducing for better risk management."
            )
        
        # Tax efficiency recommendations
        short_term_gains = sum(
            p['gain_loss'] for p in portfolio.values() 
            if p['holding_period_days'] <= 365 and p['gain_loss'] > 0
        )
        if short_term_gains > 0:
            recommendations.append(
                f"Consider holding appreciated positions longer to qualify for better tax treatment."
            )
        
        return recommendations

# Example usage
def main():
    advisor = FinancialAdvisorAgent()
    print(advisor.get_portfolio_analysis())

if __name__ == "__main__":
    main()