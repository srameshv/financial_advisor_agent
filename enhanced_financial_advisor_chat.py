from financial_advisor import FinancialAdvisorAgent
import cmd
import shlex
from typing import List, Dict, Optional
import re
from datetime import datetime, timedelta
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import seaborn as sns

class NLPProcessor:
    """Natural Language Processing for chat commands"""
    def __init__(self):
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')
            nltk.download('wordnet')
        
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Command patterns
        self.command_patterns = {
            r'(?i).*\b(show|analyze|check|view)\b.*\bportfolio\b.*': 'analyze',
            r'(?i).*\b(tax|taxes)\b.*': 'tax',
            r'(?i).*\b(import|load|read)\b.*\b(portfolio|data|csv)\b.*': 'import',
            r'(?i).*\b(check|analyze|show)\b.*\b(position|stock|holding)\b.*\b([A-Z]{1,5})\b.*': 'position',
            r'(?i).*\b(forecast|predict|project)\b.*': 'forecast',
            r'(?i).*\b(risk|volatility)\b.*': 'risk',
            r'(?i).*\b(compare|benchmark)\b.*': 'benchmark',
            r'(?i).*\b(optimize|rebalance)\b.*': 'optimize',
            r'(?i).*\b(performance|return)\b.*': 'performance',
            r'(?i).*\b(dividend|income)\b.*': 'dividend',
            r'(?i).*\b(help|assist|guide)\b.*': 'help',
            r'(?i).*\b(quit|exit|bye)\b.*': 'quit'
        }
    
    def process_command(self, user_input: str) -> tuple:
        """Process natural language input into command and arguments"""
        # Check against command patterns
        for pattern, command in self.command_patterns.items():
            match = re.match(pattern, user_input)
            if match:
                # Extract any symbols mentioned (for position command)
                symbols = re.findall(r'\b[A-Z]{1,5}\b', user_input.upper())
                return command, symbols[0] if symbols else None
        
        # If no pattern matches, try to understand the intent
        tokens = word_tokenize(user_input.lower())
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens if word not in self.stop_words]
        
        # Simple intent matching
        if any(word in tokens for word in ['show', 'display', 'list']):
            return 'analyze', None
        
        return 'unknown', None

class PortfolioVisualizer:
    """Creates visual representations of portfolio data"""
    @staticmethod
    def create_allocation_pie_chart(portfolio_data: Dict):
        plt.figure(figsize=(10, 6))
        values = [position['market_value'] for position in portfolio_data.values()]
        labels = list(portfolio_data.keys())
        plt.pie(values, labels=labels, autopct='%1.1f%%')
        plt.title('Portfolio Allocation')
        plt.savefig('portfolio_allocation.png')
        plt.close()
    
    @staticmethod
    def create_performance_chart(historical_data: List[Dict]):
        plt.figure(figsize=(12, 6))
        dates = [data['date'] for data in historical_data]
        values = [data['total_value'] for data in historical_data]
        plt.plot(dates, values)
        plt.title('Portfolio Performance Over Time')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value ($)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('portfolio_performance.png')
        plt.close()

class EnhancedFinancialAdvisorChat(cmd.Cmd):
    """Enhanced Interactive Financial Advisor Chat Interface"""
    
    intro = """
=== Enhanced Financial Advisor Chat Mode ===
Welcome! I'm your AI-powered financial advisor. You can talk to me naturally!

Example commands:
- "Show me my portfolio analysis"
- "What's my tax situation?"
- "Check AAPL position"
- "How risky is my portfolio?"
- "Create a performance chart"
- "Optimize my portfolio"
- "Compare with S&P 500"

Type 'help' for more information or 'quit' to exit.

What would you like to know?
    """
    prompt = '(Financial Advisor) '

    def __init__(self):
        super().__init__()
        self.advisor = FinancialAdvisorAgent()
        self.nlp = NLPProcessor()
        self.visualizer = PortfolioVisualizer()
        self.portfolio_imported = False
        self.last_analysis = None
    
    def default(self, line):
        """Handle natural language input"""
        command, arg = self.nlp.process_command(line)
        
        if command != 'unknown':
            if hasattr(self, f'do_{command}'):
                getattr(self, f'do_{command}')(arg)
            else:
                self.do_help(None)
        else:
            print("I'm not sure what you mean. Try rephrasing or type 'help' for guidance.")
    
    def do_risk(self, arg):
        """Analyze portfolio risk"""
        if not self.portfolio_imported:
            print("Please import your portfolio first.")
            return
        
        analysis = self.advisor.get_portfolio_analysis()
        
        if 'error' not in analysis:
            print("\nRisk Analysis:")
            
            # Asset concentration risk
            print("\nConcentration Risk:")
            for asset_type, percentage in analysis['diversification'].items():
                risk_level = "High" if percentage > 30 else "Medium" if percentage > 20 else "Low"
                print(f"{asset_type.capitalize()}: {percentage:.1f}% ({risk_level} concentration)")
            
            # Volatility analysis
            print("\nVolatility Analysis:")
            for symbol in analysis['tax_implications'].keys():
                position = self.advisor.portfolio_manager.analyze_position(symbol)
                print(f"{symbol}: {position['recommendation']}")
    
    def do_performance(self, arg):
        """Show portfolio performance metrics and charts"""
        if not self.portfolio_imported:
            print("Please import your portfolio first.")
            return
        
        analysis = self.advisor.get_portfolio_analysis()
        
        if 'error' not in analysis:
            print("\nPerformance Metrics:")
            print(f"Total Value: ${analysis['total_value']:,.2f}")
            print(f"Total Gain/Loss: ${analysis['total_gain_loss']:,.2f}")
            
            if 'historical_performance' in analysis:
                perf = analysis['historical_performance']
                print(f"\nTotal Return: {perf['total_return']:.2f}%")
                print(f"Investment Period: {perf['period']} days")
            
            # Create visual charts
            self.visualizer.create_allocation_pie_chart(
                self.advisor.portfolio_manager.portfolio_data
            )
            print("\nPortfolio allocation chart has been saved as 'portfolio_allocation.png'")
    
    def do_optimize(self, arg):
        """Suggest portfolio optimization strategies"""
        if not self.portfolio_imported:
            print("Please import your portfolio first.")
            return
        
        analysis = self.advisor.get_portfolio_analysis()
        
        if 'error' not in analysis:
            print("\nPortfolio Optimization Suggestions:")
            
            # Tax efficiency
            print("\nTax Optimization:")
            for symbol, tax_info in analysis['tax_implications'].items():
                if tax_info['tax_amount'] > 0:
                    print(f"- Consider holding {symbol} longer for better tax treatment")
            
            # Diversification
            print("\nDiversification Optimization:")
            for asset_type, percentage in analysis['diversification'].items():
                if percentage > 30:
                    print(f"- Consider reducing {asset_type} exposure from {percentage:.1f}%")
                elif percentage < 10:
                    print(f"- Consider increasing {asset_type} exposure from {percentage:.1f}%")
            
            # Risk-adjusted recommendations
            print("\nRisk-Adjusted Recommendations:")
            for rec in analysis['recommendations']:
                print(f"- {rec}")
    
    def do_dividend(self, arg):
        """Analyze dividend income potential"""
        if not self.portfolio_imported:
            print("Please import your portfolio first.")
            return
        
        portfolio = self.advisor.portfolio_manager.portfolio_data
        print("\nDividend Analysis:")
        
        total_dividend_income = 0
        for symbol, position in portfolio.items():
            if position['type'] == 'stock':
                # In a real implementation, fetch actual dividend yield
                estimated_yield = 0.02  # Example yield
                annual_dividend = position['market_value'] * estimated_yield
                total_dividend_income += annual_dividend
                print(f"{symbol}: Estimated Annual Dividend ${annual_dividend:,.2f}")
        
        print(f"\nTotal Estimated Annual Dividend Income: ${total_dividend_income:,.2f}")
    
    def do_benchmark(self, arg):
        """Compare portfolio performance with benchmarks"""
        if not self.portfolio_imported:
            print("Please import your portfolio first.")
            return
        
        analysis = self.advisor.get_portfolio_analysis()
        
        if 'error' not in analysis:
            print("\nBenchmark Comparison:")
            
            # In a real implementation, fetch actual benchmark data
            sp500_return = 0.10  # Example S&P 500 return
            portfolio_return = analysis['total_gain_loss'] / analysis['total_value']
            
            print(f"Your Portfolio Return: {portfolio_return*100:.1f}%")
            print(f"S&P 500 Return: {sp500_return*100:.1f}%")
            
            if portfolio_return > sp500_return:
                print(f"Your portfolio is outperforming the S&P 500 by {(portfolio_return-sp500_return)*100:.1f}%")
            else:
                print(f"Your portfolio is underperforming the S&P 500 by {(sp500_return-portfolio_return)*100:.1f}%")
    
    def do_forecast(self, arg):
        """Provide simple portfolio value forecasting"""
        if not self.portfolio_imported:
            print("Please import your portfolio first.")
            return
        
        analysis = self.advisor.get_portfolio_analysis()
        
        if 'error' not in analysis:
            current_value = analysis['total_value']
            
            print("\nPortfolio Value Forecasts:")
            print("(Based on different annual return scenarios)")
            
            scenarios = {
                'Conservative (4%)': 0.04,
                'Moderate (8%)': 0.08,
                'Aggressive (12%)': 0.12
            }
            
            periods = [1, 3, 5, 10]
            
            print("\nProjected Portfolio Values:")
            print("Years | Conservative | Moderate | Aggressive")
            print("-" * 45)
            
            for years in periods:
                projections = []
                for scenario, rate in scenarios.items():
                    future_value = current_value * (1 + rate) ** years
                    projections.append(f"${future_value:,.0f}")
                
                print(f"{years:5d} | {projections[0]:12s} | {projections[1]:8s} | {projections[2]:s}")
    
    def do_help(self, arg):
        """Enhanced help command with natural language examples"""
        print("\nYou can talk to me naturally! Here are some things you can say:")
        print("\nPortfolio Analysis:")
        print("- 'Show me my portfolio analysis'")
        print("- 'How is my portfolio doing?'")
        print("- 'What are my current holdings?'")
        
        print("\nTax Analysis:")
        print("- 'What's my tax situation?'")
        print("- 'Show me tax implications'")
        print("- 'Calculate my tax obligations'")
        
        print("\nRisk Analysis:")
        print("- 'How risky is my portfolio?'")
        print("- 'Show me risk metrics'")
        print("- 'Analyze portfolio risk'")
        
        print("\nPerformance and Optimization:")
        print("- 'How can I optimize my portfolio?'")
        print("- 'Show me performance metrics'")
        print("- 'Compare with market benchmarks'")
        
        print("\nForecasting and Planning:")
        print("- 'Forecast portfolio value'")
        print("- 'Show future projections'")
        print("- 'What will my portfolio be worth in 5 years?'")
        
        print("\nSpecific Holdings:")
        print("- 'Check AAPL position'")
        print("- 'How is my Bitcoin doing?'")
        print("- 'Analyze my tech stocks'")

if __name__ == '__main__':
    try:
        EnhancedFinancialAdvisorChat().cmdloop()
    except KeyboardInterrupt:
        print("\nThank you for using Financial Advisor. Goodbye!")