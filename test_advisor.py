from financial_advisor import FinancialAdvisorAgent
import json

def format_currency(value):
    """Helper function to format currency values"""
    return f"${value:,.2f}"

def main():
    # Initialize the agent
    advisor = FinancialAdvisorAgent()
    
    print("\n=== Financial Portfolio Analysis ===\n")
    
    try:
        # First, import the portfolio data
        result = advisor.import_portfolio_data('portfolio.csv')
        print(f"Import Status: {result}\n")
        
        # Then get the analysis
        analysis = advisor.get_portfolio_analysis()
        
        if 'error' not in analysis:
            # Print Total Portfolio Value and Gain/Loss
            print(f"Total Portfolio Value: {format_currency(analysis['total_value'])}")
            print(f"Total Gain/Loss: {format_currency(analysis['total_gain_loss'])}")
            
            # Print Diversification
            print("\nPortfolio Diversification:")
            for asset_type, percentage in analysis['diversification'].items():
                print(f"{asset_type.capitalize()}: {percentage:.1f}%")
            
            # Print Tax Implications
            print("\nTax Implications (30% rate for non-resident):")
            for symbol, tax_info in analysis['tax_implications'].items():
                tax_amount = tax_info['tax_amount']
                print(f"{symbol}: Estimated Tax {format_currency(tax_amount)}")
            
            # Print Historical Performance if available
            if 'historical_performance' in analysis and analysis['historical_performance']:
                print("\nHistorical Performance:")
                perf = analysis['historical_performance']
                print(f"Total Return: {perf['total_return']:.2f}%")
                print(f"Period: {perf['period']} days")
            
            # Print Recommendations
            print("\nRecommendations:")
            for rec in analysis['recommendations']:
                print(f"- {rec}")
        else:
            print(f"Error: {analysis['error']}")
            
    except FileNotFoundError:
        print("Error: portfolio.csv file not found. Please ensure the file exists in the current directory.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()