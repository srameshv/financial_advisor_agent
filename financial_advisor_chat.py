from financial_advisor import FinancialAdvisorAgent
import cmd
import shlex

class FinancialAdvisorChat(cmd.Cmd):
    """Interactive Financial Advisor Chat Interface"""
    
    intro = """
=== Financial Advisor Chat Mode ===
Welcome! I'm your financial advisor. Here are some things you can ask me:

1. "import portfolio" - Import your portfolio data
2. "analyze" - Get complete portfolio analysis
3. "check position SYMBOL" - Analyze specific position (e.g., "check position AAPL")
4. "tax analysis" - Get detailed tax implications
5. "help" - See all available commands
6. "quit" - Exit the chat

What would you like to know?
    """
    prompt = '(Financial Advisor) '

    def __init__(self):
        super().__init__()
        self.advisor = FinancialAdvisorAgent()
        self.portfolio_imported = False

    def do_import(self, arg):
        """Import portfolio data from CSV file"""
        try:
            result = self.advisor.import_portfolio_data('portfolio.csv')
            self.portfolio_imported = 'Successfully' in result
            print(result)
        except Exception as e:
            print(f"Error importing portfolio: {e}")

    def do_analyze(self, arg):
        """Get complete portfolio analysis"""
        if not self.portfolio_imported:
            print("Please import your portfolio first using 'import portfolio'")
            return
        
        analysis = self.advisor.get_portfolio_analysis()
        if 'error' not in analysis:
            print(f"\nPortfolio Summary:")
            print(f"Total Value: ${analysis['total_value']:,.2f}")
            print(f"Total Gain/Loss: ${analysis['total_gain_loss']:,.2f}")
            
            print("\nDiversification:")
            for asset_type, percentage in analysis['diversification'].items():
                print(f"{asset_type.capitalize()}: {percentage:.1f}%")
            
            print("\nRecommendations:")
            for rec in analysis['recommendations']:
                print(f"- {rec}")
        else:
            print(f"Error: {analysis['error']}")

    def do_position(self, arg):
        """Analyze specific position (usage: position SYMBOL)"""
        if not arg:
            print("Please specify a symbol (e.g., 'position AAPL')")
            return
        
        if not self.portfolio_imported:
            print("Please import your portfolio first using 'import portfolio'")
            return
        
        symbol = arg.upper()
        analysis = self.advisor.portfolio_manager.analyze_position(symbol)
        
        if 'error' not in analysis:
            print(f"\nAnalysis for {symbol}:")
            print(f"Current Value: ${analysis['current_value']:,.2f}")
            print(f"Gain/Loss: ${analysis['gain_loss']:,.2f} ({analysis['gain_loss_percent']:.1f}%)")
            print(f"Recommendation: {analysis['recommendation']}")
        else:
            print(f"Error: {analysis['error']}")

    def do_tax(self, arg):
        """Get tax analysis"""
        if not self.portfolio_imported:
            print("Please import your portfolio first using 'import portfolio'")
            return
        
        analysis = self.advisor.get_portfolio_analysis()
        if 'error' not in analysis:
            print("\nTax Analysis (Non-US Resident Rate: 30%):")
            for symbol, tax_info in analysis['tax_implications'].items():
                print(f"\n{symbol}:")
                print(f"Tax Rate: {tax_info['tax_rate']*100:.1f}%")
                print(f"Tax Amount: ${tax_info['tax_amount']:,.2f}")
                print(f"Type: {tax_info['type']}")
        else:
            print(f"Error: {analysis['error']}")

    def do_help(self, arg):
        """List available commands"""
        print("\nAvailable Commands:")
        print("1. import - Import portfolio data")
        print("2. analyze - Get complete portfolio analysis")
        print("3. position SYMBOL - Analyze specific position")
        print("4. tax - Get tax analysis")
        print("5. quit - Exit the chat")
        print("\nFor more details on any command, type: help COMMAND")

    def do_quit(self, arg):
        """Exit the chat"""
        print("\nThank you for using Financial Advisor. Goodbye!")
        return True

    def default(self, line):
        """Handle unknown commands"""
        print(f"I don't understand '{line}'. Type 'help' to see available commands.")

if __name__ == '__main__':
    FinancialAdvisorChat().cmdloop()