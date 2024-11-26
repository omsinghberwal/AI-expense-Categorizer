# Module for data visualization
import matplotlib.pyplot as plt

def generate_charts(data):
    """
    Generate a bar chart showing total expenses per category.
    :param data: DataFrame with 'Category' and 'Amount' columns
    """
    # Group by category and sum expenses
    category_totals = data.groupby('Category')['Amount'].sum()

    # Generate bar chart
    plt.figure(figsize=(10, 6))
    category_totals.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title('Total Expenses by Category', fontsize=16)
    plt.xlabel('Category', fontsize=14)
    plt.ylabel('Total Expense (in currency)', fontsize=14)
    plt.xticks(rotation=45, fontsize=12)
    plt.tight_layout()

    # Save and show the chart
    plt.savefig("expense_distribution.png")
    print("Expense distribution chart saved as 'expense_distribution.png'.")
    plt.show()
