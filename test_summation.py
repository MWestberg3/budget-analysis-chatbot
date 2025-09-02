import pytest
from unittest.mock import MagicMock, patch
from main import process_spending_data, calculate_50_30_recommendations
import pandas as pd
import io

@pytest.mark.asyncio
async def test_monthly_summary_calculation():
    csv_content = """
Account,Flag,Date,Payee,Category Group/Category,Category Group,Category,Memo,Outflow,Inflow,Cleared
Test Account,,08/01/2025,Paycheck,Income: Salary,Income,Salary,,$0.00,$2000.00,Cleared
Test Account,,08/05/2025,Rent,Needs: Housing,Needs,Housing,,$1000.00,$0.00,Cleared
Test Account,,08/10/2025,Groceries,Needs: Food,Needs,Food,,$200.00,$0.00,Cleared
Test Account,,08/15/2025,Concert,Wants: Entertainment,Wants,Entertainment,,$150.00,$0.00,Cleared
Test Account,,08/20/2025,Shopping,Wants: Clothes,Wants,Clothes,,$100.00,$0.00,Cleared
Test Account,,08/25/2025,Misc,Other: Misc,Other,Misc,,$50.00,$0.00,Cleared
"""
    
    mock_csv_file = MagicMock(spec=io.BytesIO)
    mock_csv_file.file = io.BytesIO(csv_content.encode('utf-8'))

    monthly_inflow = 2000.0

    df = pd.read_csv(mock_csv_file.file)
    monthly_summary = process_spending_data(df, monthly_inflow)

    assert len(monthly_summary) == 1
    summary = monthly_summary[0]

    assert summary["month"] == "August 2025"
    assert summary["total_net_spent"] == -500.00
    assert summary["needs_spent"] == 1200.00
    assert summary["wants_spent"] == 250.00
    assert summary["other_spent"] == 50.00
    assert "Info: In August 2025, your net spending ($-500.00) was within your monthly inflow ($2000.00)." in summary["recommendations"]
    assert "Warning: In August 2025, your Needs spending was 60.00% of your inflow, exceeding the 50% guideline." in summary["recommendations"]
    assert "Info: In August 2025, your Wants spending adhered to the 50/30 rule relative to your inflow." not in summary["recommendations"]

@pytest.mark.asyncio
async def test_multi_month_summary_calculation():
    csv_content = """
Account,Flag,Date,Payee,Category Group/Category,Category Group,Category,Memo,Outflow,Inflow,Cleared
Test Account,,07/01/2025,Paycheck,Income: Salary,Income,Salary,,$0.00,$1500.00,Cleared
Test Account,,07/05/2025,Rent,Needs: Housing,Needs,Housing,,$800.00,$0.00,Cleared
Test Account,,08/01/2025,Paycheck,Income: Salary,Income,Salary,,$0.00,$2000.00,Cleared
Test Account,,08/05/2025,Rent,Needs: Housing,Needs,Housing,,$1000.00,$0.00,Cleared
"""
    
    mock_csv_file = MagicMock(spec=io.BytesIO)
    mock_csv_file.file = io.BytesIO(csv_content.encode('utf-8'))

    monthly_inflow = 1500.0 # Use a consistent monthly inflow for comparison

    df = pd.read_csv(mock_csv_file.file)
    monthly_summary = process_spending_data(df, monthly_inflow)

    assert len(monthly_summary) == 2

    # July recommendations
    summary_july = monthly_summary[0]
    assert summary_july["month"] == "July 2025"
    assert summary_july["total_net_spent"] == -700.00
    assert summary_july["needs_spent"] == 800.00
    assert summary_july["wants_spent"] == 0.00
    assert summary_july["other_spent"] == 0.00
    assert "Info: In July 2025, your net spending ($-700.00) was within your monthly inflow ($1500.00)." in summary_july["recommendations"]
    assert "Warning: In July 2025, your Needs spending was 53.33% of your inflow, exceeding the 50% guideline." in summary_july["recommendations"]
    assert "Info: In July 2025, your Wants spending adhered to the 50/30 rule relative to your inflow." not in summary_july["recommendations"]

    # August recommendations
    summary_august = monthly_summary[1]
    assert summary_august["month"] == "August 2025"
    assert summary_august["total_net_spent"] == -1000.00
    assert summary_august["needs_spent"] == 1000.00
    assert summary_august["wants_spent"] == 0.00
    assert summary_august["other_spent"] == 0.00
    assert "Info: In August 2025, your net spending ($-1000.00) was within your monthly inflow ($1500.00)." in summary_august["recommendations"]
    assert "Warning: In August 2025, your Needs spending was 66.67% of your inflow, exceeding the 50% guideline." in summary_august["recommendations"]
    assert "Info: In August 2025, your Wants spending adhered to the 50/30 rule relative to your inflow." not in summary_august["recommendations"]

@pytest.mark.asyncio
async def test_inflow_outflow_impact():
    csv_content = """
Account,Flag,Date,Payee,Category Group/Category,Category Group,Category,Memo,Outflow,Inflow,Cleared
Test Account,,08/01/2025,Paycheck,Income: Salary,Income,Salary,,$0.00,$1000.00,Cleared
Test Account,,08/05/2025,Expense,Needs: Essential,Needs,Essential,,$200.00,$0.00,Cleared
Test Account,,08/10/2025,Refund,Income: Refund,Income,Refund,,$0.00,$50.00,Cleared
"""
    
    mock_csv_file = MagicMock(spec=io.BytesIO)
    mock_csv_file.file = io.BytesIO(csv_content.encode('utf-8'))

    monthly_inflow = 1000.0

    df = pd.read_csv(mock_csv_file.file)
    monthly_summary = process_spending_data(df, monthly_inflow)

    assert len(monthly_summary) == 1
    summary = monthly_summary[0]

    assert summary["month"] == "August 2025"
    assert summary["total_net_spent"] == -850.00
    assert summary["needs_spent"] == 200.00
    assert summary["wants_spent"] == 0.00
    assert summary["other_spent"] == 0.00
    assert "Info: In August 2025, your net spending ($-850.00) was within your monthly inflow ($1000.00)." in summary["recommendations"]
    assert "Info: In August 2025, your Needs and Wants spending adhered to the 50/30 rule relative to your inflow." in summary["recommendations"]