import pytest
from unittest.mock import MagicMock, patch
from main import calculate_50_30_recommendations
import pandas as pd
import io

@pytest.mark.asyncio
async def test_50_30_rule_violation():
    # Mock DataFrame that violates the 50/30 rule (high wants)
    df_data = {
        'Category Group': ['Needs', 'Wants', 'Wants'],
        'Outflow': [200.00, 400.00, 300.00],
        'Inflow': [0.00, 0.00, 0.00]
    }
    df = pd.DataFrame(df_data)

    recommendations = calculate_50_30_recommendations(df, 1000.0) # Pass monthly_inflow
    
    assert len(recommendations) == 2
    assert "Your spending on Wants is 70.00% of your inflow, which is over the recommended 30%. Look for areas to cut back on discretionary spending." in recommendations
    assert "You have a surplus of $100.00 this month. Consider saving or investing this amount." in recommendations

@pytest.mark.asyncio
async def test_50_30_rule_adherence():
    # Mock DataFrame that adheres to the 50/30 rule
    df_data = {
        'Category Group': ['Needs', 'Wants', 'Other'],
        'Outflow': [500.00, 300.00, 200.00],
        'Inflow': [0.00, 0.00, 0.00]
    }
    df = pd.DataFrame(df_data)

    recommendations = calculate_50_30_recommendations(df, 1000.0) # Pass monthly_inflow
    
    assert len(recommendations) == 2
    assert "Great job! Your spending on Needs and Wants is within the recommended 50/30 rule relative to your inflow." in recommendations
    assert "You have a surplus of $0.00 this month. Consider saving or investing this amount." in recommendations

@pytest.mark.asyncio
async def test_no_outflow_data():
    # Mock DataFrame with no outflow
    df_data = {
        'Category Group': ['Needs'],
        'Outflow': [0.00],
        'Inflow': [0.00]
    }
    df = pd.DataFrame(df_data)

    recommendations = calculate_50_30_recommendations(df, 1000.0) # Pass monthly_inflow
    
    assert len(recommendations) == 2
    assert "Great job! Your spending on Needs and Wants is within the recommended 50/30 rule relative to your inflow." in recommendations
    assert "You have a surplus of $1000.00 this month. Consider saving or investing this amount." in recommendations

@pytest.mark.asyncio
async def test_50_30_rule_double_violation():
    # Mock DataFrame that violates both Needs and Wants percentages
    df_data = {
        'Category Group': ['Needs', 'Wants'],
        'Outflow': [600.00, 400.00],
        'Inflow': [0.00, 0.00]
    }
    df = pd.DataFrame(df_data)

    recommendations = calculate_50_30_recommendations(df, 1000.0) # Pass monthly_inflow
    
    assert len(recommendations) == 3 # Expecting three recommendations
    assert "Your spending on Needs is 60.00% of your inflow, which is over the recommended 50%. Consider reviewing these essential expenses." in recommendations
    assert "Your spending on Wants is 40.00% of your inflow, which is over the recommended 30%. Look for areas to cut back on discretionary spending." in recommendations
    assert "You have a surplus of $0.00 this month. Consider saving or investing this amount." in recommendations

@pytest.mark.asyncio
async def test_monthly_inflow_exceeds_outflow():
    # Mock DataFrame where total outflow exceeds monthly inflow
    df_data = {
        'Category Group': ['Needs'],
        'Outflow': [1200.00],
        'Inflow': [0.00]
    }
    df = pd.DataFrame(df_data)

    recommendations = calculate_50_30_recommendations(df, 1000.0) # monthly_inflow = 1000
    
    assert len(recommendations) == 2
    assert "Your spending on Needs is 120.00% of your inflow, which is over the recommended 50%. Consider reviewing these essential expenses." in recommendations
    assert "Your total net spending ($1200.00) exceeds your monthly inflow ($1000.00). You are spending more than you earn." in recommendations

@pytest.mark.asyncio
async def test_monthly_inflow_surplus():
    # Mock DataFrame where monthly inflow has a surplus
    df_data = {
        'Category Group': ['Needs'],
        'Outflow': [500.00],
        'Inflow': [0.00]
    }
    df = pd.DataFrame(df_data)

    recommendations = calculate_50_30_recommendations(df, 1000.0) # monthly_inflow = 1000
    
    assert len(recommendations) == 2
    assert "Great job! Your spending on Needs and Wants is within the recommended 50/30 rule relative to your inflow." in recommendations
    assert "You have a surplus of $500.00 this month. Consider saving or investing this amount." in recommendations

@pytest.mark.asyncio
async def test_monthly_inflow_not_provided():
    # Mock DataFrame where monthly inflow is 0
    df_data = {
        'Category Group': ['Needs'],
        'Outflow': [500.00],
        'Inflow': [0.00]
    }
    df = pd.DataFrame(df_data)

    recommendations = calculate_50_30_recommendations(df, 0.0) # monthly_inflow = 0
    
    assert len(recommendations) == 2
    assert "Monthly inflow not provided, cannot assess spending against 50/30 rule." in recommendations
    assert "Monthly inflow not provided, cannot assess spending against income." in recommendations