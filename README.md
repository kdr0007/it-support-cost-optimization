# IT Support Cost Optimization

Python analysis of IT support ticket costs using **generated sample data**.

## What it does
- Creates 20,000 sample support tickets (category, priority, resolution time, engineer cost, client type, escalations, CSAT score)
- Cleans missing values and estimates the labour cost of each ticket
- Reports cost by issue category, client type, and priority
- Calculates a 20% cost-reduction target
- Trains a Random Forest model to flag tickets likely to need repeated escalation
- Prints practical recommendations to investigate

## Files
| File | Description |
|------|-------------|
| `it_support_cost_optimization.py` | The full analysis script |
| `IT_Support_Cost_Optimization_Code_and_Output.pdf` | Source code and the complete output |

## How to run
1. Install Python 3.8 or newer.
2. Install the libraries:
```
   python -m pip install pandas numpy scikit-learn
```
3. Run the script:
```
   python it_support_cost_optimization.py
```

## Sample results
- Tickets analyzed: 20,000
- Highest-cost category: Network
- Slowest category on average: Hardware
- Tickets with 2 or more escalations: 6.3%

## Important note
All figures come from **generated sample data** (fixed random seed 42). They are not findings about a real company. To use real data, replace `make_sample_data()` with `pd.read_csv("your_tickets.csv")`.

A Python script alone does not prove a system can process 50,000 events per second. For production streaming, use a platform such as Kafka and benchmark it.

## Built with
Python, pandas, NumPy, scikit-learn
