# Mortgage Credit-Risk Decision Support Prototype

Proof of concept AI/ML decision support system for a UK mortgage lender.
It predicts the probability of major repayment difficulty within 12 months and uses documented decision rules to offer an interpretable recommendation to a human reviewer. It does not automatically approve/reject applications.

## Installation ```bash python3 -m venv .venv source.venv/bin/activate # Windows:.venv\Scripts\activate install pip -r requirements.txt ```

## Run ```bash python main.py # train, evaluate and score applications pytest # run test suite ```
