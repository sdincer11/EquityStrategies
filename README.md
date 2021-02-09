# EquityStrategies

# NOTES FOR Momentum.py:
# Momentum.py has the implementation of Momentum class that calculates the momentum strategy returns
#   and momentum profitability. It is implemented following Jegadeesh (1993). 
# Momentum is the strategy of winner (loser) stocks in the past period (let's say 6 months ) continues to win (lose)
#   for another 6 months.  
# Check MomentumExample.py to see an example for how to call the Momentum class and its functions

# You need to provide a CSV File with its full path, and specify the strategy parameters as in Jegadeesh(1993):
# 1) pastReturnPeriod
# 2) holdingPeriod
# 3) skipMonths: Jegadeesh (1993) skips one month, so skipMonths should be set to 2.
# There are other parameters that need to be input while creating an instance of the Momentum class:
# 1) securityIdentifier = 'PERMNO',
# 2) dateIdentifier = 'YYYYMM',
# 3) dateFormat='%Y%m' for 202010, i.e, October 2020,
# 4) returnIdentifier='TOTRET',
# 5) binSize=10
# WARNING: You need to ensure that the CSV File does not have any duplicate records for each (securityIdentifier, dateIdentifier) combination.

 
