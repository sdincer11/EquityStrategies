# EquityStrategies

# NOTES FOR Momentum.py:
# Momentum.py has the implementation of Momentum class that calculates the momentum strategy returns
# and momentum profitability. It is implemented following Jegadeesh (1993). 
# Momentum is the strategy of buying (short selling) winner (loser) stocks according to their performance in the past period (let's say 6 months ). 
# It has been documented that winner(loser) stocks continues to win (lose) for another 3, 6, or 9 months.
# for another 6 months.  
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


# NOTES FOR OptionSignals.py
# OptionSignals.py has the codes for calculating option signals to predict future stock returns.
# ATM Call and Put Volatility Spread: Bali and Hovakimian (2009) show the spread between implied volatilities of ATM call and put option contracts predict the future underlying stock returns positively.
# OTM Put and ATM Call Volatility Spread: Xing (2010) shows that the OTM Puts carry information about the underlying stock. Specifically, the slope of the individual option volatility smirk (i.e, the spread between OTM puts and ATM calls of the same underlying) predicts the future stock returns negatively at the weekly level.
# Put-Call Parity Deviations: Cremers and Weinbaum (2010) shows that the put call parity deviations of individual stock options carry some useful information about the future stock performance.

 
