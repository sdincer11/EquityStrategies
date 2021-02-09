from Momentum import *
# You need to provide a CSV File with its full path, and specify the strategy parameters as in Jegaadeesh(1993):
# 1) pastReturnPeriod = J
# 2) holdingPeriod = K
# 3) skipMonths: Jegadeesh (1993) skips one month, so skipMonths should be set to 2.
# There are other parameters that need to be input while creating an instance of the Momentum class:
# 1) securityIdentifier = 'PERMNO',
# 2) dateIdentifier = 'YYYYMM',
# 3) dateFormat='%Y%m',
# 4) returnIdentifier='TOTRET',
# 5) binSize=10
# WARNING: You need to ensure that the CSV File does not have any duplicate records for each (securityIdentifier, dateIdentifier) combination.
currentFolder = os.getcwd('\\','/') + '/'
momentum = Momentum(csvFile = currentFolder + 'monthly stock returns.csv',
                    pastReturnPeriod=6, holdingPeriod=6,
                    securityIdentifier='PERMNO', dateIdentifier='YYYYMM',
                    dateFormat='%Y%m', returnIdentifier='TOTRET', binSize=10)
