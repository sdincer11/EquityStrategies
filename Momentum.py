import os
os.system('python -m venv venv && venv\\Scripts\\activate.bat && pip install pipreqs && pipreqs "' + os.getcwd() +'" && pip install -r requirements.txt')
import pandas as pd
from numpy import array,full
import numpy as np

class Momentum():
    def __init__(self, csvFile, pastReturnPeriod=6, holdingPeriod=6, skipMonths=2, securityIdentifier = 'PERMNO', dateIdentifier = 'YYYYMM', dateFormat='%Y%m', returnIdentifier='TOTRET', binSize=10):
        self.df = pd.read_csv(csvFile)
        aa = os.getcwd()
        self.holdingPeriod = holdingPeriod
        self.pastReturnPeriod = pastReturnPeriod
        self.skipMonths = skipMonths
        self.securityIdentifier = securityIdentifier
        self.dateIdentifier = dateIdentifier
        self.returnIdentifier = returnIdentifier
        self.binSize = binSize
        self.cumretDF = self.getPastCumulativeReturns()
        self.dateFormat = dateFormat
        self.strategyReturnStatistics = self.getMomentumReturns()

    def getPastCumulativeReturns(self):
        pastCumReturnsFile = os.getcwd().replace('\\','/') + '/past cumulative returns '+ str(self.pastReturnPeriod) + ' months.csv'
        if os.path.exists(pastCumReturnsFile):
            print('Past cumulative returns are NOT computed since there is already a file.')
            cumretDF = pd.read_csv(pastCumReturnsFile)
        else:
            print('Calculating past cumulative returns has been started...')
            try:
                data = self.df[[self.securityIdentifier, self.dateIdentifier, self.returnIdentifier]].drop_duplicates().sort_values(by=[self.securityIdentifier, self.returnIdentifier])
                securityIdentifiers = data[self.securityIdentifier].drop_duplicates().sort_values().values
                dates = data[self.dateIdentifier].drop_duplicates().sort_values().values
                returns = array(data.pivot(index = self.dateIdentifier, columns = self.securityIdentifier, values = self.returnIdentifier))
                T, N = returns.shape
                cumulative_returns = full([T, N], np.nan)
                nobs = full([T, N], np.nan)
                cumretDF = pd.DataFrame()
                for securityIdentifierIndex in range(0, N):
                    try:
                        return_vector = returns[:, securityIdentifierIndex]
                        nanmissing_returns = np.where(~np.isnan(return_vector))[0]
                        if len(nanmissing_returns) > 0:
                            min_yyyymm = min(nanmissing_returns)
                            max_yyyymm = max(nanmissing_returns)
                            for month in range(min_yyyymm, max_yyyymm + 1):
                                time_window = np.arange(month - (self.pastReturnPeriod - 1), month + 1)
                                overlap = np.where(np.isin(time_window, nanmissing_returns))[0]
                                time_window_index = time_window[overlap]
                                if len(time_window_index) > 0:
                                    cumulative_returns[month, securityIdentifierIndex] = np.prod(return_vector[time_window_index] + 1) - 1
                                    nobs[month, securityIdentifierIndex] = len(time_window_index)
                            yyyymm = np.arange(min_yyyymm, max_yyyymm + 1)
                            securityIdentifierColumn = array([securityIdentifiers[securityIdentifierIndex]] * len(yyyymm))
                            cumretDF = cumretDF.append(pd.DataFrame({
                                 self.securityIdentifier: securityIdentifierColumn,
                                 self.dateIdentifier: dates[yyyymm],
                                 'NOBS': nobs[yyyymm, securityIdentifierIndex],
                                 'CUMRET_' + str(self.pastReturnPeriod): cumulative_returns[yyyymm, securityIdentifierIndex]
                            }), ignore_index=True)
                            cumretDF = cumretDF[cumretDF['CUMRET_' + str(self.pastReturnPeriod)].notnull()]
                    except Exception as e:
                        print(e)
                        pass
                print('Calculating past cumulative returns has been finished...')
                cumretDF.to_csv(pastCumReturnsFile,index=False)
            except Exception as e:
                print("An exception is thrown in calculating past cumulative returns: " + e)
        return cumretDF

    def getMomentumReturns(self):
        try:
            strategyReturnStatistics = {'Mean Return': 'N/A',
                                        'Standard Error': 'N/A',
                                        'T-statistic':'N/A',
                                        'Strategy DataFrame':pd.DataFrame()}
            self.df = pd.merge(self.df, self.cumretDF, how = 'left', on = [self.securityIdentifier, self.dateIdentifier])
            momentumField =  'CUMRET_' + str(self.pastReturnPeriod)
            df2 = self.df[[self.dateIdentifier, self.securityIdentifier, momentumField, self.returnIdentifier]].drop_duplicates()
            outputDF = pd.DataFrame()
            for lag in range(self.skipMonths , self.holdingPeriod + self.skipMonths):
                df2[self.dateIdentifier + '_' + str(lag)] = (pd.to_datetime(df2[self.dateIdentifier], format = self.dateFormat, errors='ignore') + pd.DateOffset(months=-1 * lag)).apply(lambda x: x.year * 100 + x.month)
                df2[self.dateIdentifier + '_FINISH'] = (pd.to_datetime(df2[self.dateIdentifier],format = self.dateFormat, errors='ignore') + pd.DateOffset(months=1 * lag)).apply(lambda x: x.year * 100 + x.month)
                df3 = pd.merge(df2, df2[[self.securityIdentifier, self.dateIdentifier + '_' + str(lag), self.returnIdentifier]], how='left', left_on=[self.securityIdentifier, self.dateIdentifier], right_on=[self.securityIdentifier, self.dateIdentifier + '_' + str(lag)], suffixes=('', '_' + str(lag)))
                df3.rename(columns={self.dateIdentifier: self.dateIdentifier+'_START'}, inplace=True)
                df3.drop(columns=[self.dateIdentifier + '_' + str(lag) + '_' + str(lag), self.returnIdentifier], inplace=True, errors='ignore')
                df3.rename(columns={self.returnIdentifier + '_' + str(lag): self.returnIdentifier}, inplace=True)
                outputDF = outputDF.append(df3[[self.securityIdentifier, self.dateIdentifier +'_START',  self.dateIdentifier + '_FINISH', self.returnIdentifier,momentumField ]])
            sortedDF = pd.DataFrame()
            groups = outputDF.groupby([ self.dateIdentifier + '_START'])
            for group_idx, group in groups:
                try:
                    pct = list(np.arange(0.0, 100.0, 100.0 / self.binSize))
                    pct.append(100.0)
                    pct = np.array(pct)
                    group = group[group[momentumField].notnull()]
                    percentiles = np.nanpercentile(group[momentumField].values, pct)
                    group[momentumField + '_RANK'] = pd.cut(group[momentumField], percentiles, right=False, labels=False)
                    group[momentumField + '_RANK'] = group[momentumField + '_RANK'] + 1
                    sortedDF = sortedDF.append(group, ignore_index=True)
                except Exception as e:
                    print(e)
                    pass
            portfolio_returns = pd.DataFrame()
            for portfolio_id, portfolio in sortedDF.groupby([ self.dateIdentifier + '_START',  self.dateIdentifier + '_FINISH', momentumField + '_RANK']):
                portfolio['PORTFOLIO_RETURN'] = np.nanmean(portfolio[self.returnIdentifier])
                portfolio_returns = portfolio_returns.append(portfolio[[self.dateIdentifier + '_START', self.dateIdentifier + '_FINISH', momentumField + '_RANK', 'PORTFOLIO_RETURN']].drop_duplicates())
            final_portfolio_returns = pd.DataFrame()
            for portfolio_id, portfolio in portfolio_returns.groupby([self.dateIdentifier + '_FINISH', momentumField + '_RANK']):
                portfolio['PORTFOLIO_RETURN'] = np.nanmean(portfolio['PORTFOLIO_RETURN'])
                final_portfolio_returns = final_portfolio_returns.append(portfolio[[self.dateIdentifier + '_FINISH', momentumField + '_RANK', 'PORTFOLIO_RETURN']].drop_duplicates())

            winner = final_portfolio_returns[final_portfolio_returns[momentumField + '_RANK'] == self.binSize]
            loser = final_portfolio_returns[final_portfolio_returns[momentumField + '_RANK'] == 1]
            momentum = pd.merge(winner, loser, how='left', on=[ self.dateIdentifier + '_FINISH'], suffixes=('_WINNER', '_LOSER'))
            momentum['STRATEGY_RETURN'] = momentum['PORTFOLIO_RETURN_WINNER'] - momentum['PORTFOLIO_RETURN_LOSER']
            momentum.sort_values(by= self.dateIdentifier + '_FINISH', inplace=True)
            momentum['CUMULATIVE_STRATEGY_RETURN'] = (momentum['STRATEGY_RETURN'] + 1).cumprod() * 100

            strategyReturnStatistics['Standard Error'] = (np.nanstd(momentum['STRATEGY_RETURN'], ddof=1) / np.sqrt(len(momentum)))
            strategyReturnStatistics['Mean Return'] = np.nanmean(momentum['STRATEGY_RETURN'])
            strategyReturnStatistics['T-statistic'] = strategyReturnStatistics['Mean Return'] /strategyReturnStatistics['Standard Error']

        except Exception as e:
            strategyReturnStatistics = {'Mean Return': 'N/A',
                                        'Standard Error': 'N/A',
                                        'T-statistic': 'N/A',
                                        'Strategy DataFrame': pd.DataFrame()}
        return strategyReturnStatistics

