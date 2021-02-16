
import pandas as pd
import numpy as np
import os
os.system('python -m venv venv && venv\\Scripts\\activate.bat && pip install pipreqs && pipreqs "' + os.getcwd() +'" && pip install -r requirements.txt')


def numpyToDataFrame(arr, firmIDs, dates, columnName, firmField='TICKER', dateField='DATE'):
    # This is the function to convert the format of beta array from numpy to CSV

    # INPUT:
    # arr: Calculated beta array
    # firmIDs: Numpy array of firm identifiers used to look-up permnos while creating the DataFrame from the Numpy array
    # dates: Numpy array of trading dates
    # firmField: Name of the header for the firm identifier in the output file
    # dateField: Name of the header for the trading day in the output file

    # OUTPUT:
    # outputDF: a DataFrame object is returned
    try:
        nanmissing_idx = np.where(~np.isnan(arr))
        nanmissing_values = arr[nanmissing_idx].reshape(-1, 1)
        nanmissing_firmIDs = firmIDs[nanmissing_idx[1]].reshape(-1, 1)
        nanmissing_dates = dates[nanmissing_idx[0]].reshape(-1, 1)
        output_array = np.hstack((nanmissing_firmIDs, nanmissing_dates, nanmissing_values))
        outputDF = pd.DataFrame(columns=[firmField, dateField, columnName], data=output_array)
        outputDF[firmField] = outputDF[firmField]
        outputDF[dateField] = outputDF[dateField]

        return outputDF

    except Exception as e:
        print(e)


def marketBeta(dailyData, marketData, months=-12, startDate=196401, firmField='TICKER', dateField='DATE',
                 returnField='TOTRET', minObs=180):
    # This function calculates the downside beta of a stock by Ang. et. al. (2006) within rolling windows of 12-month period.
    # INPUT:
    # dailyData: Daily stock data in the CSV format
    # marketData: Daily market data in the CSV format that has DATE, DAILY RISK FREE RATE and SP 500 market return as columns
    # months: The rolling window size
    # startDate: The first date to start computing the betas. Its format must be YYYYMM.
    # firmField: The firm identifier.
    # dateField: The trading date identifier. Keep it in the YYYYMMDD format
    # minObs: Minimum number of daily observations necessary to calculate betas
    # OUTPUT:
    # - a DataFrame object that contains security ID, YYYYMM, and beta values, i.e, market beta, down market beta, up market beta:
    # - down(up)( market days are identified as the trading days in the past 12-month period where the excess market return is below (above)
    #   than the average market excess return within the rolling window, 12-month period..

    try:
        dailyData = dailyData[[firmField, dateField, returnField]].drop_duplicates()
        dailyData = dailyData[~dailyData.duplicated(subset=[firmField, dateField])]
        startdt = pd.to_datetime(startDate, errors='coerce', format='%Y%m') + pd.DateOffset(months=months + 1)
        startdt = startdt.year * 100 + startdt.month
        dailyData['YYYYMM'] = (dailyData[dateField] / 100).astype(int)
        dailyData['ID'] = dailyData.groupby([firmField]).ngroup()
        lookup_firmField = firmField
        firmField = 'ID'
        # Get trading days from the Daily Data set
        dates = dailyData[[dateField, 'YYYYMM']].drop_duplicates()
        dates['DATETIME'] = pd.to_datetime(dates[dateField], format='%Y%m%d', errors='coerce')
        dates.sort_values(by=[dateField], inplace=True)
        dates_array = dates.as_matrix()

        dailyData.sort_values(by=[firmField, dateField], inplace=True)

        firmIDs = np.sort(dailyData[firmField].unique())
        yyyymm_unique = np.sort(dailyData['YYYYMM'].drop_duplicates().values)
        T = yyyymm_unique.shape[0]

        # Initialize the beta matrices for all securities
        downbetas = np.full([yyyymm_unique.size, firmIDs.size], np.nan)
        upbetas = np.full([yyyymm_unique.size, firmIDs.size], np.nan)
        betas = np.full([yyyymm_unique.size, firmIDs.size], np.nan)

        # Convert the market and stock data to np.array form from the CSV format
        dups = dailyData[dailyData.duplicated(subset=[firmField, dateField], keep=False)]
        dailyData_array = np.array(dailyData.pivot(index=dateField, columns=firmField, values=returnField).as_matrix())
        marketData = marketData[['DATE', 'RF', 'MKT']]
        marketData['MKTRF'] = marketData['MKT'] - marketData['RF']
        marketData['YYYYMM'] = (marketData['DATE'] / 100).astype(int)
        marketData_array = marketData.values
        factor_dates = marketData_array[:, 4]
    except Exception as e:
        print(e)
    T, N = dailyData_array.shape
    # Loop over months for rolling window calculations
    for t in range(months * -1 - 1, T):
        try:
            last_month = yyyymm_unique[t]
            # month_end_date is the index of the month-end date in daily trading "dates".
            last_month_date = dates_array[np.where(dates_array == last_month)[0][0]][2]
            first_month = last_month_date + pd.DateOffset(months=months + 1)
            first_month = max(first_month.year * 100 + first_month.month, yyyymm_unique[0])

            # first_day_idx(last_day_idx) is the index to the first(last) day of the beginning(ending) month of the rolling window
            first_day_idx = np.where(dates_array == first_month)[0][0]
            last_day_idx = np.where(dates_array == last_month)[0][-1]
            window = dailyData_array[first_day_idx:last_day_idx + 1, :]

            # If a security does not have the minimum number of observations necessary, then do not compute the beta for that security
            nobs = (~np.isnan(window)).sum(axis=0)
            less_than_minObs = np.where(nobs < minObs)[0]
            window[:, less_than_minObs] = np.nan

            # Prepare the risk free rate and excess market return for beta computations

            factors_first_day_idx = np.where(factor_dates == first_month)[0][0]
            factors_last_day_idx = np.where(factor_dates == last_month)[0][-1]
            rf = marketData_array[factors_first_day_idx:factors_last_day_idx + 1, 1].reshape(-1, 1)
            rf = np.repeat(rf, firmIDs.size, axis=1)
            excess_stock_returns = window - rf
            mktrf = marketData_array[factors_first_day_idx:factors_last_day_idx + 1, 3].reshape(-1, 1)

            # Call the method to compute betas for downside, upside, and all
            def calculate_market_betas(window, mktrf, permnos, direction='up'):
                # INPUT:
                # window: The rolling window excess return array of size TxN, where T represents the number of trading days within the
                #       rolling window and N represents the number of all securities
                # rf: Risk free rate to compute the excess return used as the dependent variable in the following CAPM regression:
                #   Excess Stock Return = a0 + a1 x Excess Market Return + error term
                #   The regression is run for each stock altogether at the same time using matrix multiplication instead of one-by-one
                # permnos: is used to repeat the market excess return vector for N times (N securities) to do the matrix operation
                # direction:
                #   'up'-> When 'up' option is chosen, the function calculates the market beta of securities in up market times
                #       where up market times are defined as the trading days when the excess market return is above its average
                #       within the rolling window period
                #   'down'-> When 'up' option is chosen, the function calculates the market beta of securities in down market times
                #       where down market times are defined as the trading days when the excess market return is below its average
                #       within the rolling window period.
                #
                #
                #
                # OUTPUT:
                # It returns a Numpy array of desired betas
                try:
                    avg_mktrf = np.nanmean(mktrf)
                    if direction == 'up':
                        days = np.where(mktrf > avg_mktrf)[0]
                    elif direction == 'down':
                        days = np.where(mktrf < avg_mktrf)[0]
                    elif direction == 'all':
                        days = np.arange(0, mktrf.shape[0], 1)
                    else:
                        return
                    excess = window[days, :]
                    excess_mean = np.nanmean(excess, axis=0)
                    excess_demeaned = excess - excess_mean
                    nobs = (np.isnan(excess_demeaned)).sum(axis=0)
                    nan_ones = np.where(nobs == excess_demeaned.shape[0])[0]
                    market_excess = np.repeat(np.copy(mktrf[days, :]), permnos.size, axis=1)
                    market_excess[np.where(np.isnan(excess))] = np.nan
                    market_excess_mean = np.nanmean(market_excess, axis=0)
                    market_excess_demeaned = market_excess - market_excess_mean

                    excess_demeaned[np.where(np.isnan(excess_demeaned))] = 0
                    market_excess_demeaned[np.isnan(market_excess_demeaned)] = 0

                    beta_nominator = np.diag(np.dot(excess_demeaned.T, market_excess_demeaned))
                    beta_denominator = np.diag(np.dot(market_excess_demeaned.T, market_excess_demeaned))
                    beta = np.divide(beta_nominator, beta_denominator)
                    beta = beta.T
                    beta[nan_ones] = np.nan
                except Exception as e:
                    print(e)
                return beta

            downbetas[t, :] = calculate_market_betas(excess_stock_returns, mktrf, firmIDs, 'down')
            upbetas[t, :] = calculate_market_betas(excess_stock_returns, mktrf, firmIDs, 'up')
            betas[t, :] = calculate_market_betas(excess_stock_returns, mktrf, firmIDs, 'all')
        except Exception as e:
            print(e)
            pass
    upbetasDF = numpyToDataFrame(upbetas, dailyData[lookup_firmField].unique(), yyyymm_unique, 'UP_MARKET_BETA',
                         firmField=lookup_firmField)
    downbetasDF = numpyToDataFrame(downbetas, dailyData[lookup_firmField].unique(), yyyymm_unique, 'DOWN_MARKET_BETA',
                           firmField=lookup_firmField)
    betasDF = numpyToDataFrame(betas, dailyData[lookup_firmField].unique(), yyyymm_unique, 'MARKET_BETA',
                       firmField=lookup_firmField)
    allDF = (upbetasDF[[lookup_firmField, dateField]].drop_duplicates()).append(
        (downbetasDF[[lookup_firmField, dateField]].drop_duplicates()).append(
            betasDF[[lookup_firmField, dateField]].drop_duplicates())).drop_duplicates()
    allDF = pd.merge(allDF, upbetasDF, how='left', on=[lookup_firmField, dateField])
    allDF = pd.merge(allDF, downbetasDF, how='left', on=[lookup_firmField, dateField])
    allDF = pd.merge(allDF, betasDF, how='left', on=[lookup_firmField, dateField])
    allDF['DATE'] = allDF['DATE'].astype(int)
    # There might be multiple monthly betas for each security. So, keep only the most recent available one:
    allDF = allDF.loc[allDF.groupby(lookup_firmField)[dateField].idxmax()]

    return allDF

