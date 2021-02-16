import pandas as pd
import numpy as np

# This measure of tail risk is proposed by Kelly and Jiang (2014). Tail risk loading is not exactly like market beta
# since the independent variable in the regression, market risk measure (lambda), is not a "return" variable. So, it might be
# better to assign a bin rank for each security to have a meaningful interpretation. The stocks with high positive
# tail risk loading down-perform more when the market faces a tail event. The price of such stock goes down more than
# a stock with lower positive tail risk loading when the tail event occurs.
# 1) Tail risk measure is computed by exploiting a statistical concept, called "Power Law".
# 2) Within each month, daily returns of all stocks traded in NYSE, NASDAQ, and AMEX are used.
# 3) Monthly market tail threshold, u, is calculated as the bottom Xth percentile of all daily returns of all stocks
#  within that month.
# 4) All daily stock returns within a month that are below the threshold, u, are filtered out to be used in step5.
# 5) Market tail risk measure, lambda, is calculated as : Avg[ ln (R_stock/ u) ] where R_stock comes from step 4.
# 6) Tail risk loading for each stock, then, is calculated by regressing the stock's monthly raw returns on the
#   market tail risk measure, lambda,  within the past 120-month window.

# NOTE: The main function to call below is tail_risk_loadings(). It calls for calculate_tail_risk_loadings, which in turn
#   calls for np_to_csv() to write the generated Numpy array to CSV.
def numpyToDataFrame(arr,firmIDs,dates,firmField='TICKER',dateField='YYYYMM',columnName='BETA'):
    # This is the function to convert the format of beta array from numpy to CSV

    # INPUT:
    # arr: Calculated beta array
    # firmIDs: Numpy array of firm identifiers used to look-up permnos while creating the DataFrame from the Numpy array
    # dates: Numpy array of trading dates
    # firmField: Name of the header for the firm identifier in the output file
    # dateField: Name of the header for the trading day in the output file

    #OUTPUT:
    # outFilePath: is the full file path of the CSV file to write the resulting DF into
    try:
        nanmissing_idx=np.where(~np.isnan(arr))
        nanmissing_values=arr[nanmissing_idx].reshape(-1,1)
        nanmissing_firmIDs=firmIDs[nanmissing_idx[1]].reshape(-1,1)
        nanmissing_dates=dates[nanmissing_idx[0]].reshape(-1,1)
        output_array=np.hstack((nanmissing_firmIDs,nanmissing_dates,nanmissing_values))
        outputDF=pd.DataFrame(columns=[firmField,dateField,columnName],data=output_array)
        outputDF[firmField]=outputDF[firmField].astype(int)
        outputDF[dateField]=outputDF[dateField].astype(int)
        #outputDF.sort_values(by=[firmField,dateField],inplace=True)
        return outputDF

    except Exception as e:
        print(e)
def tailBeta(msf,dsf,firmField='TICKER',dateField='YYYYMM',returnField='RET',tailThreshold=0.05,rolling_window_months=120,minObs=20):
    # INPUT:
    #   msf: a Pandas DataFrame object that has month-end stock returns
    #   dsf: a Pandas DataFrame object that has daily stock returns
    #   firmField: the column name to identify firms.
    #   dateField: the column name to identify monthly dates.
    #   returnField: the column name for returns
    #   tailThreshold: the bottom percentile of all cross sectional daily returns within a month to identify the tail.
    #   rolling_winod_months: Number of months over which the tail risk loading is calculated for each security
    #   minObs: Minimum number of monthly returns necessary to identify tail risk loading
    msf.columns=[col.upper() for col in msf.columns]
    dsf.columns=[col.upper() for col in dsf.columns]
    dsf['YYYYMM']=(dsf['DATE']/100).astype(int)

    groups=dsf.groupby([dateField])
    dates=np.sort(dsf[dateField].unique())
    lambdas=np.full(dates.size,np.nan)
    i=0
    # For each month, calculate the tail lambda of the market by using all stocks' daily returns within each month.
    # Find the daily returns of all stocks that are below the threshold, i.e, within tail risk region.
    # Calculate the monthly market tail risk: Lambda
    for group_id,group in groups:
        threshold=group[returnField].quantile(q=tailThreshold)
        group=group[group[returnField]<=threshold]
        lambdas[i]=np.nanmean(np.log(group[returnField]/threshold))
        i+=1
    lambdas=np.vstack((dates,lambdas)).T
    lambdasDF=pd.DataFrame(columns=['YYYYMM','TAIL_LAMBDA'],data=lambdas)

    # Create a dataframe for the rolling window of monthly returns for each month.
    msf=pd.merge(msf,lambdasDF,how='left',left_on=[dateField],right_on=[dateField],suffixes=('','_y'))
    def tailRiskLoadings(msf,firmField,dateField,returnField,rolling_window_size=rolling_window_months,minObs=minObs):
        # This function is the main function to calculate tail risk loadings of each security within rolling windows
        # and returns a DataFrame object
        #
        # INPUT:
        # msf: A Pandas DataFrame object that has monthly stock information
        # firmIDs: A Numpy object that contains the list of firm identifiers
        # dates: A Numpy object that contains the list of trading dates.
        # firmField: The firm identifier column name
        # dateField: The date identifier column name
        # rolling_window_size: The window of months to compute the tail risk on.
        # minObs: The minimum number of monthly return observations within the rolling window required for a stock to compute
        #   its tail risk loading.
        # returnDF: If True, then return a DataFrame object, and do not write the data to a CSV File. If False, then
        #       don't return any DataFrame object, but write the data to a CSV file
        msf.sort_values(by=[firmField,dateField],inplace=True)
        returns_array=np.array(msf.pivot(index=dateField,columns=firmField,values=returnField).as_matrix())
        tail_risk_array=np.array(msf.pivot(index=dateField,columns=firmField,values='TAIL_LAMBDA').as_matrix())
        firmIDs=np.sort(msf[firmField].unique())
        dates=np.sort(msf[dateField].unique())
        T=len(dates)
        N=len(firmIDs)
        tail_betas=np.full([T,N],np.nan)
        for t in range(minObs,T):
            rolling_window_returns=returns_array[max(0,t-rolling_window_size+1):t,:]
            tail_risk_window=tail_risk_array[max(0,t-rolling_window_size+1):t,:]
            nobs=(~np.isnan(rolling_window_returns)).sum(axis=0)
            rolling_window_returns_mean=np.nanmean(rolling_window_returns,axis=0)
            rolling_window_returns_demeaned=rolling_window_returns-rolling_window_returns_mean
            rolling_window_returns_demeaned[np.where(np.isnan(rolling_window_returns_demeaned))]=0

            tail_risk_mean=np.nanmean(tail_risk_window,axis=0)
            tail_risk_demeaned=tail_risk_window-tail_risk_mean
            tail_risk_demeaned[np.where(np.isnan(tail_risk_demeaned))]=0

            beta_nominator=np.diag(np.dot(rolling_window_returns_demeaned.T,tail_risk_demeaned))
            beta_denominator=np.diag(np.dot(tail_risk_demeaned.T,tail_risk_demeaned))
            betas=np.divide(beta_nominator,beta_denominator)
            betas=betas.T
            nan_ones=np.where(nobs<minObs)
            betas[nan_ones]=np.nan
            tail_betas[t,:]=betas
        return numpyToDataFrame(tail_betas,firmIDs,dates,firmField,dateField,columnName='TAIL_BETA')
    tailDF=tailRiskLoadings(msf,firmField,dateField,returnField)
    return tailDF