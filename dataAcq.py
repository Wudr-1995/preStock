import pandas as pd
import pandas_datareader.data as web   
import datetime as dt

start = dt.datetime(2000, 1, 1)
end = dt.datetime.today()
codes = open('./data/stockCode.txt')
codeData = pd.DataFrame()
for code in codes.readlines():
	code = code.split('\n')
	code = code[0]
	print(code)
	try:
		tmp = web.DataReader(code, 'yahoo', start, end)
		tmp = tmp.Close.to_frame()
		tmp.columns=[code]
		if len(tmp) > 4000:
			codeData = pd.concat([codeData, tmp], axis=1)
	except:
		print('No ' + code)
codeData = codeData.dropna()
codeData.to_csv("./data/PriceAna.csv")


# import tushare as ts

# token = 'a64a79ad5c8a9b293aa1f297f3f7f0c1c440d787e4181e5273b96d92'
# ts.set_token(token)
# pro = ts.pro_api()
# data = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,area,industry,list_date')
# data.to_csv('/Users/wudiru/Documents/pyProject/preStock/data/stockCode.csv')