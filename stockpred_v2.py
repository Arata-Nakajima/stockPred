! git clone https://github.com/google-research/timesfm.git
%cd timesfm
!pip install -e .
!pip install utilsforecast
!pip install transformers accelerate bitsandbytes
!huggingface-cli login --token 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
!pip install --upgrade numpy

import datetime
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timesfm
from timesfm import TimesFm

# 調整が必要なパラメータ(シフト量と抑制閾値はほぼ毎回調整が必要です)
# フラット抑制を高めると精度は上がるが、horizon(予測期間)が短くなる
date_m = 6 # month
date_d = 14 # day
shift_d = 5 # 時間軸のシフト量
discard_th = 0.01 # フラット抑制閾値(この変動率以下の予測値は切り捨て)
codelist = ["2558.T"] # 銘柄コード
ohlc = "Adj Close" # 4値

# 現在値(手動設定, 必要があれば)
date_str1 = "2024-6-13"
date_str2 = "2024-6-14"
prices1 = [24240.0, 24285.0, 24225.0, 24475.0]
prices2 = [24430.0, 24500.0, 24420.0, 24685.0]
bk_str = "cpu" # Backend

ohlc2i = {"Open": 0, "High": 1, "Low": 2, "Adj Close": 3}
cur_price1 = prices1[ohlc2i[ohlc]]
cur_price2 = prices2[ohlc2i[ohlc]]

start = datetime.date(2020, 1, 1)
end_t = datetime.date(2024, date_m, date_d )
#end_t = datetime.date.today()

data_train = yf.download(codelist, start = start, end = end_t)
data_all = yf.download(codelist, start = start, end = end_t)
print( data_train )

data_all = data_all[ohlc].dropna()  #欠損値を除去

if data_all.empty:
    raise ValueError("データが空です。期間を変更して再度試してください。")
#data_all.loc[date_str1] = cur_price1
data_all.loc[date_str2] = cur_price2

context_len = 512  # コンテキスト長の設定
horizon_len = 1  # 予測する期間の長さの設定
#horizon_len = 128  # 予測する期間の長さの設定

# TimesFMモデルの初期化と読み込み
tfm = TimesFm(
    context_len = context_len,
    horizon_len = horizon_len,
    input_patch_len = 32,
    output_patch_len = 128,
    num_layers = 20,
    model_dims = 1280,
    backend = bk_str,
)
tfm.load_from_checkpoint(repo_id="google/timesfm-1.0-200m")

# データの前処理
data_train = data_train[ohlc].dropna()  #欠損値を除去

if data_train.empty:
    raise ValueError("データが空です。期間を変更して再度試してください。")
#data_train.loc[date_str1] = cur_price1
data_train.loc[date_str2] = cur_price2
#print("data length", len(data_train))
print( data_train )

if len(data_train) < context_len:
    raise ValueError(f"データの長さがコンテキスト長（{context_len}）より短いです。")

frequency_input = [0]  # データの頻度を設定（0は高頻度のデータ）

pred_len = 60
prev_pf = 23000
concat_pf = np.array([[]])
context_start = datetime.datetime(year = 2024, month = date_m, day = date_d )

for i in range(pred_len):
  tail = context_len + i
  context_begin = context_start - datetime.timedelta( days = i )
  context_end = context_start - datetime.timedelta( days = tail )
  context_data = data_train.loc[ context_end : context_begin ]

  # データの準備
  forecast_input = [context_data.values]

  # 予測の実行
  point_forecast, experimental_quantile_forecast = tfm.forecast(
    forecast_input,
    freq = frequency_input,
  )
  if i > 5 :
    if abs( 100 * ( point_forecast- prev_pf ) / prev_pf ) > discard_th:
      concat_pf = np.append( concat_pf, point_forecast )
      prev_pf = point_forecast
  else :
      concat_pf = np.append( concat_pf, point_forecast )
      prev_pf = point_forecast

concat_pf = np.flip( concat_pf )

#print("point_forcast", point_forecast[0], point_forecast[0].shape)
print("concat_pf", concat_pf, concat_pf.shape)

# 予測結果の表示
pred_len = concat_pf.shape[0]
forecast_dates = pd.date_range( start = data_train.index[ - pred_len ] + pd.Timedelta( days = shift_d ), periods = pred_len, freq='B' )
forecast_series = pd.Series(concat_pf, index=forecast_dates)
#print("forecast_series", forecast_series, forecast_series.index, forecast_series.values)
print("forecast_series", forecast_series)

plt.figure(figsize=(14, 7))
#plt.plot(data_train.index, data_train.values, label="Actual Prices")
plt.plot(data_all.index, data_all.values, label="Actual Prices")
plt.plot(forecast_series.index, forecast_series.values, label="Forecasted Prices")
start_date = datetime.date(2023, 12, 1)
end_date = datetime.date(2024, 7, 1)
plt.xlim(start_date, end_date)
plt.ylim(21000, 25000)
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.show()
