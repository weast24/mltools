# from sklearn import preprocessing
# 標準化
# https://note.nkmk.me/python-list-ndarray-dataframe-normalize-standardize/
# 参考
# https://facebook.github.io/prophet/docs/quick_start.html#python-api

import matplotlib.pyplot as plt

# import numpy as np
import pandas as pd
from sklearn import preprocessing

from fbprophet import Prophet

# from fbprophet.diagnostics import cross_validation, performance_metrics
# from fbprophet.plot import plot_cross_validation_metric


def plot_prophet(input_filepath: str, preriods: int) -> None:
    df = pd.read_csv(input_filepath)
    df = df.reset_index().rename(columns={"Date": "ds", "CV": "y"})

    # 差分取ってみる
    # df["y"] = df["y"].diff().fillna(0)

    # yの自然対数を取ってみる
    # df["y_origin"] = df["y"]
    # df["y"] = np.log(df["y"])

    # 標準化
    # x = df["ds"]
    # y = df["y"]
    # ds_ss = (x - x.mean()) / x.std(ddof=0)
    # y_ss = (y - y.mean()) / y.std(ddof=0)
    # df["ds"] = ds_ss
    # df["y"] = y_ss

    sscaler = preprocessing.StandardScaler()
    sscaler.fit(df)

    # optionaly 過剰適合しないように月周期性を付与
    # m = Prophet(weekly_seasonality=False)
    # model.add_seasonality(name='monthly', period=30.5, fourier_order=5)

    # 曜日周期性を付与
    m = Prophet(
        growth="linear",
        n_changepoints=25,
        weekly_seasonality=True,
        yearly_seasonality=False,
        changepoint_prior_scale=0.1,
    )

    # 外れ値の除去
    # df.loc[(df['ds'] > '2010-01-01') & (df['ds'] < '2011-01-01'), 'y'] = None

    m.fit(df)

    # periods: 何日先まで予測するか
    future = m.make_future_dataframe(periods=preriods)
    forecast = m.predict(future)

    m.plot(forecast)
    plt.ylim(0, preriods)
    """

    fig2 = m.plot_components(forecast)
    # plt.show()

    # cross validation
    df_cv = cross_validation(m, initial="1 days", period="14 days", horizon="98 days")
    fig = plot_cross_validation_metric(df_cv, metric="mape")

    # 統計量の計算
    df_p = performance_metrics(df_cv)
    df_p.head()
    """
