import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# データ読み込み
df = pd.read_csv("data.csv")
X = df[['面積', '築年数']]
y = df['価格']

# モデル学習
model = LinearRegression()
model.fit(X, y)

# モデル保存
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
