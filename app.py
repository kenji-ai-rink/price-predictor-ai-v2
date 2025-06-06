import os
import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle
from flask import Flask, request, render_template

# モデルファイルが存在しない場合は新規学習
if not os.path.exists("model.pkl"):
    df = pd.read_csv("data.csv")
    X = df[['面積', '築年数']]
    y = df['価格']
    model = LinearRegression()
    model.fit(X, y)
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
else:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    price = None
    if request.method == 'POST':
        try:
            area = float(request.form['area'])
            age = float(request.form['age'])
            prediction = model.predict([[area, age]])
            price = round(prediction[0], 1)
        except:
            price = "計算エラー"
    return render_template('index.html', price=price)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
