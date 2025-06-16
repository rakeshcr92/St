"""Flask web interface for InsiderNet demo."""
from __future__ import annotations

from flask import Flask, render_template

from .pipeline import get_predictions, get_prediction_history

app = Flask(__name__)


@app.route('/')
def index():
    df = get_predictions()
    rows = df.to_dict(orient='records')
    hist_df = get_prediction_history()
    history = hist_df.to_dict(orient='records')
    return render_template('index.html', rows=rows, history=history)


if __name__ == '__main__':
    app.run(debug=True)
