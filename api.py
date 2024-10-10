import pandas as pd
import pickle
from flask import Flask, request, render_template

app = Flask(__name__)

# טען את המודל
best_model = pickle.load(open('trained_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # קבלת הנתונים מהפורם
    manufactor = request.form.get('manufactor')
    year = int(request.form.get('year'))
    hand = int(request.form.get('hand'))
    gear = request.form.get('gear')
    capacity_engine = float(request.form.get('capacity_engine'))
    engine_type = request.form.get('engine_type')
    prev_ownership = request.form.get('prev_ownership')
    curr_ownership = request.form.get('curr_ownership')
    city = request.form.get('city')
    dial_code = request.form.get('dial_code')
    color = request.form.get('color')
    km = float(request.form.get('km'))

    # הכנת הנתונים לפלט של המודל
    data = {
        'manufactor': [manufactor],
        'Year': [year],
        'Hand': [hand],
        'Gear': [gear],
        'capacity_Engine': [capacity_engine],
        'Engine_type': [engine_type],
        'Prev_ownership': [prev_ownership],
        'Curr_ownership': [curr_ownership],
        'City': [city],
        'Dial_code': [dial_code],
        'Color': [color],
        'Km': [km]
    }
    df = pd.DataFrame(data)

    # ביצוע ניבוי
    prediction = best_model.predict(df)[0]

    return render_template('index.html', prediction_text=f'Estimated Price: ₪{prediction:.2f}')

if __name__ == "__main__":
    app.run(debug=True)
