from flask import Flask, render_template, request, redirect, url_for
import json
import os
from models.growth_model import predict_growth_stage, predict_harvest_days, evaluate_plant_condition, generate_care_recommendations, compute_health_score

app = Flask(__name__)
DATA_FILE = os.path.join(app.root_path, 'data', 'weekly_data.json')

PLANTS = {
    'cabai': {
        'label': 'Cabai',
        'description': 'Tanaman cabai membutuhkan sinar matahari penuh, tanah gembur, dan penyiraman teratur.'
    },
    'tomat': {
        'label': 'Tomat',
        'description': 'Tomat tumbuh baik pada suhu hangat dan kelembapan seimbang dengan pemupukan rutin.'
    },
    'terong': {
        'label': 'Terong',
        'description': 'Terong membutuhkan paparan sinar matahari setidaknya 6 jam per hari dan drainase yang baik.'
    }
}


def load_data():
    if not os.path.exists(DATA_FILE):
        return {plant: [] for plant in PLANTS}

    with open(DATA_FILE, 'r', encoding='utf-8') as data_file:
        return json.load(data_file)


def save_data(data):
    os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
    with open(DATA_FILE, 'w', encoding='utf-8') as data_file:
        json.dump(data, data_file, ensure_ascii=False, indent=2)


@app.route('/')
def index():
    data = load_data()
    return render_template('index.html', plants=PLANTS, data=data)


@app.route('/plant/<plant_name>')
def plant_page(plant_name):
    plant_info = PLANTS.get(plant_name)
    if plant_info is None:
        return redirect(url_for('index'))

    data = load_data()
    records = data.get(plant_name, [])
    prediction = None
    harvest_estimate = None
    plant_condition = None
    care_recommendations = None

    if records:
        last = records[-1]
        # compute health score automatically from available measurements
        health_score = compute_health_score(
            float(last.get('age_days', 0)),
            float(last.get('height_cm', 0)),
            int(last.get('leaf_count', 0)),
            float(last.get('temperature_c', 0)),
            float(last.get('humidity_pct', 0)),
        )
        prediction = predict_growth_stage(
            float(last.get('height_cm', 0)),
            int(last.get('leaf_count', 0)),
            health_score,
        )
        if all(key in last for key in ('age_days', 'temperature_c', 'humidity_pct')):
            plant_condition = evaluate_plant_condition(
                float(last.get('age_days', 0)),
                float(last.get('temperature_c', 0)),
                float(last.get('humidity_pct', 0)),
            )
            care_recommendations = generate_care_recommendations(
                plant_condition,
                float(last.get('temperature_c', 0)),
                float(last.get('humidity_pct', 0)),
                float(last.get('age_days', 0)),
            )
        if plant_condition is not None and all(key in last for key in ('age_days', 'humidity_pct', 'temperature_c')):
            try:
                harvest_estimate = predict_harvest_days(
                    float(last.get('age_days', 0)),
                    float(last.get('height_cm', 0)),
                    int(last.get('leaf_count', 0)),
                    float(last.get('humidity_pct', 0)),
                    float(last.get('temperature_c', 0)),
                    plant_condition,
                    plant_name,
                )
            except Exception:
                harvest_estimate = None

    return render_template(
        'plant.html',
        plant_name=plant_name,
        plant_info=plant_info,
        records=records,
        prediction=prediction,
        harvest_estimate=harvest_estimate,
        plant_condition=plant_condition,
        care_recommendations=care_recommendations,
    )


@app.route('/report/<plant_name>')
def report(plant_name):
    plant_info = PLANTS.get(plant_name)
    if plant_info is None:
        return redirect(url_for('index'))

    data = load_data()
    records = data.get(plant_name, [])
    growth_stage = None
    harvest_estimate = None
    plant_condition = None
    care_recommendations = None
    if records:
        last = records[-1]
        health_score = compute_health_score(
            float(last.get('age_days', 0)),
            float(last.get('height_cm', 0)),
            int(last.get('leaf_count', 0)),
            float(last.get('temperature_c', 0)),
            float(last.get('humidity_pct', 0)),
        )
        growth_stage = predict_growth_stage(
            float(last.get('height_cm', 0)),
            int(last.get('leaf_count', 0)),
            health_score,
        )
        if all(key in last for key in ('age_days', 'temperature_c', 'humidity_pct')):
            plant_condition = evaluate_plant_condition(
                float(last.get('age_days', 0)),
                float(last.get('temperature_c', 0)),
                float(last.get('humidity_pct', 0)),
            )
            care_recommendations = generate_care_recommendations(
                plant_condition,
                float(last.get('temperature_c', 0)),
                float(last.get('humidity_pct', 0)),
                float(last.get('age_days', 0)),
            )
        if plant_condition is not None and all(key in last for key in ('age_days', 'humidity_pct', 'temperature_c')):
            try:
                harvest_estimate = predict_harvest_days(
                    float(last.get('age_days', 0)),
                    float(last.get('height_cm', 0)),
                    int(last.get('leaf_count', 0)),
                    float(last.get('humidity_pct', 0)),
                    float(last.get('temperature_c', 0)),
                    plant_condition,
                    plant_name,
                )
            except Exception:
                harvest_estimate = None

    return render_template(
        'report.html',
        plant_name=plant_name,
        plant_info=plant_info,
        records=records,
        growth_stage=growth_stage,
        harvest_estimate=harvest_estimate,
        plant_condition=plant_condition,
        care_recommendations=care_recommendations,
    )


@app.route('/save', methods=['POST'])
def save_record():
    plant_name = request.form.get('plant_name')
    if plant_name not in PLANTS:
        return redirect(url_for('index'))

    record = {
        'week': request.form.get('week', '').strip(),
        'age_days': int(request.form.get('age_days') or 0),
        'height_cm': float(request.form.get('height_cm') or 0),
        'leaf_count': int(request.form.get('leaf_count') or 0),
        'humidity_pct': float(request.form.get('humidity_pct') or 0),
        'temperature_c': float(request.form.get('temperature_c') or 0),
    }

    data = load_data()
    data.setdefault(plant_name, []).append(record)
    save_data(data)

    return redirect(url_for('plant_page', plant_name=plant_name))


@app.route('/delete/<plant_name>/<int:record_idx>', methods=['POST'])
def delete_record(plant_name, record_idx):
    if plant_name not in PLANTS:
        return redirect(url_for('index'))

    data = load_data()
    records = data.get(plant_name, [])
    if 0 <= record_idx < len(records):
        records.pop(record_idx)
        data[plant_name] = records
        save_data(data)

    return redirect(url_for('plant_page', plant_name=plant_name))


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
