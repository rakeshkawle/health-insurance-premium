from flask import Flask, request, render_template

from src.pipelines.prediction_pipelines import CustomData, PredictPipeline

app = Flask(__name__)

@app.route('/')
def home_page():
    return render_template('form.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoints():
    if request.method == 'GET':
        return render_template('form.html')
    else:
        data = CustomData(
            age=request.form.get('Age'),
            sex=request.form.get('Sex'),
            bmi=request.form.get('Bmi'),
            children=request.form.get('Children'),
            smoker=request.form.get('Smoker'),
            region=request.form.get('Region')
        )

        final_new_data = data.get_data_as_dataframe()
        predict_pipeline = PredictPipeline()
        pred = predict_pipeline.predict(final_new_data)

        result = "{} $".format(round(pred[0], 2))

        return render_template('result.html', final_result=result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
