from flask import Flask, render_template, request
import predict_flask
import speech_recognition as sr

config = {
    "DEBUG": True
}

app = Flask(__name__)

app.config.from_mapping(config)
app.config["TEMPLATES_AUTO_RELOAD"] = True


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/model")
def model():
    return render_template("model.html")

@app.route("/analytics")
def analytics():
    return render_template("analytics.html")

@app.route("/notebook")
def notebook():
    return render_template("data_ravdess.html")


@app.route("/dataset")
def dataset():
    return render_template("dataset.html")

@app.route('/', methods=['POST'])
def my_post():
    # Results
    if request.form['submit_button'] == 'Load Audio File':
        transcript=""
        output=""

        file = request.files['file']

        if file:
            name = file.filename
            filename = "./" + name
            r = sr.Recognizer()
            fn = sr.AudioFile(file)
            with fn as source:
                audio = r.record(source)
            transcript = r.recognize_google(audio, key=None)
            output = predict_flask.predict(filename)

        return render_template('result.html', text=transcript, output=output)
    
    if request.form['submit_button'] == 'Record Live Audio':
        output, text = predict_flask.get_audio()
        result = render_template("result.html", output=output, text=text)

    elif request.form['submit_button'] == 'Explore Model':
        result = render_template("model.html")
    else:
        result = render_template("info.html")
    return result

if __name__ == "__main__":
    app.jinja_env.auto_reload = True
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run()
