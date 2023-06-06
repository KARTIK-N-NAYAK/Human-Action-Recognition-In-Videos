from flask import Flask, render_template, request, url_for, redirect, session
from pytube import YouTube
from LRCN import LRCN
from ConvLSTM import ConvLSTM

app = Flask(__name__)
app.config['SECRET_KEY'] = "654c0fb3968af9d5e6a9b3edcbc7051b"
UPLOAD_FOLDER = 'static/'

@app.route("/", methods = ["GET", "POST"])
def home():
    if request.method == "POST":
        session['link'] = request.form.get('url')
        try:
            url = YouTube(session['link'])
            url.check_availability()
        except:
            return render_template("home.html")
        return render_template("download.html", url = url)
    return render_template("home.html")

@app.route("/download", methods = ["GET", "POST"])
def download_video():
    if request.method == "POST":
        url = YouTube(session['link'])
        title = 'test.mp4'
        itag = request.form.get("itag")
        video = url.streams.get_by_itag(itag)   #itag resolution we choose
        video.download(UPLOAD_FOLDER, title)
        LRCN_ClassName, LRCN_conf = LRCN.predict_LRCN('static/'+title)
        ConvLSTM_ClassName, ConvLSTM_conf = ConvLSTM.predict_ConvLSTM('static/'+title)
        # ConvLSTM_result = ConvLSTM.predict_ConvLSTM('static/'+title)
    # return redirect(url_for("home"))
    return render_template("predict.html", filename = title, LRCN_ClassName = LRCN_ClassName, LRCN_conf = LRCN_conf, ConvLSTM_ClassName = ConvLSTM_ClassName, ConvLSTM_conf = ConvLSTM_conf )

# @app.route('/display/<filename>')
# def display_video(filename):
# 	#print('display_video filename: ' + filename)
# 	return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == '__main__':
    app.run(debug=True)