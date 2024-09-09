from flask import Flask, request, render_template, jsonify, redirect, url_for
import os
import numpy as np
import librosa
import tensorflow as tf
from features import extract_features
import pickle

app = Flask(__name__)

# Ensure the uploads directory exists
if not os.path.exists('uploads'):
    os.makedirs('uploads')

model = tf.keras.models.load_model('best_model_weights.keras')  # Load your Keras model

def extract_mfcc(file_path):
    # Load audio file
    y, sr = librosa.load(file_path, sr=None)
    # Extract MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    # Average MFCCs over time
    mfccs_mean = np.mean(mfccs.T, axis=0)
    return mfccs_mean


@app.after_request
def add_header(response):
    response.cache_control.no_store = True
    return response


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_page')
def upload_page():
    return render_template('upload.html')

@app.route('/record_page')
def record_page():
    return render_template('record.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'audio_file' not in request.files:
        return "No file part"
    file = request.files['audio_file']
    if file.filename == '':
        return "No selected file"
    if file:
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)
        # Redirect to results page with file path
        return redirect(url_for('results', file_path=file_path))

@app.route('/record', methods=['POST'])
def record():
    if 'audio_data' not in request.files:
        return "No audio data"
    
    file = request.files['audio_data']
    file_path = "uploads/recording.wav"
    file.save(file_path)
    
    # Redirect to results page with file path
    return redirect(url_for('results', file_path=file_path))




@app.route('/results')
def results():

    file_path = request.args.get('file_path')
    if not file_path:
        return "No file provided"
    
    with open('scaler2.pickle', 'rb') as f:
        scaler2 = pickle.load(f)
    
    with open('encoder2.pickle', 'rb') as f:
        encoder2 = pickle.load(f)

    y, sr = librosa.load(file_path, sr=22050, duration=2.5,offset=0.5)

    print(f"SR is {sr}")



    if sr != 22050:
        y = librosa.resample(y, orig_sr=sr, target_sr=22050)
        sr = 22050

    target_length = int(2.5* sr)

    y = librosa.util.fix_length(y, size=target_length)

    res = extract_features(y)
    result=np.array(res)
    result=np.reshape(result,newshape=(1,2376))
    i_result = scaler2.transform(result)
    final_result=np.expand_dims(i_result, axis=2)


    # Load the pre-trained Keras model
    def load_model():
        model = tf.keras.models.load_model("best_model1_weights.h5")
        return model

    model = load_model()


    emotions = ['Angry'  , 'Disgust'  , 'Fear', 'Happy' , 'Neutral' , 'Sad' ,  'Surprise']
    predictions=model.predict(final_result)
    print(predictions)
    y_pred = np.argmax(predictions)
    #print(emotions[y_pred])
    return render_template('result.html', emotion=emotions[y_pred])



    """
    mfccs = extract_mfcc(file_path)
    mfccs_reshaped = mfccs.reshape(-1,20,1)
    prediction = model.predict(mfccs_reshaped)
    emotion = np.argmax(prediction)
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad']
    #st.write(f"Predicted Emotion: {emotion_labels[emotion]}")
    return render_template('result.html', emotion=emotion_labels[emotion])
    """





@app.route('/model_info', methods=['GET'])
def model_info():
    return jsonify({"model": "Keras LSTM", "accuracy": "90%"})

@app.route('/dataset_info', methods=['GET'])
def dataset_info():
    return jsonify({"dataset": "CREMA-D", "samples": 7442})

if __name__ == '__main__':
    app.run(debug=True)
