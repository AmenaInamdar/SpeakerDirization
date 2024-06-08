from werkzeug.utils import secure_filename
from flask import Flask, request, render_template, flash, redirect, url_for
from Database import *
from flask import Flask, render_template, request, redirect, url_for
import pvfalcon as backbon
import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import librosa.display
import os
import matplotlib.pyplot as plt
import numpy as np
import librosa.display
from datetime import datetime

from flask import Flask, request, jsonify
import speech_recognition as sr

app = Flask(__name__)



global filename
filename = ""

app.secret_key = 'super secret key'
app.config['SESSION_TYPE'] = SYSTEM_SECRET_KEY
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def index():
    return render_template("index.html", xx= -1)


@app.route('/index')
def index1():
   return render_template('index.html')


@app.route('/home')
def home():
   return render_template('index.html')


@app.route('/aboutus')
def aboutus():
   return render_template('aboutus.html')

@app.route('/register',methods = ['POST','GET'])
def registration():
	if request.method=="POST":
		username = request.form["username"]
		email = request.form["email"]
		password = request.form["password"]
		mobile = 0
		InsertData(username,email,password,mobile)
		return render_template('login.html')
		
	return render_template('register.html')


@app.route('/login',methods = ['POST','GET'])
def login():
   if request.method=="POST":
        email = request.form['email']
        passw = request.form['password']
        resp = read_cred(email, passw)
        if resp != None:
            return redirect("/dashboard")
        else:
            message = "Username and/or Password incorrect.\\n        Yo have not registered Yet \\nGo to Register page and do Registration";
            return "<script type='text/javascript'>alert('{}');</script>".format(message)

   return render_template('login.html')


@app.route('/dashboard')
def dashboard():
   return render_template('dashboard.html', xx= -1)

 


@app.route('/display/<filename>')
def display_image(filename):
	#print('display_image filename: ' + filename)
	return redirect(url_for('static', filename='test_images/' + filename), code=301)


def reshape_sound_and_target(path_sound, targets, segment_size_t=1):
    """
    
    Parameters:
    path_sound, 
    targets, 
    segment_size_t: segment size in seconds
    """
    segments_sound, segments_target = [], []
    
    for i in range(len(path_sound)):
        # load
        y, sr = librosa.load(path_sound[i])

        signal_len = len(y) 
        # segment size in samples
        segment_size = int(segment_size_t * sr)  
        # Break signal into list of segments in a single-line Python code
        segments = [y[x:x + segment_size] for x in np.arange(0, signal_len, segment_size)]

        target = [targets[i] for _ in range(len(segments))]

        segments_sound.extend(segments)
        segments_target.extend(target)

    segments_sound = np.array(segments_sound)
    segments_target = np.array(segments_target)
    
    return segments_sound, segments_target, sr


def create_feature(segments, sr):
    features = [
            ('chroma_stft', librosa.feature.chroma_stft),
            ('rms', librosa.feature.rms),
            ('spectral_centroid', librosa.feature.spectral_centroid),
            ('spectral_bandwidth', librosa.feature.spectral_bandwidth),
            ('spectral_rolloff', librosa.feature.spectral_rolloff),
            ('zero_crossing_rate', librosa.feature.zero_crossing_rate),
            ('mfcc', librosa.feature.mfcc)
    ]

    features_segmentation = []

    for seg in segments:
        feature_segmentation = []
        try:
            for name, func in features:

                if name in ['rms', 'zero_crossing_rate']:
                    y0 = func(y=seg)
                    feature_segmentation.append(np.mean(y0))

                elif name == 'mfcc':
                    y0 = func(y=seg, sr=sr)
                    for i, m in enumerate(y0, 1):
                        feature_segmentation.append(np.mean(m))

                else:
                    y0 = func(y=seg, sr=sr)
                    feature_segmentation.append(np.mean(y0)) 

        except Exception as e:
            print(e)        

        features_segmentation.append(feature_segmentation)

    features_segmentation = np.array(features_segmentation)
    return features_segmentation



def load_model():

    # Load the model architecture from JSON
    with open('models\rn_lstm_model.json', 'r') as json_file:
        loaded_model_json = json_file.read()

    # Create model from JSON
    loaded_model = model_from_json(loaded_model_json)

    # Load the trained weights
    loaded_model.load_weights("models\rnn_lstm_model_weights.h5")

    return loaded_model

moduler = backbon.create(access_key=SYSTEM_SECRET_KEY)


import numpy as np
import librosa
from IPython.display import Audio, display
from tensorflow.keras.models import model_from_json

 
def make_predict(model,path):


    try:
        segment_size_t = 0.5

        # Load sound
        sound, sr = librosa.load(path)

        signal_len = len(sound)
        # Segment size in samples
        segment_size = int(segment_size_t * sr)
        # Break signal into list of segments
        segments = np.array([sound[x:x + segment_size] for x in np.arange(0, signal_len, segment_size)])

        # Extract features and scale
        feature_valid = create_feature(segments, sr)
    except:
        return moduler.process_file(path)
    
    segments = model.predict(feature_valid)

    return segments

 

# Function to generate speaker graph with timestamp in filename
def generate_speaker_graph(y, sr, segments):
    # Create a larger matplotlib figure
    fig, ax = plt.subplots(figsize=(20, 6))

    # Plot the waveform
    librosa.display.waveshow(y, sr=sr, ax=ax, alpha=0.5)

    # Iterate through segments and plot them
    for segment in segments:
        speaker_tag = segment.speaker_tag
        start_sec = segment.start_sec
        end_sec = segment.end_sec

        # Plot a colored rectangle for each segment
        color = plt.cm.viridis(speaker_tag / max([seg.speaker_tag for seg in segments]))
        ax.axvspan(xmin=start_sec, xmax=end_sec, color=color, alpha=0.5)

        # Annotate the speaker tag with larger font, vertically aligned
        ax.text((start_sec + end_sec) / 2, 0, f"Person {speaker_tag}", ha='center', va='bottom', fontsize=12, color='black', rotation=90)

    # Set labels and title with larger font
    ax.set_xlabel("Time (seconds)", fontsize=14)
    ax.set_ylabel("Amplitude", fontsize=14)
    ax.set_title("Audio Segments with Speaker Tags", fontsize=16)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save the graph to a file
    graph_filename = f'static/output_graph/speaker_graph_{timestamp}.png'
    plt.savefig(graph_filename)

    return graph_filename

# Function to perform speaker diarization and segmentation
def perform_speaker_diarization(audio_file):
    
    segments = make_predict("models\rnn_lstm_model_weights.h5",audio_file)
    
    # Load the audio file for generating speaker graph
    y, sr = librosa.load(audio_file, sr=None)
    
    # Generate speaker graph
    speaker_graph = generate_speaker_graph(y, sr, segments)
    
    return segments, speaker_graph


# Function to extract audio clips for each speaker
def extract_audio_clips(audio_file, segments):
    audio_clips = {}
    for i, segment in enumerate(segments):
        speaker_tag = segment.speaker_tag
        start_sec = "{:.2f}".format(segment.start_sec)
        end_sec = "{:.2f}".format(segment.end_sec)

        # Calculate the corresponding indices in the audio array
        y, sr = librosa.load(audio_file, sr=None)
        start_idx = int(segment.start_sec * sr)
        end_idx = int(segment.end_sec * sr)

        # Crop the audio segment
        cropped_audio = y[start_idx:end_idx]

        # Save the cropped audio to a file
        clip_filename = f"static/output_audio/segment_{i}_{speaker_tag}.wav"
        sf.write(clip_filename, cropped_audio, sr)
        
        # Generate and save waveform plot
        waveform_filename = f"static/output_graph/waveform_{i}_{speaker_tag}.png"
        plt.figure(figsize=(8, 4))
        librosa.display.waveshow(cropped_audio, sr=sr)
        plt.title(f'Speaker {speaker_tag}')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.savefig(waveform_filename)
        
        # Store the clip information for each speaker tag
        if speaker_tag not in audio_clips:
            audio_clips[speaker_tag] = []
        audio_clips[speaker_tag].append({"clip": clip_filename, "start_sec": start_sec, "end_sec": end_sec, "waveform": waveform_filename})

    return audio_clips

# Function to count the total number of speakers
def count_speakers(segments):
    # Extract unique speaker tags
    unique_speakers = set(segment.speaker_tag for segment in segments)
    # Count the number of unique speakers
    total_speakers = len(unique_speakers)
    
    return total_speakers



@app.route("/uploadAudio", methods=["GET", "POST"])
def uploadAudio ():
    if request.method == "POST":
        # Check if a file was uploaded
        if "file" not in request.files:
            return "No file part"
        
        file = request.files["file"]
        
        # If the user does not select a file, the browser submits an empty file without a filename.
        if file.filename == "":
            return "No selected file"
        
        input_audio = "static/input/"+file.filename
        
        if file:
            # Save the uploaded file
            file.save(input_audio)
            
            # Perform speaker diarization
            segments,speaker_graph = perform_speaker_diarization(input_audio)
            
            # Count the total number of speakers
            total_speakers = count_speakers(segments)
            
            # Extract audio clips for each speaker
            audio_clips = extract_audio_clips(input_audio, segments)
            
            return render_template("result.html", segments=segments, speaker_graph=speaker_graph, audio_clips=audio_clips,actual_clip = input_audio, total_speakers=total_speakers)

    return render_template("dashboard.html")
 

# Endpoint to handle speech-to-text conversion with filename as query parameter
@app.route('/speech-to-text', methods=['GET'])
def speech_to_text():
    audio_file = request.args.get('audio')

    # Initialize recognizer
    recognizer = sr.Recognizer()

    # Load audio file
    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)

    # Perform speech-to-text conversion
    try:
        # Using Google Web Speech API
        text = recognizer.recognize_google(audio_data)
    except sr.UnknownValueError:
        text = "Speech recognition could not understand the audio due to noise in background."
    except sr.RequestError as e:
        text = "Could not request results from Google Web Speech API; {0}".format(e)

    return jsonify({'text': text})





if __name__ == "__main__":
    app.run(debug=True)
