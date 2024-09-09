let mediaRecorder;
let audioChunks = [];

function startRecording() {
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        navigator.mediaDevices.getUserMedia({ audio: true })
            .then(function (stream) {
                mediaRecorder = new MediaRecorder(stream);

                mediaRecorder.ondataavailable = function (event) {
                    audioChunks.push(event.data);
                };

                mediaRecorder.onstop = function () {
                    const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                    audioChunks = [];

                    const formData = new FormData();
                    formData.append('audio_data', audioBlob, 'recording.wav');

                    fetch('/record', {
                        method: 'POST',
                        body: formData
                    })
                    .then(response => response.text())
                    .then(data => {
                        window.location.href = `/results?file_path=uploads/recording.wav`;
                    })
                    .catch((error) => {
                        console.error('Error:', error);
                    });
                };

                mediaRecorder.start();
                console.log("Recording started...");
            })
            .catch(function (err) {
                console.error('The following error occurred: ' + err);
            });
    } else {
        alert("Your browser does not support audio recording.");
    }
}

function stopRecording() {
    if (mediaRecorder && mediaRecorder.state === "recording") {
        mediaRecorder.stop();
        console.log("Recording stopped.");
    }
}

function stopRecording() {
    if (mediaRecorder && mediaRecorder.state === "recording") {
        mediaRecorder.stop();
        console.log("Recording stopped.");
    }
}

// Clear any previous data after stopping recording or uploading
function resetState() {
    mediaRecorder = null;
    audioChunks = [];
}

