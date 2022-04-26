//webkitURL is deprecated but nevertheless 

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

function getCookie(cname) {
    var name = cname + "=";
    var ca = document.cookie.split(';');
    for(var i=0; i<ca.length; i++) {
       var c = ca[i];
       while (c.charAt(0)==' ') c = c.substring(1);
       if(c.indexOf(name) == 0)
          return c.substring(name.length,c.length);
    }
    return "";
  }

URL = window.URL || window.webkitURL;
var gumStream;
//stream from getUserMedia() 
var rec;
//Recorder.js object 
var input;
// //MediaStreamAudioSourceNode we'll be recording 
// // shim for AudioContext when it's not avb. 
// var AudioContext = window.AudioContext || window.webkitAudioContext;
// var audioContext = new AudioContext;
//new audio context to help us record 
var recordButton = document.getElementById("recordButton");
var stopButton = document.getElementById("stopButton");
var pauseButton = document.getElementById("pauseButton");
//add events to those 3 buttons 
recordButton.addEventListener("click", startRecording);
stopButton.addEventListener("click", stopRecording);
pauseButton.addEventListener("click", pauseRecording);


function startRecording() { 
    //MediaStreamAudioSourceNode we'll be recording 
// shim for AudioContext when it's not avb. 
    var AudioContext = window.AudioContext || window.webkitAudioContext;
    var audioContext = new AudioContext;
    console.log("recordButton clicked"); 


    /* Simple constraints object, for more advanced audio features see

    https://addpipe.com/blog/audio-constraints-getusermedia/ */

    var constraints = {
        audio: true,
        video: false
    } 
    /* Disable the record button until we get a success or fail from getUserMedia() */

    recordButton.disabled = true;
    stopButton.disabled = false;
    pauseButton.disabled = false

    /* We're using the standard promise based getUserMedia()

    https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia */

    navigator.mediaDevices.getUserMedia(constraints).then(function(stream) {
        console.log("getUserMedia() success, stream created, initializing Recorder.js ..."); 
        /* assign to gumStream for later use */
        gumStream = stream;
        /* use the stream */
        input = audioContext.createMediaStreamSource(stream);
        /* Create the Recorder object and configure to record mono sound (1 channel) Recording 2 channels will double the file size */
        rec = new Recorder(input, {
            numChannels: 2
        }) 
        //start the recording process 
        rec.record(1400)
        console.log("Recording started");
    }).catch(function(err) {
        //enable the record button if getUserMedia() fails 
        recordButton.disabled = false;
        stopButton.disabled = true;
        pauseButton.disabled = true
    });

    sleep(1400).then(() => { stopRecording() });
}

function pauseRecording() {
    console.log("pauseButton clicked rec.recording=", rec.recording);
    if (rec.recording) {
        //pause 
        rec.stop();
        pauseButton.innerHTML = "Resume";
    } else {
        //resume 
        rec.record()
        pauseButton.innerHTML = "Pause";
    }
}

function stopRecording() {
    console.log("stopButton clicked");
    //disable the stop button, enable the record too allow for new recordings 
    stopButton.disabled = true;
    recordButton.disabled = true;
    pauseButton.disabled = true;
    //reset button just in case the recording is stopped while paused 
    pauseButton.innerHTML = "Pause";
    //tell the recorder to stop the recording 
    rec.stop(); //stop microphone access 
    gumStream.getAudioTracks()[0].stop();
    //create the wav blob and pass it on to createDownloadLink 
    rec.exportWAV(createDownloadLink);
}

function createDownloadLink(blob) {
    const url = URL.createObjectURL(blob);
    const au = document.createElement('audio');
    const li = document.createElement('div');
    const divDel = document.createElement('div')
    const divSub = document.createElement('div')
    const deleteButton = document.createElement('button');
    const submitButton = document.createElement('button');
    li.classList.add('row');
    divDel.classList.add('col-md-6');
    divSub.classList.add('col-md-6');
    deleteButton.classList.add('btn');
    deleteButton.classList.add('btn-secondary');
    submitButton.classList.add('btn');
    submitButton.classList.add('btn-info');
    li.classList.add('p-3');
    divDel.classList.add('p-3');
    divSub.classList.add('p-3');
    deleteButton.textContent = 'Delete';
    submitButton.textContent = 'Submit';
    const link = document.createElement('a');
    //add controls to the <audio> element 
    au.controls = true;
    au.src = url;
    //link the a element to the blob 
    link.href = url;
    link.download = new Date().toISOString() + '.wav';
    link.innerHTML = link.download;
    //add the new audio and a elements to the li element 
    li.appendChild(au);
    //li.appendChild(link);
    divDel.appendChild(deleteButton);
    divSub.appendChild(submitButton);
    li.appendChild(divDel)
    li.appendChild(divSub)

    //filename to send to server without extension 
    //upload link 
    let csrftoken = getCookie('csrftoken');
    //var upload = document.createElement('a');
    //upload.href = "#";
    //upload.innerHTML = "Upload";
    var wavFile = new File([ blob ], "audio.wav");  
    var form    = new FormData();
    form.append("myAudio", blob);
    submitButton.addEventListener("click", function(event) {
        document.getElementById("result").textContent = "Loading...";
        $.ajax(
            {
                url: "/process/",
                headers: {'X-CSRFToken':csrftoken},
                type: "POST",
                data: form,
                contentType: false,
                processData: false,
                success: function(getData)
                {
                    console.log(getData);
                }
            }).done(function(response){
                        let temp = response.split(";");
                        let word = temp[0];
                        let prob = temp[1];
                        let result = "The word is "
                        result = result.concat(word, " with probability ", prob.toString(), "%");
                        document.getElementById("result").textContent = result;
                        })
    })
    li.appendChild(document.createTextNode(" ")) //add a space in between 
    //li.appendChild(upload) //add the upload link to li

    //add the li element to the ordered list 
    recordingsList.appendChild(li);

    deleteButton.addEventListener("click", function(event) {
        let evtTgt = event.target;
        evtTgt.parentNode.parentNode.parentNode.removeChild(evtTgt.parentNode.parentNode);
        recordButton.disabled = false;
    })
}