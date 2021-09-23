let video = document.getElementById("video");
let model;
// declare a canvas variable and get its context
let canvas = document.getElementById("canvas");
let ctx = canvas.getContext("2d");

const setupCamera = () => {
  navigator.mediaDevices.getUserMedia({
      video: { width: 600, height: 400 },
      audio: false,
    })
    .then((stream) => {
      video.srcObject = stream;
    });
};

const detectFaces = async () => {
  const prediction = await model.estimateFaces(video, false);

  // console.log(prediction);

  // draw the video first
  ctx.drawImage(video, 0, 0, 600, 400);

  prediction.forEach((pred) => {

        /*
    `prediction` is an array of objects describing each detected face, for example:

    [
      {
        topLeft: [232.28, 145.26],
        bottomRight: [449.75, 308.36],
        probability: [0.998],
        landmarks: [
          [295.13, 177.64], // right eye
          [382.32, 175.56], // left eye
          [341.18, 205.03], // nose
          [345.12, 250.61], // mouth
          [252.76, 211.37], // right ear
          [431.20, 204.93] // left ear
        ]
      }
    ]
    */
    
    // draw the rectangle enclosing the face
    ctx.beginPath();
    ctx.lineWidth = "4";
    ctx.strokeStyle = "blue";
    // the last two arguments are width and height
    // since blazeface returned only the coordinates, 
    // we can find the width and height by subtracting them.
    ctx.rect(
      pred.topLeft[0],
      pred.topLeft[1],
      pred.bottomRight[0] - pred.topLeft[0],
      pred.bottomRight[1] - pred.topLeft[1]
    );
    ctx.stroke();
    
    // drawing small rectangles for the face landmarks
    ctx.fillStyle = "red";
    pred.landmarks.forEach((landmark) => {
      ctx.fillRect(landmark[0], landmark[1], 5, 5);
    });
    
  });
  requestAnimationFrame(detectFaces);
};

setupCamera();
video.addEventListener("loadeddata", async () => {
  model = await blazeface.load();
  // call detect faces every 100 milliseconds or 10 times every second
  // setInterval(detectFaces, 100);
  detectFaces();
});