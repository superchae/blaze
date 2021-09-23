const stats = new Stats();
stats.showPanel(0);
document.body.prepend(stats.domElement);

let model, ctx, videoWidth, videoHeight, video, canvas;
const eyesCanvas = document.getElementById('eyes');
const eyesCC = eyesCanvas.getContext('2d');


async function setupCamera() {
	video = document.getElementById('video');
	const stream = await navigator.mediaDevices.getUserMedia({
		'audio': false,
		'video': { facingMode: 'user' },
	});
	video.srcObject = stream;
  
	return new Promise((resolve) => {
		video.onloadedmetadata = () => {
		resolve(video);
		};
	});
}


const renderPrediction = async () => {
	stats.begin();
  
	const returnTensors = false;
	const flipHorizontal = false;
	const annotateBoxes = true;
	const predictions = await model.estimateFaces(
		video, returnTensors, flipHorizontal, annotateBoxes);
  
	if (predictions.length > 0) {
		ctx.clearRect(0, 0, canvas.width, canvas.height);
  
		for (let i = 0; i < predictions.length; i++) {
			if (returnTensors) {
				predictions[i].topLeft = predictions[i].topLeft.arraySync();
				predictions[i].bottomRight = predictions[i].bottomRight.arraySync();
				if (annotateBoxes) {
					predictions[i].landmarks = predictions[i].landmarks.arraySync();
				}
			}
	
			const start = predictions[i].topLeft;
			const end = predictions[i].bottomRight;
			const size = [end[0] - start[0], end[1] - start[1]];
			ctx.fillStyle = "rgba(255, 0, 0, 0.3)";
			ctx.fillRect(start[0], start[1], size[0], size[1]);
	
			const resizeFactorX = video.videoWidth / video.width;
			const resizeFactorY = video.videoHeight / video.height;



			// eyesCC.drawImage(
			// 	video,
			// 	start[0] * resizeFactorX,
			// 	start[1] * resizeFactorY,
			// 	size[0] * resizeFactorX,
			// 	size[1] * resizeFactorY,
			// 	0,
			// 	0,
			// 	eyesCanvas.width,
			// 	eyesCanvas.height,
			// 	);	
			// const middle = video.offsetParent.offsetLeft + video.offsetLeft + video.width/2; 
			// let newEyeX = predictions[i].landmarks[0][0] + 2 * (middle - start[0]) + 290;

			if (annotateBoxes) {
				const landmarks = predictions[i].landmarks;
	
				ctx.fillStyle = "blue";
				for (let j = 0; j < landmarks.length; j++) {
					const x = landmarks[j][0];
					const y = landmarks[j][1];
					ctx.fillRect(x, y, 5, 5);
				}

				eyesCC.drawImage(
					video,
					landmarks[4][0] * resizeFactorX,
					(start[1] + Math.min(landmarks[1][1], landmarks[0][1])) / 2 * resizeFactorY,
					(landmarks[5][0] - landmarks[4][0]) * resizeFactorX,
					size[1] / 3.5 * resizeFactorY,
					0,
					0,
					eyesCanvas.width,
					eyesCanvas.height,
				);	
							
			}
		}
	}
  
	stats.end();
  
	requestAnimationFrame(renderPrediction);
  };

  function getImage() {
    // Capture the current image in the eyes canvas as a tensor.
    return tf.tidy(function() {
    //   const image = tf.browser.fromPixels(eyesCanvas);
	  const image = tf.image.resizeBilinear(tf.browser.fromPixels(eyesCanvas), [25, 50]);
      const batchedImage = image.expandDims(0);
      // Normalize and return it:
      return batchedImage
        .toFloat()
        .div(tf.scalar(127))
        .sub(tf.scalar(1));
    });
  }

  const dataset = {
    train: {
      n: 0,
      x: null,
      y: null,
    },
    val: {
      n: 0,
      x: null,
      y: null,
    },
  };

  var arrow = 0;

  function captureExample(arrow_code) {
    // Take the latest image from the eyes canvas and add it to our dataset.
    if (arrow_code == 'L'){
      console.log("L");
      arrow = -1;
    }
    else if (arrow_code == 'R'){
      arrow = 1;
    }
    tf.tidy(function() {
      const image = getImage();
      // const mousePos = tf.tensor1d([mouse.x, mouse.y]).expandDims(0);
      const arrowPos = tf.tensor(arrow).expandDims(0);
      // Choose whether to add it to training (80%) or validation (20%) set:
      const subset = dataset[Math.random() > 0.2 ? 'train' : 'val'];

      if (subset.x == null) {
        // Create new tensors
        subset.x = tf.keep(image);
        subset.y = tf.keep(arrowPos);
      } else {
        // Concatenate it to existing tensor
        const oldX = subset.x;
        const oldY = subset.y;

        subset.x = tf.keep(oldX.concat(image, 0));
        subset.y = tf.keep(oldY.concat(arrowPos, 0));
      }

      // Increase counter
      subset.n += 1;
    });
  }

  window.addEventListener("keydown", function (event) {
	if (event.defaultPrevented) {
	  return; // Should do nothing if the default action has been cancelled
	}
  	
	if (event.keyCode == 37) {
		captureExample('L');
	} else if (event.keyCode== 39) {
		captureExample('R');
	}
  }, true);


//   $('body').keyup(function(event) {
//     // On ArrowLeft key:
//     if (event.keyCode == 37) {
//       captureExample('L');

//       event.preventDefault();
//       return false;
//     }
    
//     // On ArrowRight key:
//     else if (event.keyCode == 39) {
//       captureExample('R');

//       event.preventDefault();
//       return false;
//     }
//   });

  let currentModel;

  function createModel() {
    const model = tf.sequential();

    model.add(
      tf.layers.conv2d({
        kernelSize: 5,
        filters: 20,
        strides: 1,
        activation: 'relu',
        // inputShape: [$('#eyes').height(), $('#eyes').width(), 3],
		inputShape: [25, 50, 3],
      }),
    );

    model.add(
      tf.layers.maxPooling2d({
        poolSize: [2, 2],
        strides: [2, 2],
      }),
    );

	model.add(
		tf.layers.conv2d({
			kernelSize: 5,
			filters: 16,
			strides: 1,
			activation: 'relu',
			kernelInitializer: 'varianceScaling'
	  	})
	);
	
	model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));

    model.add(tf.layers.flatten());

    model.add(tf.layers.dropout(0.2));

    // Two output values x and y
    model.add(
      tf.layers.dense({
        units: 1,     // 2 -> 1
        activation: 'tanh',
      }),
    );

    // Use ADAM optimizer with learning rate of 0.0005 and MSE loss
    model.compile({
      optimizer: tf.train.adam(0.0005),
      loss: 'meanSquaredError',
    });

    return model;
  }

  function fitModel() {
    let batchSize = Math.floor(dataset.train.n * 0.1);
    if (batchSize < 4) {
      batchSize = 4;
    } else if (batchSize > 64) {
      batchSize = 64;
    }

    if (currentModel == null) {
      currentModel = createModel();
    }

	tfvis.show.modelSummary({name:'Summary', tab:'Model'}, currentModel);

	var _history = [];

    currentModel.fit(dataset.train.x, dataset.train.y, {
      batchSize: batchSize,
      epochs: 20,
      shuffle: true,
      validationData: [dataset.val.x, dataset.val.y],
	  callbacks: {
		onEpochEnd:
			function(epoch, logs){
				console.log('epoch', epoch, logs, 'RMSE : ', Math.sqrt(logs.loss));
				// _history.push(logs);
				// tfvis.show.history({name:'Loss', tab:'History'}, _history, ['loss']);
			}
		},
    });
	console.log('fit');
	
  }

  const train_btn = document.getElementById('train');
  train_btn.addEventListener('click', function(){
	fitModel();
	console.log('Train!');
  });

  function showResult() {
    if (currentModel == null) {
      return;
    }
    tf.tidy(function() {
      const image = getImage();
      const prediction = currentModel.predict(image);

      // It's okay to run this async, since we don't have to wait for it.
      prediction.data().then(prediction => {
        console.log(prediction);
      });
    });
  }

  setInterval(showResult, 700);

  const setupPage = async () => {
	// await tf.setBackend(state.backend);
	await setupCamera();
	video.play();
  
	videoWidth = video.videoWidth;
	videoHeight = video.videoHeight;
	video.width = videoWidth;
	video.height = videoHeight;
  
	canvas = document.getElementById('output');
	canvas.width = videoWidth;
	canvas.height = videoHeight;
	ctx = canvas.getContext('2d');
	ctx.fillStyle = "rgba(255, 0, 0, 0.5)";
  
	model = await blazeface.load();
  
	renderPrediction();
  };
  
  setupPage();