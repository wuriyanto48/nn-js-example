var buttonProcess = document.getElementById("buttonProcess");
var canvas = document.getElementById("myCanvas");
var ctx = canvas.getContext("2d");
ctx.fillStyle = "black";
ctx.fillRect(0, 0, canvas.width, canvas.height);

var img = document.getElementById('imageResized');

var drag = false;
var colorDrawDefault = "white"

// Initialize the temp canvas and it's size
var tempCanvas = null;
var ctxTemp = null

window.addEventListener('load', function() {
    tempCanvas = document.createElement("canvas");
    tempCanvas.id = "tempCanvas";
    // Set width and height
    tempCanvas.width = 28;
    tempCanvas.height = 28;

    ctxTemp = tempCanvas.getContext("2d");
    
});

function drawRect(dx, dy) {
    var bounding = canvas.getBoundingClientRect();
    ctx.fillStyle = colorDrawDefault;
    ctx.fillRect(dx-bounding.left, dy-bounding.top, 4, 4);
}

function drawCircle(dx, dy) {
    var bounding = canvas.getBoundingClientRect();
    ctx.fillStyle = colorDrawDefault;
    ctx.beginPath();
    ctx.arc(dx-bounding.left, dy-bounding.top, 20, 0, 2 * Math.PI);
    ctx.fill();
}

canvas.addEventListener('mousedown', function(event) {
    drag = true;
});

canvas.addEventListener('mouseup', function(event) {
    drag = false;
});

canvas.addEventListener('mousemove', function(event) {
    var x = event.clientX;
    var y = event.clientY;

    if (drag) {
        drawCircle(x, y);
    }
});

function resize() {
    ctxTemp.drawImage(canvas, 0, 0, tempCanvas.width, tempCanvas.height)
    const dataURI = tempCanvas.toDataURL();
    console.log(dataURI);

    // Do something with the result, like overwrite original
    img.src = dataURI;

    var imageData = ctxTemp.getImageData(0, 0, tempCanvas.width, tempCanvas.height);
    var pixelData = imageData.data;
    console.log(imageData)

    var color = new Float32Array(tempCanvas.width * tempCanvas.height);

    var j = 0;
    for(let i = 0; i < pixelData.length; i = i + 4) {
        color[j] = Math.max(pixelData[i], pixelData[i+1], pixelData[i+2])/255.0;
        j++;
    }

    console.log(color.toString());
}

buttonProcess.addEventListener('click', function(event) {
    resize();
});
