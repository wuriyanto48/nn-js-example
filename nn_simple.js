function linear(xs, ws, b) {
    if (Array.isArray(xs) && Array.isArray(ws)) {
        let s = 0;
        for (let i = 0; i < xs.length; i++) {
            let x = xs[i];
            let w = ws[i];
            s += w * x;
        }
        return s + b;
    }
    return ws * xs + b;
}

function sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
}

function derivativeSigmoid(x) {
    return sigmoid(x) * (1 - sigmoid(x));
}

function loss(outputs, yTrains) {
    let c = 0;
    for (let i = 0; i < outputs.length; i++) {
        let y = yTrains[i];
        let o = outputs[i];
        c += (1/2) * Math.pow(y - o, 2);
    }

    return c;
}

function calculateDerivativeLoss(output, yTrue) {
    return 2 * (1/2) * (yTrue - output) * -1;
}

class NNetwork {
    constructor(epochs, learningRate) {
        this.epochs = epochs;
        this.learningRate = learningRate;
        this.w1 = Math.random();
        this.w2 = Math.random();
        this.w3 = Math.random();
        this.w4 = Math.random();
        this.w5 = Math.random();
        this.w6 = Math.random();
        this.w7 = Math.random();
        this.w8 = Math.random();
        this.w9 = Math.random();
        this.w10 = Math.random();
        this.w11 = Math.random();
        this.w12 = Math.random();
        this.w13 = Math.random();
        this.w14 = Math.random();
        this.w15 = Math.random();
        this.w16 = Math.random();
        this.w17 = Math.random();
        this.w18 = Math.random();
        this.w19 = Math.random();
        this.w20 = Math.random();
        this.w21 = Math.random();
        this.b1 = Math.random();
        this.b2 = Math.random();
        this.b3 = Math.random();
        this.b4 = Math.random();
        this.b5 = Math.random();
        this.b6 = Math.random();
        this.b7 = Math.random();
        this.b8 = Math.random();
    }

    train(xTrains, yTrains) {
        for (let e = 0; e < this.epochs; e++) {
            for (let i = 0; i < yTrains.length; i++) {
                let xTrain = xTrains[i];
                let yTrain = yTrains[i];
                // forward propagation

                // first hidden layer
                let z1 = linear([xTrain[0], xTrain[1]], [this.w1, this.w2], this.b1);
                let h1 = sigmoid(z1);
                let z2 = linear([xTrain[0], xTrain[1]], [this.w3, this.w4], this.b2);
                let h2 = sigmoid(z2);
                let z3 = linear([xTrain[0], xTrain[1]], [this.w5, this.w6], this.b3);
                let h3 = sigmoid(z3);

                // second hidden layer
                let z4 = linear([h1, h2, h3], [this.w7, this.w8, this.w9], this.b4);
                let h4 = sigmoid(z4);
                let z5 = linear([h1, h2, h3], [this.w10, this.w11, this.w12], this.b5);
                let h5 = sigmoid(z5);
                let z6 = linear([h1, h2, h3], [this.w13, this.w14, this.w15], this.b6);
                let h6 = sigmoid(z6);

                // output layer
                let z7 = linear([h4, h5, h6], [this.w16, this.w17, this.w18], this.b7);
                let o1 = sigmoid(z7);
                let z8 = linear([h4, h5, h6], [this.w19, this.w20, this.w21], this.b8);
                let o2 = sigmoid(z8);

                // calculate loss
                let cost = loss([o1, o2], yTrain);
                console.log('epochs: ', e, ' | loss: ', cost);

                // backward propagation
                let derivativeCostO1 = calculateDerivativeLoss(o1, yTrain[0]);
                let derivativeCostO2 = calculateDerivativeLoss(o2, yTrain[1]);

                let updatedW21 =
                    derivativeCostO2 * derivativeSigmoid(z8) * h6;
                this.w21 = this.w21 - this.learningRate * updatedW21;

                let updatedW20 =
                    derivativeCostO2 * derivativeSigmoid(z8) * h5;
                this.w20 = this.w20 - this.learningRate * updatedW20;

                let updatedW19 =
                    derivativeCostO2 * derivativeSigmoid(z8) * h4;
                this.w19 = this.w19 - this.learningRate * updatedW19;

                let updatedW18 =
                    derivativeCostO1 * derivativeSigmoid(z7) * h6;
                this.w18 = this.w18 - this.learningRate * updatedW18;

                let updatedW17 =
                    derivativeCostO1 * derivativeSigmoid(z7) * h5;
                this.w17 += -this.learningRate * updatedW17;

                let updatedW16 =
                    derivativeCostO1 * derivativeSigmoid(z7) * h4;
                this.w16 = this.w16 - this.learningRate * updatedW16;

                let updatedB8 = derivativeCostO2 * derivativeSigmoid(z8) * 1;
                this.b8 = this.b8 - this.learningRate * updatedB8;

                let updatedB7 = derivativeCostO1 * derivativeSigmoid(z7) * 1;
                this.b7 = this.b7 - this.learningRate * updatedB7;

                let updatedW15 =
                    derivativeCostO1 * derivativeSigmoid(z7) * this.w18 * derivativeSigmoid(z6) * h3 +
                        derivativeCostO2 * derivativeSigmoid(z8) * this.w21 * derivativeSigmoid(z6) * h3;
                this.w15 = this.w15 - this.learningRate * updatedW15;

                let updatedW14 =
                    derivativeCostO1 * derivativeSigmoid(z7) * this.w18 * derivativeSigmoid(z6) * h2 +
                        derivativeCostO2 * derivativeSigmoid(z8) * this.w21 * derivativeSigmoid(z6) * h2;
                this.w14 = this.w14 - this.learningRate * updatedW14;

                let updatedW13 =
                    derivativeCostO1 * derivativeSigmoid(z7) * this.w18 * derivativeSigmoid(z6) * h1 + 
                        derivativeCostO2 * derivativeSigmoid(z8) * this.w21 * derivativeSigmoid(z6) * h1;
                this.w13 = this.w13 - this.learningRate * updatedW13;

                let updatedB6 =
                    derivativeCostO1 * derivativeSigmoid(z7) * this.w18 * derivativeSigmoid(z6) * 1 +
                        derivativeCostO2 * derivativeSigmoid(z8) * this.w21 * derivativeSigmoid(z6) * 1;
                this.b6 = this.b6 - this.learningRate * updatedB6;

                let updatedW12 =
                    derivativeCostO1 * derivativeSigmoid(z7) * this.w17 * derivativeSigmoid(z5) * h3 +
                        derivativeCostO2 * derivativeSigmoid(z8) * this.w20 * derivativeSigmoid(z5) * h3;
                this.w12 = this.w12 - this.learningRate * updatedW12;

                let updatedW11 =
                    derivativeCostO1 * derivativeSigmoid(z7) * this.w17 * derivativeSigmoid(z5) * h2 +
                        derivativeCostO2 * derivativeSigmoid(z8) * this.w20 * derivativeSigmoid(z5) * h2;
                this.w11 = this.w11 - this.learningRate * updatedW11;

                let updatedW10 =
                    derivativeCostO1 * derivativeSigmoid(z7) * this.w17 * derivativeSigmoid(z5) * h1 +
                        derivativeCostO2 * derivativeSigmoid(z8) * this.w20 * derivativeSigmoid(z5) * h1;
                this.w10 = this.w10 - this.learningRate * updatedW10;

                let updatedB5 =
                    derivativeCostO1 * derivativeSigmoid(z7) * this.w17 * derivativeSigmoid(z5) * 1 +
                        derivativeCostO2 * derivativeSigmoid(z8) * this.w20 * derivativeSigmoid(z5) * 1;
                this.b5 = this.b5 - this.learningRate * updatedB5;

                let updatedW9 = derivativeCostO1 * derivativeSigmoid(z7) * this.w16 * derivativeSigmoid(z4) * h3 +
                        derivativeCostO2 * derivativeSigmoid(z8) * this.w19 * derivativeSigmoid(z4) * h3;
                this.w9 = this.w9 - this.learningRate * updatedW9;

                let updatedW8 =
                    derivativeCostO1 * derivativeSigmoid(z7) * this.w16 * derivativeSigmoid(z4) * h2 +
                        derivativeCostO2 * derivativeSigmoid(z8) * this.w19 * derivativeSigmoid(z4) * h2;
                this.w8 = this.w8 - this.learningRate * updatedW8;

                let updatedW7 =
                    derivativeCostO1 * derivativeSigmoid(z7) * this.w16 * derivativeSigmoid(z4) * h1 +
                        derivativeCostO2 * derivativeSigmoid(z8) * this.w19 * derivativeSigmoid(z4) * h1;
                this.w7 = this.w7 - this.learningRate * updatedW7;

                let updatedB4 =
                    derivativeCostO1 * derivativeSigmoid(z7) * this.w16 * derivativeSigmoid(z4) * 1 +
                        derivativeCostO2 * derivativeSigmoid(z8) * this.w19 * derivativeSigmoid(z4) * 1;
                this.b4 = this.b4 - this.learningRate * updatedB4;


                let updatedW6 =
                    derivativeCostO1 *
                        derivativeSigmoid(z7) *
                        this.w16 *
                        derivativeSigmoid(z4) *
                        this.w9 * derivativeSigmoid(z3) * xTrain[1] +

                        derivativeCostO1 *
                        derivativeSigmoid(z7) *
                        this.w17 *
                        derivativeSigmoid(z5) *
                        this.w12 * derivativeSigmoid(z3) * xTrain[1] +

                        derivativeCostO1 *
                        derivativeSigmoid(z7) *
                        this.w18 *
                        derivativeSigmoid(z6) *
                        this.w15 * derivativeSigmoid(z3) * xTrain[1] +

                        derivativeCostO2 *
                        derivativeSigmoid(z8) *
                        this.w19 *
                        derivativeSigmoid(z4) *
                        this.w9 * derivativeSigmoid(z3) * xTrain[1] +

                        derivativeCostO2 *
                        derivativeSigmoid(z8) *
                        this.w20 *
                        derivativeSigmoid(z5) *
                        this.w12 * derivativeSigmoid(z3) * xTrain[1] +

                        derivativeCostO2 *
                        derivativeSigmoid(z8) *
                        this.w21 *
                        derivativeSigmoid(z6) *
                        this.w15 * derivativeSigmoid(z3) * xTrain[1];
                this.w6 = this.w6 - this.learningRate * updatedW6;

                let updatedW5 =
                    derivativeCostO1 *
                        derivativeSigmoid(z7) *
                        this.w16 *
                        derivativeSigmoid(z4) *
                        this.w9 * derivativeSigmoid(z3) * xTrain[0] +

                        derivativeCostO1 *
                        derivativeSigmoid(z7) *
                        this.w17 *
                        derivativeSigmoid(z5) *
                        this.w12 * derivativeSigmoid(z3) * xTrain[0] +

                        derivativeCostO1 *
                        derivativeSigmoid(z7) *
                        this.w18 *
                        derivativeSigmoid(z6) *
                        this.w15 * derivativeSigmoid(z3) * xTrain[0] +
                        
                        derivativeCostO2 *
                        derivativeSigmoid(z8) *
                        this.w19 *
                        derivativeSigmoid(z4) *
                        this.w9 * derivativeSigmoid(z3) * xTrain[0] +

                        derivativeCostO2 *
                        derivativeSigmoid(z8) *
                        this.w20 *
                        derivativeSigmoid(z5) *
                        this.w12 * derivativeSigmoid(z3) * xTrain[0] +

                        derivativeCostO2 *
                        derivativeSigmoid(z8) *
                        this.w21 *
                        derivativeSigmoid(z6) *
                        this.w15 * derivativeSigmoid(z3) * xTrain[0];
                this.w5 = this.w5 - this.learningRate * updatedW5;

                let updatedB3 =
                    derivativeCostO1 *
                        derivativeSigmoid(z7) *
                        this.w16 *
                        derivativeSigmoid(z4) *
                        this.w9 * derivativeSigmoid(z3) * 1 +

                        derivativeCostO1 *
                        derivativeSigmoid(z7) *
                        this.w17 *
                        derivativeSigmoid(z5) *
                        this.w12 * derivativeSigmoid(z3) * 1 +

                        derivativeCostO1 *
                        derivativeSigmoid(z7) *
                        this.w18 *
                        derivativeSigmoid(z6) *
                        this.w15 * derivativeSigmoid(z3) * 1 +

                        derivativeCostO2 *
                        derivativeSigmoid(z8) *
                        this.w19 *
                        derivativeSigmoid(z4) *
                        this.w9 * derivativeSigmoid(z3) * 1 +

                        derivativeCostO2 *
                        derivativeSigmoid(z8) *
                        this.w20 *
                        derivativeSigmoid(z5) *
                        this.w12 * derivativeSigmoid(z3) * 1 +

                        derivativeCostO2 *
                        derivativeSigmoid(z8) *
                        this.w21 *
                        derivativeSigmoid(z6) *
                        this.w15 * derivativeSigmoid(z3) * 1;
                this.b3 = this.b3 - this.learningRate * updatedB3;

                let updatedW4 =
                    derivativeCostO1 *
                        derivativeSigmoid(z7) *
                        this.w16 *
                        derivativeSigmoid(z4) *
                        this.w8 * derivativeSigmoid(z2) * xTrain[1] +

                        derivativeCostO1 *
                        derivativeSigmoid(z7) *
                        this.w17 *
                        derivativeSigmoid(z5) *
                        this.w11 * derivativeSigmoid(z2) * xTrain[1] +

                        derivativeCostO1 *
                        derivativeSigmoid(z7) *
                        this.w18 *
                        derivativeSigmoid(z6) *
                        this.w14 * derivativeSigmoid(z2) * xTrain[1] +

                        derivativeCostO2 *
                        derivativeSigmoid(z8) *
                        this.w19 *
                        derivativeSigmoid(z4) *
                        this.w8 * derivativeSigmoid(z2) * xTrain[1] +

                        derivativeCostO2 *
                        derivativeSigmoid(z8) *
                        this.w20 *
                        derivativeSigmoid(z5) *
                        this.w11 * derivativeSigmoid(z2) * xTrain[1] +

                        derivativeCostO2 *
                        derivativeSigmoid(z8) *
                        this.w21 *
                        derivativeSigmoid(z6) *
                        this.w14 * derivativeSigmoid(z2) * xTrain[1];
                this.w4 = this.w4 - this.learningRate * updatedW4;

                let updatedW3 =
                    derivativeCostO1 *
                        derivativeSigmoid(z7) *
                        this.w16 *
                        derivativeSigmoid(z4) *
                        this.w8 * derivativeSigmoid(z2) * xTrain[0] +

                        derivativeCostO1 *
                        derivativeSigmoid(z7) *
                        this.w17 *
                        derivativeSigmoid(z5) *
                        this.w11 * derivativeSigmoid(z2) * xTrain[0] +

                        derivativeCostO1 *
                        derivativeSigmoid(z7) *
                        this.w18 *
                        derivativeSigmoid(z6) *
                        this.w14 * derivativeSigmoid(z2) * xTrain[0] +

                        derivativeCostO2 *
                        derivativeSigmoid(z8) *
                        this.w19 *
                        derivativeSigmoid(z4) *
                        this.w8 * derivativeSigmoid(z2) * xTrain[0] +

                        derivativeCostO2 *
                        derivativeSigmoid(z8) *
                        this.w20 *
                        derivativeSigmoid(z5) *
                        this.w11 * derivativeSigmoid(z2) * xTrain[0] +

                        derivativeCostO2 *
                        derivativeSigmoid(z8) *
                        this.w21 *
                        derivativeSigmoid(z6) *
                        this.w14 * derivativeSigmoid(z2) * xTrain[0];
                this.w3 = this.w3 - this.learningRate * updatedW3;

                let updatedB2 =
                    derivativeCostO1 *
                        derivativeSigmoid(z7) *
                        this.w16 *
                        derivativeSigmoid(z4) *
                        this.w8 * derivativeSigmoid(z2) * 1 +

                        derivativeCostO1 *
                        derivativeSigmoid(z7) *
                        this.w17 *
                        derivativeSigmoid(z5) *
                        this.w11 * derivativeSigmoid(z2) * 1 +

                        derivativeCostO1 *
                        derivativeSigmoid(z7) *
                        this.w18 *
                        derivativeSigmoid(z6) *
                        this.w14 * derivativeSigmoid(z2) * 1 +

                        derivativeCostO2 *
                        derivativeSigmoid(z8) *
                        this.w19 *
                        derivativeSigmoid(z4) *
                        this.w8 * derivativeSigmoid(z2) * 1 +

                        derivativeCostO2 *
                        derivativeSigmoid(z8) *
                        this.w20 *
                        derivativeSigmoid(z5) *
                        this.w11 * derivativeSigmoid(z2) * 1 +

                        derivativeCostO2 *
                        derivativeSigmoid(z8) *
                        this.w21 *
                        derivativeSigmoid(z6) *
                        this.w14 * derivativeSigmoid(z2) * 1;
                this.b2 = this.b2 - this.learningRate * updatedB2;

                let updatedW2 =
                    derivativeCostO1 *
                        derivativeSigmoid(z7) *
                        this.w16 *
                        derivativeSigmoid(z4) *
                        this.w7 * derivativeSigmoid(z1) * xTrain[1] +

                        derivativeCostO1 *
                        derivativeSigmoid(z7) *
                        this.w17 *
                        derivativeSigmoid(z5) *
                        this.w10 * derivativeSigmoid(z1) * xTrain[1] +

                        derivativeCostO1 *
                        derivativeSigmoid(z7) *
                        this.w18 *
                        derivativeSigmoid(z6) *
                        this.w13 * derivativeSigmoid(z1) * xTrain[1] +

                        derivativeCostO2 *
                        derivativeSigmoid(z8) *
                        this.w19 *
                        derivativeSigmoid(z4) *
                        this.w7 * derivativeSigmoid(z1) * xTrain[1] +

                        derivativeCostO2 *
                        derivativeSigmoid(z8) *
                        this.w20 *
                        derivativeSigmoid(z5) *
                        this.w10 * derivativeSigmoid(z1) * xTrain[1] +

                        derivativeCostO2 *
                        derivativeSigmoid(z8) *
                        this.w21 *
                        derivativeSigmoid(z6) *
                        this.w13 * derivativeSigmoid(z1) * xTrain[1];
                this.w2 = this.w2 - this.learningRate * updatedW2;

                let updatedW1 =
                    derivativeCostO1 *
                        derivativeSigmoid(z7) *
                        this.w16 *
                        derivativeSigmoid(z4) *
                        this.w7 * derivativeSigmoid(z1) * xTrain[0] +

                        derivativeCostO1 *
                        derivativeSigmoid(z7) *
                        this.w17 *
                        derivativeSigmoid(z5) *
                        this.w10 * derivativeSigmoid(z1) * xTrain[0] +

                        derivativeCostO1 *
                        derivativeSigmoid(z7) *
                        this.w18 *
                        derivativeSigmoid(z6) *
                        this.w13 * derivativeSigmoid(z1) * xTrain[0] +

                        derivativeCostO2 *
                        derivativeSigmoid(z8) *
                        this.w19 *
                        derivativeSigmoid(z4) *
                        this.w7 * derivativeSigmoid(z1) * xTrain[0] +

                        derivativeCostO2 *
                        derivativeSigmoid(z8) *
                        this.w20 *
                        derivativeSigmoid(z5) *
                        this.w10 * derivativeSigmoid(z1) * xTrain[0] +

                        derivativeCostO2 *
                        derivativeSigmoid(z8) *
                        this.w21 *
                        derivativeSigmoid(z6) *
                        this.w13 * derivativeSigmoid(z1) * xTrain[0];
                this.w1 = this.w1 - this.learningRate * updatedW1;
                
                let updatedB1 =
                    derivativeCostO1 *
                        derivativeSigmoid(z7) *
                        this.w16 *
                        derivativeSigmoid(z4) *
                        this.w7 * derivativeSigmoid(z1) * 1 +

                        derivativeCostO1 *
                        derivativeSigmoid(z7) *
                        this.w17 *
                        derivativeSigmoid(z5) *
                        this.w10 * derivativeSigmoid(z1) * 1 +

                        derivativeCostO1 *
                        derivativeSigmoid(z7) *
                        this.w18 *
                        derivativeSigmoid(z6) *
                        this.w13 * derivativeSigmoid(z1) * 1 +

                        derivativeCostO2 *
                        derivativeSigmoid(z8) *
                        this.w19 *
                        derivativeSigmoid(z4) *
                        this.w7 * derivativeSigmoid(z1) * 1 +

                        derivativeCostO2 *
                        derivativeSigmoid(z8) *
                        this.w20 *
                        derivativeSigmoid(z5) *
                        this.w10 * derivativeSigmoid(z1) * 1 +

                        derivativeCostO2 *
                        derivativeSigmoid(z8) *
                        this.w21 *
                        derivativeSigmoid(z6) *
                        this.w13 * derivativeSigmoid(z1) * 1;
                this.b1 = this.b1 - this.learningRate * updatedB1;
            }
        }
    }

    forward(x) {
        // input layer
        let z1 = linear([x[0], x[1]], [this.w1, this.w2], this.b1);
        let h1 = sigmoid(z1);
        let z2 = linear([x[0], x[1]], [this.w3, this.w4], this.b2);
        let h2 = sigmoid(z2);
        let z3 = linear([x[0], x[1]], [this.w5, this.w6], this.b3);
        let h3 = sigmoid(z3);
        // second hidden layer
        let z4 = linear([h1, h2, h3], [this.w7, this.w8, this.w9], this.b4);
        let h4 = sigmoid(z4);
        let z5 = linear([h1, h2, h3], [this.w10, this.w11, this.w12], this.b5);
        let h5 = sigmoid(z5);
        let z6 = linear([h1, h2, h3], [this.w13, this.w14, this.w15], this.b6);
        let h6 = sigmoid(z6);
        // output layer
        let z7 = linear([h4, h5, h6], [this.w16, this.w17, this.w18], this.b7);
        let o1 = sigmoid(z7);
        let z8 = linear([h4, h5, h6], [this.w19, this.w20, this.w21], this.b8);
        let o2 = sigmoid(z8);
        return [o1, o2];
    }
}

function main() {
    const xTrains = [
        [-2, -1], // tidak masuk
        [25, 6], // masuk
        [17, 9], // masuk
        [36, 8], // masuk
        [-15, -6], // tidak masuk
        [10, 15], // masuk
        [1, 1], // tidak masuk
        [-66, -9], // tidak masuk,
        [1, -1], // tidak masuk
        [1, 2], // tidak masuk
        [10, 7], // masuk
        [-10, -5], // tidak masuk,
        [18, 9], // masuk
    ];

    const yTrains = [
        [0,1],
        [1,0],
        [1,0],
        [1,0],
        [0,1],
        [1,0],
        [0,1],
        [0,1],
        [0,1],
        [0,1],
        [1,0],
        [0,1],
        [1,0],
    ];

    const n = new NNetwork(1000, 0.01);
    n.train(xTrains, yTrains);
    console.log(n.forward([112, 22]));
}

// execute train
main();