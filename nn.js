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

function calculateCost(outputs, yTrue) {
    let c = 0;
    for (let i = 0; i < outputs.length; i++) {
        let o = outputs[i];
        c += Math.pow(o - yTrue, 2);
    }
    return (1 / outputs.length) * c;
}

function calculateDerivativeCost(outputs, yTrue) {
    let c = 0;
    for (let i = 0; i < outputs.length; i++) {
        let o = outputs[i];
        c += 2 * (o - yTrue);
    }
    return (1 / outputs.length) * c;
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
        this.b1 = 0.0;
        this.b2 = 0.0;
        this.b3 = 0.0;
        this.b4 = 0.0;
        this.b5 = 0.0;
        this.b6 = 0.0;
        this.b7 = 0.0;
        this.b8 = 0.0;
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

                // backward propagation
                let derivativeCostO1 = o1 - yTrain[0];
                let derivativeCostO2 = o2 - yTrain[1];

                let updatedW21 =
                    derivativeCostO2 * derivativeSigmoid(z8) * h6;
                this.w21 += -this.learningRate * updatedW21;

                let updatedW20 =
                    derivativeCostO2 * derivativeSigmoid(z8) * h5;
                this.w20 += -this.learningRate * updatedW20;

                let updatedW19 =
                    derivativeCostO2 * derivativeSigmoid(z8) * h4;
                this.w19 += -this.learningRate * updatedW19;

                let updatedW18 =
                    derivativeCostO1 * derivativeSigmoid(z7) * h6;
                this.w18 += -this.learningRate * updatedW18;

                let updatedW17 =
                    derivativeCostO1 * derivativeSigmoid(z7) * h5;
                this.w17 += -this.learningRate * updatedW17;

                let updatedW16 =
                    derivativeCostO1 * derivativeSigmoid(z7) * h4;
                this.w16 += -this.learningRate * updatedW16;

                let updatedB8 = derivativeCostO2 * derivativeSigmoid(z8) * 1;
                this.b8 += -this.learningRate * updatedB8;

                let updatedB7 = derivativeCostO1 * derivativeSigmoid(z7) * 1;
                this.b7 += -this.learningRate * updatedB7;

                let updatedW15 =
                    derivativeCostO1 * derivativeSigmoid(z7) * this.w18 * derivativeSigmoid(z6) * h3 +
                        derivativeCostO2 * derivativeSigmoid(z8) * this.w21 * derivativeSigmoid(z6) * h3;
                this.w15 += -this.learningRate * updatedW15;

                let updatedW14 =
                    derivativeCostO1 * derivativeSigmoid(z7) * this.w18 * derivativeSigmoid(z6) * h2 +
                        derivativeCostO2 * derivativeSigmoid(z8) * this.w21 * derivativeSigmoid(z6) * h2;
                this.w14 += -this.learningRate * updatedW14;

                let updatedW13 =
                    derivativeCostO1 * derivativeSigmoid(z7) * this.w18 * derivativeSigmoid(z6) * h1 + 
                        derivativeCostO2 * derivativeSigmoid(z8) * this.w21 * derivativeSigmoid(z6) * h1;
                this.w13 += -this.learningRate * updatedW13;

                let updatedWB6 =
                    derivativeCostO1 * derivativeSigmoid(z7) * this.w18 * derivativeSigmoid(z6) * 1 +
                        derivativeCostO2 * derivativeSigmoid(z8) * this.w21 * derivativeSigmoid(z6) * 1;
                this.b6 += -this.learningRate * updatedWB6;

                let updatedW12 =
                    derivativeCostO1 * derivativeSigmoid(z7) * this.w17 * derivativeSigmoid(z5) * h3 +
                        derivativeCostO2 * derivativeSigmoid(z8) * this.w20 * derivativeSigmoid(z5) * h3;
                this.w12 += -this.learningRate * updatedW12;

                let updatedW11 =
                    derivativeCostO1 * derivativeSigmoid(z7) * this.w17 * derivativeSigmoid(z5) * h2 +
                        derivativeCostO2 * derivativeSigmoid(z8) * this.w20 * derivativeSigmoid(z5) * h2;
                this.w11 += -this.learningRate * updatedW11;

                let updatedW10 =
                    derivativeCostO1 * derivativeSigmoid(z7) * this.w17 * derivativeSigmoid(z5) * h1 +
                        derivativeCostO2 * derivativeSigmoid(z8) * this.w20 * derivativeSigmoid(z5) * h1;
                this.w10 += -this.learningRate * updatedW10;

                let updatedWB5 =
                    derivativeCostO1 * derivativeSigmoid(z7) * this.w17 * derivativeSigmoid(z5) * 1 +
                        derivativeCostO2 * derivativeSigmoid(z8) * this.w20 * derivativeSigmoid(z5) * 1;
                this.b5 += -this.learningRate * updatedWB5;

                let updatedW9 = derivativeCostO1 * derivativeSigmoid(z7) * this.w16 * derivativeSigmoid(z4) * h3 +
                        derivativeCostO2 * derivativeSigmoid(z8) * this.w19 * derivativeSigmoid(z4) * h3;
                this.w9 += -this.learningRate * updatedW9;

                let updatedW8 =
                    derivativeCostO1 * derivativeSigmoid(z7) * this.w16 * derivativeSigmoid(z4) * h2 +
                        derivativeCostO2 * derivativeSigmoid(z8) * this.w19 * derivativeSigmoid(z4) * h2;
                this.w8 += -this.learningRate * updatedW8;

                let updatedW7 =
                    derivativeCostO1 * derivativeSigmoid(z7) * this.w16 * derivativeSigmoid(z4) * h1 +
                        derivativeCostO2 * derivativeSigmoid(z8) * this.w19 * derivativeSigmoid(z4) * h1;
                this.w7 += -this.learningRate * updatedW7;

                let updatedWB4 =
                    derivativeCostO1 * derivativeSigmoid(z7) * this.w16 * derivativeSigmoid(z4) * 1 +
                        derivativeCostO2 * derivativeSigmoid(z8) * this.w19 * derivativeSigmoid(z4) * 1;
                this.b4 += -this.learningRate * updatedWB4;


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
                this.w6 += -this.learningRate * updatedW6;

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
                this.w5 += -this.learningRate * updatedW5;

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
                this.b3 += -this.learningRate * updatedB3;

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
                this.w4 += -this.learningRate * updatedW4;

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
                this.w3 += -this.learningRate * updatedW3;

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
                this.b2 += -this.learningRate * updatedB2;

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
                this.w2 += -this.learningRate * updatedW2;

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
                this.w1 += -this.learningRate * updatedW1;
                
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
                this.b1 += -this.learningRate * updatedB1;
            }
        }
    }

    forward(x) {
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
        [-2, -1], // Alice
        [25, 6], // Bob
        [17, 4], // Charlie
        [36, 8], // Alex
        [-15, -6], // Diana
        [10, 4], // George
        [1, 1], // Dina
        [-66, -9], // Hellen,
        [1, 1], // ina
        [1, 2], // any
        [10, 7], // dody
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
        [1,0]
    ];

    // const xTrains = [
    //     [2.496714153011233,2.324083969394795],
    //     [1.8617356988288154,1.6149177195836835],
    //     [2.6476885381006925,1.3230779996940414],
    //     [3.5230298564080256,2.6116762888408678],
    //     [1.765846625276664,3.030999522495951],
    //     [1.7658630430508195,2.931280119116199],
    //     [3.5792128155073915,1.1607824767773613],
    //     [2.767434729152909,1.6907876241487854],
    //     [1.5305256140650478,2.331263431403564],
    //     [2.5425600435859645,2.975545127122359],
    //     [1.5365823071875377,1.52082576215471],
    //     [1.534270246429743,1.814341023336183],
    //     [2.241962271566034,0.8936650259939718],
    //     [0.08671975534220211,0.8037933759193292],
    //     [0.27508216748696723,2.812525822394198],
    //     [1.4377124707590272,3.3562400285708227],
    //     [0.9871688796655762,1.927989878419666],
    //     [2.314247332595274,3.003532897892024],
    //     [1.0919759244787892,2.361636025047634],
    //     [0.5876962986647085,1.3548802453948756],
    //     [3.465648768921554,2.361395605508414],
    //     [1.7742236995134644,3.538036566465969],
    //     [2.0675282046879238,1.9641739608900484],
    //     [0.5752518137865432,3.564643655814006],
    //     [1.4556172754748173,-0.6197451040897444],
    //     [2.1109225897098662,2.821902504375224],
    //     [0.8490064225776972,2.087047068238171],
    //     [2.375698018345672,1.7009926495341325],
    //     [1.399361310081195,2.0917607765355024],
    //     [1.7083062502067232,0.0124310853991072],
    //     [1.398293387770603,1.780328112162488],
    //     [3.852278184508938,2.3571125715117462],
    //     [1.986502775262066,3.4778940447415163],
    //     [0.9422890710440996,1.4817297817263526],
    //     [2.822544912103189,1.1915063971068123],
    //     [0.7791563500289778,1.4982429564154636],
    //     [2.2088635950047553,2.915402117702074],
    //     [0.04032987612022443,2.3287511096596845],
    //     [0.6718139511015695,1.4702397962329612],
    //     [2.1968612358691235,2.513267433113356],
    //     [2.7384665799954107,2.0970775493480405],
    //     [2.1713682811899706,2.9686449905328893],
    //     [1.8843517176117595,1.2979469061226476],
    //     [1.6988963044107113,1.6723378534022317],
    //     [0.5214780096325726,1.6078918468678425],
    //     [1.2801557916052912,0.5364850518678814],
    //     [1.5393612290402126,2.296120277064576],
    //     [3.0571222262189157,2.261055272179889],
    //     [2.3436182895684614,2.005113456642461],
    //     [0.23695984463726605,1.7654128666248532],
    //     [5.584629257949586,7.250492850345877],
    //     [6.579354677234641,7.346448209496976],
    //     [6.65728548347323,6.319975278421509],
    //     [6.197722730778381,7.232253697161004],
    //     [6.838714288333991,7.293072473298682],
    //     [7.404050856814538,6.285648581973632],
    //     [8.88618590121053,8.865774511144757],
    //     [7.174577812831839,7.4738329209117875],
    //     [7.2575503907227645,5.808696502797352],
    //     [6.925554084233832,7.65655360863383],
    //     [5.081228784700959,6.025318329772679],
    //     [6.973486124550783,7.787084603742452],
    //     [7.060230209941026,8.158595579007404],
    //     [9.463242112485286,6.179317681648289],
    //     [6.807639035218878,7.963376129244322],
    //     [7.301547342333612,7.412780926936498],
    //     [6.965288230294757,7.82206015999449],
    //     [5.831321962380468,8.896792982653947],
    //     [8.142822814515021,6.754611883997129],
    //     [7.751933032686774,6.246263835642511],
    //     [7.791031947043047,6.110485570374477],
    //     [6.090612545205261,6.184189715034561],
    //     [8.402794310936098,6.922898290585896],
    //     [5.5981489372077196,7.341151974816644],
    //     [7.58685709380027,7.276690799330019],
    //     [9.190455625809978,7.827183249036024],
    //     [6.009463674869312,7.013001891877907],
    //     [6.433702270397228,8.453534077157316],
    //     [7.099651365087642,6.735343166762044],
    //     [6.496524345883801,9.720169166589619],
    //     [5.449336568933868,7.625667347765006],
    //     [7.068562974806027,6.142842443583717],
    //     [5.937696286273895,5.929107501938888],
    //     [7.473592430635182,7.482472415243185],
    //     [6.080575765766197,6.776537214674149],
    //     [8.54993440501754,7.714000494092092],
    //     [6.216746707663763,7.473237624573545],
    //     [6.677938483794325,6.927171087343127],
    //     [7.81351721736967,6.153206281931595],
    //     [5.769135683566045,5.485152775314136],
    //     [7.22745993460413,6.553485047932979],
    //     [8.307142754282427,7.856398794323472],
    //     [5.3925167654387725,7.214093744130204],
    //     [7.184633858532305,5.7542612212880115],
    //     [7.259882794248424,7.173180925851182],
    //     [7.78182287177731,7.3853173797288365],
    //     [5.763049289121918,6.116142563798867],
    //     [5.679543386915723,7.153725105945528],
    //     [7.521941565616897,7.058208718446],
    //     [7.296984673233186,5.857029702169377],
    // ];

    // const yTrains = [
    //     [1, 0],
    //     [1, 0],
    //     [1, 0],
    //     [1, 0],
    //     [1, 0],
    //     [1, 0],
    //     [1, 0],
    //     [1, 0],
    //     [1, 0],
    //     [1, 0],
    //     [1, 0],
    //     [1, 0],
    //     [1, 0],
    //     [1, 0],
    //     [1, 0],
    //     [1, 0],
    //     [1, 0],
    //     [1, 0],
    //     [1, 0],
    //     [1, 0],
    //     [1, 0],
    //     [1, 0],
    //     [1, 0],
    //     [1, 0],
    //     [1, 0],
    //     [1, 0],
    //     [1, 0],
    //     [1, 0],
    //     [1, 0],
    //     [1, 0],
    //     [1, 0],
    //     [1, 0],
    //     [1, 0],
    //     [1, 0],
    //     [1, 0],
    //     [1, 0],
    //     [1, 0],
    //     [1, 0],
    //     [1, 0],
    //     [1, 0],
    //     [1, 0],
    //     [1, 0],
    //     [1, 0],
    //     [1, 0],
    //     [1, 0],
    //     [1, 0],
    //     [1, 0],
    //     [1, 0],
    //     [1, 0],
    //     [1, 0],
    //     [0, 1],
    //     [0, 1],
    //     [0, 1],
    //     [0, 1],
    //     [0, 1],
    //     [0, 1],
    //     [0, 1],
    //     [0, 1],
    //     [0, 1],
    //     [0, 1],
    //     [0, 1],
    //     [0, 1],
    //     [0, 1],
    //     [0, 1],
    //     [0, 1],
    //     [0, 1],
    //     [0, 1],
    //     [0, 1],
    //     [0, 1],
    //     [0, 1],
    //     [0, 1],
    //     [0, 1],
    //     [0, 1],
    //     [0, 1],
    //     [0, 1],
    //     [0, 1],
    //     [0, 1],
    //     [0, 1],
    //     [0, 1],
    //     [0, 1],
    //     [0, 1],
    //     [0, 1],
    //     [0, 1],
    //     [0, 1],
    //     [0, 1],
    //     [0, 1],
    //     [0, 1],
    //     [0, 1],
    //     [0, 1],
    //     [0, 1],
    //     [0, 1],
    //     [0, 1],
    //     [0, 1],
    //     [0, 1],
    //     [0, 1],
    //     [0, 1],
    //     [0, 1],
    //     [0, 1],
    //     [0, 1],
    //     [0, 1],
    // ];

    const n = new NNetwork(1000, 0.01);
    n.train(xTrains, yTrains);
    console.log(n.forward([-20, -10]));
}

// execute train
main();