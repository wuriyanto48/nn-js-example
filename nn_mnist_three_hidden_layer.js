const fs = require("fs");
const path = require('path');
const { parse } = require("csv-parse");

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

function linearOnObj(xs, ws, b) {
    if (Array.isArray(xs) && Array.isArray(ws)) {
        let s = 0;
        for (let i = 0; i < xs.length; i++) {
            let x = xs[i];
            let w = ws[i];
            s += w * x.h;
        }
        return s + b;
    }
    return ws * xs + b;
}

function sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
}

function softmax(x, xs) {
    let denominator = 0;
    for (let i = 0; i < xs.length; i++) {
        let ex = xs[i];
        let ed = Math.exp(ex);
        denominator += ed;
    }
    return Math.exp(x) / denominator;
}

function derivativeSigmoid(x) {
    return sigmoid(x) * (1 - sigmoid(x));
}

function loss(outputs, yTrains) {
    let c = 0;
    for (let i = 0; i < outputs.length; i++) {
        let y = yTrains[i];
        let o = outputs[i];
        c += (1/2) * Math.pow(o.h - y, 2);
    }

    return c;
}

function calculateDerivativeCost(output, yTrue) {
    return 2 * (1/2) * (yTrue - output) * -1;
}

function argmax(arrs) {
    let currIndex = 0;
    let arr = arrs[0].h;
    for (let i = 0; i < arrs.length; i++) {
        if (arrs[i].h > arr) {
            currIndex = i;
            arr = arrs[i].h;
        }
    }

    return currIndex;
}

class NNetwork {
    constructor(epochs, learningRate) {
        this.epochs = epochs;
        this.learningRate = learningRate;

        this.ws1 = [];
        this.ws2 = [];
        this.ws3 = [];
        this.ws4 = [];
        this.bs1 = [];
        this.bs2 = [];
        this.bs3 = [];
        this.bs4 = [];

        for (let i = 0; i < 128; i++) {
            let ws = [];
            for (let j = 0; j < 784; j++) {
                let w = Math.random();
                w = w / Math.pow(784, 0.5);
                ws.push(w);
            }

            this.ws1.push(ws);
            
        }

        for (let i = 0; i < 128; i++) {
            let ws = [];
            for (let j = 0; j < 128; j++) {
                let w = Math.random();
                w = w / Math.pow(128, 0.5);
                ws.push(w);
            }

            this.ws2.push(ws);
            
        }

        for (let i = 0; i < 20; i++) {
            let ws = [];
            for (let j = 0; j < 128; j++) {
                let w = Math.random();
                w = w / Math.pow(128, 0.5);
                ws.push(w);
            }

            this.ws3.push(ws);
            
        }

        for (let i = 0; i < 10; i++) {
            let ws = [];
            for (let j = 0; j < 20; j++) {
                let w = Math.random();
                w = w / Math.pow(20, 0.5);
                ws.push(w);
            }

            this.ws4.push(ws);
            
        }

        for (let i = 0; i < 128; i++) {
            let b = Math.random();
            this.bs1.push(b);
        }

        for (let i = 0; i < 128; i++) {
            let b = Math.random();
            this.bs2.push(b);
        }

        for (let i = 0; i < 20; i++) {
            let b = Math.random();
            this.bs3.push(b);
        }

        for (let i = 0; i < 10; i++) {
            let b = Math.random();
            this.bs4.push(b);
        }
    }

    train(xTrains, yTrains) {
        for (let e = 0; e < this.epochs; e++) {
            console.log('epochs: ', e);

            for (let d = 0; d < yTrains.length; d++) {
                let xTrain = xTrains[d];
                let yTrain = yTrains[d];
                // forward propagation

                // first hidden layer
                let zh1 = [];
                for (let i = 0; i < this.ws1.length; i++) {
                    let ws = this.ws1[i];
                    let b = this.bs1[i];

                    let z = linear(xTrain, ws, b);
                    let h = sigmoid(z);

                    zh1.push({z: z, h: h});
                }

                // second hidden layer
                let zh2 = [];
                for (let i = 0; i < this.ws2.length; i++) {
                    let ws = this.ws2[i];
                    let b = this.bs2[i];

                    let z = linearOnObj(zh1, ws, b);
                    let h = sigmoid(z);

                    zh2.push({z: z, h: h});
                }

                // third hidden layer
                let zh3 = [];
                for (let i = 0; i < this.ws3.length; i++) {
                    let ws = this.ws3[i];
                    let b = this.bs3[i];

                    let z = linearOnObj(zh2, ws, b);
                    let h = sigmoid(z);

                    zh3.push({z: z, h: h});
                }

                // output layer
                let zh4 = [];
                for (let i = 0; i < this.ws4.length; i++) {
                    let ws = this.ws4[i];
                    let b = this.bs4[i];

                    let z = linearOnObj(zh3, ws, b);
                    let h = sigmoid(z);

                    zh4.push({z: z, h: h});
                }

                // if (e % 10 == 0) {
                //     let cost = loss(zh4, yTrain);
                //     console.log('epochs: ', e, ' | loss: ', cost);
                // }

                let cost = loss(zh4, yTrain);
                console.log('epochs: ', e, ' | loss: ', cost);

                // backward propagation
                let derivativeCosts = [];
                for (let i = 0; i < zh4.length; i++) {
                    let yt = yTrain[i];
                    let h = zh4[i].h;
                    let derivativeCost = calculateDerivativeCost(h, yt);
                    // console.log(h, ' - ', yt, ' = ', derivativeCost);
                    
                    derivativeCosts.push(derivativeCost);
                }

                for (let i = derivativeCosts.length-1; i >= 0; i--) {
                    let dc = derivativeCosts[i];
                    let ws = this.ws4[i];
                    
                    let zh = zh4[i];
                    let updatedBias = 0;
                    for (let k = ws.length-1; k >= 0; k--) {
                        let zh3Temp = zh3[k];
                        // console.log(i, ' ', k);
                        let updatedWeight = dc * derivativeSigmoid(zh.z) * zh3Temp.h;
                        if (k == 0) {
                            updatedBias = dc * derivativeSigmoid(zh.z) * 1;
                        }

                        this.ws4[i][k] = this.ws4[i][k] - this.learningRate * updatedWeight;
                    }
                    // console.log('--');

                    this.bs4[i] = this.bs4[i] - this.learningRate * updatedBias;
                }

                for (let i = zh3.length-1; i >= 0; i--) {
                    let zh3Temp = zh3[i];
                    let ws3Temps = this.ws3[i];

                    let updatedBias = 0;
                    for (let j = ws3Temps.length-1; j >= 0; j--) {
                        let zh2Temp = zh2[j];
                        let updatedWeight = 0;
                        for (let k = derivativeCosts.length-1; k >= 0; k--) {
                            // console.log(k, ' ', i, ' ', j);
                            let zh4Temp = zh4[k];
                            let dc = derivativeCosts[k];
                            updatedWeight += dc * 
                                derivativeSigmoid(zh4Temp.z) * 
                                this.ws4[k][i] * 
                                derivativeSigmoid(zh3Temp.z) * 
                                zh2Temp.h;
                            
                            if (j == 0) {
                                updatedBias += dc * 
                                    derivativeSigmoid(zh4Temp.z) * 
                                    this.ws4[k][i] * 
                                    derivativeSigmoid(zh3Temp.z) * 
                                    1;
                            }
                        }

                        this.ws3[i][j] = this.ws3[i][j] - this.learningRate * updatedWeight;
                    }
                    // console.log('--');

                    this.bs3[i] = this.bs3[i] - this.learningRate * updatedBias;
                }

                for (let i = zh2.length-1; i >= 0; i--) {
                    let ws2Temps = this.ws2[i];
                    let zh2Temp = zh2[i];

                    let updatedBias = 0;
                    for (let j = ws2Temps.length-1; j >= 0; j--) {
                        let zh1Temp = zh1[j];

                        let updatedWeight = 0;
                        for (let k = derivativeCosts.length-1; k >= 0; k--) {
                            let zh4Temp = zh4[k];
                            let dc = derivativeCosts[k];
                            
                            for (let m = zh3.length - 1; m >= 0; m--) {
                                // console.log(k, ' ', m, ' ', i, ' ', j);
                                let zh3Temp = zh3[m];
                                
                                updatedWeight += dc * 
                                    derivativeSigmoid(zh4Temp.z) * 
                                    this.ws4[k][m] * 
                                    derivativeSigmoid(zh3Temp.z) * 
                                    this.ws3[m][i] * 
                                    derivativeSigmoid(zh2Temp.z) * 
                                    zh1Temp.h;

                                if (j == 0) {
                                    updatedBias += dc * 
                                        derivativeSigmoid(zh4Temp.z) * 
                                        this.ws4[k][m] * 
                                        derivativeSigmoid(zh3Temp.z) * 
                                        this.ws3[m][i] * 
                                        derivativeSigmoid(zh2Temp.z) * 
                                        1;
                                }
                            }
                        }

                        this.ws2[i][j] = this.ws2[i][j] - this.learningRate * updatedWeight;
                    }
                    // console.log('--');
                    this.bs2[i] = this.bs2[i] - this.learningRate * updatedBias;
                }

                for (let i = zh1.length-1; i >= 0; i--) {
                    let ws1Temps = this.ws1[i];
                    let zh1Temp = zh1[i];

                    let updatedBias = 0;
                    for (let j = ws1Temps.length-1; j >= 0; j--) {
                        let x = xTrain[j];

                        let updatedWeight = 0;
                        for (let k = derivativeCosts.length-1; k >= 0; k--) {
                            let zh4Temp = zh4[k];
                            let dc = derivativeCosts[k];
                            
                            for (let l = zh2.length - 1; l >= 0; l--) {
                                let zh2Temp = zh2[l];
                                for (let m = zh3.length - 1; m >= 0; m--) {
                                    let zh3Temp = zh3[m];
                                    // console.log('k: ', k, ' m: ', m, ' i: ', i, ' j: ', j, ' l: ', l, ' o: ', o);
                                    
                                    updatedWeight += dc * 
                                        derivativeSigmoid(zh4Temp.z) * 
                                        this.ws4[k][m] * 
                                        derivativeSigmoid(zh3Temp.z) * 
                                        this.ws3[m][l] * 
                                        derivativeSigmoid(zh2Temp.z) * 
                                        this.ws2[l][i] *
                                        derivativeSigmoid(zh1Temp.z) * 
                                        x;

                                    if (j == 0) {
                                        updatedBias += dc * 
                                            derivativeSigmoid(zh4Temp.z) * 
                                            this.ws4[k][m] * 
                                            derivativeSigmoid(zh3Temp.z) * 
                                            this.ws3[m][l] * 
                                            derivativeSigmoid(zh2Temp.z) * 
                                            this.ws2[l][i] *
                                            derivativeSigmoid(zh1Temp.z) * 
                                            1;
                                    }
                                    
                                }
                            }
                        }
                        
                        // console.log('--');
                        this.ws1[i][j] = this.ws1[i][j] - this.learningRate * updatedWeight;
                    }
                    // console.log('--');
                    this.bs1[i] = this.bs1[i] - this.learningRate * updatedBias;
                }

                // if (e % 10 == 0) {
                //     this.saveModel();
                // }
            }
        }
    }

    forward(x) {
        // first hidden layer
        let zh1 = [];
        for (let i = 0; i < this.ws1.length; i++) {
            let ws = this.ws1[i];
            let b = this.bs1[i];

            let z = linear(x, ws, b);
            let h = sigmoid(z);

            zh1.push({z: z, h: h});
        }

        // second hidden layer
        let zh2 = [];
        for (let i = 0; i < this.ws2.length; i++) {
            let ws = this.ws2[i];
            let b = this.bs2[i];

            let z = linearOnObj(zh1, ws, b);
            let h = sigmoid(z);

            zh2.push({z: z, h: h});
        }

        // third hidden layer
        let zh3 = [];
        for (let i = 0; i < this.ws3.length; i++) {
            let ws = this.ws3[i];
            let b = this.bs3[i];

            let z = linearOnObj(zh2, ws, b);
            let h = sigmoid(z);

            zh3.push({z: z, h: h});
        }

        // output layer
        let zh4 = [];
        for (let i = 0; i < this.ws4.length; i++) {
            let ws = this.ws4[i];
            let b = this.bs4[i];

            let z = linearOnObj(zh3, ws, b);
            let h = sigmoid(z);

            zh4.push({z: z, h: h});
        }

        return zh4;
    }

    saveModel(name) {
        if (!name) {
            name = 'model.json';
        }

        const model = {
            layer1: {
                weights: this.ws1,
                biases: this.bs1
            },
            layer2: {
                weights: this.ws2,
                biases: this.bs2
            },
            layer3: {
                weights: this.ws3,
                biases: this.bs3
            },
            layer4: {
                weights: this.ws4,
                biases: this.bs4
            },
        };

        const curDir = process.cwd();
        fs.writeFile(path.join(curDir, name), JSON.stringify(model), 'utf8', (err) => {
            if (err) {
                console.log(err);
                return;
            }

            console.log('model saved');
        })
    }

    loadModel(name) {
        let model;
        try {
            const data = fs.readFileSync(name);
            model = JSON.parse(data);
        } catch(e) {
            console.log('open model file error ', err);
            return;
        }

        this.ws1 = model.layer1.weights;
        this.bs1 = model.layer1.biases;

        this.ws2 = model.layer2.weights;
        this.bs2 = model.layer2.biases;

        this.ws3 = model.layer3.weights;
        this.bs3 = model.layer3.biases;

        this.ws4 = model.layer4.weights;
        this.bs4 = model.layer4.biases;
    }
}

function main() {

    let labels = [];
    let datas = [];
    fs.createReadStream('mnist_full.csv')
        .pipe(parse({delimiter: ','}))
        .on('data', (row) => {
            let labelStr = row[0];
            let dataStr = row[1];

            let label = labelStr.split(',').map((v) => parseFloat(v));
            let data = dataStr.split(',').map((v) => parseFloat(v));

            labels.push(label);
            datas.push(data);


        }).on('finish', () => {
            const n = new NNetwork(3, 0.1);

            n.train(datas, labels);

            console.log(labels[0]);
            console.log(n.forward(datas[0]));

            n.saveModel();

            // ---------------------------------------------

            // n.loadModel('model.json');

            // console.log(labels[6]);

            // let r = n.forward(datas[6]);
            // console.log(r)

            // console.log(argmax(r));

        });

}

// execute train
main();
