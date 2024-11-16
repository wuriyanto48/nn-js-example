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
        c += (1/2) * Math.pow(y - o.h, 2);
    }

    return c;
}

function calculateDerivativeCost(output, yTrue, yTrains) {
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
        this.bs1 = [];
        this.bs2 = [];
        this.bs3 = [];

        for (let i = 0; i < 16; i++) {
            let ws = [];
            for (let j = 0; j < 3; j++) {
                let w = Math.random();
                ws.push(w);
            }

            this.ws1.push(ws);
            
        }

        for (let i = 0; i < 32; i++) {
            let ws = [];
            for (let j = 0; j < 16; j++) {
                let w = Math.random();
                ws.push(w);
            }

            this.ws2.push(ws);
            
        }

        for (let i = 0; i < 9; i++) {
            let ws = [];
            for (let j = 0; j < 32; j++) {
                let w = Math.random();
                ws.push(w);
            }

            this.ws3.push(ws);
            
        }

        for (let i = 0; i < 16; i++) {
            let b = Math.random();
            this.bs1.push(b);
        }

        for (let i = 0; i < 32; i++) {
            let b = Math.random();
            this.bs2.push(b);
        }

        for (let i = 0; i < 9; i++) {
            let b = Math.random();
            this.bs3.push(b);
        }
    }

    train(xTrains, yTrains) {
        
        for (let e = 0; e < this.epochs; e++) {
            console.log('epochs: ', e);

            for (let d = 0; d < yTrains.length; d++) {
                let xTrain = xTrains[d];
                let yTrain = yTrains[d];
                // forward propagation

                let [zh1, zh2, zh3] = this.forward(xTrain);

                // if (e % 10 == 0) {
                //     let cost = loss(zh3, yTrain);
                //     console.log('epochs: ', e, ' | loss: ', cost);
                // }

                let cost = loss(zh3, yTrain);
                console.log('epochs: ', e, ' | loss: ', cost);

                // backward propagation
                let derivativeCosts = [];
                for (let i = 0; i < zh3.length; i++) {
                    let yt = yTrain[i];
                    let h = zh3[i].h;
                    let derivativeCost = calculateDerivativeCost(h, yt, yTrain);
                    // console.log(h, ' - ', yt, ' = ', derivativeCost);
                    
                    derivativeCosts.push(derivativeCost);
                }

                for (let i = derivativeCosts.length-1; i >= 0; i--) {
                    let dc = derivativeCosts[i];
                    let ws = this.ws3[i];
                    
                    let zh = zh3[i];
                    let updatedBias = 0;
                    for (let k = ws.length-1; k >= 0; k--) {
                        let zh2Temp = zh2[k];
                        // console.log(i, ' ', k);
                        let updatedWeight = dc * derivativeSigmoid(zh.z) * zh2Temp.h;
                        this.ws3[i][k] = this.ws3[i][k] - this.learningRate * updatedWeight;
                        if (k == 0) {
                            updatedBias = dc * derivativeSigmoid(zh.z) * 1;
                        }
                    }
                    // console.log('--');

                    this.bs3[i] = this.bs3[i] - this.learningRate * updatedBias;
                }

                for (let i = zh2.length-1; i >= 0; i--) {
                    let zh2Temp = zh2[i];
                    let ws2Temps = this.ws2[i];

                    let updatedBias = 0;
                    for (let j = ws2Temps.length-1; j >= 0; j--) {
                        let zh1Temp = zh1[j];
                        let updatedWeight = 0;
                        for (let k = derivativeCosts.length-1; k >= 0; k--) {
                            // console.log(k, ' ', i, ' ', j);
                            let zh3Temp = zh3[k];
                            let dc = derivativeCosts[k];
                            updatedWeight += dc * 
                                derivativeSigmoid(zh3Temp.z) * 
                                this.ws3[k][i] * 
                                derivativeSigmoid(zh2Temp.z) * 
                                zh1Temp.h;
                            
                            if (j == 0) {
                                updatedBias += dc * 
                                    derivativeSigmoid(zh3Temp.z) * 
                                    this.ws3[k][i] * 
                                    derivativeSigmoid(zh2Temp.z) * 
                                    1;
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
                        for (let m = zh2.length - 1; m >= 0; m--) {
                            // console.log(k, ' ', m, ' ', i, ' ', j);
                            let zh2Temp = zh2[m];

                            for (let k = zh3.length-1; k >= 0; k--) {
                                let zh3Temp = zh3[k];
                                let dc = derivativeCosts[k];
                            
                                updatedWeight += dc * 
                                    derivativeSigmoid(zh3Temp.z) * 
                                    this.ws3[k][m] * 
                                    derivativeSigmoid(zh2Temp.z) * 
                                    this.ws2[m][i] * 
                                    derivativeSigmoid(zh1Temp.z) * 
                                    x;

                                if (j == 0) {
                                    updatedBias += dc * 
                                        derivativeSigmoid(zh3Temp.z) * 
                                        this.ws3[k][m] * 
                                        derivativeSigmoid(zh2Temp.z) * 
                                        this.ws2[m][i] * 
                                        derivativeSigmoid(zh1Temp.z) * 
                                        1;
                                }
                            }
                        }

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

        // output layer
        let zh3 = [];
        for (let i = 0; i < this.ws3.length; i++) {
            let ws = this.ws3[i];
            let b = this.bs3[i];

            let z = linearOnObj(zh2, ws, b);
            let h = sigmoid(z);

            zh3.push({z: z, h: h});
        }

        return [zh1, zh2, zh3];
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
            }
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
            console.log('open model file error ', e);
            return;
        }

        this.ws1 = model.layer1.weights;
        this.bs1 = model.layer1.biases;

        this.ws2 = model.layer2.weights;
        this.bs2 = model.layer2.biases;

        this.ws3 = model.layer3.weights;
        this.bs3 = model.layer3.biases;
    }
}

function main() {

    let labels = [];
    let datas = [];
    fs.createReadStream('color_set.csv')
        .pipe(parse({delimiter: ','}))
        .on('data', (row) => {
            let dataStr = row[0];
            let labelStr = row[1];
            

            let label = labelStr.trim().split(' ').filter(v => {
                if (!v) {
                    return false;
                }
                return true
            }).map((v) => parseFloat(v));

            let data = dataStr.trim().split(' ').filter(v => {
                if (!v) {
                    return false;
                }
                return true
            }).map((v) => parseFloat(v));

            labels.push(label);
            datas.push(data);


        }).on('finish', () => {
            const n = new NNetwork(450000, 0.09);

            n.train(datas, labels);

            n.saveModel();

            // ---------------------------------------------

            // n.loadModel('model.json');

            // let l = {'merah': 0, 'jingga': 1, 'kuning': 2, 'hijau': 3, 'biru': 4, 'nila': 5, 'ungu': 6, 'hitam': 7, 'putih': 8};
            // let rl = {};
            // for (let k in l) {
            //     rl[l[k]] = k;
            // }

            // let layers = n.forward([73, 209, 15]);
            // let r = layers[layers.length-1];
            // console.log(r)
            // console.log(argmax(r))

            // console.log(rl[argmax(r)]);

        });

}

// execute train
main();
