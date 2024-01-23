const fs = require("fs");
const path = require('path');

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

function derivativeSigmoid(x) {
    return sigmoid(x) * (1 - sigmoid(x));
}

function calculateCost(outputs, yHat) {
    let c = 0;
    for (let i = 0; i < outputs.length; i++) {
        let o = outputs[i];
        c += Math.pow(o - yHat, 2);
    }
    return (1 / outputs.length) * c;
}

function calculateDerivativeCost(outputs, yHat) {
    let c = 0;
    for (let i = 0; i < outputs.length; i++) {
        let o = outputs[i];
        c += 2 * (o - yHat);
    }
    return (1 / outputs.length) * c;
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

        for (let i = 0; i < 3; i++) {
            let ws = [];
            for (let j = 0; j < 2; j++) {
                let w = Math.random();
                ws.push(w);
            }

            this.ws1.push(ws);
            
        }

        for (let i = 0; i < 3; i++) {
            let ws = [];
            for (let j = 0; j < 3; j++) {
                let w = Math.random();
                ws.push(w);
            }

            this.ws2.push(ws);
            
        }

        for (let i = 0; i < 2; i++) {
            let ws = [];
            for (let j = 0; j < 3; j++) {
                let w = Math.random();
                ws.push(w);
            }

            this.ws3.push(ws);
            
        }

        for (let i = 0; i < 3; i++) {
            let b = 0;
            this.bs1.push(b);
        }

        for (let i = 0; i < 3; i++) {
            let b = 0;
            this.bs2.push(b);
        }

        for (let i = 0; i < 2; i++) {
            let b = 0;
            this.bs3.push(b);
        }
    }

    train(xTrains, yTrains) {
        
        for (let e = 0; e < this.epochs; e++) {
            for (let i = 0; i < yTrains.length; i++) {
                let xTrain = xTrains[i];
                let yTrain = yTrains[i];
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

                // output layer
                let zh3 = [];
                for (let i = 0; i < this.ws3.length; i++) {
                    let ws = this.ws3[i];
                    let b = this.bs3[i];

                    let z = linearOnObj(zh2, ws, b);
                    let h = sigmoid(z);

                    zh3.push({z: z, h: h});
                }

                // backward propagation
                let derivativeCosts = [];
                for (let i = 0; i < zh3.length; i++) {
                    let yt = yTrain[i];
                    let h = zh3[i].h;
                    let derivativeCost = 2 * (h - yt);
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
                        // console.log(i, ' ', k)
                        let updatedWeight = dc * derivativeSigmoid(zh.z) * zh2Temp.h;
                        if (k == 0) {
                            updatedBias = dc * derivativeSigmoid(zh.z) * 1;
                        }

                        this.ws3[i][k] = this.ws3[i][k] + (-this.learningRate * updatedWeight);
                    }
                    // console.log('--')

                    this.bs3[i] = this.bs3[i] + (-this.learningRate * updatedBias);
                }

                for (let i = zh2.length-1; i >= 0; i--) {
                    let zh2Temp = zh2[i];
                    let ws2Temps = this.ws2[i];

                    let updatedBias = 0;
                    for (let j = ws2Temps.length-1; j >= 0; j--) {
                        let zh1Temp = zh2[j];
                        let updatedWeight = 0;
                        for (let k = derivativeCosts.length-1; k >= 0; k--) {
                            // console.log(k, ' ', i, ' ', j)
                            let zh3Temp = zh3[k];
                            let dc = derivativeCosts[k];
                            updatedWeight += dc * derivativeSigmoid(zh3Temp.z) * this.ws3[k][i] * derivativeSigmoid(zh2Temp.z) * zh1Temp.h;
                            
                            if (j == 0) {
                                updatedBias += dc * derivativeSigmoid(zh3Temp.z) * this.ws3[k][i] * derivativeSigmoid(zh2Temp.z) * 1;
                            }
                        }

                        this.ws2[i][j] = this.ws2[i][j] + (-this.learningRate * updatedWeight);
                    }
                    // console.log('--')

                    this.bs2[i] = this.bs2[i] + (-this.learningRate * updatedBias);
                }

                for (let i = zh1.length-1; i >= 0; i--) {
                    let ws1Temps = this.ws1[i];
                    let zh1Temp = zh1[i];

                    let updatedBias = 0;
                    for (let j = ws1Temps.length-1; j >= 0; j--) {
                        let x = xTrain[j];

                        let updatedWeight = 0;
                        for (let k = derivativeCosts.length-1; k >= 0; k--) {
                            let zh3Temp = zh3[k];
                            let dc = derivativeCosts[k];
                            
                            for (let m = zh2.length - 1; m >= 0; m--) {
                                // console.log(k, ' ', j, ' ', i, ' ', m);
                                let zh2Temp = zh2[m];
                                
                                updatedWeight += dc * derivativeSigmoid(zh3Temp.z) * this.ws3[k][m] * derivativeSigmoid(zh2Temp.z) * this.ws2[m][i] * derivativeSigmoid(zh1Temp.z) * x;

                                if (j == 0) {
                                    updatedBias += dc * derivativeSigmoid(zh3Temp.z) * this.ws3[k][m] * derivativeSigmoid(zh2Temp.z) * this.ws2[m][i] * derivativeSigmoid(zh1Temp.z) * 1;
                                }
                            }
                        }

                        this.ws1[i][j] = this.ws1[i][j] + (-this.learningRate * updatedWeight);
                    }
                    // console.log('--');
                    this.bs1[i] = this.bs1[i] + (-this.learningRate * updatedBias);
                }
            }
        }
    }

    forward(x) {
        // forward propagation

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

        return zh3;
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
            console.log('open model file error ', err);
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
    // const xTrains = [
    //     [-2, -1], // Alice
    //     // [25, 6], // Bob
    //     // [17, 4], // Charlie
    //     // [36, 8], // Alex
    //     // [-15, -6], // Diana
    //     // [10, 4], // George
    //     // [1, 1], // Dina
    //     // [-66, -9], // Hellen,
    //     // [1, 1], // ina
    //     // [1, 2], // any
    //     // [10, 7], // dody
    // ];
    // const yTrains = [
    //     [0,1],
    //     // [1,0],
    //     // [1,0],
    //     // [1,0],
    //     // [0,1],
    //     // [1,0],
    //     // [0,1],
    //     // [0,1],
    //     // [0,1],
    //     // [0,1],
    //     // [1,0]
    // ];

    const xTrains = [
        [2.496714153011233,2.324083969394795],
        [1.8617356988288154,1.6149177195836835],
        [2.6476885381006925,1.3230779996940414],
        [3.5230298564080256,2.6116762888408678],
        [1.765846625276664,3.030999522495951],
        [1.7658630430508195,2.931280119116199],
        [3.5792128155073915,1.1607824767773613],
        [2.767434729152909,1.6907876241487854],
        [1.5305256140650478,2.331263431403564],
        [2.5425600435859645,2.975545127122359],
        [1.5365823071875377,1.52082576215471],
        [1.534270246429743,1.814341023336183],
        [2.241962271566034,0.8936650259939718],
        [0.08671975534220211,0.8037933759193292],
        [0.27508216748696723,2.812525822394198],
        [1.4377124707590272,3.3562400285708227],
        [0.9871688796655762,1.927989878419666],
        [2.314247332595274,3.003532897892024],
        [1.0919759244787892,2.361636025047634],
        [0.5876962986647085,1.3548802453948756],
        [3.465648768921554,2.361395605508414],
        [1.7742236995134644,3.538036566465969],
        [2.0675282046879238,1.9641739608900484],
        [0.5752518137865432,3.564643655814006],
        [1.4556172754748173,-0.6197451040897444],
        [2.1109225897098662,2.821902504375224],
        [0.8490064225776972,2.087047068238171],
        [2.375698018345672,1.7009926495341325],
        [1.399361310081195,2.0917607765355024],
        [1.7083062502067232,0.0124310853991072],
        [1.398293387770603,1.780328112162488],
        [3.852278184508938,2.3571125715117462],
        [1.986502775262066,3.4778940447415163],
        [0.9422890710440996,1.4817297817263526],
        [2.822544912103189,1.1915063971068123],
        [0.7791563500289778,1.4982429564154636],
        [2.2088635950047553,2.915402117702074],
        [0.04032987612022443,2.3287511096596845],
        [0.6718139511015695,1.4702397962329612],
        [2.1968612358691235,2.513267433113356],
        [2.7384665799954107,2.0970775493480405],
        [2.1713682811899706,2.9686449905328893],
        [1.8843517176117595,1.2979469061226476],
        [1.6988963044107113,1.6723378534022317],
        [0.5214780096325726,1.6078918468678425],
        [1.2801557916052912,0.5364850518678814],
        [1.5393612290402126,2.296120277064576],
        [3.0571222262189157,2.261055272179889],
        [2.3436182895684614,2.005113456642461],
        [0.23695984463726605,1.7654128666248532],
        [5.584629257949586,7.250492850345877],
        [6.579354677234641,7.346448209496976],
        [6.65728548347323,6.319975278421509],
        [6.197722730778381,7.232253697161004],
        [6.838714288333991,7.293072473298682],
        [7.404050856814538,6.285648581973632],
        [8.88618590121053,8.865774511144757],
        [7.174577812831839,7.4738329209117875],
        [7.2575503907227645,5.808696502797352],
        [6.925554084233832,7.65655360863383],
        [5.081228784700959,6.025318329772679],
        [6.973486124550783,7.787084603742452],
        [7.060230209941026,8.158595579007404],
        [9.463242112485286,6.179317681648289],
        [6.807639035218878,7.963376129244322],
        [7.301547342333612,7.412780926936498],
        [6.965288230294757,7.82206015999449],
        [5.831321962380468,8.896792982653947],
        [8.142822814515021,6.754611883997129],
        [7.751933032686774,6.246263835642511],
        [7.791031947043047,6.110485570374477],
        [6.090612545205261,6.184189715034561],
        [8.402794310936098,6.922898290585896],
        [5.5981489372077196,7.341151974816644],
        [7.58685709380027,7.276690799330019],
        [9.190455625809978,7.827183249036024],
        [6.009463674869312,7.013001891877907],
        [6.433702270397228,8.453534077157316],
        [7.099651365087642,6.735343166762044],
        [6.496524345883801,9.720169166589619],
        [5.449336568933868,7.625667347765006],
        [7.068562974806027,6.142842443583717],
        [5.937696286273895,5.929107501938888],
        [7.473592430635182,7.482472415243185],
        [6.080575765766197,6.776537214674149],
        [8.54993440501754,7.714000494092092],
        [6.216746707663763,7.473237624573545],
        [6.677938483794325,6.927171087343127],
        [7.81351721736967,6.153206281931595],
        [5.769135683566045,5.485152775314136],
        [7.22745993460413,6.553485047932979],
        [8.307142754282427,7.856398794323472],
        [5.3925167654387725,7.214093744130204],
        [7.184633858532305,5.7542612212880115],
        [7.259882794248424,7.173180925851182],
        [7.78182287177731,7.3853173797288365],
        [5.763049289121918,6.116142563798867],
        [5.679543386915723,7.153725105945528],
        [7.521941565616897,7.058208718446],
        [7.296984673233186,5.857029702169377],
    ];

    const yTrains = [
        [1, 0],
        [1, 0],
        [1, 0],
        [1, 0],
        [1, 0],
        [1, 0],
        [1, 0],
        [1, 0],
        [1, 0],
        [1, 0],
        [1, 0],
        [1, 0],
        [1, 0],
        [1, 0],
        [1, 0],
        [1, 0],
        [1, 0],
        [1, 0],
        [1, 0],
        [1, 0],
        [1, 0],
        [1, 0],
        [1, 0],
        [1, 0],
        [1, 0],
        [1, 0],
        [1, 0],
        [1, 0],
        [1, 0],
        [1, 0],
        [1, 0],
        [1, 0],
        [1, 0],
        [1, 0],
        [1, 0],
        [1, 0],
        [1, 0],
        [1, 0],
        [1, 0],
        [1, 0],
        [1, 0],
        [1, 0],
        [1, 0],
        [1, 0],
        [1, 0],
        [1, 0],
        [1, 0],
        [1, 0],
        [1, 0],
        [1, 0],
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 1],
    ];

    const n = new NNetwork(1000, 0.01);
    n.train(xTrains, yTrains);
    // console.log(n.forward([5, 4]));
    // n.saveModel();

    // n.loadModel('model.json');

    console.log(n.forward([-1, -4]));
}

// https://theneuralblog.com/forward-pass-backpropagation-example/
main();