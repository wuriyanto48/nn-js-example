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

function mse(outputs, yTrains) {
    let c = 0;
    for (let i = 0; i < outputs.length; i++) {
        let y = yTrains[i];
        let o = outputs[i];
        c += Math.pow(o.h - y, 2);
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

                // output layer
                let zh3 = [];
                for (let i = 0; i < this.ws3.length; i++) {
                    let ws = this.ws3[i];
                    let b = this.bs3[i];

                    let z = linearOnObj(zh2, ws, b);
                    let h = sigmoid(z);

                    zh3.push({z: z, h: h});
                }

                let cost = mse(zh3, yTrain);
                console.log('mean squared error: ', cost);

                // backward propagation
                let derivativeCosts = [];
                for (let i = 0; i < zh3.length; i++) {
                    let yt = yTrain[i];
                    let h = zh3[i].h;
                    let derivativeCost = 2 * (1/yTrain.length) * (yt - h) * -1;
                    console.log(h, ' - ', yt, ' = ', derivativeCost);
                    
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
                        if (k == 0) {
                            updatedBias = dc * derivativeSigmoid(zh.z) * 1;
                        }

                        this.ws3[i][k] = this.ws3[i][k] - this.learningRate * updatedWeight;
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
                        for (let k = derivativeCosts.length-1; k >= 0; k--) {
                            let zh3Temp = zh3[k];
                            let dc = derivativeCosts[k];
                            
                            for (let m = zh2.length - 1; m >= 0; m--) {
                                // console.log(k, ' ', m, ' ', i, ' ', j);
                                let zh2Temp = zh2[m];
                                
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
    const xTrains = [
        [-2, -1], // Alice
        // [25, 6], // Bob
        // [17, 4], // Charlie
        // [36, 8], // Alex
        // [-15, -6], // Diana
        // [10, 4], // George
        // [1, 1], // Dina
        // [-66, -9], // Hellen,
        // [1, 1], // ina
        // [1, 2], // any
        // [10, 7], // dody
    ];
    const yTrains = [
        [0,1],
        // [1,0],
        // [1,0],
        // [1,0],
        // [0,1],
        // [1,0],
        // [0,1],
        // [0,1],
        // [0,1],
        // [0,1],
        // [1,0]
    ];

    // const xTrains = [
    //     [0.24967141530112327,0.2324083969394795],
    //     [0.18617356988288156,0.16149177195836836],
    //     [0.26476885381006926,0.13230779996940412],
    //     [0.35230298564080254,0.2611676288840868],
    //     [0.17658466252766641,0.30309995224959513],
    //     [0.17658630430508196,0.29312801191161986],
    //     [0.35792128155073916,0.11607824767773615],
    //     [0.2767434729152909,0.16907876241487854],
    //     [0.1530525614065048,0.2331263431403564],
    //     [0.25425600435859647,0.29755451271223593],
    //     [0.15365823071875379,0.152082576215471],
    //     [0.1534270246429743,0.18143410233361829],
    //     [0.22419622715660342,0.08936650259939719],
    //     [0.008671975534220222,0.08037933759193293],
    //     [0.027508216748696718,0.28125258223941985],
    //     [0.14377124707590275,0.33562400285708227],
    //     [0.09871688796655763,0.19279898784196664],
    //     [0.2314247332595274,0.30035328978920245],
    //     [0.10919759244787891,0.23616360250476343],
    //     [0.05876962986647086,0.13548802453948758],
    //     [0.3465648768921554,0.2361395605508414],
    //     [0.17742236995134644,0.35380365664659696],
    //     [0.2067528204687924,0.19641739608900485],
    //     [0.05752518137865434,0.35646436558140066],
    //     [0.14556172754748176,-0.06197451040897445],
    //     [0.21109225897098663,0.2821902504375224],
    //     [0.08490064225776972,0.20870470682381714],
    //     [0.23756980183456722,0.17009926495341327],
    //     [0.13993613100811952,0.20917607765355023],
    //     [0.17083062502067234,0.00124310853991072],
    //     [0.13982933877706033,0.1780328112162488],
    //     [0.3852278184508938,0.23571125715117466],
    //     [0.19865027752620662,0.34778940447415163],
    //     [0.09422890710440997,0.14817297817263528],
    //     [0.2822544912103189,0.11915063971068124],
    //     [0.07791563500289778,0.14982429564154637],
    //     [0.22088635950047555,0.29154021177020745],
    //     [0.004032987612022448,0.23287511096596847],
    //     [0.06718139511015694,0.14702397962329614],
    //     [0.21968612358691236,0.25132674331133564],
    //     [0.27384665799954105,0.20970775493480404],
    //     [0.21713682811899707,0.29686449905328893],
    //     [0.18843517176117597,0.12979469061226478],
    //     [0.16988963044107114,0.16723378534022318],
    //     [0.052147800963257274,0.16078918468678424],
    //     [0.12801557916052914,0.053648505186788153],
    //     [0.15393612290402126,0.2296120277064576],
    //     [0.30571222262189157,0.22610552721798893],
    //     [0.23436182895684615,0.2005113456642461],
    //     [0.023695984463726616,0.1765412866624853],
    //     [0.5584629257949585,0.7250492850345877],
    //     [0.657935467723464,0.7346448209496975],
    //     [0.665728548347323,0.6319975278421509],
    //     [0.6197722730778381,0.7232253697161003],
    //     [0.683871428833399,0.7293072473298681],
    //     [0.7404050856814538,0.6285648581973632],
    //     [0.888618590121053,0.8865774511144756],
    //     [0.7174577812831838,0.7473832920911787],
    //     [0.7257550390722763,0.580869650279735],
    //     [0.6925554084233833,0.7656553608633829],
    //     [0.5081228784700957,0.6025318329772678],
    //     [0.6973486124550783,0.7787084603742451],
    //     [0.7060230209941026,0.8158595579007404],
    //     [0.9463242112485286,0.6179317681648289],
    //     [0.6807639035218878,0.7963376129244322],
    //     [0.7301547342333612,0.7412780926936497],
    //     [0.6965288230294756,0.782206015999449],
    //     [0.5831321962380467,0.8896792982653947],
    //     [0.814282281451502,0.6754611883997129],
    //     [0.7751933032686774,0.6246263835642509],
    //     [0.7791031947043047,0.6110485570374476],
    //     [0.6090612545205261,0.6184189715034561],
    //     [0.8402794310936099,0.6922898290585895],
    //     [0.5598148937207719,0.7341151974816643],
    //     [0.758685709380027,0.7276690799330019],
    //     [0.9190455625809978,0.7827183249036024],
    //     [0.6009463674869311,0.7013001891877907],
    //     [0.6433702270397228,0.8453534077157316],
    //     [0.709965136508764,0.6735343166762043],
    //     [0.6496524345883801,0.9720169166589618],
    //     [0.5449336568933867,0.7625667347765006],
    //     [0.7068562974806026,0.6142842443583717],
    //     [0.5937696286273895,0.5929107501938887],
    //     [0.7473592430635181,0.7482472415243184],
    //     [0.6080575765766196,0.6776537214674149],
    //     [0.8549934405017539,0.7714000494092091],
    //     [0.6216746707663763,0.7473237624573544],
    //     [0.6677938483794323,0.6927171087343127],
    //     [0.781351721736967,0.6153206281931595],
    //     [0.5769135683566045,0.5485152775314135],
    //     [0.7227459934604129,0.6553485047932979],
    //     [0.8307142754282428,0.7856398794323471],
    //     [0.5392516765438772,0.7214093744130203],
    //     [0.7184633858532303,0.5754261221288012],
    //     [0.7259882794248423,0.7173180925851181],
    //     [0.778182287177731,0.7385317379728836],
    //     [0.5763049289121918,0.6116142563798866],
    //     [0.5679543386915723,0.7153725105945528],
    //     [0.7521941565616898,0.7058208718446],
    //     [0.7296984673233186,0.5857029702169376]
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
    //     [0, 1]
    // ];

    const n = new NNetwork(1, 0.01);
    n.train(xTrains, yTrains);
    console.log(n.forward([0.7, 0.9]));
    n.saveModel();

    // n.loadModel('model.json');

    // console.log(n.forward([0.9, 0.6]));
}

// execute train
main();