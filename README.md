### Repository untuk research Neural Network from scratch dengan Javascript

https://wuriyanto48.github.io/nnjs/

#### Run Training
```shell
node nn_mnist_two_hidden_layer.js
```

Operasi diatas akan menghasilkan satu Machine Learning Model file `model.json` dalam format `JSON`. Yang Pada dasarnya file tersebut berisi `parameters` `weights` dan `biases` pada setiap layer.

#### Inference
Untuk mencoba model yang telah terbentuk, salin file `model.json` ke folder `public/js`. Kemudian jalankan HTTP Server Inferencenya.
```
npm start
```

Atau anda bisa mencoba contoh yang sudah berjalan pada

https://wuriyanto48.github.io/nnjs/