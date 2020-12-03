import * as tf from '@tensorflow/tfjs'
import * as tfvis from '@tensorflow/tfjs-vis'
window.onload = async () => {
    const xs = [1, 2, 3, 4, 5]
    const ys = [3, 5, 7, 9, 11]
    tfvis.render.scatterplot(
        { name: 'y=2x+1' },
        {
            values: xs.map((x, i) => ({
                x, y: ys[i]
            }))
        },
        { xAxisDomain: [0, 6], yAxisDomain: [1, 15] }
    )//画表格
    const model = tf.sequential();//赋值给一个联续的模型
    model.add(tf.layers.dense({ units: 1, inputShape: [1] }))
    model.compile({ loss: tf.losses.meanSquaredError, optimizer: tf.train.sgd(0.1) })
    const inputs = tf.tensor(xs)
    const labels = tf.tensor(ys)
    await model.fit(inputs, labels, {
        batchSize: 1,
        epochs: 35,
        callbacks: tfvis.show.fitCallbacks(
            { name: 'Traning process' },
            ['loss']
        )

    })
    const output = model.predict(tf.tensor([6]))
    alert(`如果x=6,那么y的值为 ${output.dataSync()[0]}`)
}
