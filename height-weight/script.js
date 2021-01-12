import * as tf from '@tensorflow/tfjs'
import * as tfvis from '@tensorflow/tfjs-vis';
window.onload = async () => {
    const height =[150,160,170]
    const weight =[40,50,60]
    tfvis.render.scatterplot(
        {name:'身高体重训练'},
        {values:height.map((x,i)=>({x,y:weight[i]}))},
        {xAxisDomain:[140,180],yAxisDomain:[30,70]}
    )
    const inputs = tf.tensor(height).sub(150).div(20)
    const labels = tf.tensor(weight).sub(40).div(20)//把数据压缩到0-1之间
    const model = tf.sequential()
    model.add(tf.layers.dense({units:1,inputShape:[1]}))
    model.compile({loss:tf.losses.meanSquaredError,optimizer:tf.train.sgd(0.1)})
    await model.fit(inputs,labels,{
        batchSize:3,
        epochs:200,
        callbacks:tfvis.show.fitCallbacks(
            {name:'训练过程'},
            ['loss']
        )
    });
    //反归一化 
    const output = model.predict(tf.tensor([180]).sub(150).div(20))
    alert(`如果身高为180,那么体重为${output.mul(20).add(40).dataSync()[0]}`)
};