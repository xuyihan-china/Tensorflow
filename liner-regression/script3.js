import * as tf from '@tensorflow/tfjs'
import tf from '@tensorflow/tfjs'
import * as tfvis from '@tensorflow/tfjs-vis'
//tensorvisual 用来画图 tensorflow 用来建模
window.onload = async ()=>{
    //引入数据集合
    const xs =[1,2,3,4]
    const ys =[1,2,3,4]
    tfvis.render.scatterplot(
        {name:'线性回归'},
        {values:xs.map((x,i)=>({
            x,y:ys[i]
        }))},
        {xAxisDomain:[0,5],yAxisDomain:[0,5]}
    )
    //////////////////////////////////////////////
    const model = tf.sequential() ; //告诉模型是seqential的联续的
    model.add(tf.layers.dense({units:1,inputShape:[1]}))
    model.compile({loss:tf.losses.meanSquaredError,optimizer:tf.train.sgd(0.1
        )})
    // 联续的模型 损失函数   
    const labels = tf.tensor(ys)
    const inputs = tf.tensor(ys)
    await model.fit(inputs,labels,{
        batchSize:10,
        epochs:100,
        callbacks:tfvis.show.fitCallbacks(
            {name:'xunlian'},
            ['loss']        )
    })
    const output = model.predict(tf.tensor([6]))
    alert(
        `如果x为5 那么y的值为 ${output.dataSync()[0]}`
    )
}