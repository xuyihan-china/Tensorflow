import * as tfvis from '@tensorflow/tfjs-vis'
import * as tf from '@tensorflow/tfjs'
window.onload=async ()=>{
    const xs =[1,2,3,4];
    const ys = [1,3,5,7];
    tfvis.render.scatterplot(
    {name:'线性回归样本集'},
    {values:xs.map((x,i)=>({x,y:ys[i]}))},
    {xAxisDomain:[0,5],yAxisDomain:[0,8]}
    )
    const model = tf.sequential();//连续模型
    model.add(tf.layers.dense({units:1,inputShape:[1]}))//添加层 全链接层 权重*一个值 + 一个值 = y 的值
    //损失函数 来告诉模型错的有多离谱 纠正错误
    //谷歌机器学习速成教程
    model.compile({loss: tf.losses.meanSquaredError , optimizer:tf.train.sgd(0.1)})//sgd学习率
    //使用均方误差
    //SGD 随机梯度下降 优化器告诉模型优化的值是多少？
    const inputs = tf.tensor(xs);
    const labels = tf.tensor(ys);
    await model.fit(inputs,labels,{
        batchSize:10, //每次模型学习的量 4个的量大 超参数 调参dog
        epochs:100, //迭代整个训练过程的数据
        callbacks:tfvis.show.fitCallbacks(//训练过程可视化
            {name:'训练过程'},
            //度量单位 想看什么
            ['loss']
            )
    });
    const output = model.predict(tf.tensor([6]));
    alert(`如果x=5 ,那么y的值为 ${output.dataSync()[0]}`);//转为普通数据
}