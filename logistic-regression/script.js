import * as tfvis from '@tensorflow/tfjs-vis'
import {getData} from './data';
window.onload = ()=>{
    const data = getData(400);
    console.log(data)
    tfvis.render.scatterplot(
        {name:'罗辑回归数据'},
        {values:[
            data.filter(p => p.label ===1),//返回一个二维数组
            data.filter(p => p.label ===0),
        ]}
    )
}