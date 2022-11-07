/*
 * @Author: zhangzhulei@baidu.com 
 * @Description: 线性回归任务 
*/
import * as tfvis from '@tensorflow/tfjs-vis';
import * as tfjs from '@tensorflow/tfjs';

const xs = [1, 2, 3, 4];
const ys = [1, 3, 5, 7];

tfvis.render.scatterplot(
    {
        name: '线性回归训练集'
    }, 
    {
        values: xs.map((x, index) => ({x, y: ys[index]}))
    }, 
    {
        xAxisDomain: [0,5],
        yAxisDomain: [-1, 17]
    }
);

// 定义一个连续的模型（某一层的输入是其上一层的输出）
const model = tfjs.sequential();

// 定义一个全链接层
model.add(tfjs.layers.dense({
    units: 1,  // 定义一个神经元
    inputShape: [1]  // 对应tfjs.tensor([...])中的shape属性
}));

// 给模型设置编译参数
model.compile({
    loss: tfjs.losses.meanSquaredError,  // 设置损失函数 均方误差
    optimizer: tfjs.train.sgd(0.1)  // 设置优化器，指定学习速率（超参数） 随机梯度下降
});

//设置输入数据和标签
const inputs = tfjs.tensor(xs);
const labels = tfjs.tensor(ys);

/**
 * batchSize和epochs都是超参数（在开始学习过程之前设置值的参数，而不是通过训练得到的参数数据）
 * callbacks 可视化展示训练过程
 */
await model.fit(inputs, labels, {
    batchSize: 1,  // 学习的小批次样本数量
    epochs: 100,  // 迭代训练数组的个数
    callbacks: tfvis.show.fitCallbacks(
        {name: '训练过程'},
        ['loss']
    )
});

const output = model.predict(tfjs.tensor([5, 6, 7, 8]));
const rawData = output.dataSync();
alert(rawData.map(item => item));