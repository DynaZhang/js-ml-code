import * as tfvis from "@tensorflow/tfjs-vis";
import * as tfjs from "@tensorflow/tfjs";

const heights = [150, 160, 170];
const weights = [40, 50, 60];

tfvis.render.scatterplot(
  { name: "身高体重训练数据" },
  {
    values: heights.map((height: number, index: number) => ({
      x: height,
      y: weights[index],
    })),
  },
  {
    xAxisDomain: [140, 220],
    yAxisDomain: [30, 100],
  }
);

const inputs = tfjs
  .tensor(heights)
  .sub(150)
  .div(170 - 150);
const labels = tfjs
  .tensor(weights)
  .sub(40)
  .div(60 - 40);

const model = tfjs.sequential();

model.add(
  tfjs.layers.dense({
    units: 1,
    inputShape: [1],
  })
);

model.compile({
  loss: tfjs.losses.meanSquaredError,
  optimizer: tfjs.train.sgd(0.1),
});

await model.fit(inputs, labels, {
  batchSize: 3,
  epochs: 100,
  callbacks: tfvis.show.fitCallbacks({ name: "训练过程" }, ["loss"]),
});

const outputs = model.predict(tfjs.tensor([180, 190, 200]).sub(150).div(20));

alert(outputs.mul(20).add(40));
