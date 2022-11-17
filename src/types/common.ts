export type PointLabel<T> = {
    x: number;
    y: number;
    label: T;
};

export type BrandTrainDataSet = {
    inputs: HTMLImageElement[];
    labels: Array<number[]>;
};
