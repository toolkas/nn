package ru.toolkas.nn;

import java.util.*;
import java.util.function.BiFunction;
import java.util.function.Function;

public class Network {
    private final List<Layer> layers = new ArrayList<>();

    private final int inputs;

    public Network(int inputs) {
        this.inputs = inputs;
    }

    public void add(Layer layer) {
        layers.add(layer);
    }

    public void init(Random random) {
        Layer prev = null;
        for (Layer layer : layers) {
            for (Neuron neuron : layer.getNeurons()) {
                double[] weights = random.doubles(prev != null ? prev.count() : inputs).toArray();
                double bias = random.nextDouble();

                neuron.setWeights(weights);
                neuron.setBias(bias);
            }
            prev = layer;
        }
    }

    public double[] input(double[] in) {
        if (this.inputs != in.length) {
            throw new IllegalArgumentException("unexpected input count: " + in.length + ", must be " + this.inputs);
        }

        double[] result = in;
        for (Layer layer : layers) {
            layer.input(result);
            result = layer.result();
        }
        return result;
    }

    public double[] result() {
        Layer layer = layers.get(layers.size() - 1);
        return layer.result();
    }

    public void trainStep(double[][] inSet, double[][] outSet, double e) {
        for (int i = 0; i < inSet.length; i++) {
            double[] in = inSet[i];
            double[] out = outSet[i];

            doTrain(in, out, e);
        }
    }

    private void doTrain(double[] in, double[] out, double e) {
        // Вычисляем предсказание
        input(in);

        // Корректировка весов
        for (int lIndex = layers.size() - 1; lIndex >= 0; lIndex--) {
            Layer layer = layers.get(lIndex);
            List<Neuron> neurons = layer.getNeurons();

            // Если текущий слой - выходной
            if (lIndex == layers.size() - 1) {
                // Если предыдущий слой - скрытый
                if (lIndex - 1 >= 0) {
                    Layer prev = layers.get(lIndex - 1);

                    for (int nIndex = 0; nIndex < neurons.size(); nIndex++) {
                        Neuron neuron = neurons.get(nIndex);
                        double[] weights = neuron.getWeights();
                        for (int wIndex = 0; wIndex < weights.length; wIndex++) {
                            double dw = -2 * e * prev.getNeurons().get(wIndex).getValue() * neuron.getValue() * (1 - neuron.getValue()) * (neuron.getValue() - out[nIndex]);
                            weights[wIndex] = weights[wIndex] + dw;
                        }

                        double bias = neuron.getBias();
                        double db = -2 * e * neuron.getValue() * (1 - neuron.getValue()) * (neuron.getValue() - out[nIndex]);
                        neuron.setBias(bias + db);
                    }
                } else {
                    // Если текущий слой - первый
                    for (int nIndex = 0; nIndex < neurons.size(); nIndex++) {
                        Neuron neuron = neurons.get(nIndex);
                        double[] weights = neuron.getWeights();

                        for (int wIndex = 0; wIndex < weights.length; wIndex++) {
                            double dw = -2 * e * in[wIndex] * neuron.getValue() * (1 - neuron.getValue()) * (neuron.getValue() - out[nIndex]);
                            weights[wIndex] = weights[wIndex] + dw;
                        }

                        double bias = neuron.getBias();
                        double db = -2 * e * neuron.getValue() * (1 - neuron.getValue()) * (neuron.getValue() - out[nIndex]);
                        neuron.setBias(bias + db);
                    }
                }
            } else {
                //Если текущий слой - скрытый
                if (lIndex - 1 >= 0) {

                } else {
                    // Если текущий слой - первый

                }
            }
        }
    }
}
