package ru.toolkas.nn;

import java.util.*;

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

    public void train(double[][] inputs, double[][] targets, double e, int iterations) {
        for (int index = 0; index < iterations; index++) {
            trainStep(inputs, targets, e);
        }
    }

    public void trainStep(double[][] inSet, double[][] targets, double e) {
        for (int i = 0; i < inSet.length; i++) {
            double[] in = inSet[i];
            double[] out = targets[i];

            doCorrect(in, out, e);
        }
    }

    private void doCorrect(double[] in, double[] out, double e) {
        // Вычисляем предсказание
        input(in);

        // Корректировка весов
        for (int lIndex = layers.size() - 1; lIndex >= 0; lIndex--) {
            Layer layer = layers.get(lIndex);
            List<Neuron> neurons = layer.getNeurons();

            for (int nIndex = 0; nIndex < neurons.size(); nIndex++) {
                Neuron neuron = neurons.get(nIndex);
                setNeuronError(neuron, lIndex, nIndex, out);

                double bias = neuron.getBias();

                double db = -e * neuron.getError();
                neuron.setBias(bias + db);

                double[] weights = neuron.getWeights();
                if (lIndex - 1 >= 0) {
                    Layer prev = layers.get(lIndex - 1);

                    double[] newWeights = new double[weights.length];
                    for (int wIndex = 0; wIndex < weights.length; wIndex++) {
                        double weight = weights[wIndex];
                        double value = prev.getNeurons().get(wIndex).getValue();
                        double dw = -e * value * neuron.getError();
                        newWeights[wIndex] = weight + dw;
                    }
                    neuron.setWeights(newWeights);
                } else {
                    double[] newWeights = new double[weights.length];
                    for (int wIndex = 0; wIndex < weights.length; wIndex++) {
                        double weight = weights[wIndex];
                        double value = in[wIndex];
                        double dw = -e * value * neuron.getError();
                        newWeights[wIndex] = weight + dw;
                    }
                    neuron.setWeights(newWeights);
                }
            }
        }
    }

    private void setNeuronError(Neuron neuron, int lIndex, int nIndex, double[] out) {
        double error = 0;
        if (lIndex == layers.size() - 1) {
            error = neuron.getValue() * (1 - neuron.getValue()) * (neuron.getValue() - out[nIndex]);
        } else {
            Layer next = layers.get(lIndex + 1);
            List<Neuron> nextNeurons = next.getNeurons();

            double s = 0;
            for (Neuron nexNeuron : nextNeurons) {
                double weight = nexNeuron.getWeights()[nIndex];
                s += weight * nexNeuron.getError();
            }

            error = neuron.getValue() * (1 - neuron.getValue()) * s;
        }
        neuron.setError(error);
    }
}
