package ru.toolkas.nn;

import java.util.function.Function;

public class Neuron {
    private final Function<Double, Double> activation;

    private double[] weights;
    private double bias;

    private double[] input;
    private double value;

    public Neuron(Function<Double, Double> activation) {
        this.activation = activation;
    }

    public double[] getWeights() {
        return weights;
    }

    public double getBias() {
        return bias;
    }

    public void setWeights(double[] weights) {
        this.weights = weights;
    }

    public void setBias(double bias) {
        this.bias = bias;
    }

    public void input(double[] input) {
        this.input = input;

        if (input.length != weights.length) {
            throw new IllegalArgumentException("input.length != " + weights.length);
        }

        double in = summarize(input);
        value = activation.apply(in);
    }

    public double[] getInput() {
        return input;
    }

    public double getValue() {
        return value;
    }

    private double summarize(double[] input) {
        double result = 0;
        for (int index = 0; index < weights.length; index++) {
            double val = input[index];
            double weight = weights[index];
            result += weight * val;
        }
        result += bias;
        return result;
    }
}
