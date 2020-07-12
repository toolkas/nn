package ru.toolkas.nn;

public class Input {
    private final double weight;
    private final Neuron neuron;

    public Input(double weight, Neuron neuron) {
        this.weight = weight;
        this.neuron = neuron;
    }

    public double getWeight() {
        return weight;
    }

    public Neuron getNeuron() {
        return neuron;
    }
}
