package ru.toolkas.nn;

import java.util.ArrayList;
import java.util.List;

public class Layer {
    private final List<Neuron> neurons = new ArrayList<>();

    public void add(Neuron neuron) {
        neurons.add(neuron);
    }

    public List<Neuron> getNeurons() {
        return neurons;
    }

    public int count() {
        return neurons.size();
    }

    public void input(double[] in) {
        for (Neuron neuron : neurons) {
            neuron.input(in);
        }
    }

    public double[] result() {
        return neurons.stream().mapToDouble(Neuron::getValue).toArray();
    }
}
