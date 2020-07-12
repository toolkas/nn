package ru.toolkas.nn;

import org.junit.Assert;
import org.junit.Test;
import ru.toolkas.nn.util.Functions;

import java.util.function.Function;

public class NeuronTest {
    @Test
    public void testLinear() {
        Function<Double, Double> activation = Functions.linear();

        Neuron neuron = new Neuron(activation);
        neuron.setWeights(new double[]{1});
        neuron.setBias(0);

        neuron.input(new double[]{1});
        Assert.assertEquals(1, neuron.getValue(), 0);
    }

    @Test
    public void testLinear_v2() {
        Function<Double, Double> activation = Functions.linear();

        Neuron neuron = new Neuron(activation);
        neuron.setWeights(new double[]{0.5});
        neuron.setBias(0.5);

        neuron.input(new double[]{1});
        Assert.assertEquals(1, neuron.getValue(), 0);
    }
}
