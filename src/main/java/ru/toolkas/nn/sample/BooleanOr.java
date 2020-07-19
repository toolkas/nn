package ru.toolkas.nn.sample;

import ru.toolkas.nn.Layer;
import ru.toolkas.nn.Network;
import ru.toolkas.nn.Neuron;
import ru.toolkas.nn.util.Functions;

import java.util.Random;
import java.util.function.Function;

/**
 * x or y
 */
public class BooleanOr {
    public static void main(String[] args) {
        Network network = new Network(2);

        Layer layer1 = new Layer();
        layer1.add(new Neuron(Functions.sigmoid()));
        layer1.add(new Neuron(Functions.sigmoid()));

        Layer layer2 = new Layer();
        layer2.add(new Neuron(Functions.sigmoid()));

        network.add(layer1);
        network.add(layer2);


        double[][] inputs = new double[][]{
                {0, 0},
                {0, 1},
                {1, 0},
                {1, 1},
        };

        double[][] targets = new double[][]{
                {0},
                {1},
                {1},
                {1},
        };

        System.out.println("Init");
        network.init(new Random());

        System.out.println("Train");
        network.train(inputs, targets, 0.2, 5000);

        Function<Double, Long> format = Math::round;

        System.out.println(format.apply(network.input(new double[]{0, 0})[0]));
        System.out.println(format.apply(network.input(new double[]{0, 1})[0]));
        System.out.println(format.apply(network.input(new double[]{1, 0})[0]));
        System.out.println(format.apply(network.input(new double[]{1, 1})[0]));
    }
}
