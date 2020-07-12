package ru.toolkas.nn;

import ru.toolkas.nn.util.Functions;

import java.util.Arrays;
import java.util.Random;

public class App {
    public static void main(String[] args) {
        Network network = new Network(2);

        Layer layer1 = new Layer();
        layer1.add(new Neuron(Functions.sigmoid()));
        layer1.add(new Neuron(Functions.sigmoid()));

        Layer layer2 = new Layer();
        layer2.add(new Neuron(Functions.sigmoid()));

        network.add(layer1);
        network.add(layer2);

        Random random = new Random();
        network.init(random);

        double[][] inputs = new double[][]{
                {0, 0},
                {0, 1},
                {1, 0},
                {1, 1},
        };

        double[][] targets = new double[][]{
                {0},
                {0},
                {0},
                {1},
        };
        network.trainStep(inputs, targets, 0.1);

        network.input(new double[]{1, 2});
        double[] result = network.result();
        System.out.println(Arrays.toString(result));
    }
}
