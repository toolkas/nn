package ru.toolkas.nn.sample;

import ru.toolkas.nn.Layer;
import ru.toolkas.nn.Network;
import ru.toolkas.nn.Neuron;
import ru.toolkas.nn.util.Functions;

import java.io.*;
import java.util.Arrays;
import java.util.Random;
import java.util.function.Function;

/**
 * x * y
 */
public class Multiply {
    public static void main(String[] args) throws IOException {
        Network network = new Network(2);

        Layer layer1 = new Layer();
        layer1.add(new Neuron(Functions.sigmoid()));
        layer1.add(new Neuron(Functions.sigmoid()));
        layer1.add(new Neuron(Functions.sigmoid()));
        layer1.add(new Neuron(Functions.sigmoid()));

        Layer layer2 = new Layer();
        layer2.add(new Neuron(Functions.sigmoid()));

        network.add(layer1);
        network.add(layer2);


        Random random = new Random();
        double[][] inputs = new double[100][2];
        double[][] targets = new double[100][1];

        for (int index = 0; index < 100; index++) {
            double x = random.nextDouble();
            double y = random.nextDouble();

            inputs[index] = new double[]{x, y};
            targets[index] = new double[]{x * y};
        }


        System.out.println("Init");
        network.init(new Random());

        System.out.println("Train");
        network.train(inputs, targets, 0.001, 500000000);

        Function<double[], double[]> format = val -> val;

        System.out.println(Arrays.toString(format.apply(network.input(new double[]{0.5, 0.5}))));
        System.out.println(Arrays.toString(format.apply(network.input(new double[]{0.5, 0.1}))));
        System.out.println(Arrays.toString(format.apply(network.input(new double[]{0.9, 0.1}))));
        System.out.println(Arrays.toString(format.apply(network.input(new double[]{0.7, 0.5}))));
    }
}
