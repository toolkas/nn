package ru.toolkas.nn.util;

import java.util.function.Function;

public class Functions {
    private Functions() {
    }

    public static Function<Double, Double> linear() {
        return aDouble -> aDouble;
    }

    public static Function<Double, Double> sigmoid() {
        return val -> 1 / (1 + Math.exp(-val));
    }
}
