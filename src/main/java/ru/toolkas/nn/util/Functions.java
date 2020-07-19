package ru.toolkas.nn.util;

import java.util.function.Function;

public class Functions {
    private Functions() {
    }

    public static Function<Double, Double> linear() {
        return aDouble -> aDouble;
    }

    public static Function<Double, Integer> step(double step, int min, int max) {
        return arg -> {
            if (arg < step) {
                return min;
            }
            return max;
        };
    }

    public static Function<Double, Double> sigmoid() {
        return val -> 1 / (1 + Math.exp(-val));
    }
}
