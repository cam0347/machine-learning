package ml.core;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Arrays;

public abstract class SupervisedNetwork {
    protected double[][] data;
    protected double[][] objectives;
    protected double[][][] weights;
    protected double[][] bias;
    protected double alpha;
    protected int epochs;

    protected double activation(NNActivation activation, double x) {
        return switch (activation) {
            case SIGMOID -> 1.00 / (1.00 + Math.exp(-x));
            case TANH -> Math.tanh(x);
            case RELU -> x > 0 ? x : 0;
            case LINEAR -> x;
        };
    }

    protected double error(NNError f, double y, double out) {
        return switch(f) {
            case MSE -> Math.pow(y - out, 2);
            case MAE -> Math.abs(y - out);
            case CROSS_ENTROPY -> -(y * Math.exp(out) + (1 - y) * Math.exp(1 - out));
        };
    }

    protected void printElapsedTime(long n) {
        long ms = n;
        int days = (int) Math.floor(ms / 86400000.0);
        ms -= days * 86400000.0;
        int hours = (int) Math.floor(ms / 3600000.0);
        ms -= hours * 3600000.0;
        int minutes = (int) Math.floor(ms / 60000.0);
        ms -= minutes * 60000.0;
        int seconds = (int) Math.floor(ms / 1000.0);
        ms -= seconds * 1000.0;

        System.out.print("Training complete [elapsed ");

        if (n > 86400000) {
            System.out.print(days + "d " + hours + "h " + minutes + "m " + seconds + "s " + ms + "ms]");
        } else if (n > 3600000) {
            System.out.print(hours + "h " + minutes + "m " + seconds + "s " + ms + "ms]");
        } else if (n > 60000) {
            System.out.print(minutes + "m " + seconds + "s " + ms + "ms]");
        } else if (n > 1000) {
            System.out.print(seconds + "s " + ms + "ms]");
        } else {
            System.out.print(ms + "ms]");
        }

        System.out.println();
    }

    protected double errorDerivative(NNError f, double y, double out) {
        double d = 1.00 / this.weights[this.weights.length - 1].length;

        d *= switch(f) {
            case CROSS_ENTROPY -> -(y / out - (1 - y) / (1 - out));
            case MSE -> -2 * (y - out);
            case MAE -> (out - y) / Math.abs(y - out);
        };

        if (Double.isNaN(d)) {
            System.out.println("NaN value found: " + f + ", y=" + y + ", out=" + out);
            System.exit(0);
        }

        return d;
    }

    protected double activationDerivative(NNActivation f, double out, double net) {
        return switch(f) {
            case LINEAR -> 1;
            case SIGMOID -> out * (1 - out);
            case TANH -> 1 - Math.pow(Math.tanh(net), 2);
            case RELU -> net > 0 ? 1 : 0;
        };
    }

    public String printWeights() {
        StringBuilder s = new StringBuilder();

        for (int l = 0; l < this.weights.length; l++) {
            s.append("Layer ").append(l).append(" (neurons: ").append(this.weights[l].length).append(")\n[\n");
            for (int n = 0; n < this.weights[l].length; n++) {
                s.append(Arrays.toString(this.weights[l][n]));
                s.append("[b: ").append(this.bias[l][n]).append("]\n");
            }

            s.append("]\n\n");
        }

        return s.toString();
    }

    public abstract void test(double[][] testSet, double[][] objectives);
    public abstract double[] getAnswer(double[] x);
    public abstract void train();
}
