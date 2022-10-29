package ml.endpoints;
import ml.core.*;

public class LinearRegressor extends NeuralNetwork {
    public LinearRegressor(double[][] data, double[] obj, int epochs, double alpha) {
        super(data, new double[][] {obj}, new NNParameters(data[0].length, new int[] {1}, epochs, alpha, new NNActivation[] {NNActivation.LINEAR}, NNError.MSE));
    }

    public LinearRegressor(double[][] data, double[] obj) {
        super(data, new double[][] {obj}, new NNParameters(data[0].length, new int[] {1}, 100, 0.001, new NNActivation[] {NNActivation.LINEAR}, NNError.MSE));
    }
}
