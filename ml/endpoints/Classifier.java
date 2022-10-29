package ml.endpoints;
import ml.core.*;

public class Classifier extends NeuralNetwork {
    private final Object[] labels;

    public Classifier(double[][] data, double[][] obj, Object[] labels, int[] hiddenLayers, int epochs, double alpha, NNActivation[] hiddenActivations) throws Exception {
        super(data, obj, new NNParameters(data[0].length, hiddenLayers, epochs, alpha, hiddenActivations, NNError.CROSS_ENTROPY));

        if (hiddenLayers[hiddenLayers.length - 1] != labels.length) {
            throw new Exception("Labels number mismatch");
        }

        this.labels = labels;
    }

    public Classifier(double[][] data, double[][] obj, Object[] labels, int inputSize) {
        super(data, obj, new NNParameters(inputSize, new int[] {(int)(((float) inputSize + (float) labels.length) / 2f), obj[0].length}, 100, 0.001, new NNActivation[] {NNActivation.LINEAR, NNActivation.SIGMOID}, NNError.CROSS_ENTROPY));
        this.labels = labels;
    }

    public Object getClass(double[] input) {
        double[] out = super.getAnswer(input);
        double max = out[0];
        int maxInd = 0;

        for (int i = 0; i < out.length; i++) {
            if (out[i] > max) {
                max = out[i];
                maxInd = i;
            }
        }

        return this.labels[maxInd];
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("Classifier network\n");
        sb.append("training samples: ").append(this.data.length).append("\n");

        long nweights = 0, nneurons = 0;
        for (double[][] weight: this.weights) {
            nneurons += weight.length;

            for (double[] doubles: weight) {
                nweights += doubles.length;
            }
        }

        sb.append("neurons: ").append(nneurons).append("\n");
        sb.append("synapses: ").append(nweights).append("\n");
        sb.append("learning rate: ").append(this.alpha).append("\n");
        sb.append("epochs: ").append(this.epochs).append("\n");
        return sb.toString();
    }
}
