package ml.utility;

public class DataEditor {
    public static double[][][] split(double[][] data, int train, int validation, int test) {
        if (train + validation + test != 100 || train <= 0 || validation < 0 || test < 0) {
            System.out.println("Illegal parameters");
            return null;
        }

        int trainLength = (int) Math.floor((double) data.length * (double) train / 100.0);
        int validationLength = (int) Math.floor((double) data.length * (double) validation / 100.0);
        int testLength = (int) Math.floor((double) data.length * (double) test / 100.0);

        double[][][] ret = new double[3][][];
        ret[0] = new double[trainLength][];
        ret[1] = new double[validationLength][];
        ret[2] = new double[testLength][];

        System.arraycopy(data, 0, ret[0], 0, trainLength);
        System.arraycopy(data, trainLength, ret[1], 0, validationLength);
        System.arraycopy(data, trainLength + validationLength, ret[2], 0, testLength);

        return ret;
    }

    private static double[] getMinMaxAvg(double[][] data) {
        double min = data[0][0];
        double max = min;
        double avg = 0.00;
        double dataLength = data.length * data[0].length;

        for (double[] v: data) {
            for (double d: v) {
                if (d > max) {
                    max = d;
                }

                if (d < min) {
                    min = d;
                }

                avg += d / dataLength;
            }
        }

        return new double[] {min, max, avg};
    }

    public static double[][] minMaxNorm(double[][] data) {
        double[] minMax = getMinMaxAvg(data);
        double diff = minMax[1] - minMax[0];

        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < data[i].length; j++) {
                data[i][j] = (data[i][j] - minMax[0]) / diff;
            }
        }

        return data;
    }

    public static double[][] zScoreNorm(double[][] data) {
        double mean = getMinMaxAvg(data)[2];
        double sigma = 0.00;

        for (double[] v : data) {
            for (double d: v) {
                sigma += (Math.pow(d - mean, 2)) / data.length;
            }
        }

        sigma = Math.sqrt(sigma);

        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < data[i].length; j++) {
                data[i][j] = (data[i][j] - mean) / sigma;
            }
        }

        return data;
    }

    public static double[][] meanNorm(double[][] data) {
        double[] values = getMinMaxAvg(data);

        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < data[i].length; j++) {
                data[i][j] = (data[i][j] - values[0]) / (values[1] - values[2]);
            }
        }

        return data;
    }

    public static double[][] logReduction(double[][] data, int log) {
        double div = Math.pow(10, log);

        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < data[i].length; j++) {
                data[i][j] /= div;
            }
        }

        return data;
    }

    public static double[][] logReduction(double[][] data) {
        double max = getMinMaxAvg(data)[1];
        return logReduction(data, (int) Math.floor(Math.log10(max)));
    }

    public static double[][] oneHotEncode(double[][] data, int featureInd, double[] possibleValues) {
        final int pvl = possibleValues.length;
        double[][] ret = new double[data.length][data[0].length - 1 + pvl];

        for (int i = 0; i < data.length; i++) {
            System.arraycopy(data[i], 0, ret[i], 0, featureInd);
            System.arraycopy(data[i], featureInd + 1, ret[i], featureInd + pvl, ret[i].length - (featureInd + pvl));

            double minDiff = Math.abs(data[i][featureInd] - possibleValues[0]);
            int minDiffInd = 0;
            double[] bins = new double[pvl];

            for (int j = 0; j < pvl; j++) {
                double diff;
                if ((diff = Math.abs(data[i][featureInd] - possibleValues[j])) < minDiff) {
                    minDiff = diff;
                    minDiffInd = j;
                }
            }

            bins[minDiffInd] = 1;
            System.arraycopy(bins, 0, ret[i], featureInd, pvl);
        }

        return ret;
    }
}
