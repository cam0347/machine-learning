package ml.utility;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.ArrayList;

public class DataLoader {
    public static double[][][] loadCSV(String filename, int outputCols) {
        try {
            BufferedReader br = new BufferedReader(new FileReader(filename));
            ArrayList<double[]>[] list = new ArrayList[] {new ArrayList(), new ArrayList()};
            double[][][] ret = new double[2][][];

            while(br.ready()) {
                String row = br.readLine();

                String regex = row.contains(";") ? ";" : ",";
                String[] data = row.split(regex);

                double[] x = new double[data.length - outputCols];
                double[] y = new double[outputCols];

                for (int i = 0; i < x.length; i++) {
                    x[i] = Double.parseDouble(data[i]);
                }

                for (int i = 0; i < outputCols; i++) {
                    y[i] = Double.parseDouble(data[x.length + i]);
                }

                list[0].add(x);
                list[1].add(y);
            }

            ret[0] = new double[list[0].size()][list[0].get(0).length];
            ret[1] = new double[list[1].size()][outputCols];

            for (int i = 0; i < list[0].size(); i++) {
                ret[0][i] = list[0].get(i);
            }

            for (int i = 0; i < list[1].size(); i++) {
                ret[1][i] = list[1].get(i);
            }

            return ret;
        } catch (FileNotFoundException fnf) {
            System.out.println("Dataset not found");
        } catch (Exception e) {
            System.out.println("There was an error: " + e);
        }

        System.exit(0);
        return null;
    }
}
