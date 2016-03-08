package edu.illinois.cs.cogcomp.lbjava.algo_test;

public class AlgoData {

    private double [] featuresVector;
    private double label;

    public AlgoData(double[] f, double l) {
        featuresVector = f;
        label = l;
    }

    public double[] getFeatures() {
        return featuresVector;
    }

    public double getLabel() {
        return label;
    }
}
