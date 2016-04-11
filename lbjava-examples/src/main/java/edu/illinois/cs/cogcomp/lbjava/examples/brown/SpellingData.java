package edu.illinois.cs.cogcomp.lbjava.examples.brown;

import java.util.*;

public class SpellingData {

    private List<Double> features;
    private double label;

    public SpellingData(String line) {
        this.features = new ArrayList<>();

        for (String each : line.split(",")) {
            each = each.replace(":", "");
            features.add(Double.parseDouble(each));
        }

        label = features.get(0);
        features.remove(0);
    }

    public List<Double> getFeatures() {
        return this.features;
    }

    public double getLabel() {
        return label;
    }

    public void printData() {
        System.out.println(features.toString());
        System.out.println(label);
        System.out.println();
    }
}
