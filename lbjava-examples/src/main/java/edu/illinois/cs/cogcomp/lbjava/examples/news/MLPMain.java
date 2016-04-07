package edu.illinois.cs.cogcomp.lbjava.examples.news;

import edu.illinois.cs.cogcomp.lbjava.learn.MultiLayerPerceptron;

public class MLPMain {

    public static void main(String[] args) {
        MultiLayerPerceptron mlp = new MultiLayerPerceptron();
        mlp.learn(new int[1], new double[1], new int[1], new double[1]);
    }
}
