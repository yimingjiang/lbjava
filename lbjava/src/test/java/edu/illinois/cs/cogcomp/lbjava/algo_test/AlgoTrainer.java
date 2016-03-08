package edu.illinois.cs.cogcomp.lbjava.algo_test;

import edu.illinois.cs.cogcomp.lbjava.learn.Learner;

public class AlgoTrainer {
    private CS446DataSet dataSet;
    private Learner learner;

    private int totalNumberOfInstances = 0;
    private int currentIndex = 0;

    private int[] featuresIndices;
    private int[] labelsIndices;

    public AlgoTrainer(CS446DataSet d, Learner l) {
        dataSet = d;
        learner = l;
        totalNumberOfInstances = dataSet.getFeatures().length;

        int numberOfFeatures = dataSet.getFeatures()[0].length;
        featuresIndices = new int[numberOfFeatures];

        labelsIndices = new int[1];

        for (int i = 0; i < numberOfFeatures; i++) {
            featuresIndices[i] = i;
        }

        labelsIndices[0] = 0;
    }

    public void train(int iterations) {

    }

    public void trainEachIteration() {
        int index = findNextIndex();

        double [] featureVector = dataSet.getFeatures()[index];
        double [] labelVector = new double[1];
        labelVector[0] = dataSet.getLabels()[index];

        learner.learn(featuresIndices, featureVector, labelsIndices, labelVector);
    }

    private int findNextIndex() {
        int ret = currentIndex;
        currentIndex = (currentIndex+1) % totalNumberOfInstances;
        return ret;
    }
}
