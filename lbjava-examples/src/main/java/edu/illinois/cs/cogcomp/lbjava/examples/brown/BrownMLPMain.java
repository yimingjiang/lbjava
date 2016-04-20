package edu.illinois.cs.cogcomp.lbjava.examples.brown;

import edu.illinois.cs.cogcomp.lbjava.classify.TestDiscrete;
import edu.illinois.cs.cogcomp.lbjava.learn.BatchTrainer;
import edu.illinois.cs.cogcomp.lbjava.learn.MultiLayerPerceptron;
import edu.illinois.cs.cogcomp.lbjava.learn.SparseNetworkLearner;

public class BrownMLPMain {
    public static void main(String[] args) {

        String name = "accept";
        int iter = 100;

        SpellingParser trainingDataSet =
                new SpellingParser(System.getProperty("user.dir")+"/data/brown/" + name +"-brown80.feat");

        BrownClassifier brownClassifier = new BrownClassifier();
        MultiLayerPerceptron.Parameters p = new MultiLayerPerceptron.Parameters();
        p.learningRateP = 0.2;
        p.hiddenLayersP = new int[] {};
        brownClassifier.setParameters(p);

        // start training
        long startTime = System.nanoTime();

        BatchTrainer trainer = new BatchTrainer(brownClassifier, trainingDataSet);
        trainer.train(iter);

        long estimatedTime = System.nanoTime() - startTime;
        double trainingTime = (double)estimatedTime / Math.pow(10, 9);
        System.out.println();
        System.out.println("============================");
        System.out.printf("Training takes: %f seconds.\n", trainingTime);
        System.out.println("============================");

        SpellingParser testingDataSet =
                new SpellingParser(System.getProperty("user.dir")+"/data/brown/" + name +"-brown20.feat");

        SpellingLabel oracle = new SpellingLabel();

        startTime = System.nanoTime();
        TestDiscrete.testDiscrete(new TestDiscrete(), brownClassifier, oracle, testingDataSet, true, 100);
        estimatedTime = System.nanoTime() - startTime;
        double testingTime = (double)estimatedTime / Math.pow(10, 9);
        System.out.println("============================");
        System.out.printf("Testing takes: %f seconds.\n", testingTime);
        System.out.println("============================");
    }
}
