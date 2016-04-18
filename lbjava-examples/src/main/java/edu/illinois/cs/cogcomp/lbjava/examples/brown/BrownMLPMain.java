package edu.illinois.cs.cogcomp.lbjava.examples.brown;

import edu.illinois.cs.cogcomp.lbjava.classify.TestDiscrete;
import edu.illinois.cs.cogcomp.lbjava.learn.BatchTrainer;
import edu.illinois.cs.cogcomp.lbjava.learn.MultiLayerPerceptron;
import edu.illinois.cs.cogcomp.lbjava.learn.SparseNetworkLearner;

public class BrownMLPMain {
    public static void main(String[] args) {

        String name = "accept";
        int iter = 1000;

        SpellingParser trainingDataSet =
                new SpellingParser(System.getProperty("user.dir")+"/data/brown/" + name +"-brown80.feat");

        BrownClassifier brownClassifier = new BrownClassifier();
        MultiLayerPerceptron.Parameters p = new MultiLayerPerceptron.Parameters();
        p.learningRateP = 0.2;
        p.hiddenLayersP = new int[] {100};
        brownClassifier.setParameters(p);

        BatchTrainer trainer = new BatchTrainer(brownClassifier, trainingDataSet);
        trainer.train(iter);

        SpellingParser testingDataSet =
                new SpellingParser(System.getProperty("user.dir")+"/data/brown/" + name +"-brown20.feat");

        SpellingLabel oracle = new SpellingLabel();

        TestDiscrete.testDiscrete(new TestDiscrete(), brownClassifier, oracle, testingDataSet, true, 100);
    }
}
