package edu.illinois.cs.cogcomp.lbjava.examples.brown;

import edu.illinois.cs.cogcomp.lbjava.classify.TestDiscrete;
import edu.illinois.cs.cogcomp.lbjava.learn.BatchTrainer;
import edu.illinois.cs.cogcomp.lbjava.learn.MultiLayerPerceptron;
import edu.illinois.cs.cogcomp.lbjava.learn.SparseNetworkLearner;

public class BrownMLPMain {
    public static void main(String[] args) {
        SpellingParser trainingDataSet =
                new SpellingParser(System.getProperty("user.dir")+"/data/their-brown80.feat");

        BrownClassifier brownClassifier = new BrownClassifier();
        SparseNetworkLearner.Parameters snp = new SparseNetworkLearner.Parameters();
        MultiLayerPerceptron mlp = new MultiLayerPerceptron();
        MultiLayerPerceptron.Parameters p = new MultiLayerPerceptron.Parameters();
        p.learningRateP = 0.2;
        p.hiddenLayersP = new int[] {200};
        mlp.setParameters(p);
        snp.baseLTU = mlp;
        brownClassifier.setParameters(snp);

        BatchTrainer trainer = new BatchTrainer(brownClassifier, trainingDataSet);
        trainer.train(10);

        SpellingParser testingDataSet =
                new SpellingParser(System.getProperty("user.dir")+"/data/their-brown20.feat");

        SpellingLabel oracle = new SpellingLabel();

        TestDiscrete.testDiscrete(new TestDiscrete(), brownClassifier, oracle, testingDataSet, true, 20000);
    }
}
