package edu.illinois.cs.cogcomp.lbjava.examples.news;

import edu.illinois.cs.cogcomp.lbjava.classify.TestDiscrete;
import edu.illinois.cs.cogcomp.lbjava.examples.NewsgroupParser;
import edu.illinois.cs.cogcomp.lbjava.learn.BatchTrainer;
import edu.illinois.cs.cogcomp.lbjava.learn.MultiLayerPerceptron;
import edu.illinois.cs.cogcomp.lbjava.learn.SparseAveragedPerceptron;
import edu.illinois.cs.cogcomp.lbjava.learn.SparseNetworkLearner;

public class NewsMLPMain {
    public static void main(String[] args) {
        NewsgroupParser trainingDataSet = new NewsgroupParser(System.getProperty("user.dir")+"/data/simple.dir");

        NewsMLPClassifier mlpClassifier = new NewsMLPClassifier();
        MultiLayerPerceptron.Parameters p = new MultiLayerPerceptron.Parameters();
        p.learningRateP = 0.2;
        p.hiddenLayersP = new int[] {};
        mlpClassifier.setParameters(p);

        BatchTrainer trainer = new BatchTrainer(mlpClassifier, trainingDataSet);
        trainer.train(1);
//
//        NewsgroupParser testingDataSet = new NewsgroupParser(System.getProperty("user.dir")+"/data/20news.test");
//
//        NewsgroupLabel oracle = new NewsgroupLabel();
//
//        TestDiscrete.testDiscrete(new TestDiscrete(), mlpClassifier, oracle, testingDataSet, true, 20000);
    }
}
