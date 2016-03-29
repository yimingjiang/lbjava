package edu.illinois.cs.cogcomp.lbjava.algo_test.main;

import edu.illinois.cs.cogcomp.lbjava.algo_test.AlgoDataSet;
import edu.illinois.cs.cogcomp.lbjava.algo_test.AlgoDiscreteLabel;
import edu.illinois.cs.cogcomp.lbjava.algo_test.AlgoParser;
import edu.illinois.cs.cogcomp.lbjava.algo_test.classifiers.SparseNetworkClassifier;
import edu.illinois.cs.cogcomp.lbjava.classify.Classifier;
import edu.illinois.cs.cogcomp.lbjava.classify.TestDiscrete;
import edu.illinois.cs.cogcomp.lbjava.learn.BatchTrainer;
import edu.illinois.cs.cogcomp.lbjava.learn.SparseAveragedPerceptron;
import edu.illinois.cs.cogcomp.lbjava.learn.SparseNetworkLearner;
import edu.illinois.cs.cogcomp.lbjava.learn.SparseWinnow;

public class SparseNetworkMain {
    public static void main(String [] args) {
        winnowDemo();
    }

    public static void perceptronDemo() {
        AlgoDataSet dataSet = new AlgoDataSet(10, 100, 1000, 50000, false);
        AlgoParser trainingSet = new AlgoParser(dataSet, true);

        SparseNetworkClassifier swn = new SparseNetworkClassifier();
        SparseNetworkLearner.Parameters snp = new SparseNetworkLearner.Parameters();
        SparseAveragedPerceptron sw = new SparseAveragedPerceptron();
        snp.baseLTU = sw;
        swn.setParameters(snp);

        BatchTrainer swTrainer = new BatchTrainer(swn, trainingSet);
        swTrainer.train(10);

        AlgoParser testingSet = new AlgoParser(dataSet, false);
        Classifier oracle = new AlgoDiscreteLabel();
        TestDiscrete.testDiscrete(new TestDiscrete(), swn, oracle, testingSet, true, 1000);
    }

    public static void winnowDemo() {
        AlgoDataSet dataSet = new AlgoDataSet(10, 100, 1000, 50000, false);
        AlgoParser trainingSet = new AlgoParser(dataSet, true);

        SparseNetworkClassifier swn = new SparseNetworkClassifier();
        SparseNetworkLearner.Parameters snp = new SparseNetworkLearner.Parameters();
        SparseWinnow sw = new SparseWinnow();
        SparseWinnow.Parameters p = new SparseWinnow.Parameters();
        p.threshold = 10;
        p.learningRate = 1.01;
        p.beta = 1 / 1.01;
        p.bias = -1000;
        sw.setParameters(p);
        snp.baseLTU = sw;
        swn.setParameters(snp);

        BatchTrainer swTrainer = new BatchTrainer(swn, trainingSet);
        swTrainer.train(20);

        AlgoParser testingSet = new AlgoParser(dataSet, false);
        Classifier oracle = new AlgoDiscreteLabel();
        TestDiscrete.testDiscrete(new TestDiscrete(), swn, oracle, testingSet, true, 1000);
    }
}
