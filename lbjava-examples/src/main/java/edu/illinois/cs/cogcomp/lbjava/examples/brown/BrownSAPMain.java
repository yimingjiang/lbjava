package edu.illinois.cs.cogcomp.lbjava.examples.brown;

import edu.illinois.cs.cogcomp.lbjava.classify.TestDiscrete;
import edu.illinois.cs.cogcomp.lbjava.learn.BatchTrainer;
import edu.illinois.cs.cogcomp.lbjava.learn.SparseAveragedPerceptron;
import edu.illinois.cs.cogcomp.lbjava.learn.SparseNetworkLearner;

public class BrownSAPMain {
    public static void main(String[] args) {
//        SpellingParser trainingDataSet =
//                new SpellingParser(System.getProperty("user.dir")+"/data/their-brown80.feat");
//
//        BrownClassifier brownClassifier = new BrownClassifier();
//        SparseNetworkLearner.Parameters snp = new SparseNetworkLearner.Parameters();
//        SparseAveragedPerceptron sap = new SparseAveragedPerceptron();
//        SparseAveragedPerceptron.Parameters p = new SparseAveragedPerceptron.Parameters();
//        p.learningRate = 0.5;
//        p.thickness = 1;
//        sap.setParameters(p);
//        snp.baseLTU = sap;
//        brownClassifier.setParameters(snp);
//
//        BatchTrainer trainer = new BatchTrainer(brownClassifier, trainingDataSet);
//        trainer.train(10);
//
//        SpellingParser testingDataSet =
//                new SpellingParser(System.getProperty("user.dir")+"/data/their-brown20.feat");
//
//        SpellingLabel oracle = new SpellingLabel();
//
//        TestDiscrete.testDiscrete(new TestDiscrete(), brownClassifier, oracle, testingDataSet, true, 20000);
    }
}
