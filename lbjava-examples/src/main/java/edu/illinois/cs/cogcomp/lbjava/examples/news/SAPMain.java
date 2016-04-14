package edu.illinois.cs.cogcomp.lbjava.examples.news;

import edu.illinois.cs.cogcomp.lbjava.classify.TestDiscrete;
import edu.illinois.cs.cogcomp.lbjava.learn.BatchTrainer;
import edu.illinois.cs.cogcomp.lbjava.learn.SparseAveragedPerceptron;
import edu.illinois.cs.cogcomp.lbjava.learn.SparseNetworkLearner;

import edu.illinois.cs.cogcomp.lbjava.examples.Post;
import edu.illinois.cs.cogcomp.lbjava.examples.NewsgroupParser;

public class SAPMain {

    public static void main(String[] args) {
//        NewsgroupParser trainingDataSet = new NewsgroupParser(System.getProperty("user.dir")+"/data/20news.train.shuffled");
//
//        SparseNetworkClassifier sn = new SparseNetworkClassifier();
//        SparseNetworkLearner.Parameters snp = new SparseNetworkLearner.Parameters();
//        SparseAveragedPerceptron sap = new SparseAveragedPerceptron();
//        SparseAveragedPerceptron.Parameters p = new SparseAveragedPerceptron.Parameters();
//        p.learningRate = 0.1;
//        p.thickness = 3;
//        sap.setParameters(p);
//        snp.baseLTU = sap;
//        sn.setParameters(snp);
//
//        BatchTrainer trainer = new BatchTrainer(sn, trainingDataSet);
//        trainer.train(40);
//
//        NewsgroupParser testingDataSet = new NewsgroupParser(System.getProperty("user.dir")+"/data/20news.test");
//
//        NewsgroupLabel oracle = new NewsgroupLabel();
//
//        TestDiscrete.testDiscrete(new TestDiscrete(), sn, oracle, testingDataSet, true, 20000);
    }
}
