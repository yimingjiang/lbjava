package edu.illinois.cs.cogcomp.lbjava.examples.newsgroup;

import edu.illinois.cs.cogcomp.lbjava.classify.Classifier;
import edu.illinois.cs.cogcomp.lbjava.classify.TestDiscrete;
import edu.illinois.cs.cogcomp.lbjava.examples.DocumentReader;
import edu.illinois.cs.cogcomp.lbjava.learn.BatchTrainer;
import edu.illinois.cs.cogcomp.lbjava.learn.SparseAveragedPerceptron;
import edu.illinois.cs.cogcomp.lbjava.learn.SparseNetworkLearner;

public class NewsGroupMain {
    public static void main(String [] args) {
        DocumentReader training = new DocumentReader(System.getProperty("user.dir")+"/data/20news/train");

        NewsGroupClassifier cl = new NewsGroupClassifier();
        SparseNetworkLearner.Parameters snp = new SparseNetworkLearner.Parameters();
        SparseAveragedPerceptron sap = new SparseAveragedPerceptron();
        SparseAveragedPerceptron.Parameters sapp = new SparseAveragedPerceptron.Parameters();
        sapp.learningRate = 0.05;
        sapp.thickness = 5;
        sap.setParameters(sapp);
        snp.baseLTU = sap;
        cl.setParameters(snp);

//        SparseAveragedPerceptron.Parameters sapp = new SparseAveragedPerceptron.Parameters();
//        SparseNetworkLearner.Parameters snp = new SparseNetworkLearner.Parameters();
//        sapp.learningRate = 0.05;
//        sapp.thickness = 5;
//        snp.baseLTU = new SparseAveragedPerceptron(sapp);
//        cl.setParameters(snp);


        BatchTrainer trainer = new BatchTrainer(cl, training);
        trainer.train(5);

        DocumentReader testing = new DocumentReader(System.getProperty("user.dir")+"/data/20news/test");
        Classifier oracle = new NewsGroupLabel();
        TestDiscrete.testDiscrete(new TestDiscrete(), cl, oracle, testing, true, 2000);
    }
}
