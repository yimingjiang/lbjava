package edu.illinois.cs.cogcomp.lbjava.examples.lense;

import edu.illinois.cs.cogcomp.lbjava.classify.Classifier;
import edu.illinois.cs.cogcomp.lbjava.classify.TestDiscrete;
import edu.illinois.cs.cogcomp.lbjava.classify.TestReal;
import edu.illinois.cs.cogcomp.lbjava.examples.news.SparseNetworkClassifier;
import edu.illinois.cs.cogcomp.lbjava.examples.regression.MyDataReader;
import edu.illinois.cs.cogcomp.lbjava.learn.BatchTrainer;
import edu.illinois.cs.cogcomp.lbjava.learn.MultiLayerPerceptron;
import edu.illinois.cs.cogcomp.lbjava.learn.SparseNetworkLearner;

public class MLPMain {

    public static void main(String[] args) {
        MyDataReader d = new MyDataReader(System.getProperty("user.dir")+"/data/lense/data.txt");

        MLPClassifier mlpClassifier = new MLPClassifier();
        SparseNetworkLearner.Parameters snp = new SparseNetworkLearner.Parameters();
        MultiLayerPerceptron mlp = new MultiLayerPerceptron();
        snp.baseLTU = mlp;
        mlpClassifier.setParameters(snp);

        BatchTrainer trainer = new BatchTrainer(mlpClassifier, d);
        trainer.train(1);

        Classifier oracle = new LenseLabel();
        TestDiscrete.testDiscrete(new TestDiscrete(), mlpClassifier, oracle, d, true, 20);
    }
}
