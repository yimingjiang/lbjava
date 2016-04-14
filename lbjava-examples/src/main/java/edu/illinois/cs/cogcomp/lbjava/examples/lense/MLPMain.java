package edu.illinois.cs.cogcomp.lbjava.examples.lense;

import edu.illinois.cs.cogcomp.lbjava.classify.Classifier;
import edu.illinois.cs.cogcomp.lbjava.classify.TestDiscrete;
import edu.illinois.cs.cogcomp.lbjava.examples.regression.MyDataReader;
import edu.illinois.cs.cogcomp.lbjava.learn.BatchTrainer;
import edu.illinois.cs.cogcomp.lbjava.learn.MultiLayerPerceptron;

public class MLPMain {

    public static void main(String[] args) {
        MyDataReader d = new MyDataReader(System.getProperty("user.dir")+"/data/lense/data.txt");

        MLPClassifier mlpClassifier = new MLPClassifier();
        MultiLayerPerceptron.Parameters p = new MultiLayerPerceptron.Parameters();
        p.learningRateP = 0.2;
        p.hiddenLayersP = new int[] {18};
        mlpClassifier.setParameters(p);

        BatchTrainer trainer = new BatchTrainer(mlpClassifier, d);
        trainer.train(180);

        Classifier oracle = new LenseLabel();
        TestDiscrete.testDiscrete(new TestDiscrete(), mlpClassifier, oracle, d, true, 30);
    }
}
