package edu.illinois.cs.cogcomp.lbjava.examples.rg;

import edu.illinois.cs.cogcomp.lbjava.classify.Classifier;
import edu.illinois.cs.cogcomp.lbjava.classify.TestReal;
import edu.illinois.cs.cogcomp.lbjava.examples.regression.MyDataReader;
import edu.illinois.cs.cogcomp.lbjava.learn.AdaGrad;
import edu.illinois.cs.cogcomp.lbjava.learn.BatchTrainer;
import edu.illinois.cs.cogcomp.lbjava.learn.StochasticGradientDescent;

public class Main {
    public static void main(String[] args) {
        sgd();
    }

    public static void adg() {
        MyDataReader trainingSet = new MyDataReader(System.getProperty("user.dir")+"/data/bike/day/train.txt");

        ADGRegressor regressor = new ADGRegressor();
        AdaGrad.Parameters p = new AdaGrad.Parameters();
        p.learningRateP = 1;
        regressor.setParameters(p);

        BatchTrainer trainer = new BatchTrainer(regressor, trainingSet);
        trainer.train(10000);

        MyDataReader testingSet = new MyDataReader(System.getProperty("user.dir")+"/data/bike/day/test.txt");

        Classifier oracle = new RLabel();

        TestReal.testReal(new TestReal(), regressor, oracle, testingSet, true, 100);
    }

    public static void sgd() {
        MyDataReader trainingSet = new MyDataReader(System.getProperty("user.dir")+"/data/bike/day/train.txt");

        SGDRegressor regressor = new SGDRegressor();
        StochasticGradientDescent.Parameters p = new StochasticGradientDescent.Parameters();
        p.learningRate = Math.pow(10, -11);
        regressor.setParameters(p);
        System.out.println(regressor.getLossFunction());


        BatchTrainer trainer = new BatchTrainer(regressor, trainingSet);
        trainer.train(10000);

        MyDataReader testingSet = new MyDataReader(System.getProperty("user.dir")+"/data/bike/day/test.txt");

        Classifier oracle = new RLabel();

        TestReal.testReal(new TestReal(), regressor, oracle, testingSet, true, 100);
    }
}
