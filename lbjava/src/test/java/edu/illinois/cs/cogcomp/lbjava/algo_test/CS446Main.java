package edu.illinois.cs.cogcomp.lbjava.algo_test;

import edu.illinois.cs.cogcomp.lbjava.learn.BatchTrainer;
import edu.illinois.cs.cogcomp.lbjava.learn.Learner;
import edu.illinois.cs.cogcomp.lbjava.learn.StochasticGradientDescent;

public class CS446Main {
    public static void main(String [ ] args) {
        CS446DataSet d = new CS446DataSet(4,6,9,7,false);
        AlgoParser p = new AlgoParser(d);

        Learner sgd = new StochasticGradientDescent();

        BatchTrainer trainer = new BatchTrainer(sgd, p);
        trainer.train(1);
    }
}
