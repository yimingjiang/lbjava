package edu.illinois.cs.cogcomp.lbjava.algo_test;

import edu.illinois.cs.cogcomp.lbjava.learn.Learner;
import edu.illinois.cs.cogcomp.lbjava.learn.StochasticGradientDescent;

public class CS446Main {
    public static void main(String [ ] args) {
//        CS446DataSet d = new CS446DataSet(4,6,9,7,false);
//
//        Learner sgd = new StochasticGradientDescent();
//
//        AlgoTrainer trainer = new AlgoTrainer(d, sgd);
//
//        trainer.trainEachIteration();
        int i = 7;
        int r = (int)(i * 0.6);
        System.out.println(r);
    }
}
