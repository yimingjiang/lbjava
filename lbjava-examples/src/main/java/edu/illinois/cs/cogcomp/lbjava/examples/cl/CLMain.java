package edu.illinois.cs.cogcomp.lbjava.examples.cl;

import edu.illinois.cs.cogcomp.lbjava.classify.Classifier;
import edu.illinois.cs.cogcomp.lbjava.classify.TestDiscrete;
import edu.illinois.cs.cogcomp.lbjava.learn.*;

public class CLMain {
    public static void main(String[] args) {
        AlgoDataSet dataSet = new AlgoDataSet(10, 100, 1000, 50000, false);
        AlgoParser trainingSet = new AlgoParser(dataSet, true);

        CLClassifier sparseNetworkLearner = new CLClassifier();
        SparseNetworkLearner.Parameters networkParameters = new SparseNetworkLearner.Parameters();

        StochasticGradientDescentCL sgdcl = new StochasticGradientDescentCL();
        StochasticGradientDescentCL.Parameters sgdclp = new StochasticGradientDescentCL.Parameters();
        sgdclp.learningRate = Math.pow(10, -2);
        sgdclp.lossFunction = "hinge";
        sgdcl.setParameters(sgdclp);
        networkParameters.baseLTU = sgdcl;

//        AdaGradCL adaGradCL = new AdaGradCL();
//        AdaGradCL.Parameters adaGradParameters = new AdaGradCL.Parameters();
//        adaGradParameters.learningRateP = 0.1;
//        adaGradParameters.lossFunctionP = "lms";
//        adaGradCL.setParameters(adaGradParameters);
//        networkParameters.baseLTU = adaGradCL;

//        SparseAveragedPerceptron sap = new SparseAveragedPerceptron();
//        networkParameters.baseLTU = sap;

        sparseNetworkLearner.setParameters(networkParameters);

        BatchTrainer adgclTrainer = new BatchTrainer(sparseNetworkLearner, trainingSet);
        adgclTrainer.train(10);


        AlgoParser testingSet = new AlgoParser(dataSet, false);
        Classifier oracle = new CLLabel();
        TestDiscrete.testDiscrete(new TestDiscrete(), sparseNetworkLearner, oracle, testingSet, true, 1000);
    }
}
