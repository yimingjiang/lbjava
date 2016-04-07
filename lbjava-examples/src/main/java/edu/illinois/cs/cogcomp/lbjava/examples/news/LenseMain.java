package edu.illinois.cs.cogcomp.lbjava.examples.news;

import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.nnet.learning.MomentumBackpropagation;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintStream;
import java.util.Iterator;

public class LenseMain {

    public static void main(String[] args) {

        System.out.println("Creating training set...");

        String trainingSetFileName = "data/lense/all_data.txt";
        int inputsCount = 9;
        int outputsCount = 3;

        System.out.println("Creating training set...");
        DataSet dataSet = DataSet.createFromFile(trainingSetFileName, inputsCount, outputsCount, " ", false);


        System.out.println("Creating neural network...");
        // create MultiLayerPerceptron neural network
        MultiLayerPerceptron neuralNet = new MultiLayerPerceptron(inputsCount, 8, outputsCount);


        // attach listener to learning rule
        MomentumBackpropagation learningRule = new MomentumBackpropagation();
        //MomentumBackpropagation learningRule = (MomentumBackpropagation) neuralNet.getLearningRule();

        // set learning rate and max error
        learningRule.setLearningRate(0.2);
        //learningRule.setMaxError(0.01);

        System.out.println("Training network...");
        // train the network with training set
        //neuralNet.learn(dataSet);
        for (int i = 0; i < 1000; i++) {
            Iterator<DataSetRow> iterator = dataSet.iterator();
            while (iterator.hasNext()) {
                DataSetRow dataSetRow = iterator.next();
                neuralNet.learn(dataSetRow);
            }
        }

        System.out.println("Training completed.");
        System.out.println("Testing network...");

        testNeuralNetwork(neuralNet, dataSet);

        System.out.println("Done.");
    }

    public static void testNeuralNetwork(NeuralNetwork neuralNet, DataSet testSet) {
        PrintStream out = null;
        File file = new File("data/lense/output.txt");
        try {
            out = new PrintStream(file);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }

        for (DataSetRow testSetRow : testSet.getRows()) {
            neuralNet.setInput(testSetRow.getInput());
            neuralNet.calculate();
            double[] networkOutput = neuralNet.getOutput();

            for (int i = 0; i < testSetRow.getInput().length; i++) {
                int r = (int) testSetRow.getInput()[i];
                if (out != null) {
                    out.printf("%d ", r);
                }
            }

            for (int i = 0; i < networkOutput.length; i++) {
                int r = networkOutput[i] >= 0.5 ? 1 : 0;
                if (i == networkOutput.length-1) {
                    assert out != null;
                    out.printf("%d", r);
                }
                else if (out != null) {
                    out.printf("%d ", r);
                }
            }
            if (out != null) {
                out.printf("\n");
            }
        }
        if (out != null) {
            out.close();
        }
    }
}
