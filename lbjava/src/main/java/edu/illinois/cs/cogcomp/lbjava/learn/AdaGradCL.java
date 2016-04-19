package edu.illinois.cs.cogcomp.lbjava.learn;

import edu.illinois.cs.cogcomp.lbjava.classify.Feature;
import edu.illinois.cs.cogcomp.lbjava.classify.FeatureVector;
import edu.illinois.cs.cogcomp.lbjava.classify.ScoreSet;

import java.io.PrintStream;
import java.util.Arrays;
import java.util.Objects;

public class AdaGradCL extends LinearThresholdUnit{

    protected double learningRateA;

    protected String lossFunctionA;

    private double[] diagonalVector;
    private double[] weightVector;
    private double[] gradientVector;

    public static final double defaultLearningRate = 0.1;

    public static final String defaultLossFunction = "lms";

    private boolean areVectorsInitialized = false;

    private boolean isLMS;

    public AdaGradCL() {
        this("");
    }

    public AdaGradCL(double r) {
        this("", r);
    }

    public AdaGradCL(Parameters p) {
        this("", p);
    }

    public AdaGradCL(String n) {
        this(n, defaultLearningRate);
    }

    public AdaGradCL(String n, double r) {
        super(n);
        Parameters p = new Parameters();
        p.learningRateP = r;
        setParameters(p);
    }

    public AdaGradCL(String n, Parameters p) {
        super(n);
        setParameters(p);
    }

    public void setParameters(Parameters p) {
        learningRateA = p.learningRateP;
        lossFunctionA = p.lossFunctionP;
        if (Objects.equals(p.lossFunctionP, "lms")) {
            isLMS = true;
        }
        else if (Objects.equals(p.lossFunctionP, "hinge")) {
            isLMS = false;
        }
        else {
            System.out.println("Undefined loss function! lms or hinge");
            System.exit(-1);
        }
    }

    public double[] getWeightVector() {
        return weightVector;
    }

    public String getLossFunction() {
        return lossFunctionA;
    }

    public double getConstantLearningRate() {
        return learningRateA;
    }

    @Override
    public void learn(int[] exampleFeatures, double[] exampleValues,
                      int[] exampleLabels, double[] labelValues) {
//        System.out.println(Arrays.toString(exampleLabels));
//        System.out.println(Arrays.toString(labelValues));
//        System.out.println();
//        if (labelValues[0] != 1) {
//            System.out.println("YOOOOOOOOO");
//        }

        /* add an additional dimension to feature dimension on W to reduce computation complexities */
        int featureDimension = exampleFeatures.length + 1;

        if (!areVectorsInitialized) {
            initializeVectors(featureDimension);
            areVectorsInitialized = true;
        }

        double labelValue = 1;
        if (exampleLabels[0] == 1) {
            labelValue = 1;
        }
        else if (exampleLabels[0] == 0) {
            labelValue = -1;
        }

        /* compute (w * x + theta) */
        double wDotProductX = 0.0;
        for (int i = 0; i < featureDimension - 1; i++) {
            wDotProductX += weightVector[i] * exampleValues[i];
        }
        wDotProductX += weightVector[featureDimension - 1];

        if (isLMS) {
            double multiplier = wDotProductX - labelValue;
            /* compute gradient vector */
            for (int i = 0; i < featureDimension - 1; i++) {
                gradientVector[i] = multiplier * exampleValues[i];
            }

            gradientVector[featureDimension - 1] = multiplier;

            /* compute diagonal vector, aka squares of gradient vector */
            for (int i = 0; i < featureDimension; i++) {

                /* compute G_t = sum from 1 to t (g_t ^2) */
                diagonalVector[i] = diagonalVector[i] + (gradientVector[i] * gradientVector[i]);

                double denominator = Math.sqrt(diagonalVector[i]);
                if (denominator == 0) {
                    denominator = Math.pow(10, -100);               // avoid denominator being 0
                }

                /* update weight vector */
                /* w_(t+1) = w_t - g_t * r/(G_t)^(1/2) */
                weightVector[i] = weightVector[i] -
                        (gradientVector[i] * learningRateA / denominator);
            }
        }
        else {
            boolean didMakeAMistake = true;

            if (wDotProductX * labelValue > 1) {
                didMakeAMistake = false;
            }

            /* compute gradient vector */
            for (int i = 0; i < featureDimension - 1; i++) {
                if (didMakeAMistake) {
                    gradientVector[i] = (-1) * labelValue * exampleValues[i];
                } else {
                    gradientVector[i] = 0;
                }
            }
            if (didMakeAMistake) {
                gradientVector[featureDimension - 1] = (-1) * labelValue;
            } else {
                gradientVector[featureDimension - 1] = 0;
            }

            /* compute diagonal vector, aka squares of gradient vector */
            for (int i = 0; i < featureDimension; i++) {

                /* compute G_t = sum from 1 to t (g_t ^2) */
                diagonalVector[i] = diagonalVector[i] + (gradientVector[i] * gradientVector[i]);

                double denominator = Math.sqrt(diagonalVector[i]);
                if (denominator == 0) {
                    denominator = Math.pow(10, -100);               // avoid denominator being 0
                }

                /* update weight vector */
                if (didMakeAMistake) {
                /* w_(t+1) = w_t - g_t * r/(G_t)^(1/2) */
                    weightVector[i] = weightVector[i] -
                            (gradientVector[i] * learningRateA / denominator);
                }
            }
        }
    }

    private void initializeVectors(int size) {
        diagonalVector = new double[size];
        weightVector = new double[size];
        gradientVector = new double[size];
        for (int i = 0; i < size; i++) {
            diagonalVector[i] = 0;
            weightVector[i] = 0;
            gradientVector[i] = 0;
        }
    }

    public double score(int[] exampleFeatures, double[] exampleValues) {
        double weightDotProductX = 0.0;
        for(int i = 0; i < exampleFeatures.length; i++) {
            weightDotProductX += weightVector[i] * exampleValues[i];
        }
        weightDotProductX += weightVector[weightVector.length-1];
        return weightDotProductX;
    }

    @Override
    public Feature featureValue(int[] f, double[] v) {
        int index = score(f, v) >= 0 ? 1 : 0;
        return predictions.get(index);
    }

    @Override
    public FeatureVector classify(int[] exampleFeatures, double[] exampleValues) {
        return new FeatureVector(featureValue(exampleFeatures, exampleValues));
    }

    @Override
    public void promote(int[] exampleFeatures, double[] exampleValues, double rate) {

    }

    @Override
    public void demote(int[] exampleFeatures, double[] exampleValues, double rate) {

    }

    @Override
    public ScoreSet scores(int[] exampleFeatures, double[] exampleValues) {
        return null;
    }

    @Override
    public void write(PrintStream printStream) {

    }

    public static class Parameters extends Learner.Parameters {

        public double learningRateP;
        public String lossFunctionP;

        public Parameters() {
            learningRateP = defaultLearningRate;
            lossFunctionP = defaultLossFunction;
        }
    }
}
