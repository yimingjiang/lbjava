/**
 * This software is released under the University of Illinois/Research and
 * Academic Use License. See the LICENSE file in the root folder for details.
 * Copyright (c) 2016
 * <p>
 * Developed by:
 * The Cognitive Computations Group
 * University of Illinois at Urbana-Champaign
 * http://cogcomp.cs.illinois.edu/
 */
package edu.illinois.cs.cogcomp.lbjava.learn;

import java.io.PrintStream;
import java.util.Objects;
import edu.illinois.cs.cogcomp.lbjava.classify.Feature;
import edu.illinois.cs.cogcomp.lbjava.classify.FeatureVector;
import edu.illinois.cs.cogcomp.lbjava.classify.RealPrimitiveStringFeature;
import edu.illinois.cs.cogcomp.lbjava.classify.ScoreSet;
import edu.illinois.cs.cogcomp.lbjava.util.ExceptionlessInputStream;
import edu.illinois.cs.cogcomp.lbjava.util.ExceptionlessOutputStream;


/**
 * Gradient descent is a batch learning algorithm for function approximation
 * in which the learner tries to follow the gradient of the error function to
 * the solution of minimal error.  This implementation is a stochastic
 * approximation to gradient descent in which the approximated function is
 * assumed to have linear form.
 *
 * <p> This algorithm's user-configurable parameters are stored in member
 * fields of this class.  They may be set via either a constructor that names
 * each parameter explicitly or a constructor that takes an instance of
 * {@link edu.illinois.cs.cogcomp.lbjava.learn.StochasticGradientDescent.Parameters Parameters} as
 * input.  The documentation in each member field in this class indicates the
 * default value of the associated parameter when using the former type of
 * constructor.  The documentation of the associated member field in the
 * {@link edu.illinois.cs.cogcomp.lbjava.learn.StochasticGradientDescent.Parameters Parameters} class
 * indicates the default value of the parameter when using the latter type of
 * constructor.
 *
 * @author Nick Rizzolo
 **/
public class StochasticGradientDescent extends StochasticGradientDescentCL {
    /**
     * The learning rate takes the default value, while the name of the
     * classifier gets the empty string.
     **/
    public StochasticGradientDescent() {
        this("");
    }

    /**
     * Sets the learning rate to the specified value, while the name of the
     * classifier gets the empty string.
     *
     * @param r  The desired learning rate value.
     **/
    public StochasticGradientDescent(double r) {
        this("", r);
    }

    /**
     * Initializing constructor.  Sets all member variables to their associated
     * settings in the {@link StochasticGradientDescent.Parameters} object.
     *
     * @param p  The settings of all parameters.
     **/
    public StochasticGradientDescent(Parameters p) {
        this("", p);
    }

    /**
     * The learning rate takes the default value.
     *
     * @param n  The name of the classifier.
     **/
    public StochasticGradientDescent(String n) {
        this(n, defaultLearningRate);
    }

    /**
     * Use this constructor to specify an alternative subclass of
     * {@link SparseWeightVector}.
     *
     * @param n  The name of the classifier.
     * @param r  The desired learning rate value.
     **/
    public StochasticGradientDescent(String n, double r) {
        super(n);
        Parameters p = new Parameters();
        p.learningRate = r;
        setParameters(p);
    }

    /**
     * Initializing constructor.  Sets all member variables to their associated
     * settings in the {@link StochasticGradientDescent.Parameters} object.
     *
     * @param n  The name of the classifier.
     * @param p  The settings of all parameters.
     **/
    public StochasticGradientDescent(String n, Parameters p) {
        super(n);
        setParameters(p);
    }

    /**
     * Returns a string describing the output feature type of this classifier.
     *
     * @return <code>"real"</code>
     **/
    public String getOutputType() {
        return "real";
    }

    /**
     * Trains the learning algorithm given an object as an example.
     *
     * @param exampleFeatures  The example's array of feature indices.
     * @param exampleValues    The example's array of feature values.
     * @param exampleLabels    The example's label(s).
     * @param labelValues      The labels' values.
     **/
    public void learn(int[] exampleFeatures, double[] exampleValues,
                      int[] exampleLabels, double[] labelValues) {
        assert exampleLabels.length == 1
                : "Example must have a single label.";

        double labelValue = labelValues[0];
        double wtx = weightVector.dot(exampleFeatures, exampleValues) + bias;

        learnUpdate(exampleFeatures, exampleValues, labelValue, wtx);
    }

    /**
     * Returns the classification of the given example as a single feature
     * instead of a {@link FeatureVector}.
     *
     * @param f  The features array.
     * @param v  The values array.
     * @return The classification of the example as a feature.
     **/
    public Feature featureValue(int[] f, double[] v) {
        return new RealPrimitiveStringFeature(containingPackage, name, "", realValue(f, v));
    }


    /**
     * Simply computes the dot product of the weight vector and the example
     *
     * @param exampleFeatures  The example's array of feature indices.
     * @param exampleValues    The example's array of feature values.
     * @return The computed real value.
     **/
    public double realValue(int[] exampleFeatures, double[] exampleValues) {
        return weightVector.dot(exampleFeatures, exampleValues) + bias;
    }

    /** Returns a deep clone of this learning algorithm. */
    public Object clone() {
        StochasticGradientDescent clone = null;

        try {
            clone = (StochasticGradientDescent) super.clone();
        } catch (Exception e) {
            System.err.println("Error cloning StochasticGradientDescent: " + e);
            System.exit(1);
        }

        clone.weightVector = (SparseWeightVector) weightVector.clone();
        return clone;
    }
}

