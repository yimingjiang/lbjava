package edu.illinois.cs.cogcomp.lbjava.learn;

import edu.illinois.cs.cogcomp.lbjava.classify.Feature;
import edu.illinois.cs.cogcomp.lbjava.classify.FeatureVector;
import edu.illinois.cs.cogcomp.lbjava.classify.ScoreSet;
import org.neuroph.core.Neuron;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.core.input.WeightedSum;
import org.neuroph.core.transfer.Linear;
import org.neuroph.nnet.comp.neuron.InputNeuron;
import org.neuroph.nnet.learning.MomentumBackpropagation;
import org.neuroph.util.ConnectionFactory;
import org.neuroph.util.NeuronFactory;
import org.neuroph.util.NeuronProperties;
import org.neuroph.util.TransferFunctionType;
import org.neuroph.util.random.RangeRandomizer;
import org.neuroph.util.random.WeightsRandomizer;

import java.io.PrintStream;
import java.util.ArrayList;

public class MultiLayerPerceptron extends LinearThresholdUnit{

    private boolean isFirstTime = true;

    private org.neuroph.nnet.MultiLayerPerceptron mlp;
    private MomentumBackpropagation learningRule;

    private static final double defaultLearningRate = 0.1;
    private static final int[] defaultHiddenLayers = {};

    private double learningRateA;
    private int[] hiddenLayersA;

    private int currentMaxIndex = 0;
    private ArrayList<Integer> labelsList;

    public MultiLayerPerceptron() {
        this("");
    }

    private MultiLayerPerceptron(String n) {
        this(n, defaultLearningRate);
    }

    private MultiLayerPerceptron(String n, double learningRate) {
        this(n, learningRate, defaultHiddenLayers);
    }

    private MultiLayerPerceptron(String n, double learningRate, int[] hiddenLayers) {
        super(n);
        Parameters p = new Parameters();
        p.learningRateP = learningRate;
        p.hiddenLayersP = hiddenLayers;
        setParameters(p);
    }

    public MultiLayerPerceptron(Parameters p) {
        this("", p);
    }

    private MultiLayerPerceptron(String n, Parameters p) {
        super(n);
        setParameters(p);
    }

    public void setParameters(Parameters p) {
        learningRateA = p.learningRateP;
        hiddenLayersA = p.hiddenLayersP;
    }

    private void initialize(int[] featuresIndices, int[] labelsIndices) {
        int[] layersCountList = new int[2+hiddenLayersA.length];

        // find max index in first example
        for (int eachIndex : featuresIndices) {
            if (eachIndex > currentMaxIndex) {
                currentMaxIndex = eachIndex;
            }
        }

        layersCountList[0] = currentMaxIndex + 1;

        System.arraycopy(hiddenLayersA, 0, layersCountList, 1, layersCountList.length - 1 - 1);
        layersCountList[layersCountList.length-1] = 1;

        mlp = new org.neuroph.nnet.MultiLayerPerceptron(layersCountList);

        learningRule = new MomentumBackpropagation();
        learningRule.setLearningRate(learningRateA);

        labelsList = new ArrayList<>();
        labelsList.add(labelsIndices[0]);
    }

    private void addMoreInputNeurons(int[] exampleIndices) {
        int prevMaxIndex = currentMaxIndex;

        for (int eachIndex : exampleIndices) {
            if (eachIndex > currentMaxIndex) {
                currentMaxIndex = eachIndex;
            }
        }

        if (prevMaxIndex == currentMaxIndex) {
            return;
        }

        WeightsRandomizer randomizer = new RangeRandomizer(-0.7, 0.7);

        for (int i = 0; i < (currentMaxIndex-prevMaxIndex); i++) {
            NeuronProperties inputNeuronProperties = new NeuronProperties(InputNeuron.class, Linear.class);
            Neuron neuron = NeuronFactory.createNeuron(inputNeuronProperties);

            // connect the new neuron to all neurons in layer[1]
            ConnectionFactory.createConnection(neuron, mlp.getLayerAt(1));

            mlp.addNeuronToInputNeurons(neuron);
            randomizer.randomize(neuron);
        }
    }

    private void addMoreOutputNeurons(int[] labelIndices, double[] labelValues) {
        for (int i = 0; i < labelsList.size(); i++) {
            if (labelIndices[0] == labelsList.get(i)) {
                return;
            }
        }

        // need to add the new label
        labelsList.add(labelIndices[0]);

        // create new output neuron
        NeuronProperties neuronProperties = new NeuronProperties();
        neuronProperties.setProperty("useBias", true);
        neuronProperties.setProperty("transferFunction", TransferFunctionType.SIGMOID);
        neuronProperties.setProperty("inputFunction", WeightedSum.class);

        Neuron neuron = NeuronFactory.createNeuron(neuronProperties);

        // connect new output neuron to every neuron in the previous layer
        ConnectionFactory.createConnection(mlp.getLayerAt(mlp.getLayersCount()-1), neuron);

        mlp.addNeuronToOutputNeurons(neuron);

        WeightsRandomizer randomizer = new RangeRandomizer(-0.7, 0.7);
        randomizer.randomize(neuron);
    }

    private double[] createFeaturesArray(int[] exampleIndices, double[] exampleValues) {
        double[] featureVector = new double[currentMaxIndex+1];

        for (int i = 0; i < exampleIndices.length; i++) {
            featureVector[exampleIndices[i]] = exampleValues[i];
        }

        return featureVector;
    }

    private double[] createLabelsArray(int[] labelIndices) {
        int dimension = labelsList.size();
        double[] labelVector = new double[dimension];

        for (int i = 0; i < dimension; i++) {
            if (labelIndices[0] == labelsList.get(i)) {
                labelVector[i] = 1;
            }
        }

        return labelVector;
    }

    /**
     * Trains the learning algorithm given an example formatted as
     * arrays of feature indices, their values, and the example labels.
     *
     * @param exampleFeatures The example's array of feature indices.
     * @param exampleValues   The example's array of feature values.
     * @param exampleLabels   The example's label(s).
     * @param labelValues     The values of the labels.
     **/
    @Override
    public void learn(int[] exampleFeatures, double[] exampleValues, int[] exampleLabels, double[] labelValues) {
//        System.out.println(Arrays.toString(exampleFeatures));
//        System.out.println(Arrays.toString(exampleValues));
//        System.out.println(Arrays.toString(exampleLabels));
//        System.out.println(Arrays.toString(labelValues));
//        System.out.println();

        if (isFirstTime) {
            initialize(exampleFeatures, exampleLabels);
            isFirstTime = false;
        }
        else {
            addMoreInputNeurons(exampleFeatures);
            addMoreOutputNeurons(exampleLabels, labelValues);
        }

        double[] featuresArray = createFeaturesArray(exampleFeatures, exampleValues);
        double[] labelsArray = createLabelsArray(exampleLabels);

        DataSetRow row = new DataSetRow(featuresArray, labelsArray);

        mlp.learn(row);
    }

    /**
     * This method makes one or more decisions about a single object, returning
     * those decisions as features in a vector.
     *
     * @param exampleFeatures The example's array of feature indices.
     * @param exampleValues   The example's array of feature values.
     * @return A vector of features about the input object.
     **/
    @Override
    public FeatureVector classify(int[] exampleFeatures, double[] exampleValues) {
        return new FeatureVector(featureValue(exampleFeatures, exampleValues));
    }

    /**
     * Classify into two categories,
     * if >= 0, predict positive
     * if <  0, predict negative
     *
     * @param f  The features array.
     * @param v  The values array.
     * @return feature
     */
    @Override
    public Feature featureValue(int[] f, double[] v) {
        int index = score(f, v) >= 0.5 ? 1 : 0;
        return predictions.get(index);
    }

    public double score(int[] exampleFeatures, double[] exampleValues) {
        double[] exampleFeaturesArray = createFeaturesArray(exampleFeatures, exampleValues);
        DataSetRow row = new DataSetRow(exampleFeaturesArray);
        mlp.setInput(row.getInput());
        mlp.calculate();
        double[] networkOutput = mlp.getOutput();
        return networkOutput[0];
    }


    /**
     * If the <code>LinearThresholdUnit</code> is mistake driven, this method
     * should be overridden and used to update the internal representation when
     * a mistake is made on a positive example.
     *
     * @param exampleFeatures The example's array of feature indices
     * @param exampleValues   The example's array of feature values
     * @param rate            The learning rate at which the weights are updated.
     **/
    @Override
    public void promote(int[] exampleFeatures, double[] exampleValues, double rate) {
        System.out.println("MLP promote function!");
    }

    /**
     * If the <code>LinearThresholdUnit</code> is mistake driven, this method
     * should be overridden and used to update the internal representation when
     * a mistake is made on a negative example.
     *
     * @param exampleFeatures The example's array of feature indices
     * @param exampleValues   The example's array of feature values
     * @param rate            The learning rate at which the weights are updated.
     **/
    @Override
    public void demote(int[] exampleFeatures, double[] exampleValues, double rate) {
        System.out.println("MLP demote function");
    }

    /**
     * Produces a set of scores indicating the degree to which each possible
     * discrete classification value is associated with the given example
     * object.  Learners that return a <code>real</code> feature or more than
     * one feature may implement this method by simply returning
     * <code>null</code>.
     *
     * @param exampleFeatures The example's array of feature indices
     * @param exampleValues   The example's array of values
     * @return A set of scores indicating the degree to which each possible
     * discrete classification value is associated with the given
     * example object.
     **/
    @Override
    public ScoreSet scores(int[] exampleFeatures, double[] exampleValues) {
        return null;
    }

    /**
     * Writes the learned function's internal representation as text.
     *
     * @param out The output stream.
     **/
    @Override
    public void write(PrintStream out) {

    }

    public static class Parameters extends Learner.Parameters {

        public double learningRateP;
        public int[] hiddenLayersP;

        public Parameters() {
            learningRateP = defaultLearningRate;
            hiddenLayersP = null;
        }
    }
}
