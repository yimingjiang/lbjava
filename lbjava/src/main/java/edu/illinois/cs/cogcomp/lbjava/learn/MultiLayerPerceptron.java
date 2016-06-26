package edu.illinois.cs.cogcomp.lbjava.learn;

import edu.illinois.cs.cogcomp.lbjava.classify.Feature;
import edu.illinois.cs.cogcomp.lbjava.classify.FeatureVector;
import edu.illinois.cs.cogcomp.lbjava.classify.ScoreSet;
import org.neuroph.core.Connection;
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
import java.lang.reflect.Array;
import java.util.*;

public class MultiLayerPerceptron extends Learner{

    private boolean isFirstTime = true;

    private org.neuroph.nnet.MultiLayerPerceptron mlp;
    private MomentumBackpropagation learningRule;

    private static final double defaultLearningRate = 0.1;
    private static final int[] defaultHiddenLayers = {};

    private double learningRateA;
    private int[] hiddenLayersA;

    //private int currentMaxIndex = 0;

    private HashMap<Integer, Integer> featuresMap;
    private ArrayList<Integer> labelsList;

    private int[][] denseMatrixBooleanMapping;
    private Random randomBinary;

    private int count = 0;

    public MultiLayerPerceptron() {
        this("");
    }

    public MultiLayerPerceptron(String n) {
        this(n, defaultLearningRate);
    }

    public MultiLayerPerceptron(String n, double learningRate) {
        this(n, learningRate, defaultHiddenLayers);
    }

    public MultiLayerPerceptron(String n, double learningRate, int[] hiddenLayers) {
        super(n);
        Parameters p = new Parameters();
        p.learningRateP = learningRate;
        p.hiddenLayersP = hiddenLayers;
        setParameters(p);
    }

    public MultiLayerPerceptron(Parameters p) {
        this("", p);
    }

    public MultiLayerPerceptron(String n, Parameters p) {
        super(n);
        setParameters(p);
    }

    public void setParameters(Parameters p) {
        learningRateA = p.learningRateP;
        hiddenLayersA = p.hiddenLayersP;
    }

    private void initialize(int[] featuresIndices, int[] labelsIndices) {
        // initialize feature hash map
        featuresMap = new HashMap<>();

        // add feature index to the hash map
        for (int i = 0; i < featuresIndices.length; i++) {
            if (!featuresMap.containsKey(featuresIndices[i])) {
                featuresMap.put(featuresIndices[i], 1);
            }
        }

        System.out.println("Hidden Layers: ");
        System.out.println(Arrays.toString(hiddenLayersA));

        // construct layer count list
        int[] layersCountList = new int[2+hiddenLayersA.length];

        layersCountList[0] = 1000;

        System.arraycopy(hiddenLayersA, 0, layersCountList, 1, layersCountList.length - 1 - 1);
        layersCountList[layersCountList.length-1] = 1;

        mlp = new org.neuroph.nnet.MultiLayerPerceptron(layersCountList);

        learningRule = new MomentumBackpropagation();
        learningRule.setLearningRate(learningRateA);

        labelsList = new ArrayList<>();
        labelsList.add(labelsIndices[0]);

        randomBinary = new Random(0);
        denseMatrixBooleanMapping = new int[10000][1000];
        fillMatrixWithBooleans();
    }

    private int generateRandomBinary() {
        return randomBinary.nextInt(2);
    }

    private void fillMatrixWithBooleans() {
        int estimatedFeatureVectorDimension = 10000;
        int denseVectorDimension = 1000;

        for (int i = 0; i < estimatedFeatureVectorDimension; i++) {
            for(int j = 0; j < denseVectorDimension; j++) {
                denseMatrixBooleanMapping[i][j] = generateRandomBinary();
            }
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
        ConnectionFactory.createConnection(mlp.getLayerAt(mlp.getLayersCount()-2), neuron);

        mlp.addNeuronToOutputNeurons(neuron);

        WeightsRandomizer randomizer = new RangeRandomizer(-0.7, 0.7);
        randomizer.randomize(neuron);

        // add new neuron to the output layer
        mlp.getLayerAt(mlp.getLayersCount()-1).addNeuron(neuron);

        // instantiate trainingData for new neuron
        for (Connection connection : neuron.getInputConnections()) {
            connection.getWeight().setTrainingData(new MomentumBackpropagation.MomentumWeightTrainingData());
        }
    }

    private double[] createFeaturesArray(int[] exampleIndices, double[] exampleValues) {

        double[] featureVector = new double[1000];

        for (int i = 0; i < exampleIndices.length; i++) {
            for (int j = 0; j < 1000; j++) {
                featureVector[j] += denseMatrixBooleanMapping[exampleIndices[i]][j];
            }
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
            addMoreOutputNeurons(exampleLabels, labelValues);
        }

        count ++;
        System.out.printf("%d, %d\n", count, mlp.getInputsCount());

        double[] featuresArray = createFeaturesArray(exampleFeatures, exampleValues);
        double[] labelsArray = createLabelsArray(exampleLabels);

        DataSetRow row = new DataSetRow(featuresArray, labelsArray);

        //System.out.println(labelsList.toString());
        //System.out.println(mlp.toString());

        mlp.learn(row);
    }

    @Override
    public FeatureVector classify(int[] exampleFeatures, double[] exampleValues) {
        return new FeatureVector(featureValue(exampleFeatures, exampleValues));
    }

    @Override
    public Feature featureValue(int[] f, double[] v) {
        int index = findLabelIndex(f, v);
        return predictions.get(index);
    }

    private int findLabelIndex(int[] exampleFeatures, double[] exampleValues) {
        double[] exampleFeaturesArray = createFeaturesArray(exampleFeatures, exampleValues);
        DataSetRow row = new DataSetRow(exampleFeaturesArray);
        mlp.setInput(row.getInput());
        mlp.calculate();
        double[] networkOutput = mlp.getOutput();

//        double[] cleanOutput = new double[networkOutput.length];
//
//        for (int i = 0; i < networkOutput.length; i++) {
//            cleanOutput[i] = networkOutput[i] >= 0.5 ? 1 : 0;
//        }

        for (int i = 0; i < networkOutput.length; i++) {
            if (networkOutput[i] >= 0.5) {
                return i;
            }
        }

        System.out.println("************** Invalid ouput!!!");
//        System.exit(-1);

        return 0;
    }

    @Override
    public ScoreSet scores(int[] exampleFeatures, double[] exampleValues) {
        return null;
    }

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
