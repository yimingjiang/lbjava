package org.neuroph.core;

import org.neuroph.util.NeuronProperties;

public class SparseInputLayer extends Layer {

    public int[] currentActiveFeatureIndexVector;
    public int[] currentActiveFeatureValueVector;

    /**
     * Creates an instance of empty Layer
     */
    public SparseInputLayer() {
        super();
    }

    /**
     * Creates an instance of input layer, with an array of sparse feature vector
     */
    public SparseInputLayer(int[] featureIndexVector,
                            double[] featureValueVector,
                            NeuronProperties neuronProperties) {
        // find max index in the feature vector, upon constructing NN
        int maxIndex = -1;
        double maxValue = 0;

        for (int i : featureIndexVector) {
            if (featureValueVector[i] > maxValue) {
                maxIndex = i;
            }
        }

        layerConstructorHelper(maxIndex+1, neuronProperties);
    }

    /**
     * Performs calculation for all neurons in this layer
     */
    @Override
    public void calculate() {
        for (int i : currentActiveFeatureIndexVector) {
            neurons.get(i).calculate();
        }
    }
}
