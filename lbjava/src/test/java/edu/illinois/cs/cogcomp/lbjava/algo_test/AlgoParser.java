package edu.illinois.cs.cogcomp.lbjava.algo_test;

import edu.illinois.cs.cogcomp.lbjava.parse.Parser;

public class AlgoParser implements Parser{

    private CS446DataSet dataSet;

    private int currentIndex = 0;
    private int totalNumberOfExamples = 0;

    public AlgoParser(CS446DataSet d) {
        dataSet = d;
        totalNumberOfExamples = dataSet.getFeatures().length;
    }

    public Object next() {
        if (currentIndex < totalNumberOfExamples) {
            AlgoData data = new AlgoData(dataSet.getFeatures()[currentIndex],
                                         dataSet.getLabels()[currentIndex]);
            currentIndex ++;
            return data;
        }
        return null;
    }

    public void close() {

    }

    public void reset() {
        currentIndex = 0;
    }
}
