package edu.illinois.cs.cogcomp.lbjava.algo_test;

import edu.illinois.cs.cogcomp.lbjava.classify.*;

public class AlgoDiscreteFeatures extends Classifier
{
  public AlgoDiscreteFeatures()
  {
    containingPackage = "";
    name = "AlgoDiscreteFeatures";
  }

  public String getInputType() { return "AlgoData"; }
  public String getOutputType() { return "discrete[]"; }

  public FeatureVector classify(Object __example)
  {
    if (!(__example instanceof AlgoData))
    {
      String type = __example == null ? "null" : __example.getClass().getName();
      System.err.println("Classifier 'AlgoDiscreteFeatures(AlgoData)' defined on line 2 of CL.lbj received '" + type + "' as input.");
      new Exception().printStackTrace();
      System.exit(1);
    }

    AlgoData d = (AlgoData) __example;

    FeatureVector __result;
    __result = new FeatureVector();
    int __featureIndex = 0;
    String __value;

    for (int i = 0; i < d.getFeatures().length; i++)
    {
      __value = "" + (d.getFeatures()[i]);
      __result.addFeature(new DiscreteArrayStringFeature(this.containingPackage, this.name, "", __value, valueIndexOf(__value), (short) 0, __featureIndex++, 0));
    }

    for (int __i = 0; __i < __result.featuresSize(); ++__i)
      __result.getFeature(__i).setArrayLength(__featureIndex);

    return __result;
  }

  public String[] discreteValueArray(Object __example)
  {
    return classify(__example).discreteValueArray();
  }

  public FeatureVector[] classify(Object[] examples)
  {
    if (!(examples instanceof AlgoData[]))
    {
      String type = examples == null ? "null" : examples.getClass().getName();
      System.err.println("Classifier 'AlgoDiscreteFeatures(AlgoData)' defined on line 2 of CL.lbj received '" + type + "' as input.");
      new Exception().printStackTrace();
      System.exit(1);
    }

    return super.classify(examples);
  }

  public int hashCode() { return "AlgoDiscreteFeatures".hashCode(); }
  public boolean equals(Object o) { return o instanceof AlgoDiscreteFeatures; }
}


