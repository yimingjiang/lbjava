// Modifying this comment will cause the next execution of LBJava to overwrite this file.
// F1B88000000000000000B49CC2E4E2A4D294550FB4D2F2E4F2ACF2D20F94C4A4DC1D808CF2E215820021A9A063ABA05DA004535A54970611DB4F4D218BA6D0D4B658A500C0A375B454000000

package edu.illinois.cs.cogcomp.lbjava.examples.news;

import edu.illinois.cs.cogcomp.lbjava.classify.*;
import edu.illinois.cs.cogcomp.lbjava.examples.NewsgroupParser;
import edu.illinois.cs.cogcomp.lbjava.examples.Post;
import edu.illinois.cs.cogcomp.lbjava.infer.*;
import edu.illinois.cs.cogcomp.lbjava.io.IOUtilities;
import edu.illinois.cs.cogcomp.lbjava.learn.*;
import edu.illinois.cs.cogcomp.lbjava.parse.*;


public class NewsgroupLabel extends Classifier
{
  public NewsgroupLabel()
  {
    containingPackage = "edu.illinois.cs.cogcomp.lbjava.examples.news";
    name = "NewsgroupLabel";
  }

  public String getInputType() { return "edu.illinois.cs.cogcomp.lbjava.examples.Post"; }
  public String getOutputType() { return "discrete"; }


  public FeatureVector classify(Object __example)
  {
    return new FeatureVector(featureValue(__example));
  }

  public Feature featureValue(Object __example)
  {
    String result = discreteValue(__example);
    return new DiscretePrimitiveStringFeature(containingPackage, name, "", result, valueIndexOf(result), (short) allowableValues().length);
  }

  public String discreteValue(Object __example)
  {
    if (!(__example instanceof Post))
    {
      String type = __example == null ? "null" : __example.getClass().getName();
      System.err.println("Classifier 'NewsgroupLabel(Post)' defined on line 18 of news.lbj received '" + type + "' as input.");
      new Exception().printStackTrace();
      System.exit(1);
    }

    Post post = (Post) __example;

    return "" + (post.getNewsgroup());
  }

  public FeatureVector[] classify(Object[] examples)
  {
    if (!(examples instanceof Post[]))
    {
      String type = examples == null ? "null" : examples.getClass().getName();
      System.err.println("Classifier 'NewsgroupLabel(Post)' defined on line 18 of news.lbj received '" + type + "' as input.");
      new Exception().printStackTrace();
      System.exit(1);
    }

    return super.classify(examples);
  }

  public int hashCode() { return "NewsgroupLabel".hashCode(); }
  public boolean equals(Object o) { return o instanceof NewsgroupLabel; }
}

