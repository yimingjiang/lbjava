// Modifying this comment will cause the next execution of LBJava to overwrite this file.
// F1B88000000000000000D4ECD4A02C0301500EBAC340521AF34DDA551CE5041C580A8B8A67ADE41D4529411B2EDDD4354090CCC26EB997928DC55395A922BCBAD6978657164CEA5361F075426911ED82B5D01CAC281B2429AB6B4F3E8F2D61FAD37F4246A802069FF6B91D63F3B73654ED2B7CDC087F653BAA0F479AE4B76519DCCD5D1E32283443E0379013098F64AA2B5B0985321CC66E7F263DD5C8F33229013791FD37B7DA9C889C9631D13F8AF3F44E01568491A139610BAB3A41F91FD7106677147E30100000

package edu.illinois.cs.cogcomp.lbjava.examples.news;

import edu.illinois.cs.cogcomp.lbjava.classify.*;
import edu.illinois.cs.cogcomp.lbjava.examples.NewsgroupParser;
import edu.illinois.cs.cogcomp.lbjava.examples.Post;
import edu.illinois.cs.cogcomp.lbjava.infer.*;
import edu.illinois.cs.cogcomp.lbjava.io.IOUtilities;
import edu.illinois.cs.cogcomp.lbjava.learn.*;
import edu.illinois.cs.cogcomp.lbjava.parse.*;


public class BagOfWords extends Classifier
{
  public BagOfWords()
  {
    containingPackage = "edu.illinois.cs.cogcomp.lbjava.examples.news";
    name = "BagOfWords";
  }

  public String getInputType() { return "edu.illinois.cs.cogcomp.lbjava.examples.Post"; }
  public String getOutputType() { return "discrete%"; }

  public FeatureVector classify(Object __example)
  {
    if (!(__example instanceof Post))
    {
      String type = __example == null ? "null" : __example.getClass().getName();
      System.err.println("Classifier 'BagOfWords(Post)' defined on line 7 of news.lbj received '" + type + "' as input.");
      new Exception().printStackTrace();
      System.exit(1);
    }

    Post post = (Post) __example;

    FeatureVector __result;
    __result = new FeatureVector();
    String __id;
    String __value;

    for (int i = 0; i < post.bodySize(); ++i)
    {
      for (int j = 0; j < post.lineSize(i); ++j)
      {
        String word = post.getBodyWord(i, j);
        if (word.length() > 0 && word.substring(0, 1).matches("[A-Za-z]"))
        {
          __id = "" + (word);
          __value = "true";
          __result.addFeature(new DiscretePrimitiveStringFeature(this.containingPackage, this.name, __id, __value, valueIndexOf(__value), (short) 0));
        }
      }
    }
    return __result;
  }

  public FeatureVector[] classify(Object[] examples)
  {
    if (!(examples instanceof Post[]))
    {
      String type = examples == null ? "null" : examples.getClass().getName();
      System.err.println("Classifier 'BagOfWords(Post)' defined on line 7 of news.lbj received '" + type + "' as input.");
      new Exception().printStackTrace();
      System.exit(1);
    }

    return super.classify(examples);
  }

  public int hashCode() { return "BagOfWords".hashCode(); }
  public boolean equals(Object o) { return o instanceof BagOfWords; }
}

