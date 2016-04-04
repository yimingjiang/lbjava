package edu.illinois.cs.cogcomp.lbjava.examples;

import edu.illinois.cs.cogcomp.lbjava.parse.LineByLine;


/**
  * This parser takes a list of file names containing newsgroup posts as input
  * and returns {@link Post} objects representing those posts.  The list of
  * file names is provided to this parser in a file containing one name per
  * line.  Each name should include at least one subdirectory, since the
  * subdirectory containing the newsgroup post is taken as its label.  For
  * example, the contents of the input file might look like this:
  *
  * <p>
  * <pre>
  * data/20news/alt.atheism/49960
  * data/20news/comp.graphics/51060
  * data/20news/talk.politics.misc/51119
  * data/20news/talk.religion.misc/51120
  * </pre>
 **/
public class NewsgroupParser extends LineByLine
{
  /**
    * Constructor.
    *
    * @param file The name of file containing names of files that contain one
    *             news group post each.
   **/
  public NewsgroupParser(String file) { super(file); }


  /**
    * Returns the next {@link Post} object representing a newsgroup post from
    * the named file, or <code>null</code> if no more remain.
   **/
  public Object next() {
    String file = readLine();
    if (file == null) return null;
    return new Post(file);
  }
}

