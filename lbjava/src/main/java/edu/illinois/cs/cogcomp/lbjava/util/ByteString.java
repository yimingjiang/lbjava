package edu.illinois.cs.cogcomp.lbjava.util;

import java.util.Arrays;


/**
  * Represents a <code>String</code> by directly storing an encoding of that
  * <code>String</code> in an array of <code>byte</code>s.  This can save a
  * lot of memory if all of the application's characters fit in a single byte
  * when encoded by, for instance, UTF-8.  In fact, the default encoding used
  * by this class is UTF-8.
  *
  * @author Nick Rizzolo
 **/
public class ByteString implements Cloneable, Comparable
{
  /** The default character encoding for instances of this class. */
  public static final String defaultEncoding = "UTF-8";
  /** A byte string representing <code>""</code>. */
  public static final ByteString emptyString =
    new ByteString("", defaultEncoding);


  /**
    * Handles exceptions generated by unsupported encodings.
    *
    * @param e  The exception.
   **/
  protected void handleEncodingException(Exception e) {
    System.err.println(
        "ERROR: Encoding \"" + encoding + "\" is not supported.");
    e.printStackTrace();
    System.exit(1);
  }


  /**
    * Reads and returns a byte string from an input stream.
    *
    * @param in The input stream.
    * @return The byte string.
   **/
  public static ByteString readByteString(ExceptionlessInputStream in) {
    ByteString result = new ByteString(false);
    result.read(in);
    return result;
  }


  /**
    * Reads and returns a byte string as written by a lexicon.
    *
    * @param in The input stream.
    * @param i  The assumed identifier.  If no identifier is given in the
    *           input stream, the instantiated feature is given this
    *           identifier.
    * @return The byte string.
   **/
  public static ByteString lexReadByteString(ExceptionlessInputStream in,
                                             ByteString i) {
    ByteString result = new ByteString(false);
    result.lexRead(in, i);
    return result;
  }


  /** The encoding method used by this instance. */
  protected String encoding;
  /** The encoded characters. */
  protected byte[] value;
  /**
    * The hash code of the <code>String</code> decoding of this byte string.
   **/
  protected int hashCode;


  /**
    * For internal use only.
    *
    * @param b  Dummy variable to make a new signature.
   **/
  protected ByteString(boolean b) { }

  /** Creates an empty byte string. */
  public ByteString() { this(""); }

  /**
    * Creates a byte string by using the default encoding to encode the
    * specified string.
    *
    * @param s  The string to encode.
   **/
  public ByteString(String s) { this(s, null); }

  /**
    * Creates a byte string by using the specified encoding to encode the
    * specified string.
    *
    * @param s  The string to encode.
    * @param e  The encoding method.
   **/
  public ByteString(String s, String e) {
    encoding = e == null ? defaultEncoding : e.intern();
    setValue(s);
  }

  /**
    * Creates a byte string with the given encoding, which may involve
    * converting the specified byte string's contents if the encodings differ.
    *
    * @param b  The original byte string.
    * @param e  The new encoding.
   **/
  public ByteString(ByteString b, String e) {
    encoding = e.intern();
    if (b.encoding == encoding) {
      value = b.value;
      hashCode = b.hashCode;
    }
    else setValue(b.toString());
  }


  /** Returns the name of the encoding method of this byte string. */
  public String getEncoding() { return encoding; }


  /**
    * Sets the value of this byte string to the byte encoding of the specified
    * string.
    *
    * @param s  The string to encode.
   **/
  public void setValue(String s) {
    try { value = s.getBytes(encoding); }
    catch (Exception e) { handleEncodingException(e); }
    hashCode = s.hashCode();
  }


  /** Returns the length of {@link #value}. */
  public int length() { return value.length; }


  /**
    * Returns the byte at index <code>i</code> of {@link #value}.
    *
    * @param i  The index of the requested byte.
    * @return The value of the requested byte.
   **/
  public byte getByte(int i) { return value[i]; }


  /**
    * Appends the encoding of the given string onto the existing encoding in
    * this object.  This operation changes the {@link #value} reference in
    * this object.
    *
    * <p> <b>Warning:</b> Depending on the character encoding in use, this may
    * introduce byte order markers into the middle of this object's byte
    * array, which usually is not desired.
    *
    * @param s  The string whose encoding will be appended.
    * @return This object.
   **/
  public ByteString append(String s) {
    hashCode = (toString() + s).hashCode();
    byte[] v = null;
    try { v = s.getBytes(encoding); }
    catch (Exception e) { handleEncodingException(e); }

    byte[] t = new byte[value.length + v.length];
    System.arraycopy(value, 0, t, 0, value.length);
    System.arraycopy(v, 0, t, value.length, v.length);
    value = t;
    return this;
  }


  /**
    * Appends the encodings of all the given strings onto the existing
    * encoding in this object.  This operation changes the {@link #value}
    * reference in this object.
    *
    * <p> <b>Warning:</b> Depending on the character encoding in use, this may
    * introduce byte order markers into the middle of this object's byte
    * array, which usually is not desired.
    *
    * @param s  The strings whose encodings will be appended.
    * @return This object.
   **/
  public ByteString append(String[] s) {
    StringBuffer buffer = new StringBuffer(toString());
    for (int i = 0; i < s.length; ++i) buffer.append(s[i]);
    hashCode = buffer.toString().hashCode();

    byte[][] v = new byte[s.length][];
    int length = 0;
    try {
      for (int i = 0; i < v.length; ++i) {
        v[i] = s[i].getBytes(encoding);
        length += v[i].length;
      }
    }
    catch (Exception e) { handleEncodingException(e); }

    byte[] t = new byte[length];
    length = value.length;
    System.arraycopy(value, 0, t, 0, length);
    for (int i = 0; i < v.length; ++i) {
      System.arraycopy(v[i], 0, t, length, v[i].length);
      length += v[i].length;
    }

    value = t;
    return this;
  }


  /**
    * Appends the string represented by the given byte string onto the
    * existing content in this object.  This operation changes the
    * {@link #value} reference in this object.
    *
    * <p> <b>Warning:</b> Depending on the character encoding in use, this may
    * introduce byte order markers into the middle of this object's byte
    * array, which usually is not desired.
    *
    * @param b  The string whose encoding will be appended.
    * @return This object.
   **/
  public ByteString append(ByteString b) {
    String s = b.toString();
    hashCode = (toString() + s).hashCode();
    if (encoding != b.encoding) return append(s);
    byte[] t = new byte[value.length + b.value.length];
    System.arraycopy(value, 0, t, 0, value.length);
    System.arraycopy(b.value, 0, t, value.length, b.value.length);
    value = t;
    return this;
  }


  /**
    * Appends the strings represented by the given byte strings onto the
    * existing content in this object.  This operation changes the
    * {@link #value} reference in this object.
    *
    * <p> <b>Warning:</b> Depending on the character encoding in use, this may
    * introduce byte order markers into the middle of this object's byte
    * array, which usually is not desired.
    *
    * @param b  The strings whose encodings will be appended.
    * @return This object.
   **/
  public ByteString append(ByteString[] b) {
    int length = 0;
    StringBuffer buffer = new StringBuffer(toString());
    for (int i = 0; i < b.length; ++i) {
      String s = b[i].toString();
      buffer.append(s);
      if (encoding != b[i].encoding)
        b[i] = new ByteString(s, encoding);
      length += b[i].value.length;
    }
    hashCode = buffer.toString().hashCode();

    byte[] t = new byte[length];
    length = value.length;
    System.arraycopy(value, 0, t, 0, length);
    for (int i = 0; i < b.length; ++i) {
      System.arraycopy(b[i].value, 0, t, length, b[i].value.length);
      length += b[i].value.length;
    }

    value = t;
    return this;
  }


  /**
    * If the argument object is a byte string, this object's byte array and
    * the argument object's byte array are compared lexicographically.
    * Otherwise, -1 is returned.  Of course, this operation is considerably
    * more expensive if the two strings do not share the same encoding.
   **/
  public int compareTo(Object o) {
    if (!(o instanceof ByteString)) return -1;
    ByteString b = (ByteString) o;
    if (encoding != b.encoding) return toString().compareTo(b.toString());

    int n1 = value.length;
    int n2 = b.value.length;
    int n = Math.min(n1, n2);

    for (int i = 0; i < n; ++i) {
      byte b1 = value[i];
      byte b2 = b.value[i];
      if (b1 != b2) return b1 - b2;
    }

    return n1 - n2;
  }


  /** Returns a hash code for this object. */
  public int hashCode() { return hashCode; }


  /**
    * Two byte strings are equivalent if they encode the same string.  This
    * operation is more expensive if the two byte strings use different
    * encodings.
   **/
  public boolean equals(Object o) {
    if (o instanceof String) return toString().equals(o);
    if (!(o instanceof ByteString)) return false;
    ByteString b = (ByteString) o;
    if (encoding != b.encoding) return toString().equals(b.toString());
    if (value.length != b.value.length) return false;

    for (int i = 0; i < value.length; ++i)
      if (value[i] != b.value[i]) return false;
    return true;
  }


  /**
    * Writes a complete binary representation of this byte string.
    *
    * @param out  The output stream.
   **/
  public void write(ExceptionlessOutputStream out) {
    out.writeString(encoding);
    out.writeInt(hashCode);
    out.writeBytes(value);
  }


  /**
    * Reads in a complete binary representation of a byte string.
    *
    * @param in The input stream.
   **/
  public void read(ExceptionlessInputStream in) {
    encoding = in.readString().intern();
    hashCode = in.readInt();
    value = in.readBytes();
  }


  /**
    * Writes a binary representation of this byte string intended for use by
    * a lexicon, omitting redundant information when possible.
    *
    * @param out  The output stream.
    * @param i    The assumed identifier string.  This byte strings value,
    *             encoding, or both may be omitted if they are equivalent to
    *             <code>i</code>.
   **/
  public void lexWrite(ExceptionlessOutputStream out, ByteString i) {
    if (i != null && encoding == i.encoding && Arrays.equals(value, i.value))
      out.writeBytes(null);
    else {
      out.writeBytes(value);
      out.writeInt(hashCode);
      out.writeString(i != null && encoding == i.encoding ? null : encoding);
    }
  }


  /**
    * Reads the representation of a byte string as stored by a lexicon,
    * overwriting the data in this object.
    *
    * <p> This method is appropriate for reading byte strings as written by
    * {@link #lexWrite(ExceptionlessOutputStream,ByteString)}.
    *
    * @param in The input stream.
    * @param i  The assumed identifier string.
   **/
  public void lexRead(ExceptionlessInputStream in, ByteString i) {
    value = in.readBytes();
    if (value == null) {
      value = i.value;
      hashCode = i.hashCode;
      encoding = i.encoding;
    }
    else {
      hashCode = in.readInt();
      encoding = in.readString();
      if (encoding == null) encoding = i.encoding;
      else encoding = encoding.intern();
    }
  }


  /** Returns a decoded string. */
  public String toString() {
    try { return new String(value, encoding); }
    catch (Exception e) { handleEncodingException(e); }
    return null;
  }


  /**
    * Returns a shallow copy of this string.  Note that this class does not
    * provide any operations that modify the contents of the objects
    * referenced by its fields, making a deep clone unnecessary.
    * ({@link #append(String)}, {@link #append(ByteString)}, and
    * {@link #setValue(String)} modify the {@link #value} field itself, but
    * the reference is merely replaced; the contents of the original array do
    * not change.)
   **/
  public Object clone() {
    Object result = null;

    try { result = super.clone(); }
    catch (Exception e) {
      System.err.println("Can't clone byte string '" + this + "':");
      e.printStackTrace();
    }

    return result;
  }
}

