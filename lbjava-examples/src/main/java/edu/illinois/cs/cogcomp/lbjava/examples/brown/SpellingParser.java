package edu.illinois.cs.cogcomp.lbjava.examples.brown;

import edu.illinois.cs.cogcomp.lbjava.parse.Parser;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

public class SpellingParser implements Parser {

    private final List<String> lines;
    private int currentLineNumber;

    public SpellingParser(String filePath) {
        this.lines = new ArrayList<>();
        this.currentLineNumber = 0;

        Reader reader;
        try {
            reader = new FileReader(filePath);
            BufferedReader bufferedReader = new BufferedReader(reader);
            String eachLine;
            while ((eachLine = bufferedReader.readLine()) != null) {
                lines.add(eachLine);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public Object next() {
        if (currentLineNumber < lines.size()) {
            SpellingData ret = new SpellingData(lines.get(currentLineNumber));
            this.currentLineNumber ++;
            return ret;
        }
        return null;
    }

    public void close() {
    }

    public void reset() {
        this.currentLineNumber = 0;
    }
}
