import java.io.IOException;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.KeyValueTextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;

/**
 * <p>
 * Hadoop MapReduce TF-IDF Calculation Program (Step 3)<br>
 * This program calculates the final TF-IDF value for each word in each
 * document, as the third and final step of the TF-IDF computation process:
 * </p>
 * 
 * <ul>
 * <li>Input comes from Step 2, in the format:
 * <code>word|filename -&gt; count|totalWordsInFile</code></li>
 * <li>Mapped as: <code>word -&gt; filename=count|totalWordsInFile</code></li>
 * <li>Reduced as: <code>word|filename -&gt; TF-IDF value</code></li>
 * </ul>
 * 
 * <p>
 * Where:
 * </p>
 * <ul>
 * <li>TF (Term Frequency) = <code>count/totalWordsInFile</code></li>
 * <li>IDF (Inverse Document Frequency) =
 * <code>log(totalDocuments/numberOfDocumentsContainingWord)</code></li>
 * <li>TF-IDF = <code>TF * IDF</code></li>
 * </ul>
 */
public class CalcTFIDF {
    /**
     * <p>
     * The Mapper class for the final TF-IDF calculation.
     * </p>
     * 
     * <p>
     * This mapper reorganizes the data from Step 2 to group all information about a
     * word across all documents, which allows the reducer to calculate document
     * frequency (DF) and inverse document frequency (IDF) components.
     * </p>
     * 
     * <p>
     * Input Key: <code>Text</code> (format: "word|filename") Input Value:
     * <code>Text</code> (format: "count|totalWordsInDocument")
     * </p>
     * 
     * <p>
     * Output Key: <code>Text</code> (word) Output Value: <code>Text</code> (format:
     * "filename=count|totalWordsInDocument")
     * </p>
     */
    private static class MyMapper extends Mapper<Text, Text, Text, Text> {
        @Override
        public void map(Text key, Text value, Context context) throws IOException, InterruptedException {
            String[] keyParts = key.toString().split("\\|");
            String word = keyParts[0];
            String file = keyParts[1];
            // Output format: word -> "filename=count|totalWordsInFile"
            context.write(new Text(word), new Text(file + "=" + value.toString()));
        }
    }

    /**
     * <p>
     * The Reducer class for the final TF-IDF calculation.
     * </p>
     * 
     * <p>
     * This reducer calculates the TF-IDF value for each word in each document: 1.
     * First collects all document information for the given word 2. Calculates
     * document frequency (DF) - the number of documents containing the word 3.
     * Calculates inverse document frequency (IDF) - log(totalDocuments/DF) 4. For
     * each document, calculates TF-IDF as (count/totalWords) * IDF
     * </p>
     * 
     * <p>
     * Input Key: <code>Text</code> (word) Input Value: <code>Text</code>
     * (collection of "filename=count|totalWordsInDocument")
     * </p>
     * 
     * <p>
     * Output Key: <code>Text</code> (format: "word|filename") Output Value:
     * <code>DoubleWritable</code> (the final TF-IDF value)
     * </p>
     */
    private static class MyReducer extends Reducer<Text, Text, Text, DoubleWritable> {
        private int totalDocs; // Total number of documents (needs to be pre-calculated)

        @Override
        protected void setup(Context context) {
            // Read the total document count from configuration (assumed to be
            // pre-calculated)
            totalDocs = context.getConfiguration().getInt("total.docs", 1);
        }

        @Override
        protected void reduce(Text key, Iterable<Text> values, Context context)
                throws IOException, InterruptedException {
            Map<String, String> fileInfoMap = new HashMap<>();
            Set<String> files = new HashSet<>();

            // Collect information from all files
            for (Text val : values) {
                String[] parts = val.toString().split("=");
                String file = parts[0];
                files.add(file);
                fileInfoMap.put(file, parts[1]);
            }

            // Calculate DF (Document Frequency - number of documents containing the word)
            int df = files.size();
            double idf = Math.log((double) totalDocs / df);

            // Calculate TF-IDF for each file
            for (Map.Entry<String, String> entry : fileInfoMap.entrySet()) {
                String file = entry.getKey();
                String[] countTotal = entry.getValue().split("\\|");
                double tf = (double) Integer.parseInt(countTotal[0]) / Integer.parseInt(countTotal[1]);
                double tfidf = tf * idf;
                context.write(new Text(key.toString() + "|" + file), new DoubleWritable(tfidf));
            }
        }
    }

    /**
     * Configures and returns a new Hadoop Job for calculating the final TF-IDF
     * values. This is the third and final step of the TF-IDF calculation process.
     *
     * @param conf      the Hadoop configuration to use for the job
     * @param input     the input path containing the output from Step 2 (TF
     *                  components)
     * @param output    the output path for storing the final TF-IDF values
     * @param totalDocs the total number of documents in the corpus (needed for IDF
     *                  calculation)
     * @return a configured Job instance for TF-IDF calculation
     */
    public static Job getJob(Configuration conf, Path input, Path output, int totalDocs) throws IOException {
        conf.set("total.docs", String.valueOf(totalDocs));
        Job job = Job.getInstance(conf, "Calculate TF-IDF - Final Step");
        job.setJarByClass(CalcTFIDF.class);
        job.setMapperClass(MyMapper.class);
        job.setInputFormatClass(KeyValueTextInputFormat.class);
        job.setReducerClass(MyReducer.class);
        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(Text.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(DoubleWritable.class);
        FileInputFormat.addInputPath(job, input);
        FileOutputFormat.setOutputPath(job, output);

        return job;
    }

    /**
     * The main method that sets up and runs the MapReduce job for final TF-IDF
     * calculation.
     *
     * @param args command line arguments: input path, output path, and total number
     *             of documents
     * @throws Exception if an error occurs during job execution
     */
    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        String[] otherArgs = new GenericOptionsParser(conf, args).getRemainingArgs();
        if (otherArgs.length != 3) {
            System.err.println("Usage: CalcTFIDF <in> <out> <totalDocs>");
            System.exit(2);
        }
        Job job = getJob(conf, new Path(otherArgs[0]), new Path(otherArgs[1]), Integer.parseInt(otherArgs[2]));
        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}
