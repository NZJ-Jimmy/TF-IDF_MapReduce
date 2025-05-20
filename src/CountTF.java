import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
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
 * Hadoop MapReduce Program for Term Frequency (TF) Calculation <br>
 * This is Step 2 of the TF-IDF computation process:
 * </p>
 * 
 * <ul>
 * <li>Takes input from Step 1: <code>word|filename -&gt; count</code></li>
 * <li>Maps to <code>filename -&gt; word|count</code></li>
 * <li>Reduces to <code>word|filename -&gt; count|totalWordsInFile</code></li>
 * </ul>
 * 
 * <p>
 * The output of this step will be used as input for calculating the final
 * TF-IDF in the next step of the TF-IDF computation.
 * </p>
 */
public class CountTF {
    /**
     * <p>
     * The Mapper class for Term Frequency calculation.
     * </p>
     * 
     * <p>
     * This mapper transforms the word count data from Step 1 into a format grouped
     * by document, allowing the reducer to calculate the total word count for each
     * document.
     * </p>
     * 
     * <p>
     * Input Key: <code>Text</code> (format: "word|filename") Input Value:
     * <code>Text</code> (count of word in that file)
     * </p>
     * 
     * <p>
     * Output Key: <code>Text</code> (filename) Output Value: <code>Text</code>
     * (format: "word|count")
     * </p>
     */
    private static class MyMapper extends Mapper<Text, Text, Text, Text> {
        @Override
        public void map(Text key, Text value, Context context) throws IOException, InterruptedException {
            String[] parts = key.toString().split("\\|");
            String word = parts[0];
            String fileName = parts[1];
            context.write(new Text(fileName), new Text(word + "|" + value)); // filename, word|count
        }
    }

    /**
     * <p>
     * The Reducer class for Term Frequency calculation.
     * </p>
     * 
     * <p>
     * This reducer calculates the total number of words in each document and then
     * outputs the Term Frequency components for each word in that document.
     * </p>
     * 
     * <p>
     * Input Key: <code>Text</code> (filename) Input Value: <code>Text</code>
     * (collection of "word|count" pairs)
     * </p>
     * 
     * <p>
     * Output Key: <code>Text</code> (format: "word|filename") Output Value:
     * <code>Text</code> (format: "count|totalWordsInDocument") This is effectively
     * storing the components needed for TF calculation, where TF =
     * count/totalWordsInDocument
     * </p>
     */
    private static class MyReducer extends Reducer<Text, Text, Text, Text> {
        @Override
        protected void reduce(Text key, Iterable<Text> values, Context context)
                throws IOException, InterruptedException {
            int totalWords = 0;
            List<String> wordCounts = new ArrayList<>();

            // The first pass: calculate total words in the document
            for (Text val : values) {
                String[] parts = val.toString().split("\\|");
                int count = Integer.parseInt(parts[1]);
                totalWords += count;
                wordCounts.add(val.toString());
            }

            // The second pass: output "word|filename -> count|totalWordsInDocument"
            for (String wordCount : wordCounts) {
                String[] parts = wordCount.split("\\|");
                String word = parts[0];
                int count = Integer.parseInt(parts[1]);
                context.write(new Text(word + "|" + key.toString()), new Text(count + "|" + totalWords));
            }
        }
    }

    /**
     * Configures and returns a new Hadoop Job for calculating Term Frequency
     * components. This is the second step of the TF-IDF calculation process.
     *
     * @param conf   the Hadoop configuration to use for the job
     * @param input  the input path containing the output from Step 1 (word counts
     *               per document)
     * @param output the output path for storing Term Frequency components
     * @return a configured Job instance for TF calculation
     */
    public static Job getJob(Configuration conf, Path input, Path output) throws IOException {
        Job job = Job.getInstance(conf, "Count TF - TF-IDF Step 2");
        job.setJarByClass(CountTF.class);
        job.setMapperClass(MyMapper.class);
        job.setInputFormatClass(KeyValueTextInputFormat.class);
        job.setReducerClass(MyReducer.class);
        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(Text.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);
        FileInputFormat.addInputPath(job, input);
        FileOutputFormat.setOutputPath(job, output);
        return job;
    }

    /**
     * The main method that sets up and runs the MapReduce job for TF calculation.
     *
     * @param args command line arguments: input and output paths
     * @throws Exception if an error occurs during job execution
     */
    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        String[] otherArgs = new GenericOptionsParser(conf, args).getRemainingArgs();
        if (otherArgs.length != 2) {
            System.err.println("Usage: CountTF <in> <out>");
            System.exit(2);
        }
        Job job = getJob(conf, new Path(otherArgs[0]), new Path(otherArgs[1]));
        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}
