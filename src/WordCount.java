import java.io.IOException;
import java.util.StringTokenizer;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;

/**
 * <p>
 * A Hadoop MapReduce program to count the occurrences of each word in each
 * document.<br>
 * This is Step 1 of the TF-IDF computation process:
 * </p>
 * 
 * <ul>
 * <li>Maps each word to <code>word|filename -&gt; 1</code></li>
 * <li>Reduces to
 * <code>word|filename -&gt; count of word in that file</code></li>
 * </ul>
 * 
 * <p>
 * The output of this step will be used as input for calculating the Term
 * Frequency (TF) in the next step of the TF-IDF computation.
 * </p>
 */
public class WordCount {

    /**
     * <p>
     * <code>MyMapper</code> is a <code>Mapper</code> class that extends the Hadoop
     * Mapper class. It maps each word from the input documents to a key-value pair
     * where the key is "word|filename" and the value is 1.
     * </p>
     * 
     * <p>
     * This is the first step in TF-IDF calculation, identifying each word
     * occurrence in each document to count later.
     * </p>
     * 
     * <p>
     * Key: <code>Object</code> (position in file, not used in this implementation)
     * Value: <code>Text</code> (a line of text from the input document)
     * </p>
     * 
     * <p>
     * Output Key: <code>Text</code> (format: "word|filename") Output Value:
     * <code>IntWritable</code> (always 1 for each occurrence)
     * </p>
     */
    private static class MyMapper extends Mapper<Object, Text, Text, IntWritable> {
        private final static IntWritable ONE = new IntWritable(1);
        private Text word = new Text();

        private StringTokenizer standardize_token(String token) {
            // Remove a word containing any digit
            token = token.replaceAll(".*\\d.*", "");

            // Replace common HTML entities
            token = token.replace("&amp;", "&");
            token = token.replace("&lt;", "<");
            token = token.replace("&gt;", ">");
            token = token.replace("&quot;", "\"");
            token = token.replace("&apos;", "'");
            token = token.replace("&nbsp;", " ");

            // Remove HTML tags
            token = token.replaceAll("<[^>]+>", "");

            // Remove punctuations at the beginning and end of the token
            token = token.replaceAll("^[\\pP\\$\\+\\-\\=\\<\\>]+", "");
            token = token.replaceAll("[\\pP\\$\\+\\-\\=\\<\\>]+$", "");

            // Convert to lowercase
            token = token.toLowerCase();

            StringTokenizer st = new StringTokenizer(token);
            return st;
        }

        @Override
        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            String fileName = ((FileSplit) context.getInputSplit()).getPath().getName();
            StringTokenizer itr = new StringTokenizer(value.toString());
            while (itr.hasMoreTokens()) {
                String token = itr.nextToken();
                // Standardize the token
                StringTokenizer st = standardize_token(token);
                while (st.hasMoreTokens()) {
                    token = st.nextToken();
                    word.set(token + "|" + fileName);
                    context.write(word, ONE); // word|filename, 1
                }
            }
        }
    }

    /**
     * <p>
     * A Reducer class that aggregates the counts of each word in each document.
     * </p>
     * 
     * <p>
     * For Step 1 of TF-IDF calculation, this reducer sums up the occurrences of
     * each word within each specific document. The output will be used to calculate
     * Term Frequency (TF) in the next step.
     * </p>
     * 
     * <p>
     * Input Key: <code>Text</code> (format: "word|filename") Input Value:
     * <code>IntWritable</code> (count of 1 for each occurrence)
     * </p>
     * 
     * <p>
     * Output Key: <code>Text</code> (format: "word|filename") Output Value:
     * <code>IntWritable</code> (total count of the word in that file)
     * </p>
     */
    private static class MyReducer extends Reducer<Text, IntWritable, Text, IntWritable> {
        private IntWritable result = new IntWritable();

        @Override
        public void reduce(Text key, Iterable<IntWritable> values, Context context)
                throws IOException, InterruptedException {
            int sum = 0;
            for (IntWritable val : values) {
                sum += val.get();
            }
            result.set(sum);
            context.write(key, result); // word|filename, count - Result of Step 1 for TF-IDF
        }
    }

    /**
     * Configures and returns a new Hadoop Job for counting words in documents. This
     * is the first step of the TF-IDF calculation process.
     *
     * @param conf   the Hadoop configuration to use for the job
     * @param input  the input path containing the documents to process
     * @param output the output path for storing word counts per document
     * @return a configured Job instance for word counting
     */
    public static Job getJob(Configuration conf, Path input, Path output) throws IOException {
        Job job = Job.getInstance(conf, "Word Count - TF-IDF Step 1");
        job.setJarByClass(WordCount.class);
        job.setMapperClass(MyMapper.class);
        job.setReducerClass(MyReducer.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);
        FileInputFormat.addInputPath(job, input);
        FileOutputFormat.setOutputPath(job, output);
        return job;
    }

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        String[] otherArgs = new GenericOptionsParser(conf, args).getRemainingArgs();
        if (otherArgs.length != 2) {
            System.err.println("Usage: wordcount <in> <out>");
            System.exit(2);
        }
        Job job = getJob(conf, new Path(otherArgs[0]), new Path(otherArgs[1]));
        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}