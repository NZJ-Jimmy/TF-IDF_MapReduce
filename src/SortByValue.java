import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.io.WritableComparator;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.KeyValueTextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;

/**
 * A MapReduce job that sorts key-value pairs by their value in descending order.
 * 
 * This can be used as a post-processing step after the TF-IDF calculation to sort
 * the results by TF-IDF scores, showing the most important terms first.
 */
public class SortByValue {
    /**
     * Mapper class that swaps the key and value to enable sorting by value.
     * 
     * <p>
     * The mapper takes the original key-value pairs (word|filename, TF-IDF value)
     * and outputs (TF-IDF value, word|filename) to allow sorting by TF-IDF value.
     * </p>
     * 
     * <p>
     * Input Key: <code>Text</code> (format: "word|filename")
     * Input Value: <code>Text</code> (the TF-IDF value as a string)
     * </p>
     * 
     * <p>
     * Output Key: <code>DoubleWritable</code> (the TF-IDF value as a double)
     * Output Value: <code>Text</code> (format: "word|filename")
     * </p>
     */
    public static class SortMapper extends Mapper<Text, Text, DoubleWritable, Text> {
        public void map(Text key, Text value, Context context) throws IOException, InterruptedException {
            // Key = 原始的 key（如单词），Value = 原始的 value（如 2.4 等）
            double val = Double.parseDouble(value.toString());
            context.write(new DoubleWritable(val), key); // 输出：value 作为 key，实现排序
        }
    }

    /**
     * Custom comparator class that sorts DoubleWritable keys in descending order.
     * 
     * <p>
     * This enables sorting the TF-IDF values from highest to lowest,
     * presenting the most significant terms first in the results.
     * </p>
     */
    public static class DescendingDoubleComparator extends WritableComparator {
        protected DescendingDoubleComparator() {
            super(DoubleWritable.class, true);
        }

        @Override
        public int compare(WritableComparable a, WritableComparable b) {
            DoubleWritable d1 = (DoubleWritable) a;
            DoubleWritable d2 = (DoubleWritable) b;
            return -1 * d1.compareTo(d2); // 降序
        }
    }

    /**
     * Reducer class that flips the key and value back to the original order.
     * 
     * <p>
     * After sorting is complete, this reducer restores the original format
     * (word|filename, TF-IDF value), but now the results are ordered by TF-IDF value.
     * </p>
     * 
     * <p>
     * Input Key: <code>DoubleWritable</code> (the TF-IDF value)
     * Input Value: <code>Text</code> (format: "word|filename")
     * </p>
     * 
     * <p>
     * Output Key: <code>Text</code> (format: "word|filename")
     * Output Value: <code>DoubleWritable</code> (the TF-IDF value)
     * </p>
     */
    public static class SortReducer extends Reducer<DoubleWritable, Text, Text, DoubleWritable> {
        public void reduce(DoubleWritable key, Iterable<Text> values, Context context)
                throws IOException, InterruptedException {
            for (Text val : values) {
                context.write(val, key); // 恢复输出为 (Key, Value)
            }
        }
    }

    /**
     * Configures and returns a new Hadoop Job for sorting the TF-IDF results by value.
     * 
     * <p>
     * This job uses a key-value swap technique: values become keys for sorting,
     * then are swapped back to the original format after sorting is complete.
     * </p>
     *
     * @param conf   the Hadoop configuration to use for the job
     * @param input  the input path containing the TF-IDF results to sort
     * @param output the output path for storing the sorted results
     * @return a configured Job instance for sorting
     */
    public static Job getJob(Configuration conf, Path input, Path output) throws IOException {
        Job job = Job.getInstance(conf, "Sort TF-IDF Results by Value (Descending)");
        job.setJarByClass(SortByValue.class);
        job.setMapperClass(SortMapper.class);
        job.setInputFormatClass(KeyValueTextInputFormat.class);
        job.setReducerClass(SortReducer.class);
        job.setSortComparatorClass(DescendingDoubleComparator.class);  // Use custom comparator for descending sort
        job.setMapOutputKeyClass(DoubleWritable.class);
        job.setMapOutputValueClass(Text.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(DoubleWritable.class);
        FileInputFormat.addInputPath(job, input);
        FileOutputFormat.setOutputPath(job, output);

        return job;
    }

    /**
     * The main method to run the sort job.
     *
     * @param args command line arguments: input path and output path
     * @throws Exception if an error occurs during job execution
     */
    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        String[] otherArgs = new GenericOptionsParser(conf, args).getRemainingArgs();
        if (otherArgs.length != 2) {
            System.err.println("Usage: SortByValue <in> <out>");
            System.exit(2);
        }
        Job job = getJob(conf, new Path(otherArgs[0]), new Path(otherArgs[1]));
        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}
