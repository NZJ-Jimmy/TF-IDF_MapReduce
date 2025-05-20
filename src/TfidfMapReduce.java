import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.util.GenericOptionsParser;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import org.apache.hadoop.fs.FileSystem;

/**
 * <p>
 * Main driver class for the TF-IDF MapReduce workflow.
 * </p>
 * 
 * <p>
 * This class coordinates the execution of the three-step MapReduce job chain:
 * </p>
 * 
 * <ul>
 * <li>Step 1: <code>WordCount</code> - Count occurrences of each word in each document</li>
 * <li>Step 2: <code>CountTF</code> - Calculate Term Frequency components for each word in each
 * document</li>
 * <li>Step 3: <code>CalcTFIDF</code> - Calculate the final TF-IDF values</li>
 * </ul>
 * 
 * <p>
 * Each step uses the output from the previous step as its input.
 * </p>
 */
public class TfidfMapReduce {
    /**
     * The main method that orchestrates the TF-IDF calculation workflow.
     * 
     * @param args command line arguments: input directory containing documents and output directory
     * @throws Exception if an error occurs during execution
     */
    public static void main(String[] args) throws Exception {
        // Parse input and output paths from command line arguments
        Configuration conf = new Configuration();
        String[] otherArgs = new GenericOptionsParser(conf, args).getRemainingArgs();
        if (otherArgs.length != 2) {
            System.err.println("Usage: tfidfmapreduce <in> <out>");
            System.exit(2);
        }
        Path input = new Path(otherArgs[0]);
        Path output = new Path(otherArgs[1]);

        FileSystem fs = FileSystem.get(conf);

        // Count the total number of documents in the input directory
        // This is required for IDF calculation in the third step
        int totalDocs = fs.listStatus(input).length;
        System.out.println("Total documents: " + totalDocs);

        // // Check if output directory already exists and ask for confirmation to delete it
        if (fs.exists(output)) {
            System.out.println("Output path already exists. Press Enter to delete it or Ctrl+C to cancel.");
            BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
            reader.readLine(); // Wait for Enter
            fs.delete(output, true);
        }

        // Define paths for intermediate and final outputs of each step
        Path wordCountOutput = new Path(output, "WordCount");
        Path countTFOutput = new Path(output, "CountTF");
        Path calcTFIDFOutput = new Path(output, "CalcTFIDF");
        Path sortOutput = new Path(output, "SortedTFIDF");

        // Step 1: Execute WordCount job
        // Counts occurrences of each word in each document
        // Output format: "word|filename -> count"
        Job job1 = WordCount.getJob(conf, input, wordCountOutput);
        
        // Step 2: Execute CountTF job
        // Calculates Term Frequency components
        // Output format: "word|filename -> count|totalWordsInDocument"
        Job job2 = CountTF.getJob(conf, wordCountOutput, countTFOutput);
        
        // Step 3: Execute CalcTFIDF job
        // Calculates final TF-IDF values using IDF = log(totalDocs/docFreq)
        // Output format: "word|filename -> TF-IDF value"
        Job job3 = CalcTFIDF.getJob(conf, countTFOutput, calcTFIDFOutput, totalDocs);

        // Post-processing: Sort the final output by TF-IDF values in descending order
        Job job4 = SortByValue.getJob(conf, calcTFIDFOutput, sortOutput);

        // Execute the jobs in sequence, waiting for each to complete before starting the next
        System.out.println("Starting Step 1: Word Count...");
        job1.waitForCompletion(true);
        
        System.out.println("Starting Step 2: Term Frequency calculation...");
        job2.waitForCompletion(true);
        
        System.out.println("Starting Step 3: TF-IDF calculation...");
        job3.waitForCompletion(true);
        
        System.out.println("Post-processing: Sorting by TF-IDF values...");
        job4.waitForCompletion(true);

        System.out.println("Final sorted TF-IDF output is available at: " + sortOutput.toString());
    }
}
