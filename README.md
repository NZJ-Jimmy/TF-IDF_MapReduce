# MapReduce TF-IDF统计实验报告

## 1. 实验目的

本实验旨在通过Hadoop MapReduce框架实现TF-IDF（Term Frequency-Inverse Document Frequency）算法，深入理解MapReduce编程模型和分布式计算原理。通过对大量文本文档进行处理，计算文档中词语的重要性，最终按照重要性排序输出结果。

## 2. 实验原理

TF-IDF是一种用于信息检索与文本挖掘的常用加权技术，用于评估一个词语对于一个文档集或语料库中某一文档的重要程度。其计算原理如下：

- **词频（Term Frequency，TF）**：衡量一个词语在文档中出现的频率。
  - TF(t,d) = （词语t在文档d中出现的次数）/（文档d中的总词数）

- **逆文档频率（Inverse Document Frequency，IDF）**：衡量一个词语的普遍重要性。
  - IDF(t) = log(总文档数 / 包含词语t的文档数)

- **TF-IDF**：TF与IDF的乘积。
  - TF-IDF(t,d) = TF(t,d) * IDF(t)

TF-IDF的值越高，表明该词语对于当前文档越重要，而在整个语料库中的普遍性越低。

## 3. 实验环境

- **操作系统**：Linux
- **开发工具**：Java
- **计算框架**：Hadoop MapReduce
- **数据集**：100份英文文本文档

## 4. 实现思路

TF-IDF的MapReduce实现分为三个主要步骤，每个步骤都是一个独立的MapReduce作业：

### 步骤一：词频统计（WordCount）

- **目标**：统计每个文档中每个单词出现的次数
- **输入**：原始文本文档
- **输出格式**：`word|filename -> count`
- **Map阶段**：
  - 提取文档名
  - 分词并标准化（去除数字、HTML标签、标点符号等）
  - 输出`<word|filename, 1>`键值对
- **Reduce阶段**：
  - 对同一个`word|filename`的键值对，对值求和
  - 输出`<word|filename, count>`

### 步骤二：计算TF（CountTF）

- **目标**：计算每个文档的总词数，为TF计算做准备
- **输入**：步骤一的输出
- **输出格式**：`word|filename -> count|totalWordsInDocument`
- **Map阶段**：
  - 将步骤一的输出转换为`<filename, word|count>`格式
- **Reduce阶段**：
  - 计算每个文档的总词数
  - 输出每个词的频率计算组件`<word|filename, count|totalWordsInDocument>`

### 步骤三：计算TF-IDF（CalcTFIDF）

- **目标**：计算每个单词在每个文档中的TF-IDF值
- **输入**：步骤二的输出
- **输出格式**：`word|filename -> TF-IDF value`
- **Map阶段**：
  - 将步骤二的输出转换为`<word, filename=count|totalWordsInDocument>`格式
- **Reduce阶段**：
  - 计算包含该单词的文档数（DF）
  - 计算IDF = log(totalDocs/DF)
  - 计算TF = count/totalWordsInDocument
  - 计算并输出TF-IDF值`<word|filename, TF*IDF>`

### 后处理：结果排序（SortByValue）

- **目标**：按照TF-IDF值降序排列结果
- **输入**：步骤三的输出
- **输出格式**：`word|filename -> TF-IDF value`（排序后）
- **Map阶段**：
  - 将键值对交换为`<TF-IDF value, word|filename>`
- **Reduce阶段**：
  - 使用自定义比较器进行降序排序
  - 恢复原始格式输出`<word|filename, TF-IDF value>`

## 5. 代码实现

### 5.1 主控类（TfidfMapReduce.java）

```java
public class TfidfMapReduce {
    public static void main(String[] args) throws Exception {
        // 解析输入和输出路径
        Configuration conf = new Configuration();
        String[] otherArgs = new GenericOptionsParser(conf, args).getRemainingArgs();
        Path input = new Path(otherArgs[0]);
        Path output = new Path(otherArgs[1]);

        // 计算总文档数（IDF计算需要）
        FileSystem fs = FileSystem.get(conf);
        int totalDocs = fs.listStatus(input).length;
        System.out.println("Total documents: " + totalDocs);

        // 定义各步骤的输出路径
        Path wordCountOutput = new Path(output, "WordCount");
        Path countTFOutput = new Path(output, "CountTF");
        Path calcTFIDFOutput = new Path(output, "CalcTFIDF");
        Path sortOutput = new Path(output, "SortedTFIDF");

        // 步骤1: 执行WordCount作业
        Job job1 = WordCount.getJob(conf, input, wordCountOutput);
        
        // 步骤2: 执行CountTF作业
        Job job2 = CountTF.getJob(conf, wordCountOutput, countTFOutput);
        
        // 步骤3: 执行CalcTFIDF作业
        Job job3 = CalcTFIDF.getJob(conf, countTFOutput, calcTFIDFOutput, totalDocs);

        // 后处理: 对结果按TF-IDF值降序排序
        Job job4 = SortByValue.getJob(conf, calcTFIDFOutput, sortOutput);

        // 按顺序执行作业
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
```

### 5.2 词频统计（WordCount.java）

```java
public class WordCount {
    private static class MyMapper extends Mapper<Object, Text, Text, IntWritable> {
        private final static IntWritable ONE = new IntWritable(1);
        private Text word = new Text();

        private StringTokenizer standardize_token(String token) {
            // 移除包含数字的词
            token = token.replaceAll(".*\\d.*", "");
            // 替换HTML实体
            token = token.replace("&amp;", "&");
            // 移除HTML标签
            token = token.replaceAll("<[^>]+>", "");
            // 移除词首尾的标点符号
            token = token.replaceAll("^[\\pP\\$\\+\\-\\=\\<\\>]+", "");
            token = token.replaceAll("[\\pP\\$\\+\\-\\=\\<\\>]+$", "");
            // 转为小写
            token = token.toLowerCase();
            return new StringTokenizer(token);
        }

        @Override
        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            String fileName = ((FileSplit) context.getInputSplit()).getPath().getName();
            StringTokenizer itr = new StringTokenizer(value.toString());
            while (itr.hasMoreTokens()) {
                String token = itr.nextToken();
                StringTokenizer st = standardize_token(token);
                while (st.hasMoreTokens()) {
                    token = st.nextToken();
                    word.set(token + "|" + fileName);
                    context.write(word, ONE); // 输出格式: word|filename -> 1
                }
            }
        }
    }

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
            context.write(key, result); // 输出格式: word|filename -> count
        }
    }
}
```

### 5.3 计算TF（CountTF.java）

```java
public class CountTF {
    private static class MyMapper extends Mapper<Text, Text, Text, Text> {
        @Override
        public void map(Text key, Text value, Context context) throws IOException, InterruptedException {
            String[] parts = key.toString().split("\\|");
            String word = parts[0];
            String fileName = parts[1];
            context.write(new Text(fileName), new Text(word + "|" + value)); // 输出: filename -> word|count
        }
    }

    private static class MyReducer extends Reducer<Text, Text, Text, Text> {
        @Override
        protected void reduce(Text key, Iterable<Text> values, Context context)
                throws IOException, InterruptedException {
            int totalWords = 0;
            List<String> wordCounts = new ArrayList<>();

            // 第一次遍历: 计算文档总词数
            for (Text val : values) {
                String[] parts = val.toString().split("\\|");
                int count = Integer.parseInt(parts[1]);
                totalWords += count;
                wordCounts.add(val.toString());
            }

            // 第二次遍历: 输出 "word|filename -> count|totalWordsInDocument"
            for (String wordCount : wordCounts) {
                String[] parts = wordCount.split("\\|");
                String word = parts[0];
                int count = Integer.parseInt(parts[1]);
                context.write(new Text(word + "|" + key.toString()), new Text(count + "|" + totalWords));
            }
        }
    }
}
```

### 5.4 计算TF-IDF（CalcTFIDF.java）

```java
public class CalcTFIDF {
    private static class MyMapper extends Mapper<Text, Text, Text, Text> {
        @Override
        public void map(Text key, Text value, Context context) throws IOException, InterruptedException {
            String[] keyParts = key.toString().split("\\|");
            String word = keyParts[0];
            String file = keyParts[1];
            // 输出: word -> "filename=count|totalWordsInFile"
            context.write(new Text(word), new Text(file + "=" + value.toString()));
        }
    }

    private static class MyReducer extends Reducer<Text, Text, Text, DoubleWritable> {
        private int totalDocs;

        @Override
        protected void setup(Context context) {
            totalDocs = context.getConfiguration().getInt("total.docs", 1);
        }

        @Override
        protected void reduce(Text key, Iterable<Text> values, Context context)
                throws IOException, InterruptedException {
            Map<String, String> fileInfoMap = new HashMap<>();
            Set<String> files = new HashSet<>();

            // 收集所有文件信息
            for (Text val : values) {
                String[] parts = val.toString().split("=");
                String file = parts[0];
                files.add(file);
                fileInfoMap.put(file, parts[1]);
            }

            // 计算DF和IDF
            int df = files.size();
            double idf = Math.log((double) totalDocs / df);

            // 计算每个文件的TF-IDF
            for (Map.Entry<String, String> entry : fileInfoMap.entrySet()) {
                String file = entry.getKey();
                String[] countTotal = entry.getValue().split("\\|");
                double tf = (double) Integer.parseInt(countTotal[0]) / Integer.parseInt(countTotal[1]);
                double tfidf = tf * idf;
                context.write(new Text(key.toString() + "|" + file), new DoubleWritable(tfidf));
            }
        }
    }
}
```

### 5.5 排序结果（SortByValue.java）

```java
public class SortByValue {
    public static class SortMapper extends Mapper<Text, Text, DoubleWritable, Text> {
        public void map(Text key, Text value, Context context) throws IOException, InterruptedException {
            double val = Double.parseDouble(value.toString());
            context.write(new DoubleWritable(val), key); // 输出: TF-IDF值作为key，实现排序
        }
    }

    public static class DescendingDoubleComparator extends WritableComparator {
        protected DescendingDoubleComparator() {
            super(DoubleWritable.class, true);
        }

        @Override
        public int compare(WritableComparable a, WritableComparable b) {
            DoubleWritable d1 = (DoubleWritable) a;
            DoubleWritable d2 = (DoubleWritable) b;
            return -1 * d1.compareTo(d2); // 降序排序
        }
    }

    public static class SortReducer extends Reducer<DoubleWritable, Text, Text, DoubleWritable> {
        public void reduce(DoubleWritable key, Iterable<Text> values, Context context)
                throws IOException, InterruptedException {
            for (Text val : values) {
                context.write(val, key); // 恢复原始输出格式
            }
        }
    }
}
```

## 6. 实验结果

TF-IDF统计完成后，结果按TF-IDF值降序排列，部分顶部结果如下：

```
microgreens|502053.txt	0.08014301841071711
poktmon|732053.txt	0.058491978668294864
allowances|618053.txt	0.050960911619462375
kac|366053.txt	0.048217674038663366
smiggle|313053.txt	0.04698404719122192
demetri|103053.txt	0.043187753932482045
leadstunnel|530053.txt	0.04019072492377297
cablelabs|516053.txt	0.03950777919830765
northstar|741053.txt	0.03882349818035041
lacefield|612053.txt	0.03847562191644304
alpharetta|647053.txt	0.03826375016185344
siser|426053.txt	0.036401559872669645
piano|612053.txt	0.03618940686114688
emwd|593053.txt	0.033857244135732205
homelessness|298053.txt	0.0335978410858435
bbfc|88053.txt	0.033474251073450624
aureon|735053.txt	0.033195397642891006
cartridges|338053.txt	0.03047228998022679
offender|447053.txt	0.029096572924259944
marys|515053.txt	0.028022806831760033
pacifica|273053.txt	0.026909279428005462
jiu|407053.txt	0.02679729556035027
chanimal|739053.txt	0.02657086235915801
taft|287053.txt	0.026415422820509023
offenders|447053.txt	0.026285640189516547
harford|246053.txt	0.026036141819063066
bearer|370053.txt	0.025802404909546592
blinds|355053.txt	0.02493532036775024
westmed|436053.txt	0.02469399353044266
maths|409053.txt	0.02429199769363916
```