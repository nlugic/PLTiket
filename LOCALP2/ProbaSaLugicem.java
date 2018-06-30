/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package probasalugicem;

import java.io.InputStreamReader;
import com.univocity.parsers.csv.CsvParser;
import com.univocity.parsers.csv.CsvParserSettings;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.logging.Level;
import java.util.logging.Logger;
import scala.Tuple2;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.classification.NaiveBayes;
import org.apache.spark.mllib.classification.NaiveBayesModel;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.util.MLUtils;
import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;
import static org.apache.spark.ml.r.RWrappers.session;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.sql.SparkSession;

 
public class ProbaSaLugicem{
 
    public static void main(String[] args) {
 
        // configure spark
//        SparkConf sparkConf = new SparkConf().setAppName("JavaNaiveBayesExample")
//                                        .setMaster("local[2]").set("spark.executor.memory","2g");

        SparkSession session = SparkSession
                                .builder()
                                .appName("MLBuilder")
                                .master("local")
                                .getOrCreate();

        // start a spark context
        SparkContext jsc = session.sparkContext();      
        
        ArrayList<LabeledPoint> data = new ArrayList<>();
        CsvParserSettings settings= new CsvParserSettings();
        settings.getFormat().setLineSeparator("\n");  
        CsvParser parser = new CsvParser(settings);
        
        InputStreamReader isr=null;
        try{
              isr=new InputStreamReader(new FileInputStream("E:\\FAKULTET\\IVgodina\\IIsemestar\\SPOZ\\LOCALP2\\trainingData.csv"), "UTF-8");
        }catch(Exception e){
            e.printStackTrace();
        }
        if(isr!=null)
            parser.beginParsing(isr);
                
        String[] row;
        boolean first=false;
        while((row = parser.parseNext()) != null){
            if(!first)
                first=true;
            else
            {
                double label = Double.parseDouble(row[row.length-1]);
                double[] features = new double[row.length-1];
                for(int i = 0; i < row.length-1; i++)
                {
                    try{
                    features[i] = Double.parseDouble(row[i].trim());
                    }
                    catch(Exception ex)
                    {
                        System.out.println(ex.getMessage());
                    }
                }
                data.add(new LabeledPoint(label, Vectors.dense(features)));
            }
        }
        
        JavaSparkContext jc = JavaSparkContext.fromSparkContext(session.sparkContext());
        
        JavaRDD<LabeledPoint> trainingData= jc.parallelize(data);
            
        try
        {
        // Train a Naive Bayes model
        NaiveBayesModel model = NaiveBayes.train(trainingData.rdd(), 1.0);

        // Save model to local for future use
        model.save(jsc, "E:\\FAKULTET\\IVgodina\\IIsemestar\\SPOZ\\LOCALP2\\modelExample");
        }
        catch(Exception ex)
        {
            Logger.getLogger(ProbaSaLugicem.class.getName()).log(Level.INFO, null, ex);
        }
        // stop the spark context
        session.close();
    }
}
    