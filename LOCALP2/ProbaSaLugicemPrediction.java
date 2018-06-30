/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package probasalugicemprediction;

import java.io.InputStreamReader;
import com.univocity.parsers.csv.CsvParser;
import com.univocity.parsers.csv.CsvParserSettings;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;
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
import org.apache.spark.mllib.regression.LinearRegressionModel;
import org.apache.spark.sql.SparkSession;

 
public class ProbaSaLugicemPrediction{
 
    public static void main(String[] args) {
 
        // configure spark
//        SparkConf sparkConf = new SparkConf().setAppName("JavaNaiveBayesExample")
//                                        .setMaster("local[2]").set("spark.executor.memory","2g");

        SparkSession session = SparkSession
                                .builder()
                                .appName("MLPrediction")
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
              isr=new InputStreamReader(new FileInputStream("E:\\FAKULTET\\IVgodina\\IIsemestar\\SPOZ\\LOCALP2\\testData.csv"), "UTF-8");
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
        
        JavaRDD<LabeledPoint> testData= jc.parallelize(data);
                
        NaiveBayesModel model = NaiveBayesModel.load(jsc, "E:\\FAKULTET\\IVgodina\\IIsemestar\\SPOZ\\LOCALP2\\modelExample");
                
        // predvidjanje vrednosti na osnovu ucitanih vrednosti: model.predict(vektor atributa)
        // smestanje vrednosti u JavaRDD
        JavaRDD<Double> preds = testData.map(point -> model.predict(point.features()));
                
        // upis rezultata u csv fajl
        try {
            FileWriter fw = new FileWriter("E:\\FAKULTET\\IVgodina\\IIsemestar\\SPOZ\\LOCALP2\\prediction.csv");
                     
            List<Double> predsDouble = preds.takeOrdered((int)preds.count());
            
            for(Double d : predsDouble){
                fw.write(d.toString() + "\n");
            }
            
            fw.close();
            
        } catch (Exception ex) {
            System.out.println(ex.getMessage());
        }      
        // zatvaranje sesije
        session.close();
    }
}