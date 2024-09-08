import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.{GeneralizedLinearRegression, LinearRegression}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.sql.{Row, SparkSession}

object CryotherapyPrediction {



  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder
      .master("local[*]")
      .config("spark.sql.warehouse.dir", "/Users/mykhailomarkhain/temp")
      .appName("CryotherapyPrediction")
      .getOrCreate()


    val rawTrafficDF = spark.read.option("header", "true")
      .option("inferSchema", "true")
      .option("delimiter", ";")
      .format("com.databricks.spark.csv")
      .load( "data/Sao_Paulo.csv")
       .cache

//    rawTrafficDF.show()
    rawTrafficDF.printSchema()
//    rawTrafficDF.describe().show()

    rawTrafficDF.select("Hour (Coded)", "Immobilized bus", "Broken Truck", "Vehicle excess", "Fire", "Slowness in traffic (%)")

    var newTrafficDF = rawTrafficDF.withColumnRenamed("Slowness in traffic (%)", "label")
  //    .withColumn("label",col("label").cast(DoubleType))

    newTrafficDF = newTrafficDF.withColumnRenamed("Point of flooding",
      "NoOfFloodPoint")


    newTrafficDF.createOrReplaceTempView("slDF")


    val colNames = newTrafficDF.columns.dropRight(1)
    newTrafficDF.show(5)


    val assembler = new VectorAssembler()
      .setInputCols(colNames)
      .setOutputCol("features")

    val assembleDF = assembler.transform(newTrafficDF).select("features",
      "label")

    import spark.implicits._

    assembleDF.show()

    val seed = 12345L
    val splits = assembleDF.randomSplit(Array(0.60, 0.40), seed)
    val (trainingData, validationData) = (splits(0), splits(1))

    trainingData.cache // cache in memory for quicker access
    validationData.cache // cache in memory for quicker access

    val lr = new LinearRegression()
      .setFeaturesCol("features")
      .setLabelCol("label")

    println("Building ML regression model")
    //val lrModel = lr.fit(trainingData)
    println("Evaluating the model on the test set and calculating the regression metrics")

    val glr = new GeneralizedLinearRegression()
      .setFamily("gaussian")//continuous value prediction (or gamma)
      .setLink("identity")//continuous value prediction (or inverse)
      .setFeaturesCol("features")
      .setLabelCol("label")

    println("Building ML regression model")
    val glrModel = glr.fit(trainingData)

    val trainPredictionsAndLabels = glrModel.transform(validationData).select("label", "prediction")
      .map { case Row(label: Double, prediction: Double) => (label, prediction) }.rdd


    val testRegressionMetrics = new
        RegressionMetrics(trainPredictionsAndLabels)

    val results =
      "\n=====================================================================\n"+
      s"TrainingData count: ${trainingData.count}\n" +
      s"TestData count: ${validationData.count}\n" +
//      "=====================================================================\n" +
      s"TestData MSE = ${testRegressionMetrics.meanSquaredError}\n"
//      s"TestData RMSE = ${testRegressionMetrics.rootMeanSquaredError}\n" +
//      s"TestData R-squared = ${testRegressionMetrics.r2}\n" +
//      s"TestData MAE = ${testRegressionMetrics.meanAbsoluteError}\n" +
//        s"TestData explainedVariance = ${testRegressionMetrics.explainedVariance}\n"
    println(results)


    println("Preparing K-fold Cross Validation and Grid Search")
    // ***********************************************************
    val paramGrid = new ParamGridBuilder()
      .addGrid(glr.maxIter, Array(10, 20, 30, 50, 100, 500, 1000))
      .addGrid(glr.regParam, Array(0.001, 0.01, 0.1))
      .addGrid(glr.tol, Array(0.01, 0.1))
      .build()

    println("Preparing 10-fold Cross Validation")
    val numFolds = 10 //10-fold cross-validation
       val cv = new CrossValidator()
      .setEstimator(lr)
      .setEvaluator(new RegressionEvaluator())
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(numFolds)

    println("Training model with the Linear Regression algorithm")
    val cvModel = cv.fit(trainingData)
    cvModel.write.overwrite()
      .save("model/LR_model")

    spark.stop()
  }
}
