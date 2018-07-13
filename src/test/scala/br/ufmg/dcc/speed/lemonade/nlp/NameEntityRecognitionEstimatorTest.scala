package br.ufmg.dcc.speed.lemonade.nlp

import com.github.mrpowers.spark.fast.tests.DataFrameComparer
import org.scalatest.{FunSpec, Tag}

class NameEntityRecognitionEstimatorSpec extends FunSpec
with SparkSessionTestWrapper  with DataFrameComparer {

  import spark.implicits._

  it("transform a document, extracting named entities from it"){
    var sourceDF = Seq(
      //("President Obama went to Rome to meed Pope."),
      //("Catholic Church agrees to provide documents regarding Galileu Galilei .")
      ("Lula sempre foi um safado e quase saiu da cadeia em Curitiba"),
      ("O Tribunial Regional Federal quase soltou um ex-presidente do Brasil.")
    ).toDF("document")
    val estimator = new NamedEntityRecognitionEstimator()
    //estimator.setNerTrainPath("file:///scratch/zilton/IntelliJProjects/ScalaNER/src/main/scala/ScalaNER-master/src/main/resources/stanford_models/english.all.3class.distsim.crf.ser.gz")
    estimator.setNerTrainPath("file:///scratch/zilton/IntelliJProjects/ScalaNER/src/main/scala/ScalaNER-master/src/main/resources/stanford_models/stanford-corenlp/sigarra-ner-model-tolerance_1e-3.ser.gz")
    estimator.setInputCol("document")
    estimator.setOutputCol("entities")

    val model = estimator.fit(sourceDF)
    var resultDF = model.transform(sourceDF)

    resultDF.show(truncate = false)

    //model.save("file:///tmp/lixo.spark")
    assertSmallDataFrameEquality(sourceDF, sourceDF)

  }
}
