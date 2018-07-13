package br.ufmg.dcc.speed.lemonade.nlp

import java.io.{BufferedInputStream, InputStream}
import java.util.zip.GZIPInputStream

import edu.stanford.nlp.ie.crf.CRFClassifier
import edu.stanford.nlp.ling.{CoreAnnotations, CoreLabel}
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.SparkContext
import org.apache.spark.ml.param.{Param, ParamMap, Params}
import org.apache.spark.ml.util.{Identifiable, MLWritable}
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.types.{ArrayType, StringType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Dataset}

import scala.collection.JavaConversions._
import scala.collection.mutable.ArrayBuffer
import scala.runtime.Nothing$

trait NamedEntityRecognitionParams extends Params {
  final val nerTrainPath = new Param[String](this, "nerTrainPath",
    "Path for NER train, compatible with HDFS")

  /** Define input and output column parameters */
  final val inputCol: Param[String] = new Param[String](this, "inputCol", "input column name")
  final val outputCol: Param[String] = new Param[String](this, "outputCol", "output column name")

  setDefault(inputCol, "inputCol")

  final def getInputCol: String = $(inputCol)

  setDefault(outputCol, "outputCol")

  final def getOutputCol: String = $(outputCol)

  final def getNerTrainPath: String = $(nerTrainPath)
}

class NamedEntityRecognitionEstimator(override val uid: String)
  extends Estimator[NamedEntityRecognitionModel]
    with NamedEntityRecognitionParams {

  def this() = this(Identifiable.randomUID("NamedEntityRecognitionParams"))

  def setInputCol(value: String): this.type = set(inputCol, value)

  def setOutputCol(value: String): this.type = set(outputCol, value)

  override def fit(dataset: Dataset[_]): NamedEntityRecognitionModel = {
    var train: InputStream = null

    val sc = SparkContext.getOrCreate()
    val fs = FileSystem.get(sc.hadoopConfiguration)
    val in = fs.open(new Path($(nerTrainPath)))

    if ($(nerTrainPath).endsWith(".gz")) {
      train = new GZIPInputStream(in)
    } else {
      train = new BufferedInputStream(in)
    }
    val crfClassifier = CRFClassifier.getClassifier[CoreLabel](train)
    train.close()

    new NamedEntityRecognitionModel(uid, crfClassifier).setInputCol($(inputCol)).setOutputCol($(outputCol))
  }

  override def copy(extra: ParamMap) = {
    defaultCopy(extra)
  }

  def setNerTrainPath(value: String) {
    this.set(this.nerTrainPath, value)
  }

  setDefault(outputCol, "outputCol")
  setDefault(inputCol, "documents")

  override def transformSchema(schema: StructType) = {
    val idx = schema.fieldIndex($(inputCol))
    val field = schema.fields(idx)
    if (field.dataType != StringType) throw new Exception(s"Input type ${field.dataType} did not match input type StringType")
    schema.add(StructField($(outputCol), ArrayType(StringType), false))
  }


}

class NamedEntityRecognitionModel(override val uid: String, classifier: CRFClassifier[CoreLabel])
  extends Model[NamedEntityRecognitionModel]
    with NamedEntityRecognitionParams with Serializable {

  /** @group setParam */
  def setInputCol(value: String): this.type = set(inputCol, value)

  /** @group setParam */
  def setOutputCol(value: String): this.type = set(outputCol, value)

  override def copy(extra: ParamMap): NamedEntityRecognitionModel = {
    defaultCopy(extra)
  }

  override def transform(dataset: Dataset[_]): DataFrame = {

    val classifierUdf = udf {
      document: String => {
        val sentences = classifier.classify(document)

        val results = sentences.map {(sentence) =>
            sentence.map { (tok) =>
              Seq(
                tok.get(classOf[CoreAnnotations.AnswerAnnotation]),
                tok.word()
              )
            }
        }
        results
      }
    }

    val result = dataset.select(col("*"), classifierUdf(dataset($(inputCol))).cast(
      StringType).as($(outputCol)))
    result
  }

  override def transformSchema(schema: StructType): StructType = {
    val idx = schema.fieldIndex($(inputCol))
    val field = schema.fields(idx)
    if (field.dataType != StringType) {
      throw new Exception(s"Input type ${field.dataType} did not match input type StringType")
    }
    schema.add(StructField($(outputCol), ArrayType(ArrayType(StringType)), false))
  }
}