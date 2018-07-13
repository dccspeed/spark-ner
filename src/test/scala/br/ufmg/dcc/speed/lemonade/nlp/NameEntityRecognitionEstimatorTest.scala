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
      //("Lula sempre foi um safado e quase saiu da cadeia em Curitiba"),
      //("O Tribunial Regional Federal quase soltou um ex-presidente do Brasil."),
      //("A presidente do Superior Tribunal de Justiça (STJ), ministra Laurita Vaz, rejeitou nesta quarta-feira (11) mais 143 pedidos de liberdade para o ex-presidente Luiz Inácio Lula da Silva feitos por cidadãos. Na terça, ele já havia rejeitado um dos pedidos desse tipo, em decisão na qual fez críticas ao desembargador Rogério Fraveto, que mandou soltar Lula no domingo - a decisão de Fraveto foi depois revertida pelo presidente do Tribunal Regional Federal da 4ª Região (TRF-4). Segundo Laurita Vaz, \"o Poder Judiciário não pode ser utilizado como balcão de reivindicações ou manifestações de natureza política ou ideológico-partidárias\". Ainda está nas mãos da ministra Laurita Vaz um pedido da Procuradoria Geral da República para que ela decida de quem é a competência para analisar pedidos de liberdade de Lula - o pleito foi feito após decisões divergentes de desembargadores do TRF-4, e a PGR quer que só o STJ possa analisar habeas corpus ao ex-presidente.")
      //("WASHINGTON — It is not every day that a potential constitutional showdown over a presidential subpoena coincides with a confirmation hearing for a crucial Supreme Court seat Less likely yet is a nominee who has written extensively about the very question at the heart of the dispute But that novel historical moment is here “It is not at all far-fetched to think that the question of whether President Trump must respond to a subpoena could come before the Supreme Court shortly after the confirmation process,” said Walter Dellinger, who served as acting United States solicitor general in the Clinton administration Mr Trump’s choice for the court, Judge Brett M Kavanaugh, has expressed strong support for executive power, hostility to administrative agencies and support for gun rights and religious freedom Those are all conventional positions among conservative lawyers and judges But there is one stance that sets Judge Kavanaugh apart, and it could not be more timely: his deep skepticism of the wisdom of forcing a sitting president to answer questions in criminal cases")
      ("WASHINGTON — It is not every day that a potential constitutional showdown over a presidential subpoena coincides with a confirmation hearing for a crucial Supreme Court seat. Less likely yet is a nominee who has written extensively about the very question at the heart of the dispute. But that novel historical moment is here. “It is not at all far-fetched to think that the question of whether President Trump must respond to a subpoena could come before the Supreme Court shortly after the confirmation process,” said Walter Dellinger, who served as acting United States solicitor general in the Clinton administration. Mr. Trump’s choice for the court, Judge Brett M. Kavanaugh, has expressed strong support for executive power, hostility to administrative agencies and support for gun rights and religious freedom. Those are all conventional positions among conservative lawyers and judges. But there is one stance that sets Judge Kavanaugh apart, and it could not be more timely: his deep skepticism of the wisdom of forcing a sitting president to answer questions in criminal cases.")
    ).toDF("document")
    val estimator = new NamedEntityRecognitionEstimator()
    estimator.setNerTrainPath("file:///scratch/zilton/IntelliJProjects/ScalaNER/src/main/scala/ScalaNER-master/src/main/resources/stanford_models/english.all.3class.distsim.crf.ser.gz")
    //estimator.setNerTrainPath("file:///scratch/zilton/IntelliJProjects/ScalaNER/src/main/scala/ScalaNER-master/src/main/resources/stanford_models/stanford-corenlp/sigarra-ner-model-tolerance_1e-3.ser.gz")
    estimator.setInputCol("document")
    estimator.setOutputCol("entities")

    val model = estimator.fit(sourceDF)
    val resultDF = model.transform(sourceDF)

    resultDF.show(truncate = false)
    resultDF.summary().show(truncate = false)

    //model.save("file:///tmp/lixo.spark")
    //assertSmallDataFrameEquality(sourceDF, sourceDF)

  }
}
