import sbt.project

name := "spark-core-nlp"

scalaVersion := "2.11.12"

updateOptions := updateOptions.value.withCachedResolution(true)

sparkVersion := "2.3.0"

sparkComponents ++= Seq("mllib", "sql")

resolvers += Resolver.sonatypeRepo("public")

spShortDescription := "spark-core-nlp"

spDescription := """An implementation of an spark estimator using Stanford Core NLP.""".stripMargin

credentials += Credentials(Path.userHome / ".ivy2" / ".sbtcredentials")

licenses += "Apache-2.0" -> url("http://opensource.org/licenses/Apache-2.0")

version := "1.0"

libraryDependencies ++= Seq(
  "edu.stanford.nlp" % "stanford-corenlp" % "3.6.0",
  "edu.stanford.nlp" % "stanford-corenlp" % "3.6.0" classifier "models"
)
libraryDependencies += "com.github.mrpowers" % "spark-fast-tests_2.11" % "0.11.0" % "test"
libraryDependencies += "org.scalatest" % "scalatest_2.11" % "3.0.1"
libraryDependencies += "org.scalactic" % "scalactic_2.11" % "3.0.1"

