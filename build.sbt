ThisBuild / version := "0.1.0-SNAPSHOT"

ThisBuild / scalaVersion := "2.11.12"

lazy val root = (project in file("."))
  .settings(
    name := "spark",
  )

libraryDependencies += "org.apache.spark" %% "spark-core" % "2.4.8"
libraryDependencies += "org.apache.spark" %% "spark-sql" % "2.4.8"
libraryDependencies += "org.apache.spark" %% "spark-mllib" % "2.4.8"
libraryDependencies += "org.apache.spark" %% "spark-graphx" % "2.4.8"
libraryDependencies += "org.apache.spark" %% "spark-yarn" % "2.4.8"
libraryDependencies += "org.apache.spark" %% "spark-network-shuffle" % "2.4.8"
libraryDependencies += "org.apache.spark" %% "spark-streaming-flume" % "2.4.8"

libraryDependencies += "com.databricks" %% "spark-csv" % "1.5.0"
libraryDependencies += "com.github.scopt" %% "scopt" % "4.1.0"
