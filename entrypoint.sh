#!/bin/bash

SPARK_WORKLOAD=${SPARK_WORKLOAD}

echo "SPARK_WORKLOAD: ${SPARK_WORKLOAD}"

if [ "${SPARK_WORKLOAD}" == "master" ]; then
  exec /opt/spark/bin/spark-class org.apache.spark.deploy.master.Master --host spark-master --port 7077
elif [ "${SPARK_WORKLOAD}" == "worker" ]; then
  exec /opt/spark/bin/spark-class org.apache.spark.deploy.worker.Worker spark://spark-master:7077
elif [ "${SPARK_WORKLOAD}" == "history" ]; then
  exec /opt/spark/bin/spark-class org.apache.spark.deploy.history.HistoryServer
fi