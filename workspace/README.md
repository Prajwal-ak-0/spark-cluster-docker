# Big Data Processing with PySpark: Spotify and Server Log Analysis 

Hey there!  In this guide, I'll walk you through setting up a Spark cluster using Docker and implementing two awesome data analysis projects: Spotify Track Analysis and Server Log Analysis. Let's dive in!

## Table of Contents
- [Setting Up Spark Cluster](#setting-up-spark-cluster)
- [Spotify Data Analysis](#spotify-data-analysis)
- [Server Log Analysis](#server-log-analysis)
- [Common Issues & Solutions](#common-issues--solutions)
- [Taking it Further](#taking-it-further)

## Setting Up Spark Cluster

### 1. Project Structure Setup
First, create your project directory structure:
```bash
mkdir -p workspace
cd workspace
```

### 2. Docker Configuration Files

#### a. Create docker-compose.yml
```yaml
version: '3'
services:
  spark-master:
    build: 
      context: .
      dockerfile: spark-cluster.Dockerfile
    ports:
      - "8080:8080"
      - "7077:7077"
    volumes:
      - ./workspace:/opt/workspace
    environment:
      - SPARK_MODE=master
      - SPARK_RPC_AUTHENTICATION_ENABLED=no
      - SPARK_RPC_ENCRYPTION_ENABLED=no
      - SPARK_LOCAL_STORAGE_ENCRYPTION_ENABLED=no
      - SPARK_SSL_ENABLED=no
    networks:
      - spark-network

  spark-worker-1:
    build:
      context: .
      dockerfile: spark-cluster.Dockerfile
    depends_on:
      - spark-master
    environment:
      - SPARK_MODE=worker
      - SPARK_MASTER_URL=spark://spark-master:7077
      - SPARK_WORKER_MEMORY=1G
      - SPARK_WORKER_CORES=1
    networks:
      - spark-network

  spark-worker-2:
    build:
      context: .
      dockerfile: spark-cluster.Dockerfile
    depends_on:
      - spark-master
    environment:
      - SPARK_MODE=worker
      - SPARK_MASTER_URL=spark://spark-master:7077
      - SPARK_WORKER_MEMORY=1G
      - SPARK_WORKER_CORES=1
    networks:
      - spark-network

networks:
  spark-network:
    driver: bridge
```

#### b. Create spark-cluster.Dockerfile
```dockerfile
FROM python:3.9-slim

# Install OpenJDK-11
RUN apt-get update && \
    apt-get install -y openjdk-11-jdk && \
    apt-get install -y wget && \
    apt-get clean;

# Set JAVA_HOME
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64

# Download and install Spark
ENV SPARK_VERSION=3.5.0
ENV HADOOP_VERSION=3
ENV SPARK_HOME=/opt/spark

RUN wget -q https://downloads.apache.org/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz && \
    tar xzf spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz && \
    mv spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION} /opt/spark && \
    rm spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz

# Set Spark environment variables
ENV PATH=$SPARK_HOME/bin:$PATH

# Install PySpark
RUN pip install pyspark==${SPARK_VERSION}

# Create workspace directory
RUN mkdir -p /opt/workspace

WORKDIR /opt/workspace

# Copy entrypoint script
COPY entrypoint.sh /
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
```

#### c. Create entrypoint.sh
```bash
#!/bin/bash

if [ "$SPARK_MODE" = "master" ]; then
    exec /opt/spark/bin/spark-class org.apache.spark.deploy.master.Master
else
    exec /opt/spark/bin/spark-class org.apache.spark.deploy.worker.Worker spark://spark-master:7077
fi
```

### 3. Starting the Cluster
```bash
# Build and start the cluster
docker-compose up -d

# Verify containers are running
docker ps

# Check logs if needed
docker logs spark-master
docker logs spark-worker-1
docker logs spark-worker-2
```

### 4. Accessing the Cluster
- Spark Master UI: http://localhost:8080
- Submit jobs using: spark-submit

## Spotify Data Analysis

### 1. Data Overview
We're analyzing a Spotify dataset containing track information including:
- Track name and artist(s)
- Audio features (danceability, energy, etc.)
- Popularity metrics (streams, playlist appearances)

### 2. Implementation Details

#### a. Data Loading and Cleaning
```python
def load_and_clean_data(spark):
    """Load and clean the Spotify dataset"""
    df = spark.read.csv("/opt/workspace/spotify.csv", header=True, inferSchema=True)
    
    # Clean column names
    for column in df.columns:
        new_column = column.replace('(', '').replace(')', '').replace(' ', '_').lower()
        df = df.withColumnRenamed(column, new_column)
    
    return df
```

#### b. Analysis Functions
We perform 5 different analyses:

1. **Top Songs Analysis**
```python
def analyze_top_songs(df):
    return df.select('track_name', 'artists_name', 'streams', 
                    'danceability_percent', 'energy_percent', 'valence_percent') \
        .orderBy(desc('streams')) \
        .limit(10)
```

2. **Yearly Trends**
```python
def analyze_yearly_trends(df):
    return df.groupBy('released_year') \
        .agg(
            round(avg('danceability_percent'), 2).alias('avg_danceability'),
            round(avg('energy_percent'), 2).alias('avg_energy'),
            round(avg('valence_percent'), 2).alias('avg_valence'),
            count('*').alias('number_of_songs')
        )
```

3. **Artist Statistics**
```python
def analyze_artist_stats(df):
    return df.groupBy('artists_name') \
        .agg(count('*').alias('song_count'),
             sum('streams').alias('total_streams'))
```

### 3. Running Spotify Analysis
```bash
docker exec -it spark-master spark-submit /opt/workspace/spark.py
```

## Server Log Analysis

### 1. Log Format Understanding
We're processing server logs in the Common Log Format:
```
IP - - [timestamp] "METHOD /path HTTP/1.1" status bytes "referer" "user-agent"
```

### 2. Implementation Details

#### a. Log Parsing
```python
def load_and_parse_data(spark):
    raw_logs = spark.read.text("/opt/workspace/server_log.log")
    
    parsed_logs = raw_logs.select(
        regexp_extract('value', r'^(\S+)', 1).alias('ip'),
        regexp_extract('value', r'\[(\d{2}/\w{3}/\d{4}:\d{2}:\d{2}:\d{2})', 1).alias('timestamp'),
        regexp_extract('value', r'"(\S+)', 1).alias('method'),
        regexp_extract('value', r'"\S+ (\S+)', 1).alias('url'),
        regexp_extract('value', r'" (\d{3})', 1).cast('integer').alias('status'),
        regexp_extract('value', r'" \d{3} (\d+)', 1).cast('integer').alias('bytes')
    )
    return parsed_logs
```

#### b. Analysis Functions

1. **HTTP Methods Analysis**
```python
def analyze_http_methods(df):
    return df.groupBy('method') \
        .agg(count('*').alias('count')) \
        .orderBy(desc('count'))
```

2. **Response Codes Analysis**
```python
def analyze_response_codes(df):
    return df.groupBy('status') \
        .agg(count('*').alias('count')) \
        .orderBy(desc('count'))
```

3. **Top IP Analysis**
```python
def analyze_top_ips(df):
    return df.groupBy('ip') \
        .agg(
            count('*').alias('requests'),
            sum('bytes').alias('total_bytes')
        ) \
        .orderBy(desc('requests')) \
        .limit(10)
```

### 3. Running Log Analysis
```bash
docker exec -it spark-master spark-submit /opt/workspace/server.py
```

### 4. Understanding Results

#### a. HTTP Methods Distribution
Shows the distribution of request methods:
```
+------+-----+
|method|count|
+------+-----+
|GET   |4866 |
|HEAD  |119  |
|POST  |15   |
+------+-----+
```

#### b. Response Codes
Helps identify server health:
```
+------+-----+
|status|count|
+------+-----+
|200   |4422 |
|302   |253  |
|404   |197  |
+------+-----+
```

#### c. Top IP Addresses
Identifies heavy users or potential bots:
```
+--------------+--------+-----------+
|ip            |requests|total_bytes|
+--------------+--------+-----------+
|66.249.66.194 |696     |14086474   |
|130.185.74.243|620     |19096585   |
+--------------+--------+-----------+
```

## Common Issues & Solutions

### 1. Memory Issues
If you encounter memory errors:
```python
spark.conf.set("spark.driver.memory", "4g")
spark.conf.set("spark.executor.memory", "4g")
```

### 2. File Access Issues
Ensure proper volume mounting in docker-compose.yml:
```yaml
volumes:
  - ./workspace:/opt/workspace
```

## Taking it Further

### 1. Real-time Processing
- Add Kafka integration
- Implement streaming analytics

### 2. Enhanced Analytics
- Add machine learning models
- Implement anomaly detection

### 3. Visualization
- Create dashboards using Streamlit
- Add interactive visualizations

Remember: Start small, test thoroughly, and scale gradually! 

Happy coding! 

---
*This project was created for learning distributed data processing with PySpark. Feel free to modify and enhance it!*
