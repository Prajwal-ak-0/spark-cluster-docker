import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, desc, regexp_extract, hour, date_format, sum, avg, window
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, TimestampType
import datetime

# Configure logging
logging.basicConfig(
    filename='/opt/workspace/server_analysis.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def create_spark_session():
    """Initialize Spark Session with 2 workers"""
    logger.info("Initializing Spark Session")
    return SparkSession.builder \
        .appName("ServerLogAnalysis") \
        .config("spark.ui.showConsoleProgress", "false") \
        .config("spark.log.level", "WARN") \
        .config("spark.executor.instances", "2") \
        .config("spark.executor.cores", "1") \
        .getOrCreate()

def define_schema():
    """Define schema for server log data"""
    return StructType([
        StructField("ip", StringType(), True),
        StructField("timestamp", TimestampType(), True),
        StructField("method", StringType(), True),
        StructField("url", StringType(), True),
        StructField("status", IntegerType(), True),
        StructField("bytes", IntegerType(), True),
        StructField("referer", StringType(), True),
        StructField("user_agent", StringType(), True)
    ])

def parse_logs(line):
    """Parse each log line into structured format"""
    # Common log format regex pattern
    pattern = r'^(\S+) \S+ \S+ \[([\w:/]+\s[+\-]\d{4})\] "(\S+) (\S+) \S+" (\d{3}) (\d+) "([^"]*)" "([^"]*)"'
    
    match = regexp_extract(line, pattern, 1)
    if match:
        return {
            "ip": match.group(1),
            "timestamp": datetime.datetime.strptime(match.group(2), '%d/%b/%Y:%H:%M:%S %z'),
            "method": match.group(3),
            "url": match.group(4),
            "status": int(match.group(5)),
            "bytes": int(match.group(6)),
            "referer": match.group(7),
            "user_agent": match.group(8)
        }
    return None

def load_and_parse_data(spark):
    """Load and parse the server log data"""
    logger.info("Loading server log data")
    
    # Read the log file
    raw_logs = spark.read.text("/opt/workspace/server_log.log")
    
    # Parse logs using the defined schema
    parsed_logs = raw_logs.select(
        regexp_extract('value', r'^(\S+)', 1).alias('ip'),
        regexp_extract('value', r'\[(\d{2}/\w{3}/\d{4}:\d{2}:\d{2}:\d{2} [+\-]\d{4})\]', 1).alias('timestamp'),
        regexp_extract('value', r'"(\S+)', 1).alias('method'),
        regexp_extract('value', r'"\S+ (\S+)', 1).alias('url'),
        regexp_extract('value', r'" (\d{3})', 1).cast('integer').alias('status'),
        regexp_extract('value', r'" \d{3} (\d+)', 1).cast('integer').alias('bytes'),
        regexp_extract('value', r'" \d{3} \d+ "(.*?)"', 1).alias('referer'),
        regexp_extract('value', r'" \d{3} \d+ ".*?" "(.*?)"', 1).alias('user_agent')
    )
    
    logger.info(f"Parsed {parsed_logs.count()} log entries")
    return parsed_logs

def analyze_http_methods(df):
    """Analyze distribution of HTTP methods"""
    logger.info("Analyzing HTTP methods distribution")
    return df.groupBy('method') \
        .agg(count('*').alias('count')) \
        .orderBy(desc('count'))

def analyze_response_codes(df):
    """Analyze HTTP response codes"""
    logger.info("Analyzing response codes")
    return df.groupBy('status') \
        .agg(count('*').alias('count')) \
        .orderBy(desc('count'))

def analyze_top_ips(df):
    """Analyze top IP addresses by request count"""
    logger.info("Analyzing top IP addresses")
    return df.groupBy('ip') \
        .agg(count('*').alias('requests'),
             sum('bytes').alias('total_bytes')) \
        .orderBy(desc('requests')) \
        .limit(10)

def analyze_hourly_traffic(df):
    """Analyze traffic patterns by hour"""
    logger.info("Analyzing hourly traffic patterns")
    return df.withColumn('hour', hour('timestamp')) \
        .groupBy('hour') \
        .agg(count('*').alias('requests')) \
        .orderBy('hour')

def analyze_url_patterns(df):
    """Analyze most accessed URLs"""
    logger.info("Analyzing URL patterns")
    return df.groupBy('url') \
        .agg(count('*').alias('hits')) \
        .orderBy(desc('hits')) \
        .limit(10)

def main():
    try:
        # Initialize Spark
        spark = create_spark_session()
        sc = spark.sparkContext
        sc.setLogLevel("WARN")
        
        # Load and parse data
        df = load_and_parse_data(spark)
        
        # Run analyses
        analyses = {
            "HTTP Methods Distribution": analyze_http_methods(df),
            "Response Code Distribution": analyze_response_codes(df),
            "Top 10 IP Addresses": analyze_top_ips(df),
            "Hourly Traffic Pattern": analyze_hourly_traffic(df),
            "Top 10 Accessed URLs": analyze_url_patterns(df)
        }
        
        # Display results
        print("\n=== Server Log Analysis Results ===")
        for title, result in analyses.items():
            print(f"\n{title}:")
            result.show(truncate=False)
        
        # Save results to CSV
        logger.info("Saving analysis results")
        for name, data in analyses.items():
            output_path = f"/opt/workspace/log_analysis_{name.lower().replace(' ', '_')}"
            data.coalesce(1).write.mode('overwrite') \
                .option('header', 'true') \
                .csv(output_path)
        
        logger.info("Analysis completed successfully")
        
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        raise
    finally:
        spark.stop()
        logger.info("Spark session stopped")

if __name__ == "__main__":
    main()