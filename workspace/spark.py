import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, count, desc, round, sum, regexp_replace, when, lit
import datetime

# Configure logging
logging.basicConfig(
    filename='/opt/workspace/spotify_analysis.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def create_spark_session():
    """Initialize Spark Session with 2 workers"""
    logger.info("Initializing Spark Session")
    return SparkSession.builder \
        .appName("SpotifyAnalysis2023") \
        .config("spark.ui.showConsoleProgress", "false") \
        .config("spark.log.level", "WARN") \
        .config("spark.executor.instances", "2") \
        .config("spark.executor.cores", "1") \
        .getOrCreate()

def load_and_clean_data(spark):
    """Load and clean the Spotify dataset"""
    logger.info("Loading Spotify dataset")
    df = spark.read.csv("/opt/workspace/spotify.csv", header=True, inferSchema=True)
    logger.info(f"Loaded {df.count()} records")
    
    # Clean column names
    logger.info("Cleaning column names")
    for column in df.columns:
        new_column = column.replace('(', '').replace(')', '').replace(' ', '_').replace('%', 'percent').lower()
        df = df.withColumnRenamed(column, new_column)
    
    return df

def analyze_top_songs(df):
    """Analyze top 10 songs by streams"""
    logger.info("Analyzing top 10 songs")
    return df.select('track_name', 'artists_name', 'streams', 
                    'danceability_percent', 'energy_percent', 'valence_percent') \
        .orderBy(desc('streams')) \
        .limit(10)

def analyze_yearly_trends(df):
    """Analyze audio features by release year"""
    logger.info("Analyzing yearly trends")
    return df.groupBy('released_year') \
        .agg(
            round(avg('danceability_percent'), 2).alias('avg_danceability'),
            round(avg('energy_percent'), 2).alias('avg_energy'),
            round(avg('valence_percent'), 2).alias('avg_valence'),
            count('*').alias('number_of_songs')
        ) \
        .orderBy('released_year')

def analyze_artist_stats(df):
    """Analyze artist statistics"""
    logger.info("Analyzing artist statistics")
    return df.groupBy('artists_name') \
        .agg(
            count('*').alias('song_count'),
            sum('streams').alias('total_streams')
        ) \
        .orderBy(desc('song_count')) \
        .limit(5)

def analyze_bpm_distribution(df):
    """Analyze song distribution by BPM ranges"""
    logger.info("Analyzing BPM distribution")
    return df.withColumn('bpm_range', 
        when(col('bpm') < 90, 'Slow (< 90)') \
        .when(col('bpm') < 120, 'Medium (90-120)') \
        .when(col('bpm') < 150, 'Fast (120-150)') \
        .otherwise('Very Fast (150+)')) \
        .groupBy('bpm_range') \
        .agg(count('*').alias('song_count')) \
        .orderBy('bpm_range')

def analyze_club_songs(df):
    """Analyze songs best suited for clubs"""
    logger.info("Analyzing club-friendly songs")
    return df.withColumn('club_score', (col('danceability_percent') + col('energy_percent')) / 2) \
        .select('track_name', 'artists_name', 'danceability_percent', 
                'energy_percent', 'club_score') \
        .orderBy(desc('club_score')) \
        .limit(10)

def calculate_avg_streams(df):
    """Calculate average streams per song"""
    logger.info("Calculating average streams")
    return df.select(round(avg('streams'), 2).alias('average_streams')).collect()[0]['average_streams']

def main():
    try:
        # Initialize Spark
        spark = create_spark_session()
        sc = spark.sparkContext
        sc.setLogLevel("WARN")
        
        # Load and process data
        df = load_and_clean_data(spark)
        
        # Print dataset structure
        logger.info("Analyzing dataset structure")
        df.printSchema()
        
        # Run analyses
        analyses = {
            "Top 10 Songs Analysis": analyze_top_songs(df),
            "Yearly Audio Features": analyze_yearly_trends(df),
            "Top Artists": analyze_artist_stats(df),
            "BPM Distribution": analyze_bpm_distribution(df),
            "Club-Friendly Songs": analyze_club_songs(df)
        }
        
        # Display results
        for title, result in analyses.items():
            print(f"\n{title}:")
            result.show(truncate=False)
            
        # Calculate and display average streams
        avg_streams = calculate_avg_streams(df)
        print(f"\nAverage streams per song: {avg_streams:,.2f}")
        
        # Save results
        logger.info("Saving analysis results")
        combined_results = analyses["Top 10 Songs Analysis"]
        combined_results.coalesce(1).write.mode('overwrite') \
            .option('header', 'true') \
            .csv('/opt/workspace/spotify_analysis_results')
        
        logger.info("Analysis completed successfully")
        
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        raise
    finally:
        spark.stop()
        logger.info("Spark session stopped")

if __name__ == "__main__":
    main()