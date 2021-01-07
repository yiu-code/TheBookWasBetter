import pandas as pd
import os
import argparse

import matplotlib.pyplot as plt

# spark imports
from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import col, lower
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator 

class AlsRecommender:
    """
    This a collaborative filtering recommender with Alternating Least Square
    Matrix Factorization, which is implemented by Spark
    """
    def __init__(self, spark_session, path_books, path_ratings, path_toread):
        self.spark = spark_session
        self.sc = spark_session.sparkContext
        self.booksDF = self._load_file(path_books) \
            .select(['id', 'book_id', 'best_book_id', 'books_count', 'isbn13', 'authors', 'original_publication_year', 'original_title', 'average_rating', 'ratings_count', 'work_ratings_count', 'work_text_reviews_count', 'image_url'])
        self.ratingsDF = self._load_file(path_ratings) \
            .select(['book_id', 'user_id', 'rating'])
        self.toReadsDF = self._load_file(path_toread) \
            .select(['user_id', 'book_id'])
        self.model = ALS(
            userCol='user_id',
            itemCol='book_id',
            ratingCol='rating',
            coldStartStrategy="drop",
            nonnegative=True,
            implicitPrefs = False)
        
    def _load_file(self, filepath):
        """
        load csv file into memory as spark DF
        """
        return self.spark.read.load(filepath, format='csv',
                                    header=True, inferSchema=True)
    
    def tune_model(self, maxIter, regParams, ranks, split_ratio=(6, 2, 2)):
        """
        Hyperparameter tuning for ALS model
        Parameters
        ----------
        maxIter: int, max number of learning iterations
        regParams: list of float, regularization parameter
        ranks: list of float, number of latent factors
        split_ratio: tuple, (train, validation, test)
        """
        # split data
        train, val, test = self.ratingsDF.randomSplit(split_ratio)
        # holdout tuning
        self.model = tune_ALS(self.model, train, val,
                              maxIter, regParams, ranks)
        # test model
        predictions = self.model.transform(test)
        evaluator = RegressionEvaluator(metricName="rmse",
                                        labelCol="rating",
                                        predictionCol="prediction")
        rmse = evaluator.evaluate(predictions)
        print('The out-of-sample RMSE of the best tuned model is:', rmse)
        # clean up
        del train, val, test, predictions, evaluator
        gc.collect()   
    
    def set_model_params(self, maxIter, regParam, rank):
        """
        set model params for pyspark.ml.recommendation.ALS
        Parameters
        ----------
        maxIter: int, max number of learning iterations
        regParams: float, regularization parameter
        ranks: float, number of latent factors
        """
        self.model = self.model \
            .setMaxIter(maxIter) \
            .setRank(rank) \
            .setRegParam(regParam)
        
    def _regex_matching(self, fav_book):
        """
        return the closest matches via SQL regex.
        If no match found, return None
        Parameters
        ----------
        fav_book: str, name of user input book
        Return
        ------
        list of indices of the matching books
        """
        print('You have input book:', fav_book)
        matchesDF = self.booksDF \
            .filter(
                lower(
                    col('original_title')
                ).like('%{}%'.format(fav_book.lower()))
            ) \
            .select('book_id', 'original_title')
        if not len(matchesDF.take(1)):
            print('Oops! No match is found')
        else:
            bookIds = matchesDF.rdd.map(lambda r: r[0]).collect()
            titles = matchesDF.rdd.map(lambda r: r[1]).collect()
            print('Found possible matches in our database: '
                  '{0}\n'.format([x for x in titles]))
            return bookIds
        
    def _append_ratings(self, userId, bookIds):
        """
        append a user's book ratings to ratingsDF
        Parameter
        ---------
        userId: int, userId of a user
        bookIds: int, bookIds of user's favorite books
        """
        # create new user rdd
        user_rdd = self.sc.parallelize(
            [(userId, bookId, 5.0) for bookId in bookIds])
        # transform to user rows
        user_rows = user_rdd.map(
            lambda x: Row(
                bookId=int(x[0]),
                userId=int(x[1]),
                rating=float(x[2])
            )
        )
        # transform rows to spark DF
        userDF = self.spark.createDataFrame(user_rows) \
            .select(self.ratingsDF.columns)
        # append to ratingsDF
        self.ratingsDF = self.ratingsDF.union(userDF)
        
    def _create_inference_data(self, userId, bookIds):
        """
        create a user with all books except ones were rated for inferencing
        """
        # filter books
        other_bookIds = self.booksDF \
            .filter(~col('book_id').isin(bookIds)) \
            .select(['book_id']) \
            .rdd.map(lambda r: r[0]) \
            .collect()
        # create inference rdd
        inferenceRDD = self.sc.parallelize(
            [(userId, bookId) for bookId in other_bookIds]
        ).map(
            lambda x: Row(
                userId=int(x[0]),
                bookId=int(x[1]),
            )
        )
        # transform to inference DF
        inferenceDF = self.spark.createDataFrame(inferenceRDD) \
            .select(['userId', 'bookId'])
        return inferenceDF
    
    def _inference(self, model, fav_book, n_recommendations):
        """
        return top n book recommendations based on user's input book
        Parameters
        ----------
        model: spark ALS model
        fav_book: str, name of user input book
        n_recommendations: int, top n recommendations
        Return
        ------
        list of top n similar book recommendations
        """
        # create a userId
        userId = self.ratingsDF.agg({"userId": "max"}).collect()[0][0] + 1
        # get bookIds of favorite books
        bookIds = self._regex_matching(fav_book)
        # append new user with his/her ratings into data
        self._append_ratings(userId, bookIds)
        # matrix factorization
        model = model.fit(self.ratingsDF)
        # get data for inferencing
        inferenceDF = self._create_inference_data(userId, bookIds)
        # make inference
        return model.transform(inferenceDF) \
            .select(['bookId', 'prediction']) \
            .orderBy('prediction', ascending=False) \
            .rdd.map(lambda r: (r[0], r[1])) \
            .take(n_recommendations)
    
    def make_recommendations(self, fav_book, n_recommendations):
        """
        make top n book recommendations
        Parameters
        ----------
        fav_book: str, name of user input book
        n_recommendations: int, top n recommendations
        """
        # make inference and get raw recommendations
        print('Recommendation system start to make inference ...')
        t0 = time.time()
        raw_recommends = \
            self._inference(self.model, fav_book, n_recommendations)
        bookIds = [r[0] for r in raw_recommends]
        scores = [r[1] for r in raw_recommends]
        print('It took my system {:.2f}s to make inference \n\
              '.format(time.time() - t0))
        # get book titles
        book_titles = self.booksDF \
            .filter(col('bookId').isin(bookIds)) \
            .select('title') \
            .rdd.map(lambda r: r[0]) \
            .collect()
        # print recommendations
        print('Recommendations for {}:'.format(fav_book))
        for i in range(len(book_titles)):
            print('{0}: {1}, with rating '
                  'of {2}'.format(i+1, book_titles[i], scores[i]))


class Dataset:
    """
    data object make loading raw files easier
    """
    def __init__(self, spark_session, filepath):
        """
        spark dataset constructor
        """
        self.spark = spark_session
        self.sc = spark_session.sparkContext
        self.filepath = filepath
        # build spark data object
        self.RDD = self.load_file_as_RDD(self.filepath)
        self.DF = self.load_file_as_DF(self.filepath)
        
    def load_file_as_RDD(self, filepath):
        ratings_RDD = self.sc.textFile(filepath)
        header = ratings_RDD.take(1)[0]
        return ratings_RDD \
            .filter(lambda line: line != header) \
            .map(lambda line: line.split(",")) \
            .map(lambda tokens: (int(tokens[0]), int(tokens[1]), float(tokens[2]))) # noqa

    def load_file_as_DF(self, filepath):
        ratings_RDD = self.load_file_as_rdd(filepath)
        ratingsRDD = ratings_RDD.map(lambda tokens: Row(
            userId=int(tokens[0]), bookId=int(tokens[1]), rating=float(tokens[2]))) # noqa
        return self.spark.createDataFrame(ratingsRDD)

def tune_ALS(model, train_data, validation_data, maxIter, regParams, ranks):
    """
    grid search function to select the best model based on RMSE of
    validation data
    Parameters
    ----------
    model: spark ML model, ALS
    train_data: spark DF with columns ['user_id', 'book_id', 'rating']
    validation_data: spark DF with columns ['user_id', 'book_id', 'rating']
    maxIter: int, max number of learning iterations
    regParams: list of float, one dimension of hyper-param tuning grid
    ranks: list of float, one dimension of hyper-param tuning grid
    Return
    ------
    The best fitted ALS model with lowest RMSE score on validation data
    """
    # initial
    min_error = float('inf')
    best_rank = -1
    best_regularization = 0
    best_model = None
    for rank in ranks:
        for reg in regParams:
            # get ALS model
            als = model.setMaxIter(maxIter).setRank(rank).setRegParam(reg)
            # train ALS model
            model = als.fit(train_data)
            # evaluate the model by computing the RMSE on the validation data
            predictions = model.transform(validation_data)
            evaluator = RegressionEvaluator(metricName="rmse",
                                            labelCol="rating",
                                            predictionCol="prediction")
            rmse = evaluator.evaluate(predictions)
            print('{} latent factors and regularization = {}: '
                  'validation RMSE is {}'.format(rank, reg, rmse))
            if rmse < min_error:
                min_error = rmse
                best_rank = rank
                best_regularization = reg
                best_model = model
    print('\nThe best model has {} latent factors and '
          'regularization = {}'.format(best_rank, best_regularization))
    return best_model        
        
def parse_args():
    parser = argparse.ArgumentParser(
        prog="Rating Recommender",
        description="Run ALS Rating Recommender")
    parser.add_argument('--path', nargs='?', default='../ratingRecommender',
                        help='input data path')
    parser.add_argument('--books_filename', nargs='?', default='dataset\\cleaned_data_books_version_3.csv',
                        help='provide books filename')
    parser.add_argument('--ratings_filename', nargs='?', default='dataset\\ratings.csv',
                        help='provide ratings filename')
    parser.add_argument('--toread_filename', nargs='?', default='dataset\\to_read.csv',
                        help='provide to read filename')
    parser.add_argument('--book_name', nargs='?', default='',
                        help='provide a recommended book name')
    parser.add_argument('--top_n', type=int, default=10,
                        help='top n book recommendations')
    return parser.parse_args()

if __name__ == '__main__':
    # get args
    args = parse_args()
    data_path = args.path
    books_filename = args.books_filename
    ratings_filename = args.ratings_filename
    toread_filename = args.toread_filename
    book_name = args.book_name
    top_n = args.top_n
    # initial spark
    spark = SparkSession \
        .builder \
        .appName("Rating recommender") \
        .getOrCreate()
    # initial recommender system
    recommender = AlsRecommender(
        spark,
        os.path.join(data_path, books_filename),
        os.path.join(data_path, ratings_filename),
        os.path.join(data_path, toread_filename))
    # set params
    recommender.set_model_params(10, 0.05, 20)
    # make recommendations
    recommender.make_recommendations(book_name, top_n)
    # stop
    spark.stop()