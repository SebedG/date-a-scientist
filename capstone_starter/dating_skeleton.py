import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn import preprocessing
from sklearn import metrics
from sklearn import svm

plt.ioff()


# Generic plotting function as this is done repeatedly
def plot_me(y1, y2, label):
  plt.figure()
  plt.scatter(y1, y2)
  plt.xlabel("Actual")
  plt.ylabel("Predicted")
  plt.title("Actual vs Prediction for " + label + " model")
  plt.savefig("Reg_" + label + ".png", bbox_inches="tight")


# Linear Regression model
def linear_regression(x_data, y_data):
  x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=1)
  lm = LinearRegression()
  lm.fit(x_train, y_train)
  y_predict = lm.predict(x_test)
  print("Linear Regression Score is: %.3f" % lm.score(x_test, y_test))
  print("Linear Regression Mean Squared Error is: %.3f" % metrics.mean_squared_error(y_test, y_predict))
  plot_me(y_test, y_predict, "Linear Regression")

# KNN Classifier model
def k_nearest_classifier(x_data, y_data):
  x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=1)
  scores = []

  for i in range(1, 101): # Test range of neighbour values to find optimum
    classifier = KNeighborsClassifier(n_neighbors=i, weights="distance")
    classifier.fit(x_train, y_train)
    scores.append(classifier.score(x_test, y_test))

  # Plot score vs. number of neighbours
  plt.figure()
  plt.plot(np.linspace(1, 101, 100), scores)
  plt.title("K-Nearest Neighbour Classifier Score vs. number of neighbours")
  plt.xlabel("No of Neighbours value")
  plt.ylabel("Model score")
  plt.savefig("KNNClas_Fitting.png", bbox_inches="tight")

  # Keep best value and re fit and determine scores
  best_i = scores.index(max(scores))
  print("The best number of neighbours was found to be %d" % best_i)
  classifier = KNeighborsClassifier(n_neighbors=best_i, weights="distance")
  classifier.fit(x_train, y_train)
  y_predict = classifier.predict(x_test)
  print("K-Nearest Classifier Score is: %.3f" % classifier.score(x_test, y_test))
  print("K-Nearest Classifier Accuracy: %.3f" % metrics.accuracy_score(y_test, y_predict))
  print("K-Nearest Classifier Recall: %.3f" % metrics.recall_score(y_test, y_predict, average="micro"))
  print("K-Nearest Classifier Precision: %.3f" % metrics.precision_score(y_test, y_predict, average="micro"))
  plot_me(y_test, y_predict, "KNearestClassifier")

# KNN Regressor model
def k_nearest_regressor(x_data, y_data):
  x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=1)
  scores = []

  for i in range(1, 101): # Test range of neighbour values to find optimum
    regressor = KNeighborsRegressor(n_neighbors=i, weights="distance")
    regressor.fit(x_train, y_train)
    scores.append(regressor.score(x_test, y_test))
  # Plot score vs. number of neighbours
  plt.figure()
  plt.plot(np.linspace(1, 101, 100), scores)
  plt.title("K-Nearest Neighbour Regressor Score vs. k")
  plt.xlabel("k value")
  plt.ylabel("Model score")
  plt.savefig("KNNReg_Fitting.png", bbox_inches="tight")

  # Keep best value and re fit and determine scores
  best_i = scores.index(max(scores))
  print("The best number of neighbours was found to be %d" % best_i)
  regressor = KNeighborsRegressor(n_neighbors=best_i, weights="distance")
  regressor.fit(x_train, y_train)
  y_predict = regressor.predict(x_test)
  print("K-Nearest Regressor Score is: %.3f" % regressor.score(x_test, y_test))
  print("K-Nearest Regressor Mean Squared Error is: %.3f" % metrics.mean_squared_error(y_test, y_predict))
  plot_me(y_test, y_predict, "KNearestRegressor")

# Naive Bayes model including import of data to train on
def naive_bayes(df_cleaned, sub_columns):
  sentiment_header = ["sentiment", "number", "date", "query", "user", "tweet"]
  sentiment_data = pd.read_csv("training.1600000.processed.noemoticon.csv",
                               encoding="ISO-8859-1", names=sentiment_header)

  sentiment_data.drop(["user", "number", "date", "query"], axis=1, inplace=True)  # drop unnecessary data
  counter = CountVectorizer()
  counter.fit(sentiment_data.tweet)
  sentiment_counts = counter.transform(sentiment_data.tweet)

  x_train, x_test, y_train, y_test = train_test_split(sentiment_counts, sentiment_data.sentiment, test_size=0.2,
                                                      random_state=1)
  classifier = MultinomialNB() # Create and then train the model
  classifier.fit(x_train, y_train)
  y_predict = classifier.predict(x_test)
  # Calculate scores for the model
  print("Naive Bayes Score is: %.3f" % classifier.score(x_test, y_test))
  print("Accuracy: %.3f" % metrics.accuracy_score(y_test, y_predict))
  print("Recall: %.3f" % metrics.recall_score(y_test, y_predict, average="micro"))
  print("Precision: %.3f" % metrics.precision_score(y_test, y_predict, average="micro"))

  sentiment_list = []
  # Calculate the sentiment for each essay segment
  for column in sub_columns[3:]:
    new_col_name = column + "_sentiment"
    sentiment_list.append(new_col_name)
    df_cleaned[new_col_name] = classifier.predict(counter.transform(df_cleaned[column]))

  df_cleaned["overall_sentiment"] = df_cleaned[sentiment_list].mean(axis=1)  # aggregate essay sentiments
  return df_cleaned

# SVM classifier model
def svc_classifier(x_data, y_data):
  x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=1)
  scores = []

  list_of_Cs = [0.001, 0.01, 0.1, 1, 10] # Test a range hyper-parameter C
  for value in list_of_Cs:
    classifier = svm.SVC(kernel='linear', C=value, class_weight="balanced")
    classifier.fit(x_train, y_train)
    scores.append(classifier.score(x_test, y_test))

  # Plot of model C value vs. score
  plt.figure()
  plt.plot([0.001, 0.01, 0.1, 1, 10], scores)
  plt.title("SVM Classifier Score vs. C")
  plt.xlabel("C value")
  plt.ylabel("Model score")
  plt.savefig("SVM_Class_Score.png", bbox_inches="tight")

  # Find best C parameter then use to fit model and find scores
  best_C = list_of_Cs[scores.index(max(scores))]
  print("The best C value was found to be %f" % best_C)
  classifier = svm.SVC(kernel='linear', C=best_C, class_weight="balanced")
  classifier.fit(x_train, y_train)
  y_predict = classifier.predict(x_test)
  print("SVC Classifier Score is: %.3f" % classifier.score(x_test, y_test))
  print("SVC Classifier Accuracy: %.3f" % metrics.accuracy_score(y_test, y_predict))
  print("SVC Classifier Recall: %.3f" % metrics.recall_score(y_test, y_predict, average="micro"))
  print("SVC Classifier Precision: %.3f" % metrics.precision_score(y_test, y_predict, average="micro"))
  plot_me(y_test, y_predict, "SVC_Classifier")

# Prepare data, read in, process, clean up, investigate initial information etc. here
def prepare_data():
  df = pd.read_csv("profiles.csv")  # Read profile data
  print("The dataset contains %d users" % len(df))
  print("Comprised of %d males and %d females" % (df.sex[df.sex == "m"].count(),
                                                  df.sex[df.sex == "f"].count()))

  # prepare data here
  sub_columns = ["education", "income", "religion", "essay0", "essay1",
                 "essay2", "essay3", "essay4", "essay5", "essay6",
                 "essay7", "essay8", "essay9", "speaks"]
  fill_values = {"essay0": "", "essay1": "", "essay2": "", "essay3": "",
                 "essay4": "", "essay5": "", "essay6": "", "essay7": "",
                 "essay8": "", "essay9": "", "speaks": ""}
  df.fillna(value=fill_values, inplace=True)
  df_cleaned = df.dropna(subset=sub_columns)
  print("The data set contains %d complete user results" % len(df_cleaned))

  print("The education section is split as follows:")
  print(df_cleaned.education.value_counts())

  education_map = {"ph.d program": 10, "graduated from ph.d program": 10,
                   "med school": 10, "graduated from med school": 10,
                   "working on ph.d program": 9, "working on med school": 9,
                   "graduated from masters program": 8, "masters program": 8,
                   "dropped out of ph.d program": 8, "working on masters program": 7,
                   "graduated from college/university": 6, "college/university": 6,
                   "graduated from law school": 6, "law school": 6,
                   "two-year college": 6, "graduated from two-year college": 6,
                   "dropped out of masters program": 6, "working on college/university": 5,
                   "working on two-year college": 5, "working on law school": 5,
                   "dropped out of med school": 5, "dropped out of law school": 5,
                   "dropped out of college/university": 5,
                   "dropped out of two-year college": 5, "graduated from high school": 4,
                   "high school": 4, "working on high school": 3,
                   "dropped out of high school": 2, "space camp": 2,
                   "graduated from space camp": 2, "working on space camp": 1,
                   "dropped out of space camp": 0}
  df_cleaned["mapped_education"] = df_cleaned.education.map(education_map)
  df_cleaned["scaled_education"] = preprocessing.scale(df_cleaned.mapped_education)

  print("The religion section is split as follows:")
  print(df_cleaned.religion.value_counts())

  religion_map = {"agnosticism and very serious about it": 1, "atheism": 0,
                  "atheism and laughing about it": 0,
                  "christianity and very serious about it": 2, "christianity": 2,
                  "catholicism but not too serious about it": 2,
                  "agnosticism and somewhat serious about it": 1,
                  "catholicism and laughing about it": 2, "agnosticism": 1,
                  "agnosticism but not too serious about it": 1,
                  "other and laughing about it": 2,
                  "judaism but not too serious about it": 2,
                  "other and somewhat serious about it": 2,
                  "hinduism but not too serious about it": 2,
                  "agnosticism and laughing about it": 1, "catholicism": 2,
                  "other and very serious about it": 2, "other": 2,
                  "other but not too serious about it": 2,
                  "christianity and somewhat serious about it": 2,
                  "atheism and somewhat serious about it": 0,
                  "atheism but not too serious about it": 0, "judaism": 2,
                  "christianity but not too serious about it": 2,
                  "christianity and laughing about it": 2,
                  "buddhism but not too serious about it": 2,
                  "hinduism and laughing about it": 2, "judaism and laughing about it": 2,
                  "buddhism and somewhat serious about it": 2,
                  "buddhism and laughing about it": 2,
                  "islam and very serious about it": 2,
                  "catholicism and somewhat serious about it": 2, "hinduism": 2,
                  "judaism and somewhat serious about it": 2, "buddhism": 2,
                  "atheism and very serious about it": 0,
                  "judaism and very serious about it": 2, "islam": 2,
                  "hinduism and somewhat serious about it": 2,
                  "islam but not too serious about it": 2,
                  "catholicism and very serious about it": 2,
                  "buddhism and very serious about it": 2,
                  "islam and laughing about it": 2,
                  "islam and somewhat serious about it": 2,
                  "hinduism and very serious about it": 2}
  df_cleaned["mapped_religion"] = df_cleaned.religion.map(religion_map)
  df_cleaned["scaled_religion"] = preprocessing.scale(df_cleaned.mapped_religion)
  df_cleaned["income"].where(df_cleaned["income"] > 0, 0, inplace=True)  # replace -1s with 0s
  df_cleaned["normalised_income"] = preprocessing.normalize([df_cleaned.income]).reshape(-1, 1)
  df_cleaned["scaled_income"] = preprocessing.scale(df_cleaned.income)
  df_cleaned["no_of_languages"] = df_cleaned["speaks"].str.split(",").str.len()

  print(df_cleaned["speaks"].head(10))

  male_data = df_cleaned[df_cleaned.sex == "m"]
  print("The dataset contains %d male users" % len(male_data))
  female_data = df_cleaned[df_cleaned.sex == "f"]
  print("The dataset contains %d female users" % len(female_data))
  plt.figure()
  plt.hist(male_data.mapped_education, bins=10, histtype="step", density=True, fill=False)
  plt.hist(female_data.mapped_education, bins=10, histtype="step", density=True, fill=False)
  plt.xlabel("Education Level")
  plt.ylabel("Normalised Proportion")
  plt.legend(["Male", "Female"])
  plt.xlim(0, 5)
  plt.xticks(np.arange(11), ("None", "In Space Camp", "Space Camp", "In High School", "High School",
                             "In University/College", "University/College", "Studying Masters", "Masters",
                             "Studying PhD/Med School", "PhD/Med School"), rotation=90)
  plt.savefig("EducationMapping.png", bbox_inches="tight")

  plt.figure()
  plt.hist(male_data.mapped_religion, bins=10, histtype="step", density=True, fill=False)
  plt.hist(female_data.mapped_religion, bins=10, histtype="step", density=True, fill=False)
  plt.xlabel("Religious Beliefs")
  plt.ylabel("Normalised Proportion")
  plt.legend(["Male", "Female"])
  plt.xlim(0, 2)
  plt.xticks(np.arange(3), ("Atheist", "Agnostic", "Religious"), rotation=90)
  plt.savefig("ReligionMapping.png", bbox_inches="tight")

  plt.figure()
  plt.hist(male_data.no_of_languages, bins=10, histtype="step", density=True, fill=False, cumulative=True)
  plt.hist(female_data.no_of_languages, bins=10, histtype="step", density=True, fill=False, cumulative=True)
  plt.xlabel("Number of Languages Spoken")
  plt.ylabel("Normalised Proportion")
  plt.legend(["Male", "Female"], loc="upper left")
  plt.savefig("NoOfLanguagesMF.png", bbox_inches="tight")
  return df_cleaned, sub_columns

# Run the various components from here and time them
def model_running():
  start = time.time()
  df, sub_columns = prepare_data()
  print("It took %.2f seconds to prepare the data" % (time.time() - start))
  # # Question 1
  start = time.time()
  df_cleaned = naive_bayes(df, sub_columns)
  naive_bayes_time = time.time()
  print("It took %.2f seconds to train the NB model" % (naive_bayes_time - start))
  features = ["scaled_education", "scaled_religion", "normalised_income"]
  start = time.time()
  linear_regression(df[features], df_cleaned["overall_sentiment"])
  print("It took %.2f seconds to run the linear regression model" % (time.time() - start))

  start = time.time()
  k_nearest_regressor(df[features], np.round(df_cleaned["overall_sentiment"]))
  print("It took %.2f seconds to run the KNN regressor model" % (time.time() - start))

  # Question 2 - Education vs. number of languages
  features = ["mapped_education"]
  start = time.time()
  linear_regression(df[features], df["no_of_languages"])
  print("It took %.2f seconds to run the linear regression model" % (time.time() - start))
  start = time.time()
  k_nearest_classifier(df[features], df["no_of_languages"])
  print("It took %.2f seconds to run the KNN classifier" % (time.time() - start))
  start = time.time()
  svc_classifier(df[features], df["no_of_languages"])
  print("It took %.2f seconds to run the SVC classifier" % (time.time() - start))


model_running()
