from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib 
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

import scipy.stats

import requests
import pandas as pd
import tempfile
import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn import metrics
import numpy.random as npr
from scipy.stats import wasserstein_distance, ks_2samp
from sklearn.linear_model import LogisticRegression
from datasets_fairness import *



#@title Data utilities
class DataSet:
  def __init__(self, dataset, labels, num_classes = 2):
    self.num_datapoints = dataset.shape[0]
    self.dimension = dataset.shape[1]
    self.random_state = 0
    self.dataset = dataset
    self.labels = labels
    self.num_classes = num_classes
  def get_batch(self, batch_size):
    if batch_size > self.num_datapoints:
      X = self.dataset.values
      Y = self.labels.values
    else:
      X = self.dataset.sample(batch_size, random_state = self.random_state).values
      Y = self.labels.sample(batch_size, random_state = self.random_state).values
    # Y_one_hot = np.zeros((Y.shape[0], self.num_classes))
    # for i in range(self.num_classes):
    #   Y_one_hot[:, i] = (Y == i)*1.0
    self.random_state += 1

    return (X,Y)



class MixtureGaussianDataset:
  def __init__(self, means, 
               variances, 
               probabilities, 
               theta_stars, 
               num_classes=2, 
               max_batch_size = 10000, 
               kernel = lambda a,b : np.dot(a,b)):
    self.means = means
    self.variances = variances
    self.probabilities = probabilities
    self.num_classes = num_classes
    self.theta_stars = theta_stars
    self.cummulative_probabilities = np.zeros(len(probabilities))
    cum_prob = 0
    for i,prob in enumerate(self.probabilities):
      cum_prob += prob
      self.cummulative_probabilities[i] = cum_prob
    self.dimension = theta_stars[0].shape[0]
    self.max_batch_size = max_batch_size
    self.kernel = kernel

  def get_batch(self, batch_size):
    batch_size = min(batch_size, self.max_batch_size)
    X = []
    Y = []
    for _ in range(batch_size):
      val = np.random.random()
      index = 0  
      while index <= len(self.cummulative_probabilities)-1:
        if val < self.cummulative_probabilities[index]:
          break
        index += 1

      x = np.random.multivariate_normal(self.means[index], np.eye(self.dimension)*self.variances[index])
      logit = self.kernel(x, self.theta_stars[index])
      y_val = 1 / (1 + np.exp(-logit))
      y = (np.random.random() >= y_val)*1.0
      X.append(x)
      Y.append(y)
    X = np.array(X)
    Y = np.array(Y)
    return (X,Y)





class SVMDataset:
  def __init__(self, means, 
               variances, 
               probabilities, 
               class_list_per_center,
               num_classes=2, 
               max_batch_size = 10000):
    self.means = means
    self.variances = variances
    self.probabilities = probabilities
    self.num_classes = num_classes
    self.class_list_per_center = class_list_per_center
    self.cummulative_probabilities = np.zeros(len(probabilities))
    cum_prob = 0
    for i,prob in enumerate(self.probabilities):
      cum_prob += prob
      self.cummulative_probabilities[i] = cum_prob
    self.max_batch_size = max_batch_size
    self.num_groups = len(self.means)
    self.dim = self.means[0].shape[0]

  def get_batch(self, batch_size, verbose = False):
    batch_size = min(batch_size, self.max_batch_size)
    X = []
    Y = []
    indices = []
    for _ in range(batch_size):
      val = np.random.random()
      index = 0  
      while index <= len(self.cummulative_probabilities)-1:
        if val < self.cummulative_probabilities[index]:
          break
        index += 1
      
      x = np.random.multivariate_normal(self.means[index], np.eye(self.dim)*self.variances[index])
      y = self.class_list_per_center[index]
      X.append(x)
      Y.append(y)
      indices.append(index)
    X = np.array(X)
    Y = np.array(Y)
    indices = np.array(indices)
    if verbose:
      return (X, Y, indices)
    else:
      return (X,Y)
  
  def plot(self, batch_size, model = None, names = []):
    if names == []:
      names = ["" for _ in range(self.num_groups)]
    if self.dim != 2:
      print("Unable to plot the dataset")
    else:
      colors = ["blue", "red", "green", "yellow", "black", "orange", "purple", "violet", "gray"]
      (X, Y, indices) = self.get_batch( batch_size, verbose = True)
      #print("xvals ", X, "yvals ", Y)
      min_x = float("inf")
      max_x = -float("inf")
      for i in range(self.num_groups):
        X_filtered_0 = []
        X_filtered_1 = []
        for j in range(len(X)):
          if indices[j] == i:
            X_filtered_0.append(X[j][0])
            if X[j][0] < min_x:
              min_x = X[j][0]
            if X[j][0] > max_x:
              max_x = X[j][0]
            X_filtered_1.append(X[j][1])
            
        plt.plot(X_filtered_0, X_filtered_1, 'o', color = colors[i]  , label = "{} {}".format(self.class_list_per_center[i], names[i]) )
      if model != None:
          ## Plot line
          model.plot(min_x, max_x)
      plt.grid(True)
      plt.legend(loc = "lower right")
      #IPython.embed()
      
      #plt.show()



def get_batches(protected_datasets, global_dataset, batch_size):
  global_batch = global_dataset.get_batch(batch_size)

  protected_batches = [protected_dataset.get_batch(batch_size) for protected_dataset in protected_datasets]
  return global_batch, protected_batches


def get_dataset(dataset):

  if dataset == "Mixture":
    PROTECTED_GROUPS = ["A", "B", "C", "D"]
    d = 20
    means = [ -10*np.arange(d)/np.linalg.norm(np.ones(d)), np.zeros(d),  10*np.arange(d)/np.linalg.norm(np.arange(d)), np.ones(d)/np.linalg.norm(np.ones(d))]
    variances = [.4, .41, .41, .41]
    theta_stars = [np.zeros(d),np.zeros(d), np.zeros(d), np.zeros(d)]
    probabilities = [ .3, .1, .5, .1 ]
    kernel = lambda a,b : .1*np.dot(a-b, a-b ) - 1

    protected_datasets_train = [MixtureGaussianDataset([means[i]], [variances[i]], [1], [theta_stars[i]], kernel = kernel) for i in range(len(PROTECTED_GROUPS))]
    protected_datasets_test = [MixtureGaussianDataset([means[i]], [variances[i]], [1], [theta_stars[i]], kernel = kernel) for i in range(len(PROTECTED_GROUPS))]

    train_dataset = MixtureGaussianDataset(means, variances, probabilities, theta_stars, kernel = kernel)
    test_dataset = MixtureGaussianDataset(means, variances, probabilities, theta_stars, kernel = kernel)
  elif dataset == "Adult":
    # PROTECTED_GROUPS = [
    #     'Female_White', 'Female_Black', 'Male_White', 'Male_Black'                
    # ]''

    joint_protected_groups = False
    if joint_protected_groups: 
      PROTECTED_GROUPS = AdultParams.JOINT_PROTECTED_GROUPS
    else: 
      PROTECTED_GROUPS = AdultParams.PROTECTED_GROUPS

    dataframe_all_train, dataframe_all_test, feature_names = read_and_preprocess_adult_data_uai(remove_missing=False)
    # dataframe_all_train = dataframe_all_train.sample(data_size, random_state=random_state)

    # Identify portions of the data corresponding to particuar values of specific protected attributes.
    dataframes_protected_train = collect_adult_protected_dataframes(dataframe_all_train, PROTECTED_GROUPS)
    dataframes_protected_test = collect_adult_protected_dataframes(dataframe_all_test, PROTECTED_GROUPS)

    # Split all data into features and labels.
    x_all_train = dataframe_all_train[feature_names]
    y_all_train = dataframe_all_train[AdultParams.LABEL_COLUMN]
    x_all_test = dataframe_all_test[feature_names]
    y_all_test = dataframe_all_test[AdultParams.LABEL_COLUMN]

    x_protected_train = [df[feature_names] for df in dataframes_protected_train]
    y_protected_train = [df[AdultParams.LABEL_COLUMN] for df in dataframes_protected_train]
    x_protected_test = [df[feature_names] for df in dataframes_protected_test]
    y_protected_test = [df[AdultParams.LABEL_COLUMN] for df in dataframes_protected_test]

    train_dataset = DataSet(x_all_train, y_all_train)
    test_dataset = DataSet(x_all_test, y_all_test) 

    xy_protected_train = collect_adult_protected_xy(
      x_all_train, y_all_train, PROTECTED_GROUPS)
    xy_protected_test = collect_adult_protected_xy(
      x_all_test, y_all_test, PROTECTED_GROUPS) 

    protected_datasets_train = [DataSet(x_vals, y_vals) for (x_vals, y_vals) in xy_protected_train]
    protected_datasets_test = [DataSet(x_vals, y_vals) for (x_vals, y_vals) in xy_protected_test]


    # protected_datasets_train = [DataSet(x_vals, y_vals) for (x_vals, y_vals) in protected_dataframes_train]
    # protected_datasets_test = [DataSet(x_vals, y_vals) for (x_vals, y_vals) in protected_dataframes_test]


    # train_dataset = DataSet(X_train_adult_df, y_train_adult_df)
    # test_dataset = DataSet(X_test_adult_df, y_test_adult_df)

  elif dataset == "German":
    PROTECTED_GROUPS = GermanParams.PROTECTED_THRESHOLDS

    ### LOAD ALL THE DATA ###
    dataframe_all_train, dataframe_all_test, feature_names = read_and_preprocess_german_data()
    # dataframe_all_train = dataframe_all_train.sample(data_size, random_state=random_state)

    # Identify portions of the data corresponding to particuar values of specific
    # protected attributes.
    dataframes_protected_train = collect_german_protected_dataframes(dataframe_all_train)
    dataframes_protected_test = collect_german_protected_dataframes(dataframe_all_test)

    # Split all data into features and labels.
    x_all_train = dataframe_all_train[feature_names]
    y_all_train = dataframe_all_train[GermanParams.LABEL_COLUMN]
    x_all_test = dataframe_all_test[feature_names]
    y_all_test = dataframe_all_test[GermanParams.LABEL_COLUMN]

    x_protected_train = [df[feature_names] for df in dataframes_protected_train]
    y_protected_train = [df[GermanParams.LABEL_COLUMN] for df in dataframes_protected_train]
    x_protected_test = [df[feature_names] for df in dataframes_protected_test]
    y_protected_test = [df[GermanParams.LABEL_COLUMN] for df in dataframes_protected_test]

    # In utilities_final.py: Dataset class to be able to sample batches
    train_dataset = DataSet(x_all_train, y_all_train)
    test_dataset = DataSet(x_all_test, y_all_test) 

    xy_protected_train = collect_german_protected_xy(
      x_all_train, y_all_train, PROTECTED_GROUPS)
    xy_protected_test = collect_german_protected_xy(
      x_all_test, y_all_test, PROTECTED_GROUPS) 

    protected_datasets_train = [DataSet(x_vals, y_vals) for (x_vals, y_vals) in xy_protected_train]
    protected_datasets_test = [DataSet(x_vals, y_vals) for (x_vals, y_vals) in xy_protected_test]
  


  elif dataset == "Bank":

    PROTECTED_GROUPS = BankParams.PROTECTED_GROUPS

    ### LOAD ALL THE DATA ###
    dataframe_all_train, dataframe_all_test, feature_names = read_and_preprocess_bank_data()
    # dataframe_all_train = dataframe_all_train.sample(data_size, random_state=random_state)

    # Identify portions of the data corresponding to particuar values of specific
    # protected attributes.
    dataframes_protected_train = collect_bank_protected_dataframes(dataframe_all_train, PROTECTED_GROUPS)
    dataframes_protected_test = collect_bank_protected_dataframes(dataframe_all_test, PROTECTED_GROUPS)

    # Split all data into features and regression targets.
    x_all_train = dataframe_all_train[feature_names]
    y_all_train = dataframe_all_train[BankParams.LABEL_COLUMN]
    x_all_test = dataframe_all_test[feature_names]
    y_all_test = dataframe_all_test[BankParams.LABEL_COLUMN]

    x_protected_train = [df[feature_names] for df in dataframes_protected_train]
    y_protected_train = [df[BankParams.LABEL_COLUMN] for df in dataframes_protected_train]
    x_protected_test = [df[feature_names] for df in dataframes_protected_test]
    y_protected_test = [df[BankParams.LABEL_COLUMN] for df in dataframes_protected_test]

    # In utilities_final.py: Dataset class to be able to sample batches
    train_dataset = DataSet(x_all_train, y_all_train)
    test_dataset = DataSet(x_all_test, y_all_test) 

    xy_protected_train = collect_bank_protected_xy(
      x_all_train, y_all_train, PROTECTED_GROUPS)
    xy_protected_test = collect_bank_protected_xy(
      x_all_test, y_all_test, PROTECTED_GROUPS) 

    protected_datasets_train = [DataSet(x_vals, y_vals) for (x_vals, y_vals) in xy_protected_train]
    protected_datasets_test = [DataSet(x_vals, y_vals) for (x_vals, y_vals) in xy_protected_test]




  elif dataset == "Crime":

      PROTECTED_GROUPS = CrimeParams.PROTECTED_GROUPS

      ### LOAD ALL THE DATA ###
      dataframe_all_train, dataframe_all_test, feature_names = read_and_preprocess_crime_data()
      # dataframe_all_train = dataframe_all_train.sample(data_size, random_state=random_state)

      # Identify portions of the data corresponding to particuar values of specific protected attributes.
      dataframes_protected_train = collect_crime_protected_dataframes(dataframe_all_train, PROTECTED_GROUPS)
      dataframes_protected_test = collect_crime_protected_dataframes(dataframe_all_test, PROTECTED_GROUPS)

      # Split all data into features and labels.
      x_all_train = dataframe_all_train[feature_names]
      y_all_train = dataframe_all_train[CrimeParams.LABEL_COLUMN]
      x_all_test = dataframe_all_test[feature_names]
      y_all_test = dataframe_all_test[CrimeParams.LABEL_COLUMN]

      x_protected_train = [df[feature_names] for df in dataframes_protected_train]
      y_protected_train = [df[CrimeParams.LABEL_COLUMN] for df in dataframes_protected_train]
      x_protected_test = [df[feature_names] for df in dataframes_protected_test]
      y_protected_test = [df[CrimeParams.LABEL_COLUMN] for df in dataframes_protected_test]
      
      # In utilities_final.py: Dataset class to be able to sample batches
      train_dataset = DataSet(x_all_train, y_all_train)
      test_dataset = DataSet(x_all_test, y_all_test) 

      xy_protected_train = collect_crime_protected_xy(
        x_all_train, y_all_train, PROTECTED_GROUPS)
      xy_protected_test = collect_crime_protected_xy(
        x_all_test, y_all_test, PROTECTED_GROUPS) 
     
      protected_datasets_train = [DataSet(x_vals, y_vals) for (x_vals, y_vals) in xy_protected_train]
      protected_datasets_test = [DataSet(x_vals, y_vals) for (x_vals, y_vals) in xy_protected_test]





  elif dataset == "MultiSVM":
    PROTECTED_GROUPS = ["A", "B", "C", "D"]
    d = 2
    means = [ np.array([0, 5]), np.array([0, 0]), np.array([5, -2]), np.array([5, 5]) ]
    variances = [.5, .5, .5, .5]
    probabilities = [ .3, .3, .2, .2]
    class_list_per_center = [1, 0, 1, 0]
    
    protected_datasets_train = [SVMDataset([means[i]], [variances[i]], [1],  [class_list_per_center[i]]) for i in range(len(PROTECTED_GROUPS))]
    protected_datasets_test = [SVMDataset([means[i]], [variances[i]], [1],  [class_list_per_center[i]]) for i in range(len(PROTECTED_GROUPS))]

    train_dataset = SVMDataset(means, variances, probabilities,  class_list_per_center)
    test_dataset = SVMDataset(means, variances, probabilities, class_list_per_center)


  elif dataset == "SVM":
    PROTECTED_GROUPS = ["A", "B"]
    d = 2
    means = [ -np.arange(d)/np.linalg.norm(np.arange(d)), np.ones(d)/np.linalg.norm(np.ones(d))]
    variances = [1, .1]
    probabilities = [ .5, .5]
    class_list_per_center = [0, 1]
    
    protected_datasets_train = [SVMDataset([means[i]], [variances[i]], [1],  [class_list_per_center[i]]) for i in range(len(PROTECTED_GROUPS))]
    protected_datasets_test = [SVMDataset([means[i]], [variances[i]], [1],  [class_list_per_center[i]]) for i in range(len(PROTECTED_GROUPS))]

    train_dataset = SVMDataset(means, variances, probabilities,  class_list_per_center)
    test_dataset = SVMDataset(means, variances, probabilities, class_list_per_center)
  else:
    raise ValueError("Unrecognized dataset")

  return protected_datasets_train, protected_datasets_test, train_dataset, test_dataset








