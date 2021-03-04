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
import ray
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy.random as npr
from scipy.stats import wasserstein_distance, ks_2samp
from sklearn.linear_model import LogisticRegression
import os
import pickle

from experiment_regret import *

# ray.init()


#@ray.remote
def run_experiment_parallel(dataset, logging_frequency, max_num_steps, logistic_learning_rate,threshold, biased_threshold, batch_size, 
	random_init, fit_intercept, mahalanobis_regularizer, adjust_mahalanobis, epsilon_greedy, epsilon, alpha, MLP):

	timesteps, test_biased_accuracies_cum_averages, accuracies_cum_averages, train_biased_accuracies_cum_averages, train_cum_regret, loss_validation, loss_validation_biased, loss_baseline = run_regret_experiment_pytorch( dataset, 
																					logging_frequency, 
																					max_num_steps, 
																				    logistic_learning_rate, 
																				    threshold, 
																				    biased_threshold, 
																				    batch_size, 
																				    mahalanobis_regularizer,           
																				    adjust_mahalanobis,
																				    epsilon_greedy, 
																				    epsilon, 
																				    alpha, 
																				    MLP = MLP )
	return timesteps, test_biased_accuracies_cum_averages, accuracies_cum_averages, train_biased_accuracies_cum_averages, train_cum_regret,loss_validation, loss_validation_biased, loss_baseline





def run_and_plot(dataset, logging_frequency, max_num_steps, logistic_learning_rate, threshold, 
	biased_threshold, batch_size, random_init, fit_intercept, num_experiments, mahalanobis_regularizer, adjust_mahalanobis, epsilon_greedy, epsilon, alpha, MLP ):
	path = os.getcwd()

	network_type = "MLP" if MLP else "Linear"
	base_data_directory = "{}/experiment_results/T{}/{}/data".format(path, max_num_steps, network_type)
	base_figs_directory = "{}/experiment_results/T{}/{}/figs".format(path, max_num_steps, network_type)


	if not os.path.isdir(base_data_directory):
		try:

			os.makedirs(base_figs_directory)
			os.makedirs(base_data_directory)
		except OSError:
			print("Creation of directories failed")
		else:
			print("Successfully created the directory")



	if epsilon_greedy and adjust_mahalanobis:
		raise ValueError("Both epsilon greedy and adjust mahalanobis are on!")

	# IPython.embed()
	# raise ValueError("asdlfkm")
	# experiment_summaries = [ run_experiment_parallel.remote( dataset, logging_frequency, max_num_steps, logistic_learning_rate,threshold, biased_threshold, batch_size, 
	# random_init, fit_intercept, mahalanobis_regularizer, adjust_mahalanobis, epsilon_greedy, epsilon, alpha, MLP ) for _ in range(num_experiments)]

	experiment_summaries = [ run_experiment_parallel( dataset, logging_frequency, max_num_steps, logistic_learning_rate,threshold, biased_threshold, batch_size, 
	random_init, fit_intercept, mahalanobis_regularizer, adjust_mahalanobis, epsilon_greedy, epsilon, alpha, MLP ) for _ in range(num_experiments)]


	#experiment_summaries = ray.get(experiment_summaries)


	timesteps = experiment_summaries[0][0]

	test_biased_accuracies_cum_averages_summary = np.zeros((num_experiments, int(max_num_steps/logging_frequency)))
	accuracies_cum_averages_summary = np.zeros((num_experiments, int(max_num_steps/logging_frequency)))
	train_biased_accuracies_cum_averages_summary = np.zeros((num_experiments, int(max_num_steps/logging_frequency)))
	train_cum_regret_summary = np.zeros((num_experiments, int(max_num_steps/logging_frequency)))
	loss_validation_summary = np.zeros((num_experiments, int(max_num_steps/logging_frequency)))
	loss_validation_biased_summary = np.zeros((num_experiments, int(max_num_steps/logging_frequency)))

	loss_validation_baseline_summary = np.zeros(num_experiments)


	for j in range(num_experiments):
		test_biased_accuracies_cum_averages_summary[j,:] = experiment_summaries[j][1]
		accuracies_cum_averages_summary[j,:] = experiment_summaries[j][2]
		train_biased_accuracies_cum_averages_summary[j,:] = experiment_summaries[j][3]
		train_cum_regret_summary[j, :] = experiment_summaries[j][4]
		loss_validation_summary[j,:] = experiment_summaries[j][5]
		loss_validation_biased_summary[j, :] = experiment_summaries[j][6]
		loss_validation_baseline_summary[j] = experiment_summaries[j][7]


	mean_test_biased_accuracies_cum_averages = np.mean(test_biased_accuracies_cum_averages_summary, axis = 0)
	std_test_biased_accuracies_cum_averages = np.std(test_biased_accuracies_cum_averages_summary, axis = 0)

	mean_accuracies_cum_averages = np.mean(accuracies_cum_averages_summary, axis = 0)
	std_accuracies_cum_averages = np.std(accuracies_cum_averages_summary, axis =0 )

	mean_train_biased_accuracies_cum_averages = np.mean(train_biased_accuracies_cum_averages_summary, axis = 0)
	std_train_biased_accuracies_cum_averages = np.std(train_biased_accuracies_cum_averages_summary, axis = 0)

	mean_train_cum_regret_averages = np.mean(train_cum_regret_summary, axis = 0)
	std_train_cum_regret_averages = np.std(train_cum_regret_summary, axis = 0)

	mean_loss_validation_averages = np.mean(loss_validation_summary, axis = 0)
	std_loss_validation_averages = np.std(loss_validation_summary, axis = 0)

	mean_loss_validation_biased_averages = np.mean(loss_validation_biased_summary, axis = 0)
	std_loss_validation_biased_averages = np.std(loss_validation_biased_summary, axis = 0)

	mean_loss_validation_baseline_summary = np.mean(loss_validation_baseline_summary)
	std_loss_validation_baseline_summary = np.std(loss_validation_baseline_summary)




	plt.plot(timesteps,mean_test_biased_accuracies_cum_averages, label = "Online Decision Test - no optimism", linestyle = "dashed", linewidth = 3.5, color="blue")
	plt.fill_between(timesteps, mean_test_biased_accuracies_cum_averages - .5*std_test_biased_accuracies_cum_averages, 
		mean_test_biased_accuracies_cum_averages + .5*std_test_biased_accuracies_cum_averages, color = "blue", alpha = .1 )

	plt.plot(timesteps,mean_accuracies_cum_averages, label = "All data SGD Test", linestyle = "dashed", linewidth = 3.5, color = "red")
	plt.fill_between(timesteps, mean_accuracies_cum_averages - .5*std_accuracies_cum_averages, 
		mean_accuracies_cum_averages + .5*std_accuracies_cum_averages,color = "red", alpha = .1 )

	plt.plot(timesteps, mean_train_biased_accuracies_cum_averages, label = "Online Decision Train", linestyle = "dashed", linewidth = 3.5, color = "violet"  )
	plt.fill_between(timesteps, mean_train_biased_accuracies_cum_averages+ .5*std_train_biased_accuracies_cum_averages, 
		mean_train_biased_accuracies_cum_averages - .5*std_train_biased_accuracies_cum_averages, color = "violet", alpha = .1)

	if epsilon_greedy:
		plt.title("Test and Train Accuracies {} - Epsilon Greedy {}".format(dataset, epsilon))
		plot_name = "{}_test_train_accuracies_epsgreedy_{}".format(dataset, epsilon)
	if adjust_mahalanobis:
		plt.title("Test and Train Accuracies {} - Optimism alpha {} - Mreg {}".format(dataset, alpha, mahalanobis_regularizer))
		plot_name = "{}_test_train_accuracies_optimism_alpha_{}_mahreg_{}".format(dataset, alpha, mahalanobis_regularizer)
	if not epsilon_greedy and not adjust_mahalanobis:
		plt.title("Test and Train Accuracies {} ".format(dataset))
		plot_name  = "{}_test_train_accuracies_biased".format(dataset)

	plt.xlabel("Timesteps")
	plt.ylabel("Accuracy")
	plt.legend(loc = "lower right")
	plt.savefig("{}/{}.png".format(base_figs_directory, plot_name))
	plt.close('all')


	plt.plot(timesteps, mean_train_cum_regret_averages, label = "Regret", linestyle = "dashed", linewidth = 3.5, color = "blue")
	plt.fill_between(timesteps, mean_train_cum_regret_averages - .5*std_train_cum_regret_averages, 
		mean_train_cum_regret_averages + .5*std_train_cum_regret_averages, color = "blue", alpha = .1)
	

	if epsilon_greedy:
		plt.title("Regret {} - Epsilon Greedy {}".format(dataset, epsilon))
		plot_name = "{}_regret_epsgreedy_{}".format(dataset, epsilon)
	if adjust_mahalanobis:
		plt.title("Regret {} - Optimism alpha {} - Mreg {}".format(dataset, alpha, mahalanobis_regularizer))
		plot_name = "{}_regret_optimism_alpha_{}_mahreg_{}".format(dataset, alpha, mahalanobis_regularizer)
	if not epsilon_greedy and not adjust_mahalanobis:
		plt.title("Regret {} ".format(dataset))
		plot_name  = "{}_regret_biased".format(dataset)


	#plot_name = "{}_regret".format(plot_name)

	plt.xlabel("Timesteps")
	plt.ylabel("Regret")
	plt.legend(loc = "lower right")
	plt.savefig("{}/{}.png".format(base_figs_directory, plot_name))
	plt.close('all')


	### LOSS PLOTS
	plt.plot(timesteps, mean_loss_validation_averages, label = "Unbiased model loss", linestyle = "dashed", linewidth = 3.5, color = "blue")
	plt.fill_between(timesteps, mean_loss_validation_averages - .5*std_loss_validation_averages, 
		mean_loss_validation_averages + .5*std_loss_validation_averages, color = "blue", alpha = .1)

	plt.plot(timesteps, mean_loss_validation_biased_averages, label = "Biased model loss", linestyle = "dashed", linewidth = 3.5, color = "red")
	plt.fill_between(timesteps, mean_loss_validation_biased_averages - .5*std_loss_validation_biased_averages, 
		mean_loss_validation_biased_averages + .5*std_loss_validation_biased_averages, color = "red", alpha = .1)

	plt.plot(timesteps, [mean_loss_validation_baseline_summary]*len(timesteps), label = "Baseline Loss", linestyle = "dashed", linewidth = 3.5, color = "black")
	plt.fill_between(timesteps, [mean_loss_validation_baseline_summary - .5*std_loss_validation_baseline_summary]*len(timesteps), 
		[mean_loss_validation_baseline_summary + .5*std_loss_validation_baseline_summary]*len(timesteps), color = "black", alpha = .1)

	if epsilon_greedy:
		plt.title("Loss {} - Epsilon Greedy {}".format(dataset, epsilon))
		plot_name = "{}_loss_epsgreedy_{}".format(dataset, epsilon)
	if adjust_mahalanobis:
		plt.title("Loss {} - Optimism alpha {} - Mreg {}".format(dataset, alpha, mahalanobis_regularizer))
		plot_name = "{}_loss_optimism_alpha_{}_mahreg_{}".format(dataset, alpha, mahalanobis_regularizer)
	if not epsilon_greedy and not adjust_mahalanobis:
		plt.title("Loss {} ".format(dataset))
		plot_name  = "{}_loss_biased".format(dataset)


	#plot_name = "{}_loss".format(plot_name)

	plt.xlabel("Timesteps")
	plt.ylabel("Loss")
	plt.legend(loc = "lower right")
	plt.savefig("{}/{}.png".format(base_figs_directory, plot_name))
	plt.close('all')



	pickle.dump((timesteps, mean_test_biased_accuracies_cum_averages, std_test_biased_accuracies_cum_averages, mean_accuracies_cum_averages, std_accuracies_cum_averages, 
		mean_train_biased_accuracies_cum_averages, std_train_biased_accuracies_cum_averages, 
		max_num_steps, mean_loss_validation_averages, std_loss_validation_averages,mean_loss_validation_biased_averages, 
		std_loss_validation_biased_averages ), open("{}/{}.p".format(base_data_directory, plot_name), "wb"))



def main():
	dataset = "Adult"
	logging_frequency = 10
	max_num_steps = 1000
	logistic_learning_rate = .01
	threshold = .5
	biased_threshold = .5
	batch_size = 10
	random_init = True
	fit_intercept = True

	num_experiments = 2

	adjust_mahalanobis = False
	mahalanobis_regularizer = .1
	alpha = 0

	epsilon_greedy = False
	epsilon = .1

	MLP = True

	#run without any optimism or epsilon greedy
	run_and_plot(dataset, logging_frequency, max_num_steps, logistic_learning_rate, threshold, 
				biased_threshold, batch_size, random_init, fit_intercept, num_experiments, 
				mahalanobis_regularizer, adjust_mahalanobis, epsilon_greedy, epsilon, alpha, MLP)

	for alpha in [1]:#, 2, 3, 4, 5, 10]:
		adjust_mahalanobis = True
		epsilon_greedy = False
		for mahalanobis_regularizer in [.1]:#, 1]:#[.1, 1]:
			run_and_plot(dataset, logging_frequency, max_num_steps, logistic_learning_rate, threshold, 
				biased_threshold, batch_size, random_init, fit_intercept, num_experiments, 
				mahalanobis_regularizer, adjust_mahalanobis, epsilon_greedy, epsilon, alpha, MLP)

	for epsilon in [.1]:#, .2]:#, .2, .5]:	
		adjust_mahalanobis = False
		epsilon_greedy = True
		run_and_plot(dataset, logging_frequency, max_num_steps, logistic_learning_rate, threshold, 
			biased_threshold, batch_size, random_init, fit_intercept, num_experiments, 
			mahalanobis_regularizer, adjust_mahalanobis, epsilon_greedy, epsilon, alpha, MLP)



main()