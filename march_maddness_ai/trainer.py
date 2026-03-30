from march_maddness_ai.conversions import try_similar_names

from march_maddness_ai.results import all_results, years
from march_maddness_ai import NeuralNetwork as NN

from copy import deepcopy
import numpy as np
import pandas as pd

class Trainer():
    def __init__(self, epochs : int, layers : list[NN.Layer], base_file: str = "trained_models\\2026\\NNSave", load: bool = False):
        self.epochs : int = epochs
        self.neural_network : NN.NeuralNetwork = NN.NeuralNetwork(layers)
        self.neural_network.InitializeRandomWeights(0, 0.5)
        self.training_set_data, self.training_set_labels, self.test_set_data, self.test_set_labels = self._create_data_scores()
        if load:
            # Load the data
            self.neural_network.Load(base_file)
            # Remove the extension from the filepath
            base_file = base_file[:base_file.rfind('.')]
        self.base_file = base_file

    def train(self):
        """Trains the nueral net. Plots the error history and the confusion matrix.
        """
        self.neural_network.Train(self.training_set_data, self.epochs, self.training_set_labels, self.test_set_data, self.test_set_labels, fileName=self.base_file)
        self.neural_network.PlotErrorHistory()
        self.neural_network.ConfMatrix(self.test_set_data, np.rint(self.test_set_labels).astype(int))
    
    def _create_data_scores(self):
        """Creates the training/test data/labels. And randomizes for training.
        
        Returns:
            np.array: training set
            np.array: training lebels
            np.array: test set
            np.array: test labels
        """
        temp_data = None
        temp_labels = None
        
        for i, year in enumerate(all_results):
            # Note: need to change this path for your model.
            df = pd.read_pickle(f"data_collection\stats{years[i]}.pkl")
            for j, game in enumerate(year):
                try:
                    # Get the vector given a team's name.
                    team1 = try_similar_names(df, game[0])
                    team1 = team1.values.flatten().tolist()[1:]
                                    
                    team2 = try_similar_names(df, game[1])
                    team2 = team2.values.flatten().tolist()[1:]
                    
                    if not team1 or not team2:
                        continue
                    
                    # Create two sets of data and labels. One for team 1 vs team2 and one for team2 vs team 1.
                    t1t2 = deepcopy(team1)
                    t2t1 = deepcopy(team2)
                    
                    t1t2.extend(team2)
                    t2t1.extend(team1)
                    t1t2 = np.array(t1t2, dtype=np.float64)
                    t2t1 = np.array(t2t1, dtype=np.float64)
                    
                    label1 = game[2] / (game[2] + game[3])
                    label2 = game[3] / (game[2] + game[3])
                    
                    if temp_data is not None:
                        # Append to current array
                        temp_data = np.append(temp_data, (t1t2,), axis=0)
                        temp_labels = np.append(temp_labels, [label1], axis=0)
                        temp_data = np.append(temp_data, (t2t1,), axis=0)
                        temp_labels = np.append(temp_labels, [label2], axis=0)
                    else:
                        # Create a new np array
                        temp_data = np.array((t1t2,), dtype=np.float64)
                        temp_labels = np.full((1,), label1, dtype=np.float64)
                        temp_data = np.append(temp_data, (t2t1,), axis=0)
                        temp_labels = np.append(temp_labels, [label2], axis=0)
                            
                except Exception as e:
                    print(f"{e}")
        
        if temp_labels is None:
            raise Exception("No useable data is found for training")
        
        # Randomize the data
        permutation = np.random.permutation(len(temp_labels))
        temp_labels = temp_labels[permutation]
        temp_data = temp_data[permutation]
            
        # 80% of data is for training. Other 20% is test
        trainingSize = int(len(temp_labels) * 0.8)
        
        training_set_data, test_set_data = np.split(temp_data, [trainingSize])
        training_set_labels, test_set_labels = np.split(temp_labels, [trainingSize])
        training_set_labels = training_set_labels.reshape(-1,1)
        test_set_labels = test_set_labels.reshape(-1,1)
        
        return training_set_data, training_set_labels, test_set_data, test_set_labels