from march_maddness_ai.conversions import try_similar_names
import march_maddness_ai.NeuralNetwork as NN

import numpy as np
import pandas as pd

Teams = list[str]

class Predictor():
    def __init__(self, startingTeams : Teams, year : int, modelPath: str, layers : list[NN.Layer]):
        self.year : int = year
        self.nn : NN.NeuralNetwork = NN.NeuralNetwork(layers)
        self.nn.Load(modelPath)
        self._create(startingTeams)
        
    def _predict_winner(self, teamName1, teamName2) -> str:
        df = pd.read_pickle(f"data_collection\stats{self.year}.pkl")
        team1 = try_similar_names(df, teamName1)
        team1 = team1.values.flatten().tolist()[1:]
                        
        team2 = try_similar_names(df, teamName2)
        team2 = team2.values.flatten().tolist()[1:]
        
        if not team1:
            return ValueError(f"can't find {teamName1}")
        if not team2:
            return ValueError(f"can't find {teamName2}")
        
        # Predict the result with team 1 as the front team, and then with team 2 as the front team
        input_team1_first = np.array(team1 + team2, dtype=np.float64)
        input_team1_first = input_team1_first[:, np.newaxis]
        result_team1_first = self.nn.Predict(input_team1_first)
        
        input_team2_first = np.array(team2 + team1, dtype=np.float64)
        input_team2_first = input_team2_first[:, np.newaxis]
        result_team2_first = self.nn.Predict(input_team2_first)
        
        # Average the results
        result = (result_team1_first + (1 - result_team2_first)) / 2
        
        # Round the result to get the predicted winner (0 for team 1, 1 for team 2)
        rounded_result = np.round(result[0], 0)
        rounded_result = int(result.item())
        
        if rounded_result == 0:
            return teamName1
        else:
            return teamName2
            
    def _create(self, startingTeams : list[str]):
        # clear current results
        self.results = []
        numTeams = len(startingTeams) 
        # must be even
        assert((numTeams > 2) and (numTeams & (numTeams - 1)) == 0 ) # Bitwise check for power of 2)
        
        rounds = int(np.log2(numTeams) + 1)
        matchups_per_round : dict[int, int] = {i: int(numTeams / (np.pow(2, i))) for i in range(0, rounds)} #{0: 16, 1: 8, 2: 4, 3: 2, 4: 1}
        self.results = startingTeams
        
        round_winners : list = startingTeams
        prev_round_winners : list = []
        for round in range(1, rounds):
            prev_round_winners = round_winners
            round_winners = []
            for matchup in range(0, matchups_per_round[round]):
                winningTeam : str = self._predict_winner(prev_round_winners[matchup*2], prev_round_winners[matchup*2+1])
                self.results.append(winningTeam)
                round_winners.append(winningTeam)
    
    def get_results(self) -> list[str]:
        return self.results