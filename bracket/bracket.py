import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import random

NodeKey = int
NodePositions = dict[NodeKey : tuple[int, int]]
BracketEdges = list[tuple[NodeKey,NodeKey]]

# Wants order in WEST, SOUTH, EAST, WIDWEST

"""
Note: This class is very unmaintable, I am aware. For the scope of this project the funcitonallity
  provided is sufficent. It is able to display a bracket of 2**n teams where 2 < n < inf - larger than
  2**6 will result is slow loading. It is also able to flip the branch so that the championship is in the
  middle - similar to how March Maddness Brackets look like.
  I have done by best to comment this describing general ideas, but it is not great.  
"""

class Bracket():
    def __init__(self, num_teams : int, rotate_bracket : bool, results: list[str]):
        # assert power of two
        assert num_teams > 2 and (num_teams & (num_teams - 1) == 0)
        
        self.num_teams : int = num_teams
        self.rotate_bracket : bool = rotate_bracket
        self.node_positions : NodePositions = {}
        self.bracket_edges : BracketEdges = []
        self.graph = nx.Graph()
        
        self.text_size : int = 10
        self.figure_size : tuple[int, int] = (18,9)
        self._initialize_figure()
        
        # helpful for future calculations
        self.num_matchups : int = (2 * num_teams) - 1
        self.rounds : int = int(np.log2(num_teams)) + 1
        # Determine how many rounds of matchups there will be, and how many matchups per round there will be
        self.matchups_per_round : dict[int, int] = {i: int(self.num_teams / (np.pow(2, i))) for i in range(0, self.rounds)} #{0: 16, 1: 8, 2: 4, 3: 2, 4: 1}
        
        # create results if not provided, otherwise use provided results
        if not results or len(results) == 0:
            self._generate_test_results()
        else:
            self.results = results
            
        # create the bracket
        self._create_bracket()
        
    def save(self, filename):
        plt.savefig(filename, bbox_inches='tight')

    def show(self):
        plt.show()

    def _create_bracket(self):
        self._create_bracket_nodes()

        if self.rotate_bracket:
            self._rotate_bracket(180)
        else:
            self.add_winner_edge()

        # nx.draw_networkx_nodes(G, pos, node_size=100, nodelist=[key for key in pos.keys()], node_color="tab:blue", node_shape='s')
        # nx.draw_networkx_labels(G, pos, labels={key : key for key in pos.keys()}, font_size=10, font_family="sans-serif")

        # Draw edges - need to figure out an algorythem for this.
        self.graph.add_edges_from(self.bracket_edges)
        nx.draw_networkx_edges(self.graph, self.node_positions, alpha=0.5, width=2)

        self._draw_edge_labels()

    def _initialize_figure(self):
        plt.figure(figsize=self.figure_size)
        # Customize axes
        ax = plt.gca()
        ax.margins(0.05)
        plt.tight_layout()
        plt.axis("off")

    # creates and appends to list
    # Last node to draw is the last one before this method is called
    def _create_bracket_edges(self, team1 : NodeKey, team2 : NodeKey, result : NodeKey):

        # initially was +1 +2
        middle_btm = self._get_next_available_node()
        middle_top = self._get_next_available_node() + 1

        invisible_nodes = {
            middle_btm : (self.node_positions[result][0], self.node_positions[team1][1]),
            middle_top : (self.node_positions[result][0] , self.node_positions[team2][1])
            }

        # add invisble nodes to graph
        edges = [
            (team1, middle_btm),
            (middle_btm, result),
            (middle_top, team2),
            (result, middle_top)
        ]

        self.node_positions.update(invisible_nodes)
        self.bracket_edges.extend(edges)

    def _rotate_points(self, points_to_rotate : NodePositions, pivot, angle):
        angle_rad = np.radians(angle)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        cx, cy = pivot
        rotated_points : NodePositions = {}
        for key, (x,y) in points_to_rotate.items():
            # shift to origin
            new_x = x - cx
            new_y = y - cy

            # rotate
            rotated_x = round(new_x * cos_a - new_y * sin_a)
            rotated_y = round(new_x * sin_a + new_y * cos_a)

            # shift back
            rotated_points[key] = (rotated_x + cx, rotated_y + cy)

        # update the positions with the rotated points
        self.node_positions.update(rotated_points)

    def _create_bracket_nodes(self):
        current_node_counter : int = 0

        for round in range(0,self.rounds):
            for matchup in range(0, self.matchups_per_round[round]):
                # The starting position (offset) of each round
                round_increment : int = int(max(0, np.pow(2, round) - 1))
                # How much the y position increments per matchup
                matchup_modifier : int = int(np.pow(2,round+1))
                y = (matchup*matchup_modifier) + round_increment
                pos = (round, y)

                # always want to increment & store
                self.node_positions[current_node_counter] = pos
                current_node_counter += 1

        current_node_counter : int = 0

        # Do not do the last round
        for round in range(0,self.rounds-1):
            total_matchups_in_round : int = self.matchups_per_round[round] # 8, 4, 2, 1
            for matchup in range(0, self.matchups_per_round[round]):
                # only do work if % 2. Note will do for both nodes
                if matchup % 2 == 0 and (current_node_counter < self.num_matchups - 3 or not self.rotate_bracket):
                    next_node = current_node_counter + 1
                    end_node = int(current_node_counter + (total_matchups_in_round - (matchup)/2))
                    self._create_bracket_edges(current_node_counter, next_node, end_node)
                elif self.rotate_bracket and (current_node_counter == self.num_matchups - 2 or current_node_counter == self.num_matchups - 3): # create connection at the end
                    edge_increment = 0
                    if not matchup % 2 == 0:
                        edge_increment = -1

                    end_node = int(current_node_counter + edge_increment + (total_matchups_in_round - (matchup + edge_increment)/2))
                    edge = (current_node_counter, end_node)
                    self.bracket_edges.append(edge)

                current_node_counter += 1

    def _rotate_bracket(self, angle):
        nodes_to_rotate = {}
        current_node_counter : int = 0
        # dont need ot do the last round (of 1)
        for round in range(0,self.rounds-1):
            for matchup in range(0, self.matchups_per_round[round]):
            # only add if in upper half of round.
                if (self.matchups_per_round[round] - matchup) <= self.matchups_per_round[round] / 2:
                    nodes_to_rotate[current_node_counter] = self.node_positions[current_node_counter]

                    # also rotate the appened on invisible nodes
                    if matchup % 2 == 0:
                        invisible_node_1 : int = current_node_counter + self.num_matchups
                        invisible_node_2 : int = current_node_counter + self.num_matchups + 1
                        nodes_to_rotate[invisible_node_1] = self.node_positions[invisible_node_1]
                        nodes_to_rotate[invisible_node_2] = self.node_positions[invisible_node_2]

                current_node_counter+=1

        # rotate around the central node. This will update the node positions of the nodes to rotate
        self._rotate_points(nodes_to_rotate, self.node_positions[self.num_matchups-1], angle)

        # now translate the y position of the central node
        self.node_positions[self.num_matchups-1] = (self.node_positions[self.num_matchups-1][0], self.node_positions[self.num_matchups-2][1])

        # now add winner line below 2 nodes and an edge
        # get important nodes
        next_aviable_node_pos = self._get_next_available_node()
        current_middle_node = self.node_positions[self.num_matchups - 1]
        current_left_node = self.node_positions[self.num_matchups - 2]
        current_right_node = self.node_positions[self.num_matchups - 3]
        closest_bracket_node = self.node_positions[next_aviable_node_pos - 1]

        # put it between the final two and between the final line and the next closest bracket line
        left_node_pos = ((current_left_node[0] + current_middle_node[0]) / 2, (current_left_node[1] + closest_bracket_node[1]) / 2)
        current_right_node = ((current_right_node[0] + current_middle_node[0]) / 2, (current_left_node[1] + closest_bracket_node[1]) / 2)
        self.node_positions[next_aviable_node_pos] = left_node_pos
        self.node_positions[next_aviable_node_pos + 1] = current_right_node

        self.bracket_edges.append((next_aviable_node_pos, next_aviable_node_pos + 1))

    def _draw_edge_labels(self):
        labels = {}

        results_counter : int = 0
        for i, edge in enumerate(self.bracket_edges):
            # First & second statement done by virtue of the structure creataed in create_bracket_edges
            # second statement allows for the final result to be added
            # third statement allows for the final matchup other edge on rotated bracket to be added
            if i % 2 == 0 or (i == len(self.bracket_edges) - 1) or (self.rotate_bracket and i == len(self.bracket_edges) - 2):
                labels[edge] = self.results[results_counter]
                results_counter += 1
        nx.draw_networkx_edge_labels(self.graph, self.node_positions, edge_labels=labels, alpha=0.5, font_size = self.text_size)

    def add_winner_edge(self):
        winner : NodeKey = self.num_matchups - 1
        total_pos = self._get_next_available_node()
        self.node_positions[total_pos] = (self.node_positions[winner][0] + 1, self.node_positions[winner][1])
        self.bracket_edges.append((winner, total_pos))

    def _get_next_available_node(self) -> NodeKey:
        return len(self.node_positions)
    
    def _generate_test_results(self) -> list:
        # first clear out results if there are any
        self.results = []
        
        team_counter : int = 0
        round_winners : list = []
        prev_round_winners : list = []
        for round in range(0, self.rounds):
            prev_round_winners = round_winners
            round_winners = []
            for matchup in range(0, self.matchups_per_round[round]):
                team_name : str = ""
                if round == 0:
                    team_name= f"team{team_counter}"
                    team_counter += 1
                else:
                    # choose randomly between the matchup
                    winning_team_indx = random.choice([matchup*2, matchup*2+1])
                    team_name = prev_round_winners[winning_team_indx]

                self.results.append(team_name)
                round_winners.append(team_name)