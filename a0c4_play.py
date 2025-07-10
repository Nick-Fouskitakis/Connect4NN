# %%
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(0)
from tqdm.notebook import trange, tqdm

import random
import math

# %%
class TicTacToe:
    def __init__(self):
        self.row_count = 3
        self.column_count = 3
        self.action_size = self.row_count * self.column_count

    def __repr__(self):
        return "TicTacToe"

    def get_initial_state(self):
        return np.zeros((self.row_count, self.column_count))

    def get_next_state(self, state, action, player):
        row = action // self.column_count
        column = action % self.column_count
        state[row, column] = player
        return state

    def get_valid_moves(self, state):
        return (state.reshape(-1) == 0).astype(np.uint8)

    def check_win(self, state, action):
        if action == None:
            return False

        row = action // self.column_count
        column = action % self.column_count
        player = state[row, column]

        return (
            np.sum(state[row, :]) == player * self.column_count
            or np.sum(state[:, column]) == player * self.row_count
            or np.sum(np.diag(state)) == player * self.row_count
            or np.sum(np.diag(np.flip(state, axis=0))) == player * self.row_count
        )

    def get_value_and_terminated(self, state, action):
        if self.check_win(state, action):
            return 1, True
        if np.sum(self.get_valid_moves(state)) == 0:
            return 0, True
        return 0, False

    def get_opponent(self, player):
        return -player

    def get_opponent_value(self, value):
        return -value

    def change_perspective(self, state, player):
        return state * player

    def get_encoded_state(self, state):
        encoded_state = np.stack(
            (state == -1, state == 0, state == 1)
        ).astype(np.float32)

        if len(state.shape) == 3:
            encoded_state = np.swapaxes(encoded_state, 0, 1)

        return encoded_state

# %%
class ConnectFour:
    def __init__(self):
        self.row_count = 6
        self.column_count = 7
        self.action_size = self.column_count
        self.in_a_row = 4

    def __repr__(self):
        return "ConnectFour"

    def get_initial_state(self):
        return np.zeros((self.row_count, self.column_count))

    def get_next_state(self, state, action, player):
        row = np.max(np.where(state[:, action] == 0))
        state[row, action] = player
        return state

    def get_valid_moves(self, state):
        return (state[0] == 0).astype(np.uint8)

    def check_win(self, state, action):
        if action == None:
            return False

        row = np.min(np.where(state[:, action] != 0))
        column = action
        player = state[row][column]

        def count(offset_row, offset_column):
            for i in range(1, self.in_a_row):
                r = row + offset_row * i
                c = action + offset_column * i
                if (
                    r < 0
                    or r >= self.row_count
                    or c < 0
                    or c >= self.column_count
                    or state[r][c] != player
                ):
                    return i - 1
            return self.in_a_row - 1

        return (
            count(1, 0) >= self.in_a_row - 1 # vertical
            or (count(0, 1) + count(0, -1)) >= self.in_a_row - 1 # horizontal
            or (count(1, 1) + count(-1, -1)) >= self.in_a_row - 1 # top left diagonal
            or (count(1, -1) + count(-1, 1)) >= self.in_a_row - 1 # top right diagonal
        )

    def get_value_and_terminated(self, state, action):
        if self.check_win(state, action):
            return 1, True
        if np.sum(self.get_valid_moves(state)) == 0:
            return 0, True
        return 0, False

    def get_opponent(self, player):
        return -player

    def get_opponent_value(self, value):
        return -value

    def change_perspective(self, state, player):
        return state * player

    def get_encoded_state(self, state):
        encoded_state = np.stack(
            (state == -1, state == 0, state == 1)
        ).astype(np.float32)

        if len(state.shape) == 3:
            encoded_state = np.swapaxes(encoded_state, 0, 1)

        return encoded_state

# %%
class ResNet(nn.Module):
    def __init__(self, game, num_resBlocks, num_hidden, device):
        super().__init__()

        self.device = device
        self.startBlock = nn.Sequential(
            nn.Conv2d(3, num_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU()
        )

        self.backBone = nn.ModuleList(
            [ResBlock(num_hidden) for i in range(num_resBlocks)]
        )

        self.policyHead = nn.Sequential(
            nn.Conv2d(num_hidden, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * game.row_count * game.column_count, game.action_size)
        )

        self.valueHead = nn.Sequential(
            nn.Conv2d(num_hidden, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3 * game.row_count * game.column_count, 1),
            nn.Tanh()
        )

        self.to(device)

    def forward(self, x):
        x = self.startBlock(x)
        for resBlock in self.backBone:
            x = resBlock(x)
        policy = self.policyHead(x)
        value = self.valueHead(x)
        return policy, value


class ResBlock(nn.Module):
    def __init__(self, num_hidden):
        super().__init__()
        self.conv1 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_hidden)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x


# %%
class Node:
    def __init__(self, game, args, state, parent=None, action_taken=None, prior=0, visit_count=0):
        self.game = game
        self.args = args
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.prior = prior

        self.children = []

        self.visit_count = visit_count
        self.value_sum = 0

    def is_fully_expanded(self):
        return len(self.children) > 0

    def select(self):
        best_child = None
        best_ucb = -np.inf

        for child in self.children:
            ucb = self.get_ucb(child)
            if ucb > best_ucb:
                best_child = child
                best_ucb = ucb

        return best_child

    def get_ucb(self, child):
        if child.visit_count == 0:
            q_value = 0
        else:
            q_value = 1 - ((child.value_sum / child.visit_count) + 1) / 2
        return q_value + self.args['C'] * (math.sqrt(self.visit_count) / (child.visit_count + 1)) * child.prior

    def expand(self, policy):
        for action, prob in enumerate(policy):
            if prob > 0:
                child_state = self.state.copy()
                child_state = self.game.get_next_state(child_state, action, 1)
                child_state = self.game.change_perspective(child_state, player=-1)

                child = Node(self.game, self.args, child_state, self, action, prob)
                self.children.append(child)

        return child

    def backpropagate(self, value):
        self.value_sum += value
        self.visit_count += 1

        value = self.game.get_opponent_value(value)
        if self.parent is not None:
            self.parent.backpropagate(value)


class MCTS:
    def __init__(self, game, args, model):
        self.game = game
        self.args = args
        self.model = model

    @torch.no_grad()
    def search(self, state):
        root = Node(self.game, self.args, state, visit_count=1)

        policy, _ = self.model(
            torch.tensor(self.game.get_encoded_state(state), device=self.model.device).unsqueeze(0)
        )
        policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
        policy = (1 - self.args['dirichlet_epsilon']) * policy + self.args['dirichlet_epsilon'] \
            * np.random.dirichlet([self.args['dirichlet_alpha']] * self.game.action_size)

        valid_moves = self.game.get_valid_moves(state)
        policy *= valid_moves
        policy /= np.sum(policy)
        root.expand(policy)

        for search in range(self.args['num_searches']):
            node = root

            while node.is_fully_expanded():
                node = node.select()

            value, is_terminal = self.game.get_value_and_terminated(node.state, node.action_taken)
            value = self.game.get_opponent_value(value)

            if not is_terminal:
                policy, value = self.model(
                    torch.tensor(self.game.get_encoded_state(node.state), device=self.model.device).unsqueeze(0)
                )
                policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
                valid_moves = self.game.get_valid_moves(node.state)
                policy *= valid_moves
                policy /= np.sum(policy)

                value = value.item()

                node.expand(policy)

            node.backpropagate(value)


        action_probs = np.zeros(self.game.action_size)
        for child in root.children:
            action_probs[child.action_taken] = child.visit_count
        action_probs /= np.sum(action_probs)
        return action_probs


# %%
class AlphaZero:
    def __init__(self, model, optimizer, game, args):
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.mcts = MCTS(game, args, model)

    def selfPlay(self):
        memory = []
        player = 1
        state = self.game.get_initial_state()

        while True:
            neutral_state = self.game.change_perspective(state, player)
            action_probs = self.mcts.search(neutral_state)

            memory.append((neutral_state, action_probs, player))

            temperature_action_probs = action_probs ** (1 / self.args['temperature']) # Divide temperature_action_probs with its sum in case of an error
            action = np.random.choice(self.game.action_size, p=temperature_action_probs)

            state = self.game.get_next_state(state, action, player)

            value, is_terminal = self.game.get_value_and_terminated(state, action)

            if is_terminal:
                returnMemory = []
                for hist_neutral_state, hist_action_probs, hist_player in memory:
                    hist_outcome = value if hist_player == player else self.game.get_opponent_value(value)
                    returnMemory.append((
                        self.game.get_encoded_state(hist_neutral_state),
                        hist_action_probs,
                        hist_outcome
                    ))
                return returnMemory

            player = self.game.get_opponent(player)

    def train(self, memory):
        random.shuffle(memory)
        for batchIdx in range(0, len(memory), self.args['batch_size']):
            sample = memory[batchIdx:min(len(memory) - 1, batchIdx + self.args['batch_size'])] # Change to memory[batchIdx:batchIdx+self.args['batch_size']] in case of an error
            state, policy_targets, value_targets = zip(*sample)

            state, policy_targets, value_targets = np.array(state), np.array(policy_targets), np.array(value_targets).reshape(-1, 1)

            state = torch.tensor(state, dtype=torch.float32, device=self.model.device)
            policy_targets = torch.tensor(policy_targets, dtype=torch.float32, device=self.model.device)
            value_targets = torch.tensor(value_targets, dtype=torch.float32, device=self.model.device)

            out_policy, out_value = self.model(state)

            policy_loss = F.cross_entropy(out_policy, policy_targets)
            value_loss = F.mse_loss(out_value, value_targets)
            loss = policy_loss + value_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def learn(self):
        for iteration in range(self.args['num_iterations']):
            memory = []

            self.model.eval()
            for selfPlay_iteration in trange(self.args['num_selfPlay_iterations']):
                memory += self.selfPlay()

            self.model.train()
            for epoch in trange(self.args['num_epochs']):
                self.train(memory)

            torch.save(self.model.state_dict(), f"model_{iteration}_{self.game}.pt")
            torch.save(self.optimizer.state_dict(), f"optimizer_{iteration}_{self.game}.pt")

# %%
class MCTSParallel:
    def __init__(self, game, args, model):
        self.game = game
        self.args = args
        self.model = model

    @torch.no_grad()
    def search(self, states, spGames):
        policy, _ = self.model(
            torch.tensor(self.game.get_encoded_state(states), device=self.model.device)
        )
        policy = torch.softmax(policy, axis=1).cpu().numpy()
        policy = (1 - self.args['dirichlet_epsilon']) * policy + self.args['dirichlet_epsilon'] \
            * np.random.dirichlet([self.args['dirichlet_alpha']] * self.game.action_size, size=policy.shape[0])

        for i, spg in enumerate(spGames):
            spg_policy = policy[i]
            valid_moves = self.game.get_valid_moves(states[i])
            spg_policy *= valid_moves
            spg_policy /= np.sum(spg_policy)

            spg.root = Node(self.game, self.args, states[i], visit_count=1)
            spg.root.expand(spg_policy)

        for search in range(self.args['num_searches']):
            for spg in spGames:
                spg.node = None
                node = spg.root

                while node.is_fully_expanded():
                    node = node.select()

                value, is_terminal = self.game.get_value_and_terminated(node.state, node.action_taken)
                value = self.game.get_opponent_value(value)

                if is_terminal:
                    node.backpropagate(value)

                else:
                    spg.node = node

            expandable_spGames = [mappingIdx for mappingIdx in range(len(spGames)) if spGames[mappingIdx].node is not None]

            if len(expandable_spGames) > 0:
                states = np.stack([spGames[mappingIdx].node.state for mappingIdx in expandable_spGames])

                policy, value = self.model(
                    torch.tensor(self.game.get_encoded_state(states), device=self.model.device)
                )
                policy = torch.softmax(policy, axis=1).cpu().numpy()
                value = value.cpu().numpy()

            for i, mappingIdx in enumerate(expandable_spGames):
                node = spGames[mappingIdx].node
                spg_policy, spg_value = policy[i], value[i]

                valid_moves = self.game.get_valid_moves(node.state)
                spg_policy *= valid_moves
                spg_policy /= np.sum(spg_policy)

                node.expand(spg_policy)
                node.backpropagate(spg_value)

# %%
class AlphaZeroParallel:
    def __init__(self, model, optimizer, game, args):
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.mcts = MCTSParallel(game, args, model)

    def self_play(self):
        return_memory = []
        player = 1
        sp_games = [SPG(self.game) for spg in range(self.args['num_parallel_games'])]

        progress = tqdm(total=len(sp_games), desc="Self-play Games:")
        
        while len(sp_games) > 0:
            before = len(sp_games)
            states = np.stack([spg.state for spg in sp_games])
            neutral_states = self.game.change_perspective(states, player)

            self.mcts.search(neutral_states, sp_games)

            for i in range(len(sp_games))[::-1]:
                spg = sp_games[i]

                action_probs = np.zeros(self.game.action_size)
                for child in spg.root.children:
                    action_probs[child.action_taken] = child.visit_count
                action_probs /= np.sum(action_probs)

                spg.memory.append((spg.root.state, action_probs, player))

                temperature_action_probs = action_probs ** (1 / self.args['temperature'])
                action = np.random.choice(self.game.action_size, p=temperature_action_probs/np.sum(temperature_action_probs)) # Divide temperature_action_probs with its sum in case of an error

                spg.state = self.game.get_next_state(spg.state, action, player)

                value, is_terminal = self.game.get_value_and_terminated(spg.state, action)

                if is_terminal:
                    for hist_neutral_state, hist_action_probs, hist_player in spg.memory:
                        hist_outcome = value if hist_player == player else self.game.get_opponent_value(value)
                        return_memory.append((
                            self.game.get_encoded_state(hist_neutral_state),
                            hist_action_probs,
                            hist_outcome
                        ))
                    del sp_games[i]

            player = self.game.get_opponent(player)
            after = len(sp_games)
            progress.update(before - after)

        progress.close()
        return return_memory

    def train(self, memory):
        random.shuffle(memory)
        for batchIdx in range(0, len(memory), self.args['batch_size']):
            sample = memory[batchIdx:min(len(memory) - 1, batchIdx + self.args['batch_size'])] # Change to memory[batchIdx:batchIdx+self.args['batch_size']] in case of an error
            state, policy_targets, value_targets = zip(*sample)

            state, policy_targets, value_targets = np.array(state), np.array(policy_targets), np.array(value_targets).reshape(-1, 1)

            state = torch.tensor(state, dtype=torch.float32, device=self.model.device)
            policy_targets = torch.tensor(policy_targets, dtype=torch.float32, device=self.model.device)
            value_targets = torch.tensor(value_targets, dtype=torch.float32, device=self.model.device)

            out_policy, out_value = self.model(state)

            policy_loss = F.cross_entropy(out_policy, policy_targets)
            value_loss = F.mse_loss(out_value, value_targets)
            loss = policy_loss + value_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def learn(self):
        # for iteration in range(self.args['num_iterations']):
        #     memory = []

        #     self.model.eval()
        #     for selfPlay_iteration in trange(self.args['num_selfPlay_iterations'] // self.args['num_parallel_games']):
        #         memory += self.selfPlay()

        #     self.model.train()
        #     for epoch in trange(self.args['num_epochs']):
        #         self.train(memory)

        #     torch.save(self.model.state_dict(), f"model_{iteration}_{self.game}.pt")
        #     torch.save(self.optimizer.state_dict(), f"optimizer_{iteration}_{self.game}.pt")
        
        # New training function starting from iteration 8        
        for iteration in range(8, self.args['num_iterations']):
            memory = []
        
            self.model.eval()
            for _ in trange(self.args['num_self_play_iterations'] // self.args['num_parallel_games'], desc=f"Self-play Iter {iteration}"):
                memory += self.self_play()
        
            self.model.train()
            for _ in trange(self.args['num_epochs'], desc=f"Train Iter {iteration}"):
                self.train(memory)
        
            torch.save(self.model.state_dict(), f"model_{iteration}_{self.game}.pt")
            torch.save(self.optimizer.state_dict(), f"optimizer_{iteration}_{self.game}.pt")

class SPG:
    def __init__(self, game):
        self.state = game.get_initial_state()
        self.memory = []
        self.root = None
        self.node = None

# %%
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches

# def plot_connect4(board):
#     rows, cols = board.shape

#     fig, ax = plt.subplots(figsize=(cols, rows + 1))  # Extra space for column labels
#     ax.set_xlim(0, cols)
#     ax.set_ylim(-0.5, rows + 0.5)
#     ax.set_aspect('equal')
#     ax.axis('off')

#     # Draw background
#     ax.add_patch(patches.Rectangle((0, 0), cols, rows, color='blue'))

#     # Draw circles
#     for r in range(rows):
#         for c in range(cols):
#             value = board[rows - 1 - r, c]  # Flip vertically
#             if value == 1:
#                 color = 'red'
#             elif value == -1:
#                 color = 'yellow'
#             else:
#                 color = 'white'
#             circle = patches.Circle((c + 0.5, r + 0.5), 0.4, facecolor=color, edgecolor='black', linewidth=1)
#             ax.add_patch(circle)

#     # Add column numbers
#     for c in range(cols):
#         ax.text(c + 0.5, rows + 0.1, str(c + 1), ha='center', va='bottom', fontsize=14, fontweight='bold')

#     plt.show()

# %%
# game = ConnectFour()
# player = 1

# args = {
#     'C': 2,
#     'num_searches': 600,
#     'dirichlet_epsilon': 0.,
#     'dirichlet_alpha': 0.3
# }

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model = ResNet(game, 9, 128, device)
# model.load_state_dict(torch.load("model_11_ConnectFour.pt", map_location=device))
# model.eval()

# mcts = MCTS(game, args, model)

# state = game.get_initial_state()

# move_count = 0

# while True:
#     plot_connect4(state)

#     if player == 1:
#         valid_moves = game.get_valid_moves(state)
#         print("Valid moves:", [i+1 for i in range(game.action_size) if valid_moves[i] == 1])
#         action = int(input(f"{player}:")) - 1

#         if valid_moves[action] == 0:
#             print("Action invalid, try again")
#             continue

#     else:
#         neutral_state = game.change_perspective(state, player)
#         mcts.args['num_searches'] = 600 + move_count*6
#         mcts_probs = mcts.search(neutral_state)
#         action = np.argmax(mcts_probs)

#     state = game.get_next_state(state, action, player)

#     value, is_terminal = game.get_value_and_terminated(state, action)

#     if is_terminal:
#         plot_connect4(state)
#         if value == 1:
#             print(player, "won")
#         else:
#             print("draw")
#         break

#     player = game.get_opponent(player)
#     move_count += 1

# %%
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QMessageBox
from PyQt5.QtGui import QPainter, QColor, QPen
from PyQt5.QtCore import Qt
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QIcon


class Connect4GUI(QWidget):
    def __init__(self, game, model, mcts):
        super().__init__()
        self.setWindowTitle("Play against the AlphaZero Algorithm in Connect 4!")
        self.setWindowIcon(QIcon("c4.png"))
        self.setGeometry(100, 100, 500, 550)
        self.board_size = (6, 7)
        self.cell_size = 70
        self.offset = 30
        self.radius = 30

        self.game = game
        self.model = model
        self.mcts = mcts

        self.state = game.get_initial_state()
        self.player = 1
        self.move_count = 0
        self.is_game_over = False

        self.show()

    def paintEvent(self, event):
        painter = QPainter(self)
        self.draw_board(painter)

    def draw_board(self, painter):
        rows, cols = self.board_size
        width = self.cell_size * cols
        height = self.cell_size * rows

        painter.setBrush(QColor(0, 0, 255))  # Blue background
        painter.drawRect(self.offset, self.offset, width, height)

        for r in range(rows):
            for c in range(cols):
                x = self.offset + c * self.cell_size
                y = self.offset + r * self.cell_size
                value = self.state[r][c]
                if value == 1:
                    color = QColor(255, 0, 0)  # Red
                elif value == -1:
                    color = QColor(255, 255, 0)  # Yellow
                else:
                    color = QColor(255, 255, 255)  # Empty

                painter.setBrush(color)
                painter.setPen(QPen(Qt.black, 2))
                painter.drawEllipse(x + 5, y + 5, self.radius * 2, self.radius * 2)

    def mousePressEvent(self, event):
        if self.is_game_over:
            return

        if event.button() == Qt.LeftButton:
            x = event.pos().x() - self.offset
            col = x // self.cell_size

            if 0 <= col < 7:
                self.human_move(col)

    def human_move(self, col):
        if self.is_game_over:
            return

        valid_moves = self.game.get_valid_moves(self.state)
        if valid_moves[col] == 0:
            QMessageBox.warning(self, "Invalid Move", f"Column {col+1} is full!")
            return

        self.state = self.game.get_next_state(self.state, col, self.player)
        self.check_game_end(col)

        if not self.is_game_over:
            self.player = self.game.get_opponent(self.player)

            # Update GUI immediately before AI thinks
            self.update()
            # After a delay, run AI move
            QTimer.singleShot(100, self.ai_move)  # 100ms delay


    def ai_move(self):
        neutral_state = self.game.change_perspective(self.state, self.player)
        self.mcts.args['num_searches'] = 600 + self.move_count * 6
        probs = self.mcts.search(neutral_state)
        action = np.argmax(probs)

        self.state = self.game.get_next_state(self.state, action, self.player)
        self.check_game_end(action)

        if not self.is_game_over:
            self.player = self.game.get_opponent(self.player)

    def check_game_end(self, action):
        value, is_terminal = self.game.get_value_and_terminated(self.state, action)
        self.move_count += 1
        self.update()

        if is_terminal:
            self.is_game_over = True
            if value == 1:
                winner = "You" if self.player == 1 else "AI"
                QMessageBox.information(self, "Game Over", f"{winner} won!")
            else:
                QMessageBox.information(self, "Game Over", "It's a draw!")

if __name__ == "__main__":
    game = ConnectFour()
    player = 1
    args = {'C': 2,
            'num_searches': 600, 
            'dirichlet_epsilon': 0., 
            'dirichlet_alpha': 0.3}
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ResNet(game, 9, 128, device)
    model.load_state_dict(torch.load("model_11_ConnectFour.pt", map_location=device))
    model.eval()

    mcts = MCTS(game, args, model)

    app = QApplication(sys.argv)
    window = Connect4GUI(game, model, mcts)
    sys.exit(app.exec_())

# %%