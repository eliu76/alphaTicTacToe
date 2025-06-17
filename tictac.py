import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque, defaultdict
import math
from tqdm import tqdm

BOARD_SIZE = 3
N_SIMULATIONS = 1000
NUM_SELF_PLAY_GAMES = 200
BATCH_SIZE = 32
EPOCHS = 15
EVALUATION_GAMES = 40
WIN_THRESHOLD = 0.55  # Accept new model only if >55% win rate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Seed for reproducibility ---
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed()

# --------------------- Environment ---------------------

class TicTacToe:
    def __init__(self):
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
        self.current_player = 1
        self.winner = None

    def clone(self):
        clone = TicTacToe()
        clone.board = self.board.copy()
        clone.current_player = self.current_player
        clone.winner = self.winner
        return clone

    def legal_moves(self):
        return [(i, j) for i in range(BOARD_SIZE) for j in range(BOARD_SIZE) if self.board[i, j] == 0]

    def move(self, x, y):
        if self.board[x, y] != 0:
            return False
        self.board[x, y] = self.current_player
        self.check_winner()
        self.current_player *= -1
        return True

    def check_winner(self):
        for i in range(BOARD_SIZE):
            row_sum = sum(self.board[i, :])
            col_sum = sum(self.board[:, i])
            if abs(row_sum) == BOARD_SIZE:
                self.winner = np.sign(row_sum)
                return
            if abs(col_sum) == BOARD_SIZE:
                self.winner = np.sign(col_sum)
                return
        diag1 = sum(self.board[i, i] for i in range(BOARD_SIZE))
        diag2 = sum(self.board[i, BOARD_SIZE - 1 - i] for i in range(BOARD_SIZE))
        if abs(diag1) == BOARD_SIZE:
            self.winner = np.sign(diag1)
        elif abs(diag2) == BOARD_SIZE:
            self.winner = np.sign(diag2)
        elif not self.legal_moves():
            self.winner = 0  # Draw

    def game_over(self):
        return self.winner is not None

    def encode(self):
        current = (self.board == self.current_player).astype(np.float32)
        opponent = (self.board == -self.current_player).astype(np.float32)
        return np.stack([current, opponent])

# --------------------- Neural Network ---------------------

class PolicyValueNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU()
        )
        self.policy_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * BOARD_SIZE * BOARD_SIZE, BOARD_SIZE * BOARD_SIZE),
            nn.Softmax(dim=-1)
        )
        self.value_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * BOARD_SIZE * BOARD_SIZE, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.conv(x)
        return self.policy_head(x), self.value_head(x)

# --------------------- MCTS ---------------------

class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = {}
        self.N = defaultdict(int)
        self.W = defaultdict(float)
        self.Q = defaultdict(float)
        self.P = {}

    def expand(self, policy):
        moves = self.state.legal_moves()
        eps = 1e-8
        total_prob = sum(policy.values()) + eps * len(policy)
        for move in moves:
            prob = policy.get(move, eps)
            self.P[move] = prob / total_prob
            if move not in self.children:
                next_state = self.state.clone()
                next_state.move(*move)
                self.children[move] = MCTSNode(next_state, parent=self)

    def select(self, c_puct=1.0):
        best_score = -float("inf")
        best_move = None
        total_N = sum(self.N.values())
        for move in self.state.legal_moves():
            u = c_puct * self.P.get(move, 0) * math.sqrt(total_N + 1) / (1 + self.N[move])
            score = self.Q[move] + u
            if score > best_score:
                best_score = score
                best_move = move
        return best_move

    def backup(self, move, value):
        self.N[move] += 1
        self.W[move] += value
        self.Q[move] = self.W[move] / self.N[move]

def run_mcts(root, net):
    for _ in range(N_SIMULATIONS):
        node = root
        path = []

        # Selection & Expansion
        while node.children:
            move = node.select()
            path.append((node, move))
            node = node.children[move]

        if node.state.game_over():
            value = node.state.winner
        else:
            input_tensor = torch.from_numpy(np.array([node.state.encode()], dtype=np.float32)).to(device)
            policy_tensor, value_tensor = net(input_tensor)
            value = value_tensor.item()

            policy = policy_tensor.view(-1).detach().cpu().numpy()
            moves = node.state.legal_moves()
            policy_dict = {(i, j): policy[i * BOARD_SIZE + j] for i, j in moves}

            node.expand(policy_dict)

        # Backup value
        for parent, move in reversed(path):
            parent.backup(move, value)
            value = -value

def select_move(root, temperature=1.0):
    legal_moves = root.state.legal_moves()
    counts = np.array([root.N[move] for move in legal_moves], dtype=np.float32)

    if counts.sum() == 0:
        probs = np.ones(len(legal_moves), dtype=np.float32) / len(legal_moves)
    else:
        if temperature == 0:
            move = legal_moves[np.argmax(counts)]
            return move, None
        counts = counts ** (1 / temperature)
        probs = counts / counts.sum()

    move = random.choices(legal_moves, weights=probs, k=1)[0]
    return move, probs

# --------------------- Self-Play and Training ---------------------

def self_play_game(net):
    state = TicTacToe()
    examples = []
    move_num = 0
    while not state.game_over():
        root = MCTSNode(state.clone())
        run_mcts(root, net)

        temperature = 1.0 if move_num < 10 else 0.1
        move, pi = select_move(root, temperature=temperature)

        encoded = state.encode()
        full_pi = np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.float32)
        if pi is not None:
            for (i, j), p in zip(state.legal_moves(), pi):
                full_pi[i * BOARD_SIZE + j] = p
        else:
            full_pi[move[0] * BOARD_SIZE + move[1]] = 1.0

        examples.append((encoded, full_pi, state.current_player))
        state.move(*move)
        move_num += 1

    winner = state.winner
    data = []
    for s, pi, player in examples:
        z = 1 if winner == player else -1 if winner == -player else 0
        data.append((s, pi, z))
    return data

def train(net, memory, optimizer):
    net.train()
    best_loss = float('inf')
    patience = 3
    patience_counter = 0

    for epoch in range(EPOCHS):
        batch = random.sample(memory, min(BATCH_SIZE, len(memory)))
        boards = torch.tensor(np.array([x[0] for x in batch]), dtype=torch.float32).to(device)
        pis = torch.tensor([x[1] for x in batch], dtype=torch.float32).to(device)
        zs = torch.tensor([x[2] for x in batch], dtype=torch.float32).unsqueeze(1).to(device)

        optimizer.zero_grad()
        out_pi, out_v = net(boards)
        loss_pi = -torch.sum(pis * torch.log(out_pi + 1e-8)) / len(batch)
        loss_v = nn.functional.mse_loss(out_v, zs)
        loss = loss_pi + loss_v
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {loss.item():.4f} (Policy: {loss_pi.item():.4f}, Value: {loss_v.item():.4f})")

        if loss.item() < best_loss - 1e-4:
            best_loss = loss.item()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

def evaluate_models(new_net, old_net, num_games=EVALUATION_GAMES):
    results = {"new_win": 0, "old_win": 0, "draw": 0}
    for i in range(num_games):
        state = TicTacToe()
        first_player = 1 if i % 2 == 0 else -1

        while not state.game_over():
            if state.current_player == first_player:
                root = MCTSNode(state.clone())
                run_mcts(root, new_net)
                move, _ = select_move(root, temperature=0)
            else:
                root = MCTSNode(state.clone())
                run_mcts(root, old_net)
                move, _ = select_move(root, temperature=0)
            state.move(*move)

        if state.winner == first_player:
            results["new_win"] += 1
        elif state.winner == -first_player:
            results["old_win"] += 1
        else:
            results["draw"] += 1

    win_ratio = results["new_win"] / num_games
    print(f"Evaluation - New Wins: {results['new_win']} ({win_ratio*100:.2f}%), Old Wins: {results['old_win']} ({results['old_win']/num_games*100:.2f}%), Draws: {results['draw']} ({results['draw']/num_games*100:.2f}%)")
    return win_ratio

def train_alpha_zero():
    net = PolicyValueNet().to(device)
    memory = deque(maxlen=10000)
    optimizer = optim.Adam(net.parameters(), lr=1e-3)
    previous_net = None

    for iter in range(20):
        print(f"\nIteration {iter + 1}: Self-play generating games...")
        for _ in tqdm(range(NUM_SELF_PLAY_GAMES)):
            data = self_play_game(net)
            memory.extend(data)

        print("Training network...")
        train(net, memory, optimizer)

        if previous_net is None:
            previous_net = PolicyValueNet().to(device)
            previous_net.load_state_dict(net.state_dict())
            continue

        print("Evaluating new network against old...")
        win_ratio = evaluate_models(net, previous_net)

        if win_ratio >= WIN_THRESHOLD:
            print(f"New model accepted (win ratio {win_ratio:.2f} >= {WIN_THRESHOLD})")
            previous_net.load_state_dict(net.state_dict())
        else:
            print(f"New model rejected (win ratio {win_ratio:.2f} < {WIN_THRESHOLD})")
            net.load_state_dict(previous_net.state_dict())

    return net

if __name__ == "__main__":
    trained_net = train_alpha_zero()
