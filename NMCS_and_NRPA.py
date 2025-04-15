import random
import copy
import numpy as np
import math
import collections
import sys

SIZE = 4  # 4x4 grid
UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3

##############################################################################
#                             2048 GAME LOGIC
##############################################################################
class Game2048:
    """
    Represents the 2048 game state and provides methods to manipulate it.
    """
    def __init__(self, grid=None, score=0, done=False):
        if grid is None:
            self.grid = np.zeros((SIZE, SIZE), dtype=int)
            self.add_random_tile()
            self.add_random_tile()
            self.score = 0
            self.done = False
        else:
            self.grid = np.copy(grid)
            self.score = score
            self.done = done

    def clone(self):
        return Game2048(grid=self.grid, score=self.score, done=self.done)

    def add_random_tile(self):
        empty_positions = [(r, c) for r in range(SIZE) for c in range(SIZE) if self.grid[r, c] == 0]
        if not empty_positions:
            return
        r, c = random.choice(empty_positions)
        self.grid[r, c] = 4 if random.random() < 0.1 else 2

    def can_move(self, direction):
        temp = self.clone()
        score_before = temp.score
        changed = temp.move(direction)
        return changed and (temp.score != score_before or not np.array_equal(temp.grid, self.grid))

    def move(self, direction):
        old_grid = np.copy(self.grid)
        if direction == UP:
            self.grid = self._move_up(self.grid)
        elif direction == DOWN:
            self.grid = np.flipud(self._move_up(np.flipud(self.grid)))
        elif direction == LEFT:
            self.grid = self._move_left(self.grid)
        elif direction == RIGHT:
            self.grid = np.fliplr(self._move_left(np.fliplr(self.grid)))
        else:
            raise ValueError("Invalid direction")
        changed = not np.array_equal(old_grid, self.grid)
        if changed:
            self.add_random_tile()
            if self.is_terminal():
                self.done = True
        return changed

    def _move_up(self, mat):
        out = np.zeros_like(mat)
        for col in range(SIZE):
            stack = [x for x in mat[:, col] if x != 0]
            merged_col = []
            skip = False
            for i in range(len(stack)):
                if skip:
                    skip = False
                    continue
                if i < len(stack) - 1 and stack[i] == stack[i + 1]:
                    merged_val = stack[i] * 2
                    merged_col.append(merged_val)
                    self.score += merged_val
                    skip = True
                else:
                    merged_col.append(stack[i])
            for i in range(len(merged_col)):
                out[i, col] = merged_col[i]
        return out

    def _move_left(self, mat):
        out = np.zeros_like(mat)
        for row in range(SIZE):
            stack = [x for x in mat[row, :] if x != 0]
            merged_row = []
            skip = False
            for i in range(len(stack)):
                if skip:
                    skip = False
                    continue
                if i < len(stack) - 1 and stack[i] == stack[i + 1]:
                    merged_val = stack[i] * 2
                    merged_row.append(merged_val)
                    self.score += merged_val
                    skip = True
                else:
                    merged_row.append(stack[i])
            for i in range(len(merged_row)):
                out[row, i] = merged_row[i]
        return out

    def get_legal_moves(self):
        moves = []
        for d in [UP, RIGHT, DOWN, LEFT]:
            if self.can_move(d):
                moves.append(d)
        return moves

    def is_terminal(self):
        if len([x for x in self.grid.flatten() if x == 0]) == 0:
            for d in [UP, RIGHT, DOWN, LEFT]:
                temp = self.clone()
                if temp.move(d):
                    return False
            return True
        return False

    def get_score(self):
        return self.score

    def __str__(self):
        return f"Score: {self.score}\n{self.grid}"


##############################################################################
#                    HELPER: REPLAY A MOVE SEQUENCE
##############################################################################
def replay_sequence(seq):
    """
    Replays the sequence of moves from a fresh game state.
    Only the first move of each simulated sequence is applied.
    """
    st = Game2048()
    for move in seq:
        st.move(move)
        if st.done or st.is_terminal():
            break
    return st


##############################################################################
#                    NMCS IMPLEMENTATION
##############################################################################
def random_playout(game, max_depth=50):
    """
    Level-0 rollout: simulate up to max_depth moves.
    Returns (final_score, sequence_of_moves).
    """
    state = game.clone()
    seq = []
    for i in range(max_depth):
        if state.is_terminal():
            break
        moves = state.get_legal_moves()
        if not moves:
            break
        move = random.choice(moves)
        state.move(move)
        seq.append(move)
    return state.get_score(), seq

def nested_mcs(game, level=1, max_depth=50):
    """
    NMCS recursion. At level 0, run a random rollout.
    Otherwise, for each legal move, simulate with NMCS(level-1) and select the best move.
    Returns (final_score, sequence_of_moves) from the simulation.
    """
    if level == 0:
        return random_playout(game, max_depth=max_depth)
    
    best_seq = []
    best_score = -1
    current_state = game.clone()
    depth_counter = 0
    while depth_counter < max_depth:
        if current_state.is_terminal():
            break
        moves = current_state.get_legal_moves()
        if not moves:
            break
        
        best_move = None
        local_best_score = -1
        local_best_seq = []
        for move in moves:
            child = current_state.clone()
            child.move(move)
            score_child, seq_child = nested_mcs(child, level=level-1, max_depth=max_depth)
            if score_child > local_best_score:
                local_best_score = score_child
                best_move = move
                local_best_seq = seq_child

        # Guard against the case where no move was chosen
        if best_move is None:
            break

        current_state.move(best_move)

        best_seq.append(best_move)
        best_score = local_best_score
        depth_counter += 1
    return best_score, best_seq

def choose_move_nmcs(game, simulation_depth=50, nmcs_level=1):
    """
    Runs NMCS from the current state, then returns only the first move of the simulated sequence.
    """
    score, seq = nested_mcs(game, level=nmcs_level, max_depth=simulation_depth)
    if seq:
        return seq[0]
    return None


##############################################################################
#                    NRPA IMPLEMENTATION
##############################################################################
class Policy:
    """
    A dictionary-based policy mapping (state_hash, action) -> float weight.
    """
    def __init__(self):
        self.policy_dict = collections.defaultdict(float)
    
    def get_weight(self, state_hash, action):
        return self.policy_dict[(state_hash, action)]
    
    def set_weight(self, state_hash, action, value):
        self.policy_dict[(state_hash, action)] = value
    
    def add_weight(self, state_hash, action, value):
        self.policy_dict[(state_hash, action)] += value

def state_to_string(game_state):
    return ''.join(str(x) for x in game_state.grid.flatten())

def nrpa_playout_with_legals(game, policy, max_depth=50):
    """
    Perform one playout using Gibbs sampling based on the policy,
    up to max_depth moves. Returns (final_score, sequence) where sequence
    is a list of (state_hash, legal_moves, chosen_move).
    """
    seq = []
    state = game.clone()
    for i in range(max_depth):
        if state.is_terminal():
            break
        moves = state.get_legal_moves()
        if not moves:
            break
        s_hash = state_to_string(state)
        exps = []
        total = 0.0
        for mv in moves:
            w = math.exp(policy.get_weight(s_hash, mv))
            exps.append(w)
            total += w
        pick = random.random() * total
        cumulative = 0.0
        chosen_mv = moves[0]
        for i, mv in enumerate(moves):
            cumulative += exps[i]
            if cumulative >= pick:
                chosen_mv = mv
                break
        seq.append((s_hash, moves, chosen_mv))
        state.move(chosen_mv)
    return state.get_score(), seq

def nrpa_adapt(policy, best_seq, alpha=1.0):
    """
    Adapt the policy based on the best sequence.
    For each (state_hash, legal_moves, chosen_move), decrease all weights 
    by alpha * p(m) and increase the weight of the chosen move by alpha.
    """
    new_policy = copy.deepcopy(policy)
    for (s_hash, moves, chosen_mv) in best_seq:
        sum_exp = 0.0
        for m in moves:
            sum_exp += math.exp(new_policy.get_weight(s_hash, m))
        for m in moves:
            old_w = new_policy.get_weight(s_hash, m)
            p_m = math.exp(old_w) / sum_exp if sum_exp > 0 else 0
            new_policy.set_weight(s_hash, m, old_w - alpha * p_m)
        chosen_w = new_policy.get_weight(s_hash, chosen_mv)
        new_policy.set_weight(s_hash, chosen_mv, chosen_w + alpha)
    return new_policy

def NRPA(level, policy, nb_iter=50, max_depth=50):
    """
    NRPA recursion:
      - At level 0, do a single playout.
      - Otherwise, for nb_iter iterations, call NRPA(level-1) and adapt the policy.
    Returns (final_score, sequence) from the best playout.
    """
    if level == 0:
        game = Game2048()
        sc, seq = nrpa_playout_with_legals(game, policy, max_depth)
        return sc, seq
    best_score = -1
    best_seq = []
    for i in range(nb_iter):
        pol_copy = copy.deepcopy(policy)
        sc, seq = NRPA(level-1, pol_copy, nb_iter=nb_iter, max_depth=max_depth)
        if sc > best_score:
            best_score = sc
            best_seq = seq
        policy = nrpa_adapt(policy, best_seq, alpha=1.0)
    return best_score, best_seq

def choose_move_nrpa(game, policy, nrpa_level=2, nb_iter=20, rollout_depth=50):
    """
    Runs NRPA and returns only the first move of the best sequence.
    """
    sc, seq = NRPA(nrpa_level, policy, nb_iter=nb_iter, max_depth=rollout_depth)
    if seq:
        # Each element in seq is (state_hash, legal_moves, chosen_move)
        return seq[0][2]
    return None


##############################################################################
#                      MAIN: Play a Game Using NMCS or NRPA
##############################################################################
import os

def clear_console():
    # Clear console for latest move only
    os.system('cls' if os.name == 'nt' else 'clear')

def play_game_with_nmcs(simulation_depth=50, nmcs_level=1, verbose=False):
    game = Game2048()
    moves_played = []
    move_count = 0

    while not game.is_terminal() and game.get_legal_moves():
        move = choose_move_nmcs(game, simulation_depth, nmcs_level)
        if move is None:
            break
        game.move(move)
        move_count += 1
        moves_played.append(move)

        if verbose:
            clear_console()
            print(f"Move #{move_count} - Direction: {['UP', 'RIGHT', 'DOWN', 'LEFT'][move]}")
            print(f"Score: {game.get_score()}")
            print("Board:")
            print(game.grid)

            sys.stdout.flush()

    print("\nGame Over! Final Score (NMCS):", game.get_score())
    return game, moves_played

def play_game_with_nrpa(simulation_depth=50, nrpa_level=2, nb_iter=20, verbose=False):
    game = Game2048()
    base_policy = Policy()
    moves_played = []
    move_count = 0

    while not game.is_terminal() and game.get_legal_moves():
        move = choose_move_nrpa(game, base_policy, nrpa_level, nb_iter, rollout_depth=simulation_depth)
        if move is None:
            break
        game.move(move)
        move_count += 1
        moves_played.append(move)

        if verbose:
            clear_console()
            print(f"Move #{move_count} - Direction: {['UP', 'RIGHT', 'DOWN', 'LEFT'][move]}")
            print(f"Score: {game.get_score()}")
            print("Board:")
            print(game.grid)

            sys.stdout.flush()

    print("\nGame Over! Final Score (NRPA):", game.get_score())
    return game, moves_played

def print_stats(strategy_name, game, moves):
    print(f"\n======= {strategy_name} =======")
    print(f"# of moves played: {len(moves)}")
    print(f"Score: {game.get_score()}")
    print("Board:")
    print(game.grid)


if __name__ == "__main__":
    # Set seeds for reproducibility.
    random.seed(30)
    np.random.seed(30)
    verbose = False

    print("======= Playing Game with NMCS =======")
    # game_nmcs, moves_nmcs = play_game_with_nmcs(simulation_depth=2, nmcs_level=0, verbose=verbose)
    game_nmcs, moves_nmcs = play_game_with_nmcs(simulation_depth=2, nmcs_level=3, verbose=verbose)
    # game_nmcs, moves_nmcs = play_game_with_nmcs(simulation_depth=2, nmcs_level=3, verbose=verbose)
    print_stats("NMCS", game_nmcs, moves_nmcs)

    print("\n======= Playing Game with NRPA =======")
    # game_nrpa, moves_nrpa = play_game_with_nrpa(simulation_depth=10, nrpa_level=2, nb_iter=30, verbose=verbose)
    game_nrpa, moves_nrpa = play_game_with_nrpa(simulation_depth=20, nrpa_level=1, nb_iter=40, verbose=verbose)
    if verbose:
        print_stats("NMCS", game_nmcs, moves_nmcs)
    print_stats("NRPA", game_nrpa, moves_nrpa)
