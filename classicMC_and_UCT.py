import random
import copy
import numpy as np
import math
import collections

SIZE = 4  # 4×4 grid
UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3

##############################################################################
#                               2048 GAME LOGIC
##############################################################################

class Game2048:
    """
    Represents the 2048 game state and provides methods to manipulate it.
    """
    def __init__(self, grid=None, score=0, done=False):
        if grid is None:
            # Initialize an empty board
            self.grid = np.zeros((SIZE, SIZE), dtype=int)
            # Place two random tiles at the start
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
        """
        Adds a 2- or 4-tile to a random empty square.
        90% chance for a 2, 10% chance for a 4 (typical 2048).
        """
        empty_positions = [(r, c) for r in range(SIZE) for c in range(SIZE) 
                           if self.grid[r, c] == 0]
        if not empty_positions:
            return
        r, c = random.choice(empty_positions)
        self.grid[r, c] = 4 if random.random() < 0.1 else 2

    def can_move(self, direction):
        """
        Check if a move in the given direction is possible,
        without changing this object’s grid.
        """
        temp_game = self.clone()
        score_before = temp_game.score
        changed = temp_game.move(direction)
        return changed and (temp_game.score != score_before 
                            or not np.array_equal(temp_game.grid, self.grid))

    def move(self, direction):
        """
        Executes a move in-place; returns True if the grid changed, False otherwise.
        The grid is manipulated in a direction-agnostic way by rotating/flipping 
        before combining.
        """
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
            # Check if game is over
            if self.is_terminal():
                self.done = True

        return changed

    def _move_up(self, mat):
        """
        Moves the entire matrix 'mat' up and merges tiles.
        Returns the new matrix, modifies self.score for merges.
        """
        out = np.zeros_like(mat)
        for col in range(SIZE):
            stack = [x for x in mat[:, col] if x != 0]  # extract non-zero tiles
            merged_col = []
            skip = False
            for i in range(len(stack)):
                if skip:
                    skip = False
                    continue
                if i < len(stack) - 1 and stack[i] == stack[i + 1]:
                    # Merge
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
        """
        Moves entire matrix 'mat' left and merges tiles.
        Returns the new matrix, modifies self.score for merges.
        """
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
                    # Merge
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
        """
        Returns a list of all valid moves in [UP, RIGHT, DOWN, LEFT].
        """
        moves = []
        for d in [UP, RIGHT, DOWN, LEFT]:
            if self.can_move(d):
                moves.append(d)
        return moves

    def is_terminal(self):
        """
        Check if the game is over: 
         - no valid moves remain 
         - or optionally if we define "done" after hitting 2048, etc.
        """
        # If no empty cells left and no merges possible => game over
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
#                     HELPER: REPLAYING A MOVE SEQUENCE
##############################################################################

def replay_sequence(seq):
    """
    Replays the sequence of moves from a fresh Game2048 board and 
    returns the final state.
    """
    st = Game2048()  # brand new game
    for move in seq:
        st.move(move)
        if st.done or st.is_terminal():
            break
    return st

##############################################################################
#                              MCTS ALGORITHMS                                
##############################################################################

class Node:
    def __init__(self, state, parent=None, move=None):
        self.state = state
        self.parent = parent
        self.move = move

        self.children = {}
        self.visits = 0
        self.value = 0
        self.untried_moves = state.get_legal_moves()

    def is_fully_expanded(self):
        return len(self.untried_moves) == 0
    
    def best_child(self):
        return max(self.children.values(), key=lambda c: c.value / c.visits if c.visits > 0 else float('-inf'))
    
    def best_child_uct(self, C):
        return max(self.children.values(), key= lambda c: c.uct_score(C))

    def expand(self):
        move = self.untried_moves.pop()
        next_state = self.state.clone()
        next_state.move(move)

        child_node = Node(next_state, parent=self, move=move)
        self.children[move] = child_node
        return child_node

    def update(self, score):
        self.value += score
        self.visits += 1

    def uct_score(self, C):
        if self.visits == 0:
            return float('inf')
        
        mean = self.value / self.visits
        exploration = C * math.sqrt(math.log(self.parent.visits) / self.visits)

        return mean + exploration

def random_rollout(state, max_depth=50):
    rollout_state = state.clone()
    for i in range(max_depth):
        if rollout_state.is_terminal():
            break
        legal_moves = rollout_state.get_legal_moves()
        if not legal_moves:
            break
        move = random.choice(legal_moves)
        rollout_state.move(move)
    return rollout_state.get_score()


    return rollout_state.get_score()

def mcts_search(root_state, n=100, use_uct=False, C=math.sqrt(2)):
    root = Node(root_state)

    for _ in range(n):
        # Selection
        node = root
        while node.is_fully_expanded() and node.children:
            node = node.best_child_uct(C) if use_uct else node.best_child()

        # Expansion
        if not node.is_fully_expanded():
            node = node.expand()

        # Simulation
        score = random_rollout(node.state, max_depth=1)
    
        # Backprop
        while node is not None:
            node.update(score)
            node = node.parent

    if root.children:
        best_move = max(root.children.items(), key=lambda item: item[1].value / item[1].visits)[0]
        return best_move
    else:
        return None
    

def evaluate(mcts, label, num_games=1):
    scores = []
    for i in range(num_games):
        game = Game2048()
        while not game.done and game.get_legal_moves():
            move = mcts(game)
            if move is None:
                break
            game.move(move)
        scores.append(game.get_score())
        print(f"Game {i+1} final state:")
        print(game)
    print(f'{label} - Mean Score: {np.mean(scores)}, Max Score: {np.max(scores)}')

if __name__ == "__main__":
    
    # Create a new game instance.
    # game = Game2048()
    # print("Initial game state:")
    # print(game)
    
    # # Mapping move numbers to names for better readability.
    # move_names = {UP: "UP", RIGHT: "RIGHT", DOWN: "DOWN", LEFT: "LEFT"}

    # # Play the game by applying random legal moves until the game is over.
    # move_count = 0
    # while not game.is_terminal() and game.get_legal_moves():
    #     legal = game.get_legal_moves()
    #     move = random.choice(legal)
    #     move_count += 1
    #     print(f"\nMove {move_count}: {move_names[move]}")
    #     game.move(move)
    #     print(game)
    
    # print("\nFinal game state:")
    # print(game)

    evaluate(lambda g: mcts_search(g, 20, use_uct=False), "Classic MCTS")
    evaluate(lambda g: mcts_search(g, 20, use_uct=True), "UCT MCTS")
