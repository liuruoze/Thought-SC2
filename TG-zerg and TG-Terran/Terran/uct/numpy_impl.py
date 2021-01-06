import collections
import numpy as np
import math
import lib.config as C
import lib.utils as U

MAX_ACTIONS = 3


class UCTNode():

    def __init__(self, game_state, move, parent=None, max_actions=MAX_ACTIONS):
        self.game_state = game_state
        self.move = move
        self.is_expanded = False
        self.parent = parent  # Optional[UCTNode]
        self.children = {}  # Dict[move, UCTNode]
        self.child_priors = np.zeros([max_actions], dtype=np.float32)
        self.child_total_value = np.zeros([max_actions], dtype=np.float32)
        self.child_mean_value = np.ones([max_actions], dtype=np.float32) * 1e9
        self.child_number_visits = np.zeros([max_actions], dtype=np.float32)

    @property
    def number_visits(self):
        return self.parent.child_number_visits[self.move]

    @number_visits.setter
    def number_visits(self, value):
        self.parent.child_number_visits[self.move] = value

    @property
    def total_value(self):
        return self.parent.child_total_value[self.move]

    @total_value.setter
    def total_value(self, value):
        self.parent.child_total_value[self.move] = value

    @property
    def mean_value(self):
        return self.parent.child_mean_value[self.move]

    @mean_value.setter
    def mean_value(self, value):
        self.parent.child_mean_value[self.move] = value

    @property
    def prior(self):
        return self.parent.child_priors[self.move]

    @prior.setter
    def prior(self, value):
        self.parent.child_priors[self.move] = value

    def __str__(self):
        return str(self.prior) + ', ' + str(self.mean_value) + ', ' + str(self.number_visits)

    def child_Q(self):
        # return self.child_total_value / (0.00001 + self.child_number_visits)
        return self.child_mean_value

    def child_U(self):
        # return math.sqrt(self.number_visits) * (self.child_priors / (1 + self.child_number_visits))
        return math.sqrt(self.number_visits) * self.child_priors / (1 + self.child_number_visits)

    def best_child(self):
        #print('child_Q', self.child_Q())
        # scale Q value to [0, 1]
        child_Q_nor = self.child_Q() / (1e-9 + np.sum(self.child_Q()))
        #print('child_Q_nor', child_Q_nor)
        #print('child_U', self.child_U())

        select_weight = self.child_U() + child_Q_nor
        #print('select_weight', select_weight)

        return np.argmax(select_weight)

    def select_leaf(self):
        current = self
        while current.is_expanded:
            # print('best_move')
            best_move = current.best_child()
            current = current.maybe_add_child(best_move)
        return current

    def expand(self, child_priors):
        self.is_expanded = True
        self.child_priors = child_priors
        use_batch = False
        if use_batch:
            next_states = self.game_state.play_all_move()
            for move, prior in enumerate(child_priors):
                if move not in self.children:
                    self.children[move] = UCTNode(game_state=next_states[move], move=move, parent=self)

    def maybe_add_child(self, move):
        if move not in self.children:
            self.children[move] = UCTNode(
                self.game_state.play(move), move, parent=self)
        return self.children[move]

    def backup(self, value_estimate: float):
        current = self
        while current.parent is not None:
            current.number_visits += 1
            current.total_value += (value_estimate *
                                    self.game_state.to_play)
            if True:
                v = (value_estimate * self.game_state.to_play)
                #print('v', v)
                x = current.mean_value
                #print('x', x)
                if x == 1e9:          # first get a value
                    current.mean_value = v
                else:
                    n = current.number_visits - 1
                    #print('n', n)
                    x_new = x - (x - v) / (n + 1)
                    #print('x_new', x_new)
                    if x_new != x_new:  # is x is nan
                        x_new = 0
                    current.mean_value = x_new
                #print('current.mean_value', current.mean_value)
            current = current.parent


class DummyNode(object):

    def __init__(self):
        self.parent = None
        self.child_priors = collections.defaultdict(float)
        self.child_total_value = collections.defaultdict(float)
        self.child_mean_value = collections.defaultdict(float)
        self.child_number_visits = collections.defaultdict(float)


def UCT_search(game_state, num_reads, policy_in_mcts, temperature=1):
    root = UCTNode(game_state, move=None, parent=DummyNode())
    for _ in range(num_reads):
        leaf = root.select_leaf()
        #print('leaf', leaf)
        child_priors, value_estimate = policy_in_mcts.predict(leaf.game_state)

        #print('child_priors, value_estimate', child_priors, value_estimate)
        leaf.expand(child_priors)
        leaf.backup(value_estimate)
        # print(leaf)

    # print(root.child_total_value)
    #print('root.child_mean_value', root.child_mean_value)
    # print(root.child_number_visits)
    exp = np.power(root.child_number_visits, 1 / temperature)
    prob = exp / (1e-9 + np.sum(exp))
    #print('prob', prob)
    if np.sum(prob) == 0:
        select = 0
    else:
        select = np.random.choice(prob.shape[0], 1, p=prob)
    # print(select)
    return select


class PolicyNetinMCTS():

    def __init__(self, net):
        self.net = net
        self.min_v = 0
        self.max_v = 1
        self.mean_v = 0
        self.std_v = 1

    def update_min_max_v(self, min_v, max_v):
        self.min_v = min_v
        self.max_v = max_v

    def update_mean_std_v(self, mean_v, std_v):
        self.mean_v = mean_v
        self.std_v = std_v

    def predict(self, game_state):
        tech_act_probs, v_preds = self.net.policy.get_action_probs(game_state.obs(), verbose=False)

        # because v_preds is a return estimate, so we need to rescale to [0, 1]
        #v_nor = self.min_max_normalization(v_preds)
        #v_preds = self.z_score_normalization(v_preds)

        return tech_act_probs, v_preds

    def min_max_normalization(self, v_preds):

        # if we use min max normalization
        v_preds = np.clip(v_preds, self.min_v, self.max_v, out=v_preds)
        v_nor = (v_preds - self.min_v) / (self.max_v - self.min_v)
        #print('v_nor:', v_nor)

        return v_nor

    def z_score_normalization(self, v_preds):

        # if we use min max normalization
        # print('v_preds:', v_preds)
        v_nor = (v_preds - self.mean_v) / self.std_v
        # print('v_nor:', v_nor)

        return v_nor


class GameState():

    def __init__(self, dynamic_net, to_play=1, state=U.GAME_INITIAL_SIMPLE_STATE):
        self.to_play = to_play
        self.state = state  # game first st ata
        self.dynamic_net = dynamic_net

    def play(self, move, verbose=False):
        next_state = self.dynamic_net.predict_tech(self.state, np.asarray([move]))
        if verbose:
            print('state', np.array(self.state).astype(dtype=np.int32))
            print('next_state', np.array(next_state).astype(dtype=np.int32))
        #next_state = np.array(next_state).astype(dtype=np.int32)
        return GameState(self.dynamic_net, self.to_play, next_state)

    def play_all_move(self):
        states = self.state
        moves = np.array([0])
        for i in range(MAX_ACTIONS - 1):
            states = np.vstack((states, self.state))
            moves = np.vstack((moves, np.array([i + 1])))
        # print(states)
        # print(moves)
        next_states = self.dynamic_net.predict_tech_batch(states, moves)
        #next_state = np.array(next_state).astype(dtype=np.int32)
        # print(next_states)
        gamestates = []
        for next_state in next_states:
            gamestate = GameState(self.dynamic_net, self.to_play, next_state)
            gamestates.append(gamestate)

        return gamestates

    def obs(self):
        return self.state
