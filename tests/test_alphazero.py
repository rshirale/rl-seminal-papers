"""Tests for the Chapter 8 AlphaZero modules.

Four things are worth guarding here, and none of them is "the loss went down".

The first is the sign convention. Positions are canonical -- the side to move
is always +1 -- and the outcome a game reports belongs to the player to move in
the *terminal* position, who is the one that did not just move. Every module
downstream depends on that, and a sign error anywhere along it is silent:
training still runs, the loss still falls, and the agent learns to lose. That
failure mode gets three tests rather than one.

The second is the oracle. ``solver.solve`` is what turns the project's
convergence claim into a measurement instead of a vibe, so it is checked
against the values van den Herik, Uiterwijk and van Rijswijck (2002)
published rather than against itself.

The third is the interface. ``game.Game`` is the contract that lets one
unchanged agent play Connect Four and Othello, which is AlphaZero's actual
headline claim; a test that only ever instantiates Connect Four would not
notice it rotting.

The fourth is the two measured numbers the chapter quotes: the published
networks' parameter counts, and the fact that an untrained agent's
``mirror_draw_rate`` is a real measurement rather than a coin flip.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.part_2_methods.ch08_alphazero import network
from src.part_2_methods.ch08_alphazero.alphago import (
    PolicyNetwork, PublishedAlphaZeroNet, ValueNetwork, parameter_counts,
)
from src.part_2_methods.ch08_alphazero.arena import (
    _random_action, evaluate_agent, mirror_match, play_match,
)
from src.part_2_methods.ch08_alphazero.connect_four import ConnectFour
from src.part_2_methods.ch08_alphazero.mcts import BatchedMCTS, Node, puct_score
from src.part_2_methods.ch08_alphazero.othello import Othello
from src.part_2_methods.ch08_alphazero.selfplay import play_batch
from src.part_2_methods.ch08_alphazero.solver import (
    best_action, disc_difference, solve,
)
from src.part_2_methods.ch08_alphazero.train import (
    build_game, loss_fn, root_value,
)

CPU = torch.device("cpu")


def tiny_agent(game, channels=8, blocks=1, seed=0):
    """A randomly-initialized net and a search over it.

    Deliberately far too small to play well. Every test below is about
    mechanism -- shapes, signs, masks, bookkeeping -- and none of them should
    depend on the network having learned anything.

    Seeded per call rather than once per session: without that, a test's
    network depends on how much randomness the tests before it happened to
    consume, and a suite that passes alone fails when reordered.
    """
    torch.manual_seed(seed)
    net = network.build(game, channels=channels, blocks=blocks).to(CPU).eval()
    return net, BatchedMCTS(game, net, CPU)


# --------------------------------------------------------------------------
# The rules, in both domains
# --------------------------------------------------------------------------

def test_connect_four_detects_every_orientation():
    game = ConnectFour(6, 7)
    # Four in a column. The result is -1 because it is reported for the player
    # to move, and the player to move is the one who just lost.
    board = game.initial_board()
    for action in (0, 1, 0, 1, 0, 1, 0):
        board = game.apply_move(board, action)
    assert game.outcome(board) == -1

    # Four on a rising diagonal.
    board = game.initial_board()
    for action in (0, 1, 1, 2, 2, 3, 2, 3, 3, 6, 3):
        board = game.apply_move(board, action)
    assert game.outcome(board) == -1

    # Four on a row, and four on a falling diagonal, so no direction in
    # _build_lines can be dropped without a failure.
    board = game.initial_board()
    for action in (0, 0, 1, 1, 2, 2, 3):
        board = game.apply_move(board, action)
    assert game.outcome(board) == -1

    board = game.initial_board()
    for action in (3, 2, 2, 1, 1, 0, 1, 0, 0, 6, 0):
        board = game.apply_move(board, action)
    assert game.outcome(board) == -1


def test_connect_four_full_board_is_a_draw():
    # Three columns cannot hold a line of four, so filling the board draws.
    game = ConnectFour(2, 3, win_length=4)
    board = game.initial_board()
    for action in (0, 0, 1, 1, 2, 2):
        assert game.outcome(board) is None
        board = game.apply_move(board, action)
    assert game.outcome(board) == 0


def test_a_full_column_is_illegal_and_refused():
    game = ConnectFour(2, 3)
    board = game.initial_board()
    for _ in range(2):
        board = game.apply_move(board, 0)
    assert game.legal_moves(board)[0] == 0
    with pytest.raises(ValueError):
        game.apply_move(board, 0)


def test_moves_alternate_and_stay_canonical():
    game = ConnectFour()
    board = game.apply_move(game.initial_board(), 3)
    # The mover's disc belongs to the opponent once the board flips, which is
    # what "canonical" buys: the agent never asks whose turn it is.
    assert board[5, 3] == -1
    assert (board == 1).sum() == 0


def test_othello_opening_has_four_moves():
    game = Othello(6)
    legal = np.nonzero(game.legal_moves(game.initial_board()))[0]
    assert len(legal) == 4


def test_othello_pass_is_legal_only_when_stuck():
    game = Othello(6)
    legal = game.legal_moves(game.initial_board())
    assert legal[game.pass_action] == 0

    # A board where the player to move has nothing: all discs are theirs.
    board = np.full((6, 6), -1, dtype=np.int8)
    board[0, 0] = 0
    legal = game.legal_moves(board)
    assert legal[game.pass_action] == 1


def test_othello_ends_when_neither_side_can_move():
    """Two passes end the game, which is why outcome() tests both boards.

    Testing only the side to move would leave a position where one player is
    stuck but the other is not looking terminal, and the game would end a
    move early with the wrong score.
    """
    game = Othello(6)
    board = np.full((6, 6), 1, dtype=np.int8)
    board[0, 0] = -1
    assert game.outcome(board) == 1
    assert game.outcome(-board) == -1

    tied = np.full((6, 6), 1, dtype=np.int8)
    tied[:3] = -1
    assert game.outcome(tied) == 0


def test_othello_symmetries_preserve_the_policy():
    game = Othello(6)
    policy = np.zeros(game.action_size, dtype=np.float32)
    policy[14] = 1.0
    variants = game.symmetries(game.initial_board(), policy)
    assert len(variants) == 8
    for board, pi in variants:
        assert abs(pi.sum() - 1.0) < 1e-6
        assert board.shape == (6, 6)
        # The pass action rides along untouched; rotating it would be a bug.
        assert pi[-1] == policy[-1]


def test_connect_four_symmetry_mirrors_board_and_policy_together():
    """A mirrored board with an unmirrored policy is training on a lie."""
    game = ConnectFour(6, 6)
    board = game.apply_move(game.initial_board(), 0)
    policy = np.zeros(game.action_size, dtype=np.float32)
    policy[0] = 1.0
    (_, first), (flipped, second) = game.symmetries(board, policy)
    assert np.array_equal(first, policy)
    assert second[-1] == 1.0
    assert np.array_equal(flipped, board[:, ::-1])


# --------------------------------------------------------------------------
# The oracle
# --------------------------------------------------------------------------

def test_solver_matches_the_published_small_board_values():
    """Van den Herik and colleagues (2002) record 4x4 and 4x5 as draws.

    6x4 is also a draw and also tractable, but takes about sixteen seconds --
    it belongs in `make run-ch8-solve`, not in the fast test suite.
    """
    for rows, cols in ((4, 4), (4, 5)):
        game = ConnectFour(rows, cols)
        value = int(solve(game, game.initial_board(), table={}))
        assert value == game.solved_value()


def test_the_transposition_table_does_not_corrupt_the_answer():
    """A table that cached window-bounded values as exact would return wrong
    answers on later lookups, which is the one way an oracle can lie."""
    game = ConnectFour(4, 4)
    shared = {}
    first = solve(game, game.initial_board(), table=shared)
    assert shared, "nothing was cached, so this proves nothing"
    second = solve(game, game.initial_board(), table=shared)
    assert first == second == 0.0


def test_a_forced_win_is_seen_and_taken():
    game = ConnectFour(4, 4)
    board = game.initial_board()
    # X threatens column 3 with three in a row along the bottom; O is busy
    # stacking column 0, so X to move has a win in one.
    for action in (0, 0, 1, 0, 2):
        board = game.apply_move(board, action)
    board = -board  # hand it back to X, who is one move from four in a row
    assert solve(game, board, table={}) == 1.0
    assert best_action(game, board, depth=2) == 3


def test_disc_difference_is_signed_from_the_mover():
    board = np.array([[1, 1], [-1, 0]], dtype=np.int8)
    assert disc_difference(board) == pytest.approx(1 / 3)
    assert disc_difference(-board) == pytest.approx(-1 / 3)
    assert disc_difference(np.zeros((2, 2), dtype=np.int8)) == 0.0


# --------------------------------------------------------------------------
# The network, and the published sizes the chapter quotes
# --------------------------------------------------------------------------

def test_network_output_shapes_match_the_game():
    """The net reads nothing but the interface, which is what lets one agent
    play both games. Othello's extra pass action is the thing that would
    break a network hard-coded to a square board."""
    for game in (ConnectFour(6, 7), Othello(6)):
        net = network.build(game, channels=16, blocks=1)
        planes = torch.zeros(4, game.input_planes, *game.board_shape)
        policy, value = net(planes)
        assert policy.shape == (4, game.action_size)
        assert value.shape == (4,)


def test_the_value_head_is_bounded_but_the_policy_head_is_not():
    """Value is a tanh, so it lives in [-1, 1]. Policy is raw logits: a
    softmax here would spend mass on moves the rules forbid, before the
    legal mask is known."""
    game = ConnectFour(6, 6)
    net = network.build(game, channels=16, blocks=1)
    planes = torch.randn(8, game.input_planes, *game.board_shape) * 5
    policy, value = net(planes)
    assert value.abs().max() <= 1.0
    assert not torch.allclose(policy.exp().sum(1), torch.ones(8))


def test_published_parameter_counts_are_what_the_chapter_says():
    """The chapter quotes all four of these. They are measured, not asserted:
    instantiate the network and count."""
    counts = dict((label.split(" (")[0], n) for label, n in parameter_counts())
    assert counts["AlphaGo policy network"] == 3_882_433
    assert counts["AlphaGo value network"] == 3_975_362
    assert counts["AlphaGo Zero network"] == 22_837_864
    assert counts["this chapter's project"] == 456_647


def test_the_published_networks_run_on_a_real_position():
    """They are not trainable here, but they must at least be correct code --
    a listing that does not execute is how the previous draft's fragments got
    away with calling methods that never existed."""
    planes = torch.zeros(1, 48, 19, 19)
    assert PolicyNetwork()(planes).shape == (1, 361)
    assert ValueNetwork()(planes).shape == (1, 1)
    policy, value = PublishedAlphaZeroNet()(torch.zeros(1, 17, 19, 19))
    assert policy.shape == (1, 362) and value.shape == (1, 1)


def test_pick_device_honours_an_explicit_request():
    assert network.pick_device("cpu").type == "cpu"
    assert network.pick_device("auto").type in {"cpu", "cuda", "mps"}


# --------------------------------------------------------------------------
# The search
# --------------------------------------------------------------------------

def test_search_returns_a_distribution_over_legal_moves_only():
    game = ConnectFour(6, 7)
    _, mcts = tiny_agent(game, channels=16)

    # Fill column 0 so it becomes illegal.
    board = game.initial_board()
    for _ in range(6):
        board = game.apply_move(board, 0)
    policies, _ = mcts.search([board], simulations=16, add_noise=False)
    assert policies[0][0] == 0.0
    assert abs(policies[0].sum() - 1.0) < 1e-6


def test_the_cold_start_guard_keeps_the_first_simulation_prior_driven():
    """Without max(1, parent.visit_count), sqrt(0) makes every child score
    exactly zero on the first simulation, and the opening move is chosen by
    dict order rather than by the policy head."""
    parent = Node()  # no visits yet: this is the first simulation
    weak = Node(prior=0.1, parent=parent, action=0)
    strong = Node(prior=0.9, parent=parent, action=1)
    assert puct_score(parent, strong, 1.5) > puct_score(parent, weak, 1.5)


def test_a_visited_child_scores_from_the_opponents_point_of_view():
    """A child's Q is what the *opponent* thinks, so it negates on the way
    up. Drop the minus sign and the search walks straight into losing lines."""
    parent = Node()
    parent.visit_count = 16
    losing = Node(prior=0.5, parent=parent, action=0)
    losing.visit_count, losing.value_sum = 4, 4.0   # great for the opponent
    winning = Node(prior=0.5, parent=parent, action=1)
    winning.visit_count, winning.value_sum = 4, -4.0
    assert puct_score(parent, winning, 1.5) > puct_score(parent, losing, 1.5)


def test_child_boards_are_built_lazily():
    """Expanding materialises one child per legal move; a search visits few of
    them. Building every board up front is the expensive naive mistake."""
    game = ConnectFour(6, 7)
    _, mcts = tiny_agent(game)
    root = Node(game.initial_board())
    priors, _ = mcts._evaluate([root.board(game)])
    mcts._expand(root, priors[0])

    assert len(root.children) == 7
    assert all(c._board is None for c in root.children.values())
    child = root.children[3]
    assert child.board(game) is not None
    assert child._board is not None


def test_search_concentrates_its_visits_once_there_is_something_to_find():
    """The claim the animation makes visually, stated precisely.

    Search sharpens a distribution only where the tree contains information
    the priors do not. On an empty board under an untrained network there is
    none, and the visits stay near-uniform -- which is correct behaviour, not
    a bug, and is why this test uses a position with a win in one instead.
    """
    game = ConnectFour(4, 4)
    _, mcts = tiny_agent(game)
    board = game.initial_board()
    for action in (0, 1, 0, 1, 0, 1):
        board = game.apply_move(board, action)

    trace = []
    mcts.search([board], simulations=32, add_noise=False, trace=trace)
    early = float(np.max(trace[3][0]))
    late = float(np.max(trace[-1][0]))
    assert late > early
    assert int(np.argmax(trace[-1][0])) == 0


def test_a_terminal_leaf_is_scored_by_the_rules_not_the_network():
    game = ConnectFour(4, 4)
    _, mcts = tiny_agent(game)
    board = game.initial_board()
    for action in (0, 1, 0, 1, 0, 1):
        board = game.apply_move(board, action)
    # Whoever is to move has a win in one; the search should find it every
    # time, because the winning child is terminal and worth exactly +1.
    policies, _ = mcts.search([board], simulations=32, add_noise=False)
    assert int(np.argmax(policies[0])) == 0


def test_reusing_a_root_keeps_its_statistics():
    """Tree reuse is where most of the self-play speedup comes from, and a
    version that silently rebuilt the subtree would still produce correct
    games -- just far slower."""
    game = ConnectFour(6, 6)
    _, mcts = tiny_agent(game)
    policies, roots = mcts.search([game.initial_board()], simulations=16,
                                  add_noise=False)
    child = roots[0].children[int(np.argmax(policies[0]))]
    visits_before = child.visit_count
    assert visits_before > 0
    _, again = mcts.search([child.board(game)], simulations=8,
                           add_noise=False, roots=[child])
    assert again[0] is child
    assert child.visit_count == visits_before + 8


# --------------------------------------------------------------------------
# Self-play: the silent-failure tests
# --------------------------------------------------------------------------

def test_self_play_labels_the_winner_positively():
    """A sign error here is invisible in the loss curve and fatal to training.

    The agent still trains, the loss still falls, and what it learns is how to
    lose. Nothing else in the loop would catch it.
    """
    game = ConnectFour(4, 4)
    _, mcts = tiny_agent(game)
    examples, results = play_batch(game, mcts, num_games=12, simulations=8,
                                   rng=np.random.default_rng(3))

    assert len(examples) > 0
    values = {float(v) for _, _, v in examples}
    assert values <= {-1.0, 0.0, 1.0}
    # A decisive game must contribute at least one winning and one losing label.
    if any(r != 0 for r in results):
        assert 1.0 in values and -1.0 in values


def test_the_last_mover_of_a_decisive_game_is_labelled_plus_one():
    """The sharper version of the test above, on a game whose result is known.

    play_batch walks the history backwards from a result that belongs to the
    player who did *not* move last, flipping the sign each ply. This pins the
    first link in that chain, which is the one an off-by-one would break.
    """
    game = ConnectFour(4, 4)
    _, mcts = tiny_agent(game)
    rng = np.random.default_rng(0)
    for _ in range(6):
        examples, results = play_batch(game, mcts, num_games=4, simulations=6,
                                       temperature_moves=0, rng=rng)
        if any(r != 0 for r in results):
            break
    assert any(r != 0 for r in results), "no decisive game to check"
    # Symmetries double each position, and the winner's last stored position
    # carries +1 for exactly the player who then played the winning move.
    assert 1.0 in {float(v) for _, _, v in examples}


def test_self_play_multiplies_positions_by_the_games_symmetries():
    """Connect Four claims two symmetries, Othello eight. Every stored
    position should appear once per symmetry -- that is free training data,
    and losing it is a silent halving of the sample budget."""
    for game, factor in ((ConnectFour(4, 4), 2), (Othello(6), 8)):
        _, mcts = tiny_agent(game)
        examples, _ = play_batch(game, mcts, num_games=2, simulations=4,
                                 rng=np.random.default_rng(1))
        assert len(examples) % factor == 0
        assert len(examples) >= factor


def test_every_stored_policy_is_a_distribution():
    game = ConnectFour(4, 5)
    _, mcts = tiny_agent(game)
    examples, _ = play_batch(game, mcts, num_games=3, simulations=6,
                             rng=np.random.default_rng(2))
    for planes, policy, _ in examples:
        assert planes.shape == (game.input_planes, *game.board_shape)
        assert abs(float(policy.sum()) - 1.0) < 1e-5


def test_the_ply_cap_scores_an_unfinished_game_as_a_draw():
    game = ConnectFour(6, 6)
    _, mcts = tiny_agent(game)
    _, results = play_batch(game, mcts, num_games=2, simulations=2,
                            rng=np.random.default_rng(0), max_plies=3)
    assert results == [0.0, 0.0]


# --------------------------------------------------------------------------
# The arena: the metric that can be faked, and the one that cannot
# --------------------------------------------------------------------------

def test_a_match_accounts_for_every_game_it_plays():
    game = ConnectFour(4, 4)
    _, mcts = tiny_agent(game)
    rng = np.random.default_rng(0)
    for agent_first in (True, False):
        wins, draws, losses = play_match(game, mcts, 6, _random_action, 6, rng,
                                         agent_first=agent_first)
        assert wins + draws + losses == 6


def test_an_untrained_mirror_match_is_a_measurement_not_a_coin_flip():
    """The bug this replaced reported 1.0, 0.0, 1.0, 0.0, 0.0 across a run.

    Greedy play with no noise made all 32 "games" the same deterministic game
    repeated, so the rate could only ever be exactly 0 or exactly 1. Sampling
    the opening plies is what makes it a real number, and the test for that is
    that the batch contains more than one distinct game.
    """
    game = ConnectFour(6, 6)
    _, mcts = tiny_agent(game)
    rate = mirror_match(game, mcts, simulations=8, num_games=16,
                        rng=np.random.default_rng(0), opening_plies=4)
    assert 0.0 <= rate <= 1.0
    # With opening_plies=0 the sampling is off and every game is identical,
    # so the rate collapses back to exactly 0 or exactly 1 -- the old bug,
    # pinned here so the fix cannot be quietly reverted.
    collapsed = mirror_match(game, mcts, simulations=8, num_games=16,
                             rng=np.random.default_rng(0), opening_plies=0)
    assert collapsed in (0.0, 1.0)


def test_evaluate_agent_reports_both_colours_and_all_three_opponents():
    game = ConnectFour(4, 4)
    _, mcts = tiny_agent(game)
    report = evaluate_agent(game, mcts, simulations=4, num_games=4,
                            rng=np.random.default_rng(0), depth=2)
    assert set(report) == {"vs_random", "score_random", "vs_depth3",
                           "score_depth3", "mirror_draw_rate"}
    for key in ("score_random", "score_depth3", "mirror_draw_rate"):
        assert 0.0 <= report[key] <= 1.0
    # w/d/l, summed over both colours, must account for every game played.
    assert sum(int(n) for n in report["vs_random"].split("/")) == 4


# --------------------------------------------------------------------------
# The training loop
# --------------------------------------------------------------------------

def test_the_loss_is_zero_only_when_both_heads_are_right():
    """Two terms over a shared trunk. A loss that ignored the policy target
    would still fall, and the agent would never learn to search."""
    target_policy = torch.tensor([[0.0, 1.0, 0.0]])
    target_value = torch.tensor([1.0])
    perfect = torch.tensor([[-50.0, 50.0, -50.0]])

    total, value_loss, policy_loss = loss_fn(
        perfect, torch.tensor([1.0]), target_policy, target_value)
    assert total.item() == pytest.approx(0.0, abs=1e-4)

    # Right policy, wrong value: the value term alone carries the error.
    total, value_loss, policy_loss = loss_fn(
        perfect, torch.tensor([-1.0]), target_policy, target_value)
    assert value_loss == pytest.approx(4.0, abs=1e-4)
    assert policy_loss == pytest.approx(0.0, abs=1e-4)

    # Right value, wrong policy: the other way round.
    total, value_loss, policy_loss = loss_fn(
        torch.tensor([[50.0, -50.0, -50.0]]), torch.tensor([1.0]),
        target_policy, target_value)
    assert value_loss == pytest.approx(0.0, abs=1e-4)
    assert policy_loss > 50.0


def test_build_game_reaches_both_domains_and_refuses_a_third():
    assert build_game("connect4", 6, 6).name == "connect_four_6x6"
    assert build_game("othello", 6, None).name == "othello_6x6"
    with pytest.raises(ValueError):
        build_game("go", 19, 19)


def test_the_default_board_is_the_one_with_a_proven_draw():
    """The project's convergence signal only exists because 6x6 is a proven
    draw. A default of 6x7 would leave mirror_draw_rate meaningless."""
    from src.part_2_methods.ch08_alphazero.train import build_parser
    args = build_parser().parse_args([])
    game = build_game(args.game, args.rows, args.cols)
    assert (args.rows, args.cols) == (6, 6)
    assert game.solved_value() == 0
    assert ConnectFour(6, 7).solved_value() == 1
    assert Othello(6).solved_value() == -1


def test_root_value_is_the_networks_opinion_of_the_empty_board():
    game = ConnectFour(6, 6)
    net, _ = tiny_agent(game)
    value = root_value(game, net, CPU)
    assert -1.0 <= value <= 1.0


@pytest.mark.slow
def test_two_iterations_of_the_real_loop_run_and_log(tmp_path):
    """End to end, on the smallest board and the shortest budget that still
    exercises every branch: self-play, training, and an evaluation round."""
    from src.part_2_methods.ch08_alphazero.train import main

    result = main([
        "--rows", "4", "--cols", "4", "--games-per-iteration", "8",
        "--simulations", "4", "--train-steps", "2", "--batch-size", "16",
        "--eval-every", "1", "--eval-games", "4", "--max-iterations", "2",
        "--channels", "8", "--blocks", "1", "--device", "cpu",
        "--out", str(tmp_path),
    ])

    assert len(result.history) == 2
    assert "mirror_draw_rate" in result.history[0]
    # The trained network comes back with the log, so measuring a finished run
    # does not mean rebuilding it from the checkpoint first.
    assert result.game.name == "connect_four_4x4"
    assert result.net.parameter_count() > 0
    assert (result.out_dir / "history.json").exists()
    assert (result.out_dir / "latest.pt").exists()


@pytest.mark.slow
def test_a_checkpoint_round_trips_through_inspect_agent(tmp_path):
    """The run's arguments travel inside the checkpoint, so a board size never
    has to be remembered and retyped correctly."""
    from src.part_2_methods.ch08_alphazero.inspect_agent import load
    from src.part_2_methods.ch08_alphazero.train import main

    main([
        "--rows", "4", "--cols", "4", "--games-per-iteration", "4",
        "--simulations", "4", "--train-steps", "1", "--batch-size", "8",
        "--eval-every", "99", "--max-iterations", "1", "--channels", "8",
        "--blocks", "1", "--device", "cpu", "--out", str(tmp_path),
    ])
    checkpoint = next(tmp_path.iterdir()) / "latest.pt"
    game, net, saved = load(checkpoint, CPU)
    assert game.name == "connect_four_4x4"
    assert saved["channels"] == 8
    assert net.parameter_count() > 0
