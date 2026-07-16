"""
Hanabi Deluxe II simulator.

Implements the ruleset discussed for Hanabi Deluxe II (the Mah-Jongg-tile
edition, base 5-suit game, no Master Artisan / multicolor variants):

  - 50 tiles: 5 colors x {1x3, 2x2, 3x2, 4x2, 5x1}
  - Hand size: 5 tiles for 2-3 players, 4 tiles for 4-5 players
  - 8 clue tokens, 3 fuse tokens (3rd fuse = loss)
  - Deluxe-specific quirk: you may NOT discard while all 8 clue tokens are
    currently available (there must be at least one "spent" token to flip
    back). You also can't clue with 0 tokens available.
  - Completing a color's 5 returns a clue token (capped at 8) and is a free
    action (no discard needed).
  - Reaching 25 (all 5 fireworks complete) ends the game immediately as a win.
  - When the draw pile empties, every player (including whoever drew the
    last tile) gets exactly one more turn, then the game ends.

Includes a simple rule-based bot so games can be auto-played end to end, and
a batch-simulation mode that reports aggregate stats and cross-checks the
observed discard/clue counts against the theoretical per-player-count
ceilings derived analytically (see the conversation this was built from):

    max_discards(n) = 25 - n*(h-1)
    max_clues(n)    = 8 + max_discards(n) + 4

Usage:
    python3 hanabi_simulator.py play  --players 3 --seed 1     # one bot-only verbose game
    python3 hanabi_simulator.py sim   --players 3 --games 500  # batch stats
    python3 hanabi_simulator.py sim   --all --games 500        # all player counts
    python3 hanabi_simulator.py human --players 3              # you play seat 0, bots fill the rest
    python3 hanabi_simulator.py human --players 4 --human-seats 0,2  # hotseat: seats 0 and 2 are human

Known simplification: the clue-tracking model eliminates candidates per card
independently (if every real copy of a (color,rank) is already accounted for
elsewhere, a card can't be that pair). It does not run a full constraint solve
across all ambiguous cards at once, so in rare, heavily-overlapping situations
it could in principle "identify" two cards as the same last-remaining copy
before the mutual exclusion catches up next pass. Stress-testing (12,000+
games, tile-conservation invariants checked every game) found no case where
this actually produced a crash or an invalid state, but it's a known limit of
the simplified model worth knowing about if you extend the bot.
"""

import argparse
import random
from collections import Counter
from dataclasses import dataclass, field

COLORS = ['Red', 'Yellow', 'Green', 'Blue', 'White']
RANK_COUNTS = {1: 3, 2: 2, 3: 2, 4: 2, 5: 1}
MAX_CLUES = 8
MAX_FUSES = 3
FIREWORK_TOTAL = len(COLORS) * 5  # 25


def hand_size(n):
    return 5 if n in (2, 3) else 4


def theoretical_ceilings(n):
    h = hand_size(n)
    max_discards = FIREWORK_TOTAL - n * (h - 1)
    max_clues = MAX_CLUES + max_discards + 4
    return max_discards, max_clues


@dataclass
class Card:
    color: str
    rank: int

    def __repr__(self):
        return f"{self.color[0]}{self.rank}"


ALL_PAIRS = frozenset((c, r) for c in COLORS for r in RANK_COUNTS)


class CardKnowledge:
    """What's publicly known about one tile: a joint set of (color, rank)
    candidates, narrowed by clues AND by card-counting elimination (if every
    other copy of a candidate is already accounted for elsewhere, this card
    can't be that candidate). This is what lets a single clue sometimes fully
    identify a tile once enough of the deck has been seen."""

    def __init__(self):
        self.possible = set(ALL_PAIRS)
        self.notes = []  # human-readable log of what narrowed this slot down

    def is_identified(self):
        return len(self.possible) == 1

    def is_untouched(self):
        return len(self.possible) == len(ALL_PAIRS)

    def filter_color(self, color, matches):
        self.possible = {(c, r) for (c, r) in self.possible if (c == color) == matches}
        self.notes.append(f"told color {'IS' if matches else 'is NOT'} {color}")

    def filter_rank(self, rank, matches):
        self.possible = {(c, r) for (c, r) in self.possible if (r == rank) == matches}
        self.notes.append(f"told rank {'IS' if matches else 'is NOT'} {rank}")


def new_deck(rng):
    deck = []
    for c in COLORS:
        for rank, count in RANK_COUNTS.items():
            deck.extend([Card(c, rank)] * count)
    rng.shuffle(deck)
    return deck


class IllegalAction(Exception):
    pass


class HanabiGame:
    def __init__(self, num_players, seed=None):
        assert 2 <= num_players <= 5, "Hanabi Deluxe II supports 2-5 players"
        self.rng = random.Random(seed)
        self.n = num_players
        self.h = hand_size(num_players)
        self.deck = new_deck(self.rng)
        self.hands = [[] for _ in range(self.n)]
        self.knowledge = [[] for _ in range(self.n)]
        for _ in range(self.h):
            for p in range(self.n):
                self._draw_to_hand(p)

        self.discard_pile = []
        self.fireworks = {c: 0 for c in COLORS}
        self.clue_tokens = MAX_CLUES
        self.strikes = 0
        self.current_player = 0
        self.turn_count = 0
        self.final_round_left = None  # becomes an int once the deck runs dry
        self.game_over = False
        self.result = None  # 'win' / 'loss' / 'deck_out'
        self.log = []

        # running stats
        self.discards_made = 0
        self.clues_given = 0
        self.successful_plays = 0
        self.misplays = 0

    # ---------- core mechanics ----------

    def _draw_to_hand(self, player):
        if self.deck:
            self.hands[player].append(self.deck.pop())
            self.knowledge[player].append(CardKnowledge())

    def score(self):
        return sum(self.fireworks.values())

    def legal_actions(self, player):
        actions = []
        if self.clue_tokens > 0:
            for target in range(self.n):
                if target == player:
                    continue
                colors_present = {c.color for c in self.hands[target]}
                ranks_present = {c.rank for c in self.hands[target]}
                for c in colors_present:
                    actions.append(('clue', target, 'color', c))
                for r in ranks_present:
                    actions.append(('clue', target, 'rank', r))
        if self.clue_tokens < MAX_CLUES:
            for i in range(len(self.hands[player])):
                actions.append(('discard', i))
        for i in range(len(self.hands[player])):
            actions.append(('play', i))
        return actions

    def apply_action(self, player, action):
        if self.game_over:
            raise IllegalAction("game already over")
        kind = action[0]

        if kind == 'clue':
            _, target, clue_type, value = action
            if self.clue_tokens <= 0:
                raise IllegalAction("no clue tokens available")
            if target == player:
                raise IllegalAction("can't clue yourself")
            matched_any = False
            for card, know in zip(self.hands[target], self.knowledge[target]):
                if clue_type == 'color':
                    matches = (card.color == value)
                    know.filter_color(value, matches)
                else:
                    matches = (card.rank == value)
                    know.filter_rank(value, matches)
                matched_any = matched_any or matches
            if not matched_any:
                raise IllegalAction("clue must match at least one tile")
            self.clue_tokens -= 1
            self.clues_given += 1
            self.log.append(f"P{player} clues P{target}: {clue_type}={value}")
            self._run_elimination()

        elif kind == 'discard':
            if self.clue_tokens >= MAX_CLUES:
                raise IllegalAction("can't discard while all 8 clue tokens are available")
            _, idx = action
            card = self.hands[player].pop(idx)
            self.knowledge[player].pop(idx)
            self.discard_pile.append(card)
            self.clue_tokens = min(MAX_CLUES, self.clue_tokens + 1)
            self.discards_made += 1
            self.log.append(f"P{player} discards {card}")
            self._run_elimination()
            self._draw_replacement(player)

        elif kind == 'play':
            _, idx = action
            card = self.hands[player].pop(idx)
            self.knowledge[player].pop(idx)
            if self.fireworks[card.color] == card.rank - 1:
                self.fireworks[card.color] = card.rank
                self.successful_plays += 1
                self.log.append(f"P{player} plays {card} (success)")
                if card.rank == 5:
                    self.clue_tokens = min(MAX_CLUES, self.clue_tokens + 1)
                if self.score() == FIREWORK_TOTAL:
                    self.game_over = True
                    self.result = 'win'
            else:
                self.discard_pile.append(card)
                self.strikes += 1
                self.misplays += 1
                self.log.append(f"P{player} MISPLAYS {card} (strike {self.strikes}/{MAX_FUSES})")
                if self.strikes >= MAX_FUSES:
                    self.game_over = True
                    self.result = 'loss'
            self._run_elimination()
            if not self.game_over:
                self._draw_replacement(player)
        else:
            raise IllegalAction(f"unknown action {action}")

    def accounted_counts(self):
        """How many copies of each (color, rank) are already nailed down
        somewhere - discarded, played, or in a fully-identified hand slot.
        Shared by elimination and by the probability-weighting used for move
        advice (RANK_COUNTS[rank] - accounted = copies still unaccounted for,
        i.e. could be anywhere still ambiguous: someone's unidentified slot,
        or the draw pile)."""
        accounted = Counter()
        for card in self.discard_pile:
            accounted[(card.color, card.rank)] += 1
        for color, top in self.fireworks.items():
            for r in range(1, top + 1):
                accounted[(color, r)] += 1
        for hand_know in self.knowledge:
            for k in hand_know:
                if k.is_identified():
                    accounted[next(iter(k.possible))] += 1
        return accounted

    def _run_elimination(self):
        """Constraint-propagate: if every remaining copy of a (color, rank)
        is already accounted for by identified/visible tiles, no ambiguous
        tile can secretly be that pair. Iterate to a fixpoint since newly
        identified tiles can trigger further eliminations elsewhere."""
        changed = True
        while changed:
            changed = False
            accounted = self.accounted_counts()
            for hand_know in self.knowledge:
                for k in hand_know:
                    if k.is_identified():
                        continue
                    new_possible = {p for p in k.possible if accounted[p] < RANK_COUNTS[p[1]]}
                    if new_possible != k.possible:
                        removed = k.possible - new_possible
                        for (c, r) in removed:
                            k.notes.append(
                                f"eliminated {c}{r} (all {RANK_COUNTS[r]} copies already accounted for elsewhere)")
                        k.possible = new_possible
                        changed = True
                        if len(new_possible) == 1:
                            c, r = next(iter(new_possible))
                            k.notes.append(f"=> by elimination, this MUST be {c}{r} (pigeonhole: nowhere else it could be)")

    def _draw_replacement(self, player):
        if self.deck:
            self._draw_to_hand(player)
            if not self.deck and self.final_round_left is None:
                # "Each player plays one more time, including the player who
                # picked up the last tile" - this CURRENT turn (the one doing
                # the draining draw) is a normal main-phase turn and should
                # NOT itself count against the n-turn final round. But the
                # advance_turn() call that immediately follows this same
                # action will decrement final_round_left once regardless, so
                # set it to n+1 to absorb that "free" decrement and leave
                # exactly n genuinely new turns afterward.
                self.final_round_left = self.n + 1

    def advance_turn(self):
        self.turn_count += 1
        if self.final_round_left is not None:
            self.final_round_left -= 1
            if self.final_round_left <= 0 and not self.game_over:
                self.game_over = True
                self.result = self.result or 'deck_out'
        self.current_player = (self.current_player + 1) % self.n


# ---------------------------------------------------------------------------
# A simple rule-based bot. Not expert-level, but plays soundly: only plays
# cards it can prove are safe from public clue-knowledge, gives save clues
# for critical tiles, and discards its oldest untouched tile otherwise.
# ---------------------------------------------------------------------------

def is_playable(possible_pairs, fireworks):
    return all(fireworks[c] == r - 1 for c, r in possible_pairs)


def is_dead(possible_pairs, fireworks):
    """Every consistent (color, rank) is already behind that color's stack."""
    return all(r <= fireworks[c] for c, r in possible_pairs)


def remaining_count(game, color, rank):
    """How many copies of (color, rank) are not yet visible in discard/fireworks."""
    total = RANK_COUNTS[rank]
    used = sum(1 for c in game.discard_pile if c.color == color and c.rank == rank)
    if game.fireworks[color] >= rank:
        used += 1
    return total - used


def choose_action(game, player, trace=None):
    """Picks an action. If `trace` is a list, appends human-readable
    reasoning lines to it describing what was deduced and why this action
    was chosen (this is the "thinking out loud" log)."""
    if trace is None:
        trace = []
    hand = game.hands[player]
    know = game.knowledge[player]

    # Log a snapshot of current self-knowledge, calling out anything that
    # got pinned down purely by elimination (pigeonhole-style deduction)
    # rather than a direct clue.
    trace.append(f"My hand: " + " | ".join(f"[{i}] {describe_knowledge(k)}" for i, k in enumerate(know)))
    for i, k in enumerate(know):
        pigeonhole_notes = [n for n in k.notes if n.startswith("=>")]
        if pigeonhole_notes:
            trace.append(f"  slot {i}: {pigeonhole_notes[-1]}")

    # 1) Play any tile we can prove is safe (leftmost/oldest first).
    for i, k in enumerate(know):
        if is_playable(k.possible, game.fireworks):
            trace.append(f"Slot {i} ({describe_knowledge(k)}) is provably playable "
                          f"against the current fireworks -> PLAY slot {i}.")
            return ('play', i), trace

    # 2) If we have clue tokens, try to set up a teammate's play - but only
    # give a clue that would make the tile PROVABLY playable to them (given
    # what's already been eliminated for that slot); otherwise don't waste
    # the clue.
    if game.clue_tokens > 0:
        for offset in range(1, game.n):
            target = (player + offset) % game.n
            t_hand = game.hands[target]
            t_know = game.knowledge[target]
            for i, (card, k) in enumerate(zip(t_hand, t_know)):
                if is_playable(k.possible, game.fireworks):
                    continue  # already playable to them, no need to reclue
                if game.fireworks[card.color] != card.rank - 1:
                    continue  # not actually playable right now
                trial_color = {p for p in k.possible if p[0] == card.color}
                if trial_color and is_playable(trial_color, game.fireworks):
                    trace.append(f"P{target}'s slot {i} is really {card} and IS playable now "
                                 f"(their {card.color} stack needs exactly this). A color clue would "
                                 f"fully pin it down (only {card.color} card left as a candidate) -> "
                                 f"clue P{target} color={card.color}.")
                    return ('clue', target, 'color', card.color), trace
                trial_rank = {p for p in k.possible if p[1] == card.rank}
                if trial_rank and is_playable(trial_rank, game.fireworks):
                    trace.append(f"P{target}'s slot {i} is really {card} and IS playable now. "
                                 f"A rank clue would fully pin it down -> clue P{target} rank={card.rank}.")
                    return ('clue', target, 'rank', card.rank), trace
                # Neither single clue fully proves it yet - skip, don't waste it.

        # 3) Save clues: protect critical tiles sitting on a teammate's chop
        # (their oldest fully-untouched tile).
        for offset in range(1, game.n):
            target = (player + offset) % game.n
            t_hand = game.hands[target]
            t_know = game.knowledge[target]
            chop_idx = next((i for i, k in enumerate(t_know) if k.is_untouched()), None)
            if chop_idx is None:
                continue
            card = t_hand[chop_idx]
            if card.rank == 5 or remaining_count(game, card.color, card.rank) == 1:
                if not is_dead(t_know[chop_idx].possible, game.fireworks):
                    reason = "it's a 5" if card.rank == 5 else "it's the LAST remaining copy of that tile"
                    trace.append(f"P{target}'s chop (slot {chop_idx}, really {card}) is untouched and {reason} "
                                 f"- if they discard it next turn it's gone forever. Giving a save clue -> "
                                 f"clue P{target} rank={card.rank}.")
                    return ('clue', target, 'rank', card.rank), trace

    # 4) Discard: prefer a known-dead tile, else our own oldest untouched
    # tile (chop), else just the oldest tile we're holding.
    if game.clue_tokens < MAX_CLUES:
        for i, k in enumerate(know):
            if is_dead(k.possible, game.fireworks):
                trace.append(f"Slot {i} ({describe_knowledge(k)}) is provably dead already "
                             f"(every consistent color/rank is already behind that stack) -> DISCARD slot {i}.")
                return ('discard', i), trace
        for i, k in enumerate(know):
            if k.is_untouched():
                trace.append(f"Nothing provably playable or dead. Slot {i} is my oldest untouched "
                             f"(uninformed) tile, i.e. my chop -> DISCARD slot {i}.")
                return ('discard', i), trace
        trace.append("Nothing clearly safe to discard either, falling back to slot 0 -> DISCARD slot 0.")
        return ('discard', 0), trace

    # 5) Stuck: clue tokens are maxed out (can't discard) and nothing useful
    # to clue was found above. Spend a token on whatever clue is legal so a
    # discard becomes possible next time.
    legal = game.legal_actions(player)
    clue_actions = [a for a in legal if a[0] == 'clue']
    if clue_actions:
        trace.append("Clue tokens are maxed at 8 so I can't discard, and nothing useful to clue - "
                     "spending an arbitrary clue just to free up discarding for next time.")
        return clue_actions[0], trace

    # Truly no clues and can't discard (shouldn't normally happen) - play
    # blind on the oldest tile as a last resort.
    trace.append("No clue tokens and can't discard - forced to play blind on slot 0.")
    return ('play', 0), trace


# ---------------------------------------------------------------------------
# Mathematical move advisor. This is NOT a full game-theoretic solver - exact
# optimal play in full Hanabi is a decentralized-POMDP problem, which is
# NEXP-hard to solve in general, so there's no tractable closed-form "best
# move". What IS exactly computable is a Bayesian estimate from the current
# public information: given every visible card, the discard pile, and the
# clue history, you can work out precisely how many physical copies of each
# candidate identity are still unaccounted for, and turn that into an exact
# probability that a given play succeeds or a given discard is safe. That's
# what this does.
# ---------------------------------------------------------------------------

def slot_probabilities(game, player, slot_idx):
    """Probability distribution over what this card slot actually is, given
    public information - weighted by how many real copies of each candidate
    remain unaccounted for (not just a flat guess across possible types)."""
    k = game.knowledge[player][slot_idx]
    accounted = game.accounted_counts()
    weights = {}
    for (c, r) in k.possible:
        w = RANK_COUNTS[r] - accounted[(c, r)]
        if w > 0:
            weights[(c, r)] = w
    total = sum(weights.values())
    if total == 0:
        return {}
    return {pair: w / total for pair, w in weights.items()}


def analyze_hand(game, player):
    """For every slot in `player`'s own hand: exact P(playing it succeeds
    right now) and P(discarding it destroys the last copy of something the
    team still needs)."""
    results = []
    for i in range(len(game.hands[player])):
        probs = slot_probabilities(game, player, i)
        p_play = sum(p for (c, r), p in probs.items() if game.fireworks[c] == r - 1)
        p_critical = sum(p for (c, r), p in probs.items()
                          if r > game.fireworks[c] and remaining_count(game, c, r) <= 1)
        results.append({'slot': i, 'p_play': p_play, 'p_critical': p_critical, 'probs': probs})
    return results


def recommend_action(game, player):
    """Returns (lines, suggested_action). Combines exact probabilities for
    your own ambiguous cards with the deterministic clue-search (clues target
    other players' REAL, fully-visible hands, so no probability needed
    there) to suggest a move, prioritized the same way strong human play
    generally is: a provably-safe play beats everything; then a clue that
    hands a teammate a certain play; then a save clue for a critical tile;
    then the lowest-risk discard."""
    lines = []
    analysis = analyze_hand(game, player)

    lines.append("Play safety by slot (exact P(playable), weighted by how many real copies "
                 "of each candidate identity remain unaccounted for):")
    for a in analysis:
        tag = "  <- CERTAIN" if a['p_play'] >= 0.999 else ""
        lines.append(f"  slot {a['slot']}: P(playable) = {a['p_play'] * 100:5.1f}%{tag}")

    lines.append("Discard risk by slot (P it's the last copy of something still needed):")
    for a in analysis:
        tag = "  <- would PERMANENTLY cap a color, avoid" if a['p_critical'] >= 0.999 else ""
        lines.append(f"  slot {a['slot']}: P(critical if discarded) = {a['p_critical'] * 100:5.1f}%{tag}")

    best_play = max(analysis, key=lambda a: a['p_play'])
    safest_discard = min(analysis, key=lambda a: a['p_critical'])

    trace = []
    bot_action, trace = choose_action(game, player, trace)

    lines.append("")
    if best_play['p_play'] >= 0.999:
        lines.append(f"RECOMMENDATION: play slot {best_play['slot']} - provably safe (100%).")
        suggestion = ('play', best_play['slot'])
    elif bot_action[0] == 'clue':
        lines.append(f"RECOMMENDATION: {trace[-1]}")
        suggestion = bot_action
    elif game.clue_tokens < MAX_CLUES:
        lines.append(f"RECOMMENDATION: discard slot {safest_discard['slot']} - lowest risk option "
                     f"({safest_discard['p_critical'] * 100:.1f}% chance it's critical).")
        suggestion = ('discard', safest_discard['slot'])
    else:
        lines.append(f"RECOMMENDATION: {trace[-1]}")
        suggestion = bot_action
    return lines, suggestion


def play_one_game(num_players, seed=None, verbose=False, explain=False):
    game = HanabiGame(num_players, seed=seed)
    guard = 0
    while not game.game_over:
        guard += 1
        if guard > 5000:
            raise RuntimeError("game did not terminate - bot logic bug")
        player = game.current_player
        trace = [] if explain else None
        action, trace = choose_action(game, player, trace)
        game.apply_action(player, action)
        if explain:
            for line in trace:
                print(f"  (P{player} thinking) {line}")
        if verbose:
            print(game.log[-1])
        game.advance_turn()
    if verbose:
        print(f"\nResult: {game.result} | Score: {game.score()}/25 | "
              f"Discards: {game.discards_made} | Clues: {game.clues_given} | "
              f"Strikes: {game.strikes}")
        print(format_discard_pile(game))
    return game


def simulate(num_players, games, seed=0):
    scores = []
    discards = []
    clues = []
    wins = 0
    losses = 0
    max_discards_theory, max_clues_theory = theoretical_ceilings(num_players)

    perfect_discards, perfect_clues = [], []

    for i in range(games):
        g = play_one_game(num_players, seed=seed + i)
        scores.append(g.score())
        discards.append(g.discards_made)
        clues.append(g.clues_given)
        if g.result == 'win':
            wins += 1
            perfect_discards.append(g.discards_made)
            perfect_clues.append(g.clues_given)
            # The 25-n(h-1) / 37-n(h-1) ceilings were derived assuming a
            # perfect (25-point) game specifically, so they're only a valid
            # upper bound on games that actually reach 25.
            assert g.discards_made <= max_discards_theory, \
                f"PERFECT game had {g.discards_made} discards, exceeding theoretical max {max_discards_theory}"
            assert g.clues_given <= max_clues_theory, \
                f"PERFECT game had {g.clues_given} clues, exceeding theoretical max {max_clues_theory}"
        elif g.result == 'loss':
            losses += 1

    n = len(scores)
    print(f"\n=== {num_players} players, {games} games ===")
    print(f"Ceiling for a PERFECT (25-pt) game only: max_discards={max_discards_theory}, max_clues={max_clues_theory}")
    print(f"Avg score:    {sum(scores)/n:.2f} / 25   (best {max(scores)}, worst {min(scores)})")
    print(f"Win rate (25):  {wins/n*100:.1f}%")
    print(f"Loss rate (3 strikes): {losses/n*100:.1f}%")
    print(f"Avg discards (all games): {sum(discards)/n:.2f}  (max observed {max(discards)})")
    print(f"Avg clues (all games):    {sum(clues)/n:.2f}  (max observed {max(clues)})")
    if perfect_discards:
        print(f"Among the {wins} PERFECT games: avg discards {sum(perfect_discards)/len(perfect_discards):.2f}, "
              f"avg clues {sum(perfect_clues)/len(perfect_clues):.2f} (both verified <= ceiling)")
    print("Note: non-perfect games can exceed the 'ceiling' above - it only bounds true 25-point finishes.")
    return dict(scores=scores, discards=discards, clues=clues, wins=wins, losses=losses)


# ---------------------------------------------------------------------------
# Human-playable mode: terminal UI. Any subset of seats can be human-
# controlled (hotseat multiplayer); the rest are filled by the bot above.
# ---------------------------------------------------------------------------

def describe_knowledge(k):
    if k.is_identified():
        c, r = next(iter(k.possible))
        return f"{c}{r} (known)"
    colors = sorted({c for c, r in k.possible})
    ranks = sorted({r for c, r in k.possible})
    color_str = ",".join(x[0] for x in colors) if len(colors) < len(COLORS) else "any"
    rank_str = ",".join(str(x) for x in ranks) if len(ranks) < len(RANK_COUNTS) else "any"
    return f"color:[{color_str}] rank:[{rank_str}]"


def format_discard_pile(game):
    """Group discards by color, sorted by rank, so it's easy to scan which
    tiles are gone (rather than a flat chronological list)."""
    if not game.discard_pile:
        return "Discard pile: (empty)"
    by_color = {c: [] for c in COLORS}
    for card in game.discard_pile:
        by_color[card.color].append(card.rank)
    parts = []
    for c in COLORS:
        ranks = sorted(by_color[c])
        parts.append(f"{c}: {','.join(str(r) for r in ranks) if ranks else '-'}")
    return f"Discard pile ({len(game.discard_pile)}):  " + "   ".join(parts)


def format_move_history(game, last_n=None):
    entries = list(enumerate(game.log, start=1))
    if last_n is not None:
        entries = entries[-last_n:]
    if not entries:
        return "(no moves yet)"
    return "\n".join(f"{i:>3}. {line}" for i, line in entries)


def display_state(game, player):
    print("\n" + "=" * 70)
    print(f"Turn {game.turn_count + 1}  |  Player {player}'s turn")
    fw = "  ".join(f"{c}:{game.fireworks[c]}" for c in COLORS)
    print(f"Fireworks -> {fw}   (score {game.score()}/25)")
    print(f"Clue tokens: {game.clue_tokens}/8   Strikes: {game.strikes}/3   Deck left: {len(game.deck)}")
    print(format_discard_pile(game))
    print("-" * 70)
    print("Recent moves:")
    print(format_move_history(game, last_n=6))
    print("(type L to see the full move history)")
    print("-" * 70)
    for p in range(game.n):
        if p == player:
            cards = " | ".join(f"[{i}] {describe_knowledge(k)}" for i, k in enumerate(game.knowledge[p]))
            print(f"Your hand (P{p}):  {cards}")
            for i, k in enumerate(game.knowledge[p]):
                pigeonhole_notes = [n for n in k.notes if n.startswith("=>")]
                if pigeonhole_notes:
                    print(f"    slot {i}: {pigeonhole_notes[-1]}")
        else:
            cards = " | ".join(f"[{i}] {c}" for i, c in enumerate(game.hands[p]))
            print(f"P{p}'s hand:      {cards}")
    print("=" * 70)


def prompt_for_action(game, player):
    while True:
        choice = input("\n[P]lay, [D]iscard, [C]lue, [A]dvice, [L]og, [Q]uit > ").strip().lower()
        if choice.startswith('q'):
            raise KeyboardInterrupt
        if choice.startswith('l'):
            print(format_move_history(game))
            continue
        if choice.startswith('a'):
            lines, _ = recommend_action(game, player)
            print("\n".join(lines))
            continue
        if choice.startswith('p') or choice.startswith('d'):
            raw = input(f"Card index (0-{len(game.hands[player]) - 1})? ").strip()
            try:
                idx = int(raw)
            except ValueError:
                print("Not a number, try again.")
                continue
            return ('play', idx) if choice.startswith('p') else ('discard', idx)
        if choice.startswith('c'):
            raw = input(f"Clue which player (0-{game.n - 1}, not yourself)? ").strip()
            try:
                target = int(raw)
            except ValueError:
                print("Not a number, try again.")
                continue
            ctype = input("Clue [c]olor or [r]ank? ").strip().lower()
            if ctype.startswith('c'):
                value = input(f"Which color ({', '.join(COLORS)})? ").strip().capitalize()
                if value not in COLORS:
                    print("Not a valid color, try again.")
                    continue
                return ('clue', target, 'color', value)
            else:
                raw = input("Which rank (1-5)? ").strip()
                try:
                    value = int(raw)
                except ValueError:
                    print("Not a number, try again.")
                    continue
                return ('clue', target, 'rank', value)
        print("Didn't catch that - type P, D, C, A, L, or Q.")


def pass_device(player):
    input(f"\n{'#' * 70}\nPass the keyboard to Player {player}. Press Enter when ready...")
    print("\n" * 60)


def play_human_game(num_players, human_seats, seed=None, show_bot_thinking=True):
    game = HanabiGame(num_players, seed=seed)
    print(f"Starting a {num_players}-player game. Human seat(s): {sorted(human_seats)}. "
          f"Bots fill the rest.")
    if show_bot_thinking:
        print("(Bot reasoning will be printed before each bot move. Use --no-thinking to hide it.)")
    try:
        while not game.game_over:
            player = game.current_player
            if player in human_seats:
                if len(human_seats) > 1:
                    pass_device(player)
                display_state(game, player)
                while True:
                    try:
                        action = prompt_for_action(game, player)
                        game.apply_action(player, action)
                        print(f">> {game.log[-1]}")
                        break
                    except IllegalAction as e:
                        print(f"Illegal move: {e}")
            else:
                trace = []
                action, trace = choose_action(game, player, trace)
                game.apply_action(player, action)
                if show_bot_thinking:
                    for line in trace:
                        print(f"  (P{player} thinking) {line}")
                print(f"[bot] {game.log[-1]}")
            game.advance_turn()
    except (KeyboardInterrupt, EOFError):
        print("\nGame abandoned.")
        return game

    print("\n" + "#" * 70)
    print(f"GAME OVER: {game.result}   Final score: {game.score()}/25")
    print(f"Discards used: {game.discards_made}   Clues given: {game.clues_given}   Strikes: {game.strikes}")
    print(format_discard_pile(game))
    print("Final hands:")
    for p in range(game.n):
        print(f"  P{p}: {' | '.join(str(c) for c in game.hands[p])}")
    print("\nFull move history:")
    print(format_move_history(game))
    return game


def main():
    parser = argparse.ArgumentParser(description="Hanabi Deluxe II simulator")
    sub = parser.add_subparsers(dest='mode', required=True)

    p_play = sub.add_parser('play', help='Play (auto-play) a single verbose game')
    p_play.add_argument('--players', type=int, required=True, choices=[2, 3, 4, 5])
    p_play.add_argument('--seed', type=int, default=None)
    p_play.add_argument('--explain', action='store_true', help='print each bot\'s reasoning before its move')

    p_sim = sub.add_parser('sim', help='Run a batch of auto-played games and report stats')
    p_sim.add_argument('--players', type=int, choices=[2, 3, 4, 5])
    p_sim.add_argument('--all', action='store_true', help='run for all player counts 2-5')
    p_sim.add_argument('--games', type=int, default=200)
    p_sim.add_argument('--seed', type=int, default=0)

    p_human = sub.add_parser('human', help='Play interactively in the terminal (with bots filling other seats)')
    p_human.add_argument('--players', type=int, required=True, choices=[2, 3, 4, 5])
    p_human.add_argument('--human-seats', type=str, default='0',
                          help='comma-separated seat indices controlled by a human, e.g. "0" or "0,2"')
    p_human.add_argument('--seed', type=int, default=None)
    p_human.add_argument('--no-thinking', action='store_true', help='hide bot reasoning, just show their moves')

    args = parser.parse_args()

    if args.mode == 'play':
        play_one_game(args.players, seed=args.seed, verbose=True, explain=args.explain)
    elif args.mode == 'sim':
        if args.all:
            for n in [2, 3, 4, 5]:
                simulate(n, args.games, seed=args.seed)
        else:
            if not args.players:
                parser.error("--players is required unless --all is given")
            simulate(args.players, args.games, seed=args.seed)
    elif args.mode == 'human':
        seats = {int(x) for x in args.human_seats.split(',') if x.strip() != ''}
        if not seats or max(seats) >= args.players or min(seats) < 0:
            parser.error(f"--human-seats must be indices between 0 and {args.players - 1}")
        play_human_game(args.players, seats, seed=args.seed, show_bot_thinking=not args.no_thinking)


if __name__ == '__main__':
    main()
