"""
From Character-level DFA to Token-level FSM
============================================

A regular expression defines valid *character* sequences. But LLMs don't
generate characters — they generate *tokens* (which can be multi-character
strings like "ing", ".2", or "123").

The key insight is: we can **precompile** the character-level DFA into a
new FSM whose transitions are over token IDs instead of characters.

Pipeline:
  1. Parse the regex → character-level DFA  (interegular)
  2. Clean up the DFA                       (make_deterministic_fsm)
  3. Walk each token through every state     (create_fsm_index_tokenizer)
  → Token-level FSM ready for constrained decoding
"""

import math
import numpy as np
import interegular
from interegular import fsm as fsm_module
from scipy.special import softmax
from collections import defaultdict
from dataclasses import dataclass, field


# ──────────────────────────────────────────────────────────────────────────────
# Step 1: Clean up the character-level DFA
# ──────────────────────────────────────────────────────────────────────────────

def make_deterministic_fsm(fsm):
    """Normalize the character-level DFA from interegular.

    interegular's FSMs can contain:
      - A special "oblivion" / dead state (often the object `None` or an
        unreachable sink) that we want to remap to a clean integer.
      - An `anything_else` sentinel in the alphabet for characters not
        explicitly listed.

    This function:
      1. Collects all reachable states (including the dead/sink state).
      2. Remaps them to contiguous integers 0, 1, 2, ...
      3. Returns a new clean FSM and the state mapping.

    Returns
    -------
    new_fsm : A cleaned-up FSM with integer states and no implicit sink.
    state_mapping : dict mapping old state → new integer state.
    """
    # --- 收集转移表中所有可达的状态 ---
    all_states = set()
    all_states.add(fsm.initial)
    all_states.update(fsm.finals)

    for state, transitions in fsm.map.items():
        all_states.add(state)
        for symbol_idx, target in transitions.items():
            all_states.add(target)

    # interegular 用一个 "oblivion" 状态表示死胡同。它可能出现在
    # 转移表中，也可能不出现。我们专门加一个死状态。
    dead_state = max(s for s in all_states if isinstance(s, int)) + 1 if all(isinstance(s, int) for s in all_states) else len(all_states)

    # --- 构建连续的整数映射 ---
    # 把初始状态放在最前（映射到 0），其余按顺序。
    ordered = [fsm.initial]
    for s in sorted(all_states - {fsm.initial}, key=lambda x: (not isinstance(x, int), str(x))):
        ordered.append(s)

    state_mapping = {old: new for new, old in enumerate(ordered)}
    # 把任何没见过的状态（例如 oblivion）映射到死状态
    dead_new = len(ordered)

    def map_state(s):
        return state_mapping.get(s, dead_new)

    # --- 用新的状态 ID 重建转移表 ---
    new_map = {}
    for old_state, transitions in fsm.map.items():
        new_state = map_state(old_state)
        new_map[new_state] = {}
        for symbol_idx, old_target in transitions.items():
            new_map[new_state][symbol_idx] = map_state(old_target)

    new_initial = map_state(fsm.initial)
    new_finals = frozenset(map_state(s) for s in fsm.finals)

    # 用 interegular 的 FSM 类构建新的 FSM
    new_fsm = fsm_module.FSM(
        alphabet=fsm.alphabet,       # 相同的字符 → symbol_index 映射
        states=set(range(len(ordered))),
        initial=new_initial,
        finals=new_finals,
        map=new_map,
    )

    return new_fsm, state_mapping


# ──────────────────────────────────────────────────────────────────────────────
# Step 2: Build the token-level FSM index
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TokenFSM:
    """A finite-state machine whose transitions are over token IDs.

    Attributes
    ----------
    initial : int
        The initial state (same as the character DFA's initial state).
    finals : set of int
        The accept states.
    map : dict[int, dict[int, int]]
        Transition table: map[state][token_id] → next_state
    """
    initial: int
    finals: set
    map: dict = field(default_factory=dict)

    def allowed_token_ids(self, state):
        """Return the set of token IDs valid from `state`."""
        return set(self.map.get(state, {}).keys())

    def next_state(self, state, token_id):
        """Transition to the next state given a token ID."""
        return self.map[state][token_id]


def _walk_token_through_fsm(fsm, state, token):
    """Walk a token (string) through the character-level DFA from `state`.

    For each character in the token, we look up the corresponding symbol
    index in the DFA's alphabet and attempt to transition. If any character
    has no valid transition, the token is incompatible with this state.

    Parameters
    ----------
    fsm : the character-level DFA (cleaned up by make_deterministic_fsm)
    state : int, starting state
    token : str, the token string to walk

    Returns
    -------
    final_state : int or None
        The state we land on after consuming all characters, or None if
        the token is rejected from this starting state.
    """
    for char in token:
        # 查这个字符映射到哪个符号索引
        if char in fsm.alphabet:
            symbol_idx = fsm.alphabet[char]
        elif fsm_module.anything_else in fsm.alphabet:
            symbol_idx = fsm.alphabet[fsm_module.anything_else]
        else:
            return None  # 字符根本不在字母表里

        # 尝试转移
        transitions = fsm.map.get(state, {})
        if symbol_idx not in transitions:
            return None  # 没有有效转移 → 拒绝
        state = transitions[symbol_idx]

    return state


def create_fsm_index_tokenizer(fsm, tokenizer):
    """Build a token-level FSM from a character-level DFA and a tokenizer.

    For every (state, token) pair, we check whether the token can be
    "walked" through the character DFA starting from that state. If yes,
    we record the transition: state --token_id--> landing_state.

    This is O(|states| × |vocabulary| × avg_token_length) but is done
    **once** and then reused for all generation steps.

    Parameters
    ----------
    fsm : the character-level DFA (output of make_deterministic_fsm)
    tokenizer : list of str (vocabulary), or any object with a
                `convert_ids_to_tokens` method.

    Returns
    -------
    token_fsm : TokenFSM
        The compiled token-level FSM.
    index : dict[int, dict[int, int]]
        Same as token_fsm.map (for backward compatibility).
    """
    # 同时支持简单列表和 HuggingFace 风格的分词器
    if isinstance(tokenizer, list):
        vocabulary = tokenizer
    else:
        vocabulary = [tokenizer.convert_ids_to_tokens(i)
                      for i in range(tokenizer.vocab_size)]

    index = defaultdict(dict)  # state → {token_id → next_state}

    for state in fsm.states:
        for token_id, token in enumerate(vocabulary):
            landing = _walk_token_through_fsm(fsm, state, token)
            if landing is not None:
                index[state][token_id] = landing

    token_fsm = TokenFSM(
        initial=fsm.initial,
        finals=set(fsm.finals),
        map=dict(index),
    )

    return token_fsm, dict(index)


# ══════════════════════════════════════════════════════════════════════════════
# Demo
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # --- 配置 ---
    regex_pattern = r"([0-9]+)?\.[0-9]+"
    vocabulary = ["a", ".", ".2", "1"]

    print("=" * 60)
    print("Character-level DFA → Token-level FSM")
    print("=" * 60)
    print(f"\nRegex:      {regex_pattern}")
    print(f"Vocabulary: {vocabulary}\n")

    # --- 第 1 步：把正则解析成字符级 DFA ---
    raw_fsm = interegular.parse_pattern(regex_pattern).to_fsm()
    print("── Raw character-level DFA ──")
    print(f"  States:  {raw_fsm.states}")
    print(f"  Initial: {raw_fsm.initial}")
    print(f"  Finals:  {raw_fsm.finals}")
    print(f"  Transitions:")
    for state, trans in sorted(raw_fsm.map.items(), key=lambda x: str(x[0])):
        print(f"    State {state}: {dict(trans)}")

    # --- 第 2 步：清理 ---
    clean_fsm, state_mapping = make_deterministic_fsm(raw_fsm)
    print(f"\n── Cleaned DFA (state mapping: {state_mapping}) ──")
    print(f"  States:  {clean_fsm.states}")
    print(f"  Initial: {clean_fsm.initial}")
    print(f"  Finals:  {clean_fsm.finals}")

    # --- 第 3 步：构建 token 级 FSM ---
    token_fsm, index = create_fsm_index_tokenizer(clean_fsm, vocabulary)

    print(f"\n── Token-level FSM ──")
    print(f"  Initial state: {token_fsm.initial}")
    print(f"  Accept states: {token_fsm.finals}")
    print(f"\n  Transition table (state → token → next_state):")
    for state in sorted(token_fsm.map.keys()):
        for tid, next_s in sorted(token_fsm.map[state].items()):
            print(f"    State {state} --[{tid}: '{vocabulary[tid]}']→ State {next_s}")

    # --- 第 4 步：用 token FSM 做受限生成 ---
    print(f"\n── Constrained generation ──")

    np.random.seed(12349)
    logits = np.ones(len(vocabulary))

    completion = ""
    state = token_fsm.initial

    for step in range(7):
        # 屏蔽现在只是一次集合查找 — 不需要正则！
        allowed = token_fsm.allowed_token_ids(state)
        mask = np.full(len(vocabulary), -np.inf)
        mask[list(allowed)] = 0.0

        masked_logits = logits + mask
        probs = softmax(masked_logits)
        next_id = np.random.choice(len(vocabulary), p=probs)

        next_state = token_fsm.next_state(state, next_id)
        print(f"  Step {step}: state={state}, "
              f"allowed={[vocabulary[i] for i in sorted(allowed)]}, "
              f"sampled='{vocabulary[next_id]}' → state={next_state}")

        state = next_state
        completion += vocabulary[next_id]

    print(f"\n  Final completion: '{completion}'")
    is_full_match = state in token_fsm.finals
    print(f"  In accept state?  {is_full_match}")
