"""
Реализация класса HMM, имеющего 4 необходимых метода:
1) алгоритм Витерби
2) алгоритм Forward 
3) алгоритм Backward
4) Апостериорное декодирование

Все вычисления проводятся в логарифмическом пространстве. Где необходимо - проводится обратное преобразование. ТАКЖЕ ДОБАВЛЕНА ПОДДЕРЖКА КОНЦА ПОСЛЕДОВАТЕЛЬНОСТИ. те если есть последотвальность, то после нее если указать соотв параметр идет конец последовательности. и тогда соответственно она должна заканчиваться интроном лол
"""

import numpy as np
from dataclasses import dataclass, field 
# дата классы использованы для наиболее удобной реализации классов, филд - чтобы не было проблем с параметрами по умолчанию
from typing import Sequence    # обобщенная последовательность


# ── tiny helper ──────────────────────────────────────────────────────────────

def _log(x) -> np.ndarray:
    """Safe log: replaces 0 → -inf without warning."""
    with np.errstate(divide="ignore"):
        return np.log(np.asarray(x, dtype=float))


LOG_ZERO = -np.inf


# ── result containers ─────────────────────────────────────────────────────────

@dataclass
class ViterbiResult:
    path: list[str]      # most probable hidden-state sequence
    log_prob: float      # log P(best path, O | λ)

@dataclass
class ForwardResult:
    alpha: np.ndarray    # shape (T, N) — log α
    log_prob: float      # log P(O | λ)

@dataclass
class BackwardResult:
    beta: np.ndarray     # shape (T, N) — log β
    log_prob: float      # log P(O | λ) computed from β

@dataclass
class PosteriorResult:
    gamma: np.ndarray            # shape (T, N) — P(state i at t | O, λ) [linear]
    states_at_each_step: list[str]
    log_prob: float              # log P(O | λ)


# ── main class ────────────────────────────────────────────────────────────────

class HiddenMarkovModel:
    """
    Parameters
    ----------
    states : list[str]
        Names of hidden states, e.g. ['E', '5', 'I'].
    symbols : list[str]
        Observation alphabet, e.g. ['A', 'T', 'G', 'C'].
    pi : array-like, shape (N,)
        Initial state probabilities. Must sum to 1.
    A : array-like, shape (N, N)
        Transition matrix. A[i, j] = P(state j | state i).
        Rows do NOT have to sum to 1 if end_prob is provided —
        the remaining probability mass goes to END.
    B : array-like, shape (N, K)
        Emission matrix. B[i, k] = P(symbol k | state i).
        Each row must sum to 1.
    end_prob : array-like, shape (N,), optional
        Probability of ending the sequence from each state.
        end_prob[i] = P(END | state i).
        If None, termination is not modelled (uniform end, classic HMM).
    """

    def __init__(
        self,
        states:   list[str],
        symbols:  list[str],
        pi:       "array-like",         # NUMPY DOCUMENTATION STYLE LEGOOOOO
        A:        "array-like",
        B:        "array-like",
        end_prob: "array-like | None" = None,
    ):
        self.states  = list(states)
        self.symbols = list(symbols)
        self.N = len(states)
        self.K = len(symbols)

        self._sym_index = {s: i for i, s in enumerate(symbols)}

        self._log_pi = _log(pi)
        self._log_A  = _log(A)
        self._log_B  = _log(B)

        # end_prob: если не задан — log(1) = 0 для всех состояний
        # (умножение на 1 ничего не меняет — классический HMM)
        if end_prob is None:
            self._log_end = np.zeros(self.N)   # log(1) = 0
        else:
            self._log_end = _log(end_prob)

        self._validate()

    # проверка на длину и на корректность алфавита

    def _validate(self):
        assert self._log_pi.shape  == (self.N,),        "pi must have length N"
        assert self._log_A.shape   == (self.N, self.N), "A must be (N, N)"
        assert self._log_B.shape   == (self.N, self.K), "B must be (N, K)"
        assert self._log_end.shape == (self.N,),        "end_prob must have length N"

    def _obs_indices(self, sequence: Sequence[str]) -> np.ndarray:
        try:
            return np.array([self._sym_index[s] for s in sequence], dtype=int)
        except KeyError as e:
            raise ValueError(f"Symbol {e} not in alphabet {self.symbols}") from e

    # ── log-sum-exp ───────────────────────────────────────────────────────────

    @staticmethod
    def _logsumexp(log_values: np.ndarray) -> float:
        m = np.max(log_values)
        if m == LOG_ZERO:
            return LOG_ZERO
        return m + np.log(np.sum(np.exp(log_values - m)))

    # =========================================================================
    # 1. VITERBI
    # выбираем максимум и при этом запоминаем что выбирали. Потом восстанавливаем путь
    # =========================================================================

    def viterbi(self, sequence: Sequence[str]) -> ViterbiResult:
        """
        Most probable hidden-state path for *sequence*.

        На последнем шаге умножаем на end_prob[i], чтобы учесть
        вероятность того, что последовательность заканчивается именно здесь.
        """
        obs = self._obs_indices(sequence)
        T   = len(obs)

        delta = np.full((T, self.N), LOG_ZERO)
        psi   = np.zeros((T, self.N), dtype=int)

        # initialisation
        delta[0] = self._log_pi + self._log_B[:, obs[0]]

        # ДИНАМИЧЕСКОЕ ПРОГРАММИРОВАНИЕ заполняем матрицу
        for t in range(1, T):
            for j in range(self.N):
                scores    = delta[t - 1] + self._log_A[:, j]
                best_i    = int(np.argmax(scores))
                psi[t, j] = best_i
                delta[t, j] = scores[best_i] + self._log_B[j, obs[t]]

        # учитываем вероятность конца на последнем шаге
        delta_end = delta[-1] + self._log_end

        # back-tracking
        path_idx     = np.zeros(T, dtype=int)
        path_idx[-1] = int(np.argmax(delta_end))
        for t in range(T - 2, -1, -1):
            path_idx[t] = psi[t + 1, path_idx[t + 1]]

        path     = [self.states[i] for i in path_idx]
        log_prob = float(delta_end[path_idx[-1]])
        return ViterbiResult(path=path, log_prob=log_prob)

    # =========================================================================
    # 2. FORWARD.
    # тут суммируем и ниче восстанавливать не надо, те халява
    # =========================================================================

    def forward(self, sequence: Sequence[str]) -> ForwardResult:
        """
        Полная вероятность последовательности через суммирование по всем путям.

        На последнем шаге суммируем с учётом end_prob.
        """
        obs = self._obs_indices(sequence)
        T   = len(obs)

        log_alpha = np.full((T, self.N), LOG_ZERO)

        # initialisation
        log_alpha[0] = self._log_pi + self._log_B[:, obs[0]]

        # recursion
        for t in range(1, T):
            for j in range(self.N):
                log_alpha[t, j] = (
                    self._logsumexp(log_alpha[t - 1] + self._log_A[:, j])
                    + self._log_B[j, obs[t]]
                )

        # P(O | λ) = sum_i alpha_T(i) * end_prob(i)
        log_prob = self._logsumexp(log_alpha[-1] + self._log_end)
        return ForwardResult(alpha=log_alpha, log_prob=float(log_prob))

    
    # =========================================================================
    # 3. BACKWARD
    # суммируем в обратную сторону
    # =========================================================================

    def backward(self, sequence: Sequence[str]) -> BackwardResult:
        """
        Полная вероятность последовательности через обратный проход.

        Инициализация последнего шага: β_T(i) = end_prob(i)
        (вместо 1), потому что после последнего символа модель
        должна перейти в END.
        """
        obs = self._obs_indices(sequence)
        T   = len(obs)

        log_beta = np.full((T, self.N), LOG_ZERO)

        # initialisation: β_T(i) = end_prob(i)
        log_beta[-1] = self._log_end

        # recursion
        for t in range(T - 2, -1, -1):
            for i in range(self.N):
                log_beta[t, i] = self._logsumexp(
                    self._log_A[i, :]
                    + self._log_B[:, obs[t + 1]]
                    + log_beta[t + 1]
                )

        log_prob = self._logsumexp(
            self._log_pi + self._log_B[:, obs[0]] + log_beta[0]
        )
        return BackwardResult(beta=log_beta, log_prob=float(log_prob))

    
    # =========================================================================
    # 4. POSTERIOR DECODING
    # тут просто произведение считаем
    # =========================================================================

    def posterior_decoding(self, sequence: Sequence[str]) -> PosteriorResult:
        """
        γ_t(i) = P(state_t = i | O, λ) для каждой позиции.

        γ_t(i) = α_t(i) · β_t(i) / P(O | λ)
        """
        fwd = self.forward(sequence)
        bwd = self.backward(sequence)

        log_prob  = fwd.log_prob
        log_gamma = fwd.alpha + bwd.beta - log_prob
        gamma     = np.exp(log_gamma)

        # численная нормировка
        row_sums = gamma.sum(axis=1, keepdims=True)
        gamma   /= np.where(row_sums == 0, 1, row_sums)

        best_states = [self.states[int(np.argmax(gamma[t]))] for t in range(len(gamma))]

        return PosteriorResult(
            gamma=gamma,
            states_at_each_step=best_states,
            log_prob=float(log_prob),
        )


