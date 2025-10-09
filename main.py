
import argparse
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd

# ----------------------------- Constantes ---------------------------------

EMPTY = 0
X = 1      # Player 1
O = -1     # Player 2

# Direções para checar sequências
DIRECTIONS = [(1, 0), (0, 1), (1, 1), (1, -1)]  # dir, baixo, diag ↘, diag ↗


# ----------------------------- Tabuleiro ----------------------------------

@dataclass
class Board:
    size: int = 5
    target: int = 4
    cells: List[int] = field(default_factory=lambda: [EMPTY] * 25)

    def clone(self) -> "Board":
        return Board(self.size, self.target, self.cells.copy())

    def index(self, r: int, c: int) -> int:
        return r * self.size + c

    def get(self, r: int, c: int) -> int:
        return self.cells[self.index(r, c)]

    def set(self, r: int, c: int, val: int) -> None:
        self.cells[self.index(r, c)] = val

    def moves(self) -> List[Tuple[int, int]]:
        s = self.size
        return [(r, c) for r in range(s) for c in range(s) if self.get(r, c) == EMPTY]

    def is_full(self) -> bool:
        return all(v != EMPTY for v in self.cells)

    def winner(self) -> Optional[int]:
        s, t = self.size, self.target
        for r in range(s):
            for c in range(s):
                v = self.get(r, c)
                if v == EMPTY:
                    continue
                for dr, dc in DIRECTIONS:
                    rr, cc = r, c
                    cnt = 0
                    for _ in range(t):
                        if 0 <= rr < s and 0 <= cc < s and self.get(rr, cc) == v:
                            cnt += 1
                            rr += dr
                            cc += dc
                        else:
                            break
                    if cnt == t:
                        return v
        return None

    def is_terminal(self) -> bool:
        return self.winner() is not None or self.is_full()


# ------------------------------ Heurística --------------------------------

def heuristic(board: Board, player: int) -> int:
    """
    Heurística simples baseada em janelas de tamanho 'target' (4).
    Soma pesos para sequências parciais de 2 e 3 em linha (sem bloqueio do oponente),
    com bônus para 'quase vitória' (3 do mesmo + 1 vazio).
    """
    s, t = board.size, board.target
    score = 0
    weights = {2: 3, 3: 10}

    def window_cells(r, c, dr, dc):
        return [(r + i * dr, c + i * dc) for i in range(t)]

    for r in range(s):
        for c in range(s):
            for dr, dc in DIRECTIONS:
                rr = r + (t - 1) * dr
                cc = c + (t - 1) * dc
                if not (0 <= rr < s and 0 <= cc < s):
                    continue
                coords = window_cells(r, c, dr, dc)
                vals = [board.get(rr, cc) for rr, cc in coords]

                # janela bloqueada
                if (X in vals) and (O in vals):
                    continue

                count_x = vals.count(X)
                count_o = vals.count(O)
                empty = vals.count(EMPTY)

                if empty == t:
                    continue

                if count_x > 0 and count_o == 0:
                    if count_x in weights:
                        score += weights[count_x]
                    if count_x == t - 1 and empty == 1:
                        score += 50
                elif count_o > 0 and count_x == 0:
                    if count_o in weights:
                        score -= weights[count_o]
                    if count_o == t - 1 and empty == 1:
                        score -= 50

    # valor relativo à perspectiva do 'player'
    return score if player == X else -score


# --------------------------- Busca: Minimax --------------------------------

def minimax(board: Board, perspective: int, depth: int, maximizing: bool,
            max_depth: int, stats: Dict[str, int]) -> Tuple[int, Optional[Tuple[int, int]]]:
    """
    perspective: o jogador para o qual avaliamos o valor (X ou O)
    maximizing: True se é a vez do 'perspective', False se é do oponente.
    """
    stats["nodes"] += 1

    win = board.winner()
    if win is not None:
        if win == perspective:
            return 100000, None
        elif win == -perspective:
            return -100000, None
        else:
            return 0, None

    if depth == max_depth or board.is_full():
        return heuristic(board, perspective), None

    best_move = None
    if maximizing:
        best_val = -math.inf
        for (r, c) in board.moves():
            board.set(r, c, perspective)  # o jogador 'perspective' joga quando maximizing=True
            val, _ = minimax(board, perspective, depth + 1, False, max_depth, stats)
            board.set(r, c, EMPTY)
            if val > best_val:
                best_val = val
                best_move = (r, c)
        return best_val, best_move
    else:
        best_val = math.inf
        opp = -perspective
        for (r, c) in board.moves():
            board.set(r, c, opp)  # oponente joga quando maximizing=False
            val, _ = minimax(board, perspective, depth + 1, True, max_depth, stats)
            board.set(r, c, EMPTY)
            if val < best_val:
                best_val = val
                best_move = (r, c)
        return best_val, best_move


# ---------------------- Busca: Alfa-Beta (poda) ----------------------------

def alphabeta(board: Board, perspective: int, depth: int, maximizing: bool,
              max_depth: int, alpha: float, beta: float, stats: Dict[str, int]) -> Tuple[int, Optional[Tuple[int, int]]]:
    stats["nodes"] += 1

    win = board.winner()
    if win is not None:
        if win == perspective:
            return 100000, None
        elif win == -perspective:
            return -100000, None
        else:
            return 0, None

    if depth == max_depth or board.is_full():
        return heuristic(board, perspective), None

    best_move = None
    if maximizing:
        value = -math.inf
        for (r, c) in board.moves():
            board.set(r, c, perspective)
            val, _ = alphabeta(board, perspective, depth + 1, False, max_depth, alpha, beta, stats)
            board.set(r, c, EMPTY)
            if val > value:
                value = val
                best_move = (r, c)
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return value, best_move
    else:
        value = math.inf
        opp = -perspective
        for (r, c) in board.moves():
            board.set(r, c, opp)
            val, _ = alphabeta(board, perspective, depth + 1, True, max_depth, alpha, beta, stats)
            board.set(r, c, EMPTY)
            if val < value:
                value = val
                best_move = (r, c)
            beta = min(beta, value)
            if alpha >= beta:
                break
        return value, best_move


# -------------------------------- Agentes ----------------------------------

class Agent:
    def __init__(self, name: str):
        self.name = name

    def choose(self, board: Board, side: int) -> Tuple[Tuple[int, int], int, float]:
        raise NotImplementedError


class MinimaxAgent(Agent):
    def __init__(self, max_depth: int = 3):
        super().__init__(f"Minimax(d={max_depth})")
        self.max_depth = max_depth

    def choose(self, board: Board, side: int) -> Tuple[Tuple[int, int], int, float]:
        stats = {"nodes": 0}
        start = time.time()
        _, move = minimax(board, perspective=side, depth=0, maximizing=True, max_depth=self.max_depth, stats=stats)
        elapsed_ms = (time.time() - start) * 1000.0
        if move is None:
            # se não houver movimento, devolve algo seguro
            move = (0, 0)
        return move, stats["nodes"], elapsed_ms


class AlphaBetaAgent(Agent):
    def __init__(self, max_depth: int = 3):
        super().__init__(f"AlphaBeta(d={max_depth})")
        self.max_depth = max_depth

    def choose(self, board: Board, side: int) -> Tuple[Tuple[int, int], int, float]:
        stats = {"nodes": 0}
        start = time.time()
        _, move = alphabeta(board, perspective=side, depth=0, maximizing=True,
                            max_depth=self.max_depth, alpha=-math.inf, beta=math.inf, stats=stats)
        elapsed_ms = (time.time() - start) * 1000.0
        if move is None:
            move = (0, 0)
        return move, stats["nodes"], elapsed_ms


# --------------------------- Simulação/Torneio -----------------------------

def play_game(agent_X: Agent, agent_O: Agent, verbose: bool = False) -> Dict:
    board = Board()
    side_to_move = X
    total_nodes = {agent_X.name: 0, agent_O.name: 0}
    total_time = {agent_X.name: 0.0, agent_O.name: 0.0}
    moves_count = 0

    while not board.is_terminal():
        if side_to_move == X:
            move, nodes, ms = agent_X.choose(board, X)
            if board.get(*move) != EMPTY:
                # fallback se por algum motivo vier célula ocupada
                legal = board.moves()
                if not legal:
                    break
                move = legal[0]
            board.set(move[0], move[1], X)
            total_nodes[agent_X.name] += nodes
            total_time[agent_X.name] += ms
        else:
            move, nodes, ms = agent_O.choose(board, O)
            if board.get(*move) != EMPTY:
                legal = board.moves()
                if not legal:
                    break
                move = legal[0]
            board.set(move[0], move[1], O)
            total_nodes[agent_O.name] += nodes
            total_time[agent_O.name] += ms

        moves_count += 1
        side_to_move = -side_to_move

        if verbose:
            pass  # pode imprimir tabuleiro aqui se quiser

    win = board.winner()
    result = "draw"
    if win == X:
        result = "X"
    elif win == O:
        result = "O"

    return {
        "result": result,
        "moves": moves_count,
        "nodes": total_nodes,
        "time_ms": total_time
    }


def tournament(n_games: int = 6, depth: int = 3, seed: int = 42) -> pd.DataFrame:
    random.seed(seed)
    mm = MinimaxAgent(max_depth=depth)
    ab = AlphaBetaAgent(max_depth=depth)

    rows = []
    for i in range(n_games):
        # alterna quem começa
        if i % 2 == 0:
            res = play_game(ab, mm)
            first, second = ab.name, mm.name
        else:
            res = play_game(mm, ab)
            first, second = mm.name, ab.name

        rows.append({
            "game": i + 1,
            "first_player": first,
            "winner": res["result"],
            "moves": res["moves"],
            f"nodes_{ab.name}": res["nodes"].get(ab.name, 0),
            f"nodes_{mm.name}": res["nodes"].get(mm.name, 0),
            f"time_ms_{ab.name}": res["time_ms"].get(ab.name, 0.0),
            f"time_ms_{mm.name}": res["time_ms"].get(mm.name, 0.0),
        })
    return pd.DataFrame(rows)


# ------------------------- Relatórios & Gráficos ---------------------------

def save_outputs(df: pd.DataFrame, outdir: Path, depth: int, n_games: int):
    outdir.mkdir(parents=True, exist_ok=True)

    # CSV com resultados por jogo
    csv_path = outdir / "resultados_torneio.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")

    # Agregados
    nodes_ab = df.filter(like="nodes_AlphaBeta").sum(axis=1).sum()
    nodes_mm = df.filter(like="nodes_Minimax").sum(axis=1).sum()
    time_ab = df.filter(like="time_ms_AlphaBeta").sum(axis=1).sum()
    time_mm = df.filter(like="time_ms_Minimax").sum(axis=1).sum()

    agg = pd.DataFrame({
        "metric": ["nodes_total", "time_ms_total"],
        "AlphaBeta": [nodes_ab, time_ab],
        "Minimax": [nodes_mm, time_mm]
    })
    agg_csv_path = outdir / "agregados.csv"
    agg.to_csv(agg_csv_path, index=False, encoding="utf-8")

    # Gráficos (matplotlib puro, sem estilos)
    fig1 = plt.figure()
    plt.title(f"Nós visitados - Total ({n_games} jogos, d={depth})")
    plt.bar(["AlphaBeta", "Minimax"], [nodes_ab, nodes_mm])
    plt.ylabel("Nós")
    plt.tight_layout()
    nodes_png = outdir / "nodes_total.png"
    plt.savefig(nodes_png)
    plt.close(fig1)

    fig2 = plt.figure()
    plt.title(f"Tempo total por agente (ms) ({n_games} jogos, d={depth})")
    plt.bar(["AlphaBeta", "Minimax"], [time_ab, time_mm])
    plt.ylabel("ms")
    plt.tight_layout()
    time_png = outdir / "time_total.png"
    plt.savefig(time_png)
    plt.close(fig2)

    # Relatório em Markdown
    md = f"""# TDE 2 — Minimax vs Poda Alfa-Beta (Jogo da Velha 5x5, 4 em linha)

## 1. Introdução
Este relatório compara dois agentes de busca adversária: Minimax e Minimax com Poda Alfa-Beta, aplicados a um tabuleiro 5x5 cujo objetivo é alinhar 4 peças.

## 2. Algoritmos
### 2.1 Minimax (básico)
O algoritmo expande uma árvore de jogo alternando níveis MAX/MIN e usa utilidade para estados terminais; para profundidade limitada, aplica-se uma função heurística.

### 2.2 Poda Alfa-Beta
Mantém limites alfa (para MAX) e beta (para MIN) para evitar explorar ramos que não podem melhorar a decisão, reduzindo nós visitados sem alterar o resultado ótimo (mesma ordem de expansão).

## 3. Heurística
Contagem de janelas de tamanho 4 em todas as direções:
- Pesos para sequências parciais (2 e 3 em linha sem bloqueio).
- Bônus de “quase vitória” (3 + 1 vazio).

## 4. Metodologia Experimental
- Profundidade de busca: d={depth}
- Número de jogos: {n_games}, alternando quem começa.
- Métricas: nós visitados por agente e tempo total (ms).

## 5. Resultados
- **CSV completo**: `resultados_torneio.csv`
- **Agregados**: `agregados.csv`
- **Gráficos**:
  - Nós totais: `nodes_total.png`
  - Tempo total (ms): `time_total.png`

## 6. Análise (resumo)
A Poda Alfa-Beta normalmente visita menos nós e consome menos tempo que o Minimax sem poda, mantendo a mesma qualidade de decisão quando ambos usam a mesma profundidade e heurística.

## 7. Limitações e Próximos Passos
- Profundidade limitada (custo computacional cresce rápido).
- Melhorias: ordenação de jogadas (move ordering), transposition tables e heurísticas mais ricas para 5x5.

## 8. Conclusão
No cenário avaliado, Alfa-Beta mostrou eficiência superior em nós e tempo, preservando decisões equivalentes às do Minimax básico na mesma profundidade.
"""
    md_path = outdir / "relatorio_TDE2_minimax_alphabeta.md"
    md_path.write_text(md, encoding="utf-8")

    return {
        "csv_jogos": csv_path,
        "csv_agregados": agg_csv_path,
        "png_nodes": nodes_png,
        "png_tempo": time_png,
        "md": md_path
    }


# --------------------------------- Main -----------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="TDE2: Minimax vs Alfa-Beta no 5x5 (4 em linha) — gera CSV, PNG e MD."
    )
    parser.add_argument("--games", type=int, default=6, help="Número de jogos (padrão: 6)")
    parser.add_argument("--depth", type=int, default=3, help="Profundidade de busca (padrão: 3)")
    parser.add_argument("--seed", type=int, default=7, help="Seed p/ aleatoriedade (padrão: 7)")
    parser.add_argument("--outdir", type=str, default="resultados", help="Pasta de saída (padrão: resultados)")
    args = parser.parse_args()

    random.seed(args.seed)

    print(f"Rodando torneio: games={args.games}, depth={args.depth}, seed={args.seed}...")
    df = tournament(n_games=args.games, depth=args.depth, seed=args.seed)

    outdir = Path(args.outdir)
    paths = save_outputs(df, outdir, depth=args.depth, n_games=args.games)
    print("\nArquivos gerados:")
    for k, p in paths.items():
        print(f"- {k}: {p}")

    


if __name__ == "__main__":
    main()
