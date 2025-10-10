
import argparse
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
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

    # =========================
    # CSV com resultados por jogo
    # =========================
    csv_path = outdir / "resultados_torneio.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")

    # =========================
    # Agregados
    # =========================
    nodes_ab = df.filter(like="nodes_AlphaBeta").sum(axis=1).sum()
    nodes_mm = df.filter(like="nodes_Minimax").sum(axis=1).sum()
    time_ab = df.filter(like="time_ms_AlphaBeta").sum(axis=1).sum()
    time_mm = df.filter(like="time_ms_Minimax").sum(axis=1).sum()

    agg = pd.DataFrame({
        "Métrica": ["Nós visitados (total)", "Tempo total (ms)"],
        "AlphaBeta": [nodes_ab, time_ab],
        "Minimax": [nodes_mm, time_mm]
    })
    agg_csv_path = outdir / "agregados.csv"
    agg.to_csv(agg_csv_path, index=False, encoding="utf-8")

    # =========================
    # Gráficos
    # =========================
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

    # =========================
    # Relatório em Markdown (já com imagens inline)
    # =========================
    md = f"""# TDE 2 — Minimax vs Poda Alfa-Beta (Jogo da Velha 5x5, 4 em linha)

## 1. Introdução
Este relatório compara dois agentes de busca adversária: **Minimax** e **Minimax com Poda Alfa-Beta**, aplicados a um tabuleiro 5x5 cujo objetivo é alinhar 4 peças.

## 2. Algoritmos
- **Minimax**: expande a árvore de jogo alternando níveis MAX/MIN e usa utilidade para estados terminais.  
- **Poda Alfa-Beta**: mantém limites alfa/beta para evitar explorar ramos inúteis, reduzindo nós visitados sem alterar o resultado ótimo.

## 3. Heurística
- Contagem de janelas de tamanho 4 em todas as direções.  
- Pesos para sequências de 2 ou 3 em linha sem bloqueio.  
- Bônus para “quase vitória” (3 + 1 vazio).  

## 4. Metodologia
- Profundidade: d={depth}  
- Jogos simulados: {n_games} (alternando quem começa)  
- Métricas: nós visitados e tempo total.  

## 5. Resultados
### Tabela Resumida

### Gráficos
![Nós visitados]({nodes_png.name})  
![Tempo total]({time_png.name})  

## 6. Análise Crítica
A Poda Alfa-Beta visitou **menos nós** e consumiu **menos tempo**, sem perder qualidade de decisão.  
Isso confirma a eficiência da poda em domínios combinatórios. Entretanto:
- O ganho depende da ordem das jogadas exploradas.  
- Para profundidades muito grandes, ainda há explosão combinatória.  

## 7. Conclusão
- **Alfa-Beta** > **Minimax** em desempenho (nós e tempo).  
- Ambos mantêm a mesma qualidade de jogada.  
- Próximos passos: ordenação de movimentos, uso de *transposition tables* e heurísticas mais sofisticadas.  
"""
    md_path = outdir / "relatorio_TDE2_minimax_alphabeta.md"
    md_path.write_text(md, encoding="utf-8")

    # =========================
    # Relatório em PDF
    # =========================
    pdf_path = outdir / "relatorio_TDE2_minimax_alphabeta.pdf"
    doc = SimpleDocTemplate(str(pdf_path))
    styles = getSampleStyleSheet()
    story = []

    # Capa simples
    story.append(Paragraph("TDE 2 — Minimax vs Poda Alfa-Beta (Jogo da Velha 5x5, 4 em linha)", styles["Title"]))
    story.append(Spacer(1, 20))
    story.append(Paragraph(f"Profundidade: {depth} | Jogos: {n_games}", styles["Normal"]))
    story.append(Spacer(1, 40))

    # Seções resumidas
    story.append(Paragraph("Introdução", styles["Heading2"]))
    story.append(Paragraph("Comparação entre Minimax e Poda Alfa-Beta em jogo da velha 5x5 (4 em linha).", styles["Normal"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Algoritmos", styles["Heading2"]))

    story.append(Paragraph(
        "O algoritmo Minimax é uma técnica clássica de tomada de decisão em jogos de dois jogadores, "
        "particularmente em jogos de soma zero. Seu funcionamento baseia-se em simular todas as jogadas "
        "possíveis a partir do estado atual até alcançar estados terminais, avaliando-os por uma função de "
        "utilidade. O jogador MAX busca maximizar sua pontuação, enquanto o jogador MIN tenta minimizar, "
        "resultando em uma árvore de busca onde cada nível alterna entre os interesses dos dois jogadores. "
        "Apesar de garantir a escolha da jogada ótima, o Minimax puro é altamente custoso em termos de tempo "
        "de execução, pois necessita explorar todos os nós da árvore de decisão, tornando-se inviável para "
        "problemas de maior complexidade ou profundidade.", styles["Normal"])
    )

    story.append(Spacer(1, 12))

    story.append(Paragraph(
        "A Poda Alfa-Beta surge como uma otimização essencial ao algoritmo Minimax, reduzindo "
        "significativamente o número de nós explorados sem comprometer a qualidade da decisão final. "
        "Ela funciona descartando ramos da árvore que não podem alterar o resultado da escolha ótima, "
        "com base em limites denominados alfa (o valor mínimo garantido para MAX) e beta (o valor máximo "
        "garantido para MIN). Quando um ramo atinge valores que não podem superar esses limites, ele é "
        "interrompido imediatamente, economizando tempo e recursos computacionais. Na prática, essa técnica "
        "torna possível aplicar o raciocínio do Minimax em profundidades maiores, aumentando a eficiência da "
        "busca e permitindo que estratégias mais avançadas, como heurísticas de ordenação de jogadas, sejam "
        "implementadas para potencializar ainda mais os ganhos de desempenho.", styles["Normal"])
    )

    story.append(Spacer(1, 12))


    story.append(Paragraph("Resultados", styles["Heading2"]))
    # Tabela com agregados
    table_data = [agg.columns.tolist()] + agg.values.tolist()
    t = Table(table_data, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
        ("GRID", (0,0), (-1,-1), 0.5, colors.black),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold")
    ]))
    story.append(t)
    story.append(Spacer(1, 12))

    # Imagens
    story.append(Image(str(nodes_png), width=400, height=300))
    story.append(Spacer(1, 12))
    story.append(Image(str(time_png), width=400, height=300))
    story.append(Spacer(1, 20))

        # =========================
    # Análise Crítica
    # =========================
    story.append(Paragraph("Análise Crítica", styles["Heading2"]))
    story.append(Paragraph(
        "A comparação entre os dois algoritmos evidencia claramente que a poda Alfa-Beta não apenas "
        "reduz o número de nós visitados, mas também influencia diretamente o tempo de execução. "
        "Enquanto o Minimax puro explora todos os ramos possíveis até a profundidade estabelecida, "
        "o Alfa-Beta elimina antecipadamente regiões da árvore que não poderiam alterar a decisão final. "
        "Isso faz com que, na prática, o Alfa-Beta apresente desempenho significativamente superior em cenários "
        "com profundidade de busca intermediária, como o jogo da velha 5x5 com quatro em linha."
    , styles["Normal"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph(
        "Apesar do ganho em eficiência, é importante notar que a poda Alfa-Beta não altera a complexidade assintótica "
        "do problema, que continua exponencial em função da profundidade e do fator de ramificação. "
        "Assim, conforme a profundidade aumenta, o crescimento do espaço de busca ainda impõe um limite prático. "
        "Neste sentido, técnicas complementares como ordenação de jogadas (move ordering), armazenamento de estados repetidos "
        "em tabelas de transposição e funções de avaliação mais sofisticadas tornam-se indispensáveis "
        "para garantir a viabilidade computacional em jogos mais complexos."
    , styles["Normal"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph(
        "Outro aspecto crítico é a qualidade da heurística empregada. Uma heurística simples, baseada apenas em contagem de sequências, "
        "pode ser suficiente para capturar vantagens táticas básicas, mas não é capaz de compreender nuances estratégicas mais profundas. "
        "Isso significa que, mesmo com a poda, a qualidade das decisões pode ser limitada pela simplicidade da função de avaliação. "
        "Logo, em experimentos futuros, seria recomendável investigar heurísticas mais expressivas que incorporem noções de bloqueio, "
        "potencial de expansão e até aprendizado automático a partir de dados de partidas simuladas."
    , styles["Normal"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph(
        "Finalmente, deve-se considerar que os experimentos aqui relatados foram conduzidos em condições controladas, "
        "com número limitado de jogos e profundidade fixa. Em ambientes competitivos ou de maior escala, "
        "a robustez do algoritmo frente a diferentes estilos de oponente e condições de tempo real seria um aspecto relevante "
        "para análise. Assim, embora os resultados confirmem as vantagens conhecidas da poda Alfa-Beta, "
        "eles também apontam para a necessidade de um ecossistema mais completo de técnicas para explorar todo o potencial da busca adversária."
    , styles["Normal"]))

    # =========================
    # Conclusão
    # =========================
    story.append(Spacer(1, 20))
    story.append(Paragraph("Conclusão", styles["Heading2"]))
    story.append(Paragraph(
        "No contexto do jogo da velha 5x5 com condição de vitória em quatro peças, a implementação da poda Alfa-Beta "
        "demonstrou-se claramente superior ao Minimax puro em termos de eficiência. "
        "A redução drástica no número de nós visitados e a consequente diminuição do tempo de execução confirmam "
        "o papel essencial da poda em problemas de busca combinatória."
    , styles["Normal"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph(
        "Entretanto, a análise mais aprofundada revela que tais ganhos, embora expressivos, não são suficientes por si só "
        "para lidar com a explosão combinatória em jogos mais complexos. "
        "A integração com técnicas de otimização adicionais, aliada ao desenvolvimento de heurísticas mais inteligentes, "
        "aponta como caminho natural para trabalhos futuros."
    , styles["Normal"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph(
        "Portanto, este estudo não apenas reforça o valor da poda Alfa-Beta como técnica fundamental, "
        "mas também ressalta que a construção de agentes de jogo realmente competitivos exige uma combinação de abordagens "
        "– desde algoritmos de busca até funções de avaliação e estratégias de pré-processamento. "
        "Assim, este trabalho pode ser visto como uma base sólida para experimentos posteriores mais sofisticados "
        "em jogos de maior complexidade."
    , styles["Normal"]))
    story.append(Spacer(1, 12))
    

    doc.build(story)

    return {
        "csv_jogos": csv_path,
        "csv_agregados": agg_csv_path,
        "png_nodes": nodes_png,
        "png_tempo": time_png,
        "md": md_path,
        "pdf": pdf_path
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
