# TDE2 – Minimax vs Alfa-Beta em Jogo da Velha 5x5
---
## Alunos: 
  
  - Ricardo Vinicius Moreira Vianna
  - Ricardo Ryu Makino Magalhães
  - Crystofer Samuel Demetino Dos Santos
  ---

Este repositório contém a implementação e análise comparativa entre os algoritmos **Minimax** e **Minimax com Poda Alfa-Beta**, aplicados a um tabuleiro 5x5 (objetivo: alinhar 4 peças).

O projeto gera automaticamente:
- Resultados de partidas em formato **CSV**
- Métricas agregadas (nós visitados e tempo total)
- Gráficos em **PNG**
- Relatório em **Markdown** pronto para edição/exportação em PDF

---

## Estrutura do Projeto

```
.
├── maina.py   # Script principal
├── resultados/                 # Pasta padrão de saída
│   ├── resultados_torneio.csv
│   ├── agregados.csv
│   ├── nodes_total.png
│   ├── time_total.png
│   └── relatorio_TDE2_minimax_alphabeta.md
```

---

## Pré-requisitos

- **Python 3.8+**
- Dependências listadas abaixo (instaladas via `pip`):
  - `pandas`
  - `matplotlib`

---

## Instalação

Clone o repositório e instale as dependências:

```bash
git clone https://github.com/ricaprof/MinimaxVSalphabetaO.git
cd MinimaxVSalphabeta

# (opcional) criar ambiente virtual
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -U pip
pip install pandas matplotlib
```

---

## Uso

Execute o script principal para rodar o torneio entre os agentes e gerar os relatórios.

### Exemplo (configuração padrão)
```bash
python main.py
```
- Roda **6 jogos**
- Profundidade de busca **3**
- Resultados em `./resultados`

### Personalizando
```bash
# 12 jogos, profundidade 4, saída em ./meus_resultados
python tde2_minimax_alphabeta.py --games 12 --depth 4 --outdir meus_resultados
```

Parâmetros disponíveis:
- `--games` → número de jogos (padrão: 6)
- `--depth` → profundidade da busca (padrão: 3)
- `--seed`  → semente de aleatoriedade (padrão: 7)
- `--outdir` → pasta de saída (padrão: `resultados`)

---

## Saída Gerada

Após a execução, você terá:
- `resultados_torneio.csv` → dados de cada jogo (nós visitados, tempo por agente, vencedor etc.)
- `agregados.csv` → métricas agregadas (totais de nós e tempo por agente)
- `nodes_total.png` → gráfico comparando nós visitados
- `time_total.png` → gráfico comparando tempo total (ms)
- `relatorio_TDE2_minimax_alphabeta.md` → relatório estruturado em Markdown

---

## Relatório

O relatório gerado contém:
1. Introdução
2. Explicação dos algoritmos (Minimax e Alfa-Beta)
3. Heurística utilizada
4. Metodologia experimental
5. Resultados (com links para CSV e imagens)
6. Análise crítica
7. Limitações e próximos passos
8. Conclusão

Você pode abrir o arquivo `.md` em qualquer editor de texto ou Markdown e exportar para PDF.

---

## Licença

Este projeto é distribuído sob a licença MIT. Consulte o arquivo `LICENSE` para mais detalhes.
