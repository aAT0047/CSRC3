# CSRC3
# Chain Surgical Risk Calculator (CSRC3)

Traditional surgical risk calculators typically focus solely on mortality as the final predictive outcome, neglecting the interrelationships between other adverse events, which limits their effectiveness in practical applications. In response, this project proposes the **Chain Surgical Risk Calculator (CSRC3)**, a tool based on multi-label ensemble learning theory that focuses on analyzing the unidirectional chain relationships between adverse event labels. CSRC3 performs particularly well in scenarios of data imbalance.

## Features

- **Multi-label Ensemble Learning**: Utilizes the multi-path AdaCost framework to handle multiple labels and improve predictive accuracy.
- **Chain Relationship Modeling**: Analyzes unidirectional chain relationships between adverse events to address label dependencies.
- **Handling Data Imbalance**: Effectively deals with data imbalance, which is common in healthcare datasets.
- **Validation**: The model was tested on data from the **Surgical Outcome Monitoring and Improvement Program (SOMIP)** in Hong Kong, demonstrating its utility in real-world applications.

## Getting Started

### Prerequisites

- Python 3.9
- Required libraries (install via `requirements.txt`):
# Chain Surgical Risk Calculator (CSRC3)

Traditional surgical risk calculators typically focus solely on mortality as the final predictive outcome, neglecting the interrelationships between other adverse events, which limits their effectiveness in practical applications. In response, this project proposes the **Chain Surgical Risk Calculator (CSRC3)**, a tool based on multi-label ensemble learning theory that focuses on analyzing the unidirectional chain relationships between adverse event labels. CSRC3 performs particularly well in scenarios of data imbalance.

## Features

- **Multi-label Ensemble Learning**: Utilizes the multi-path AdaCost framework to handle multiple labels and improve predictive accuracy.
- **Chain Relationship Modeling**: Analyzes unidirectional chain relationships between adverse events to address label dependencies.
- **Handling Data Imbalance**: Effectively deals with data imbalance, which is common in healthcare datasets.
- **Validation**: The model was tested on data from the **Surgical Outcome Monitoring and Improvement Program (SOMIP)** in Hong Kong, demonstrating its utility in real-world applications.

## Algorithm: Chain Surgical Risk Calculator (CSRC3)

Below is the pseudocode for the **CSRC3** algorithm:

```pseudo
\begin{algorithm}[H]
\SetAlgoLined
\KwResult{a multi-label classifier \(h_{\sim q}\)}
 \textbf{Input:}
 \begin{itemize}
     \item A set \( S = \{s_1, s_2, \ldots, s_n\} \)
     \item A set \( Y = \{y_1, y_2, \ldots, y_q\} \)
 \end{itemize}
 Order classifier chain: \( \min \limits_{i=1}^q \frac{1}{(y_i)^2} \)

 Initialize distribution \(D_{1}(j){=}z_{i}/\sum_{j}^{m}z_{j}\)

 \For{\( t = 1, \ldots, T \)}{
  Train weak learner using distribution \( D_t \), weak learner = order CC\\
  Compute weak hypothesis \( h_{tq} \)\\
  Choose \( \alpha_{tq} = \frac{1}{2} \ln \frac{1+\varepsilon_{tq}}{1-\varepsilon_{tq}} \) and \( \beta = \beta( \text{sign} (y_i h_{tq}(x_i)), c_i ) \)\\
  Update: \( D_{t+1,q} = \beta ( \text{sign} (y_i h_{tq}(x_i)), c_i ) e^\frac{-\alpha_{tq} \text{sign}(y_i h_{tq }(x_i)) c_i}{Z_t}D_{t,q} \)\\
  cost adjustment function \( \beta ( \text{sign} (y_i h_{tq}(x_i)), c_i ) \) in the range of [0, +1], \( D_{t+1,q} \) will be a distribution\\

  \If{Judges the single-label hypothesis \( h_{tq} \) in CC if end of iterations, \( \varepsilon_{tq} \leq 0.05 \)}{
   Output a multi-label classifier\\
   \( h_{\sim q} = \sum \limits_{t=1}^T \beta ( \text{sign} (y_i h_{tq}(x_i)), c_i ) \alpha_{tq} \)\\
  }
 }
\caption{CSRC3}
\end{algorithm}

\documentclass{article}
\usepackage[linesnumbered,ruled,vlined]{algorithm2e}

\begin{document}

\begin{algorithm}[H]
\caption{DP\_order}
\SetAlgoLined
\KwIn{
  A set of labels \( \Lambda = \{q_1, q_2, \ldots, q_5\} \),\\
  A vector \( y = [y_1, y_2, \ldots, y_5] \)
}
\KwOut{Optimal value \( Q^*_{\text{opt}} \)}

\textbf{Initialization:}\\
\For{\(i = 1\) \KwTo \(5\)}{
  Set \( V(i, 1) = \frac{1}{(y_i)^2} \)
}

\textbf{Recursive Computation:}\\
\For{\(n = 2\) \KwTo \(5\)}{
  \For{\(i = 1\) \KwTo \(5\)}{
    Set \( V(i, n) = \infty \)\\
    \For{\(j = 1\) \KwTo \(5\)}{
      \If{\( j \neq i \) \textbf{and} \( q_i \notin M^{n-1} \)}{
        Compute cost \( = \frac{1}{(y_i)^{n+1}} + V(j, n - 1) \)\\
        \If{\( cost < V(i, n) \)}{
          Set \( V(i, n) = cost \)\\
          Update \( M^n \)
        }
      }
    }
  }
}

\textbf{Solution Extraction:}\\
Set \( Q^*_{\text{opt}} = \min\limits_{i} V(i, 5) \)

\end{algorithm}

\end{document}

