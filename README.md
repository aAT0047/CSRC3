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
  ```bash
  pip install -r requirements.txt
