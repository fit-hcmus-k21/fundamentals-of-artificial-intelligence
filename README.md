# 🧩HavardX: CS50's Introduction to Artificial Intelligence With Python

## Overview:
```mermaid
graph LR
  
  A[CS50AI 2023] --> B[Search]
  A --> C[Knowledge]
  A --> D[Uncertainty]
  A --> E[Optimization]
  A --> F[Learning]
  A --> G[Neural Networks]
  A --> H[Language]
```

### Course 's content:

![image](https://github.com/fit-hcmus-k21/fundamentals-of-artificial-intelligence/assets/93416202/deb25a90-2cd2-4894-bd3b-947c8a238759)

___

## Search

```mermaid
graph TD
  A[Agent] -->|Interacts with| B[Environment]
  B -->|Has| C[State]
  A -->|Perceives| C
  A -->|Chooses| D[Actions]
  C -->|Results in| E[New State]
  D -->|Impacts| E
  C -->|Forms| F[State Space]
  F -->|Subset of| G[State Space]
  G -->|Satisfies| H[Goal Test]
  D -->|Incurs| I[Path Cost]
  
  J[Search Algorithm] -->|Uses| F
  J -->|Explores| G
  J -->|Applies| H
  J -->|Evaluates| I

  K[Search Algorithms]
  K -->|DFS| L[Depth-First Search]
  K -->|BFS| M[Breadth-First Search]
  K -->|Greedy Best First Search| N[Greedy Best-First Search]
  K -->|A* Search| O[A* Search]
  K -->|Adversarial Search| P[Adversarial Search]
  P -->|Uses| Q[Minimax]
  P -->|Optimized by| R[Alpha-Beta Pruning]
  P -->|Limited Depth| S[Depth-Limited Minimax]

```

- `Agent` represents an entity interacting with the environment.
- `Environment` is the space in which the agent interacts and induces changes.
- `State` is the current condition of the environment.
- `Actions` are the steps or behaviors that the agent can perform.
- `State Space` is the collection of all possible states.
- `Goal Test` is the criterion to check if a state achieves the objective.
- `Path Cost` is the cost associated with a sequence of actions.
- `Search Algorithm` refers to the algorithm used to explore the state space.

___

## Knowledge

```mermaid
graph LR
  A[Knowledge-Based Agents]
  B[Sentence]
  C[Propositional Logic]
  D[Logical Connectives]
  E[Model]
  F[Knowledge Base]
  G[Entailment]
  H[Inference]
  I[Inference Rules]
  J[Modus Ponens]
  K[And Elimination]
  L[Double Negation Elimination]
  M[Implication Elimination]
  N[Biconditional Elimination]
  O[De Morgan's Law]
  P[Distributive Property]
  Q[Resolution]
  R[First Order Logic]
  S[Universal Quantification]
  T[Existential Quantification]

  A -->|Leads to| B
  B -->|Is a part of| C
  C -->|Includes| D
  C -->|Involves| E
  C -->|Contains| F
  C -->|Implies| G
  B -->|Relates to| H
  H -->|Uses| I
  I -->|Includes| J
  I -->|Utilizes| K
  I -->|Applies| L
  I -->|Employs| M
  I -->|Derives| N
  I -->|Utilizes| O
  I -->|Applies| P
  H -->|Utilizes| Q
  B -->|Involves| R
  R -->|Utilizes| S
  R -->|Involves| T


```

___

## Uncertainty

```mermaid
graph TD
  A[Uncertainty]
  B[Probability]
  C[Possible Worlds]
  C -->|Described by| D[Axioms of Probability]
  C -->|Includes| E[Unconditional Probability]
  B -->|Involves| F[Conditional Probability]
  G[Random Variables]
  G -->|Related to| H[Independence]
  I[Bayes's Rule]
  J[Joint Probability]
  K[Probability Rules]
  K -->|Include| L[Negation]
  K -->|Include| M[Inclusion-Exclusion]
  K -->|Include| N[Marginalization]
  K -->|Include| O[Conditioning]
  P[Bayesian Networks]
  P -->|Utilize| Q[Inference]
  Q -->|Implemented by| R[Inference by Enumeration]
  S[Sampling]
  S -->|Involves| T[Likelihood Weighting]
  U[Markov Models]
  U -->|Based on| V[Markov Assumption]
  U -->|Describes| W[Markov Chain]
  X[Hidden Markov Models]
  X -->|Assumes| Y[Sensor Markov Assumption]

  A -->|Comprises| B
  B -->|Includes| C
  B -->|Involves| G
  B -->|Utilizes| I
  B -->|Describes| J
  B -->|Utilizes| K
  B -->|Utilizes| P
  B -->|Involves| U
  B -->|Involves| X
  G -->|Involves| H
  I -->|Involves| J
  I -->|Involves| G
  P -->|Involves| Q
  S -->|Utilizes| Q
  S -->|Utilizes| T
  U -->|Based on| V
  U -->|Describes| W
  X -->|Assumes| Y
```

___

## Optimization

```mermaid
graph LR
  A[Optimization]
  B[Local Search]
  C[Objective Function]
  D[Cost Function]
  E[Current State]
  F[Neighbor State]
  B -->|Involves| C
  B -->|Involves| D
  B -->|Utilizes| E
  B -->|Involves| F
  G[Hill Climbing]
  H[Local and Global Minima and Maxima]
  I[Hill Climbing Variants]
  I -->|Include| J[Steepest-Ascent]
  I -->|Include| K[Stochastic]
  I -->|Include| L[First Choice]
  I -->|Include| M[Random Restart]
  I -->|Include| N[Local Beam Search]
  O[Simulated Annealing]
  O -->|Applied to| P[Traveling Salesman Problem]
  Q[Linear Programming]
  R[Constraint Satisfaction]
  S[Hard Constraint]
  T[Soft Constraint]
  U[Unary Constraint]
  V[Binary Constraint]
  W[Node Consistency]
  X[Arc Consistency]
  Y[Backtracking Search]
  Y -->|Utilizes| Z[Inference]
  Y -->|Uses| AA[MRV]
  Y -->|Uses| AB[Degree Heuristic]
  Y -->|Involves| AC[Constraining Values]

  A -->|Includes| B
  A -->|Includes| G
  G -->|Describes| H
  G -->|Includes| I
  O -->|Applies to| P
  A -->|Involves| Q
  A -->|Involves| R
  R -->|Involves| S
  R -->|Involves| T
  R -->|Involves| U
  R -->|Involves| V
  R -->|Utilizes| W
  R -->|Utilizes| X
  R -->|Involves| Y
```

- Local Search:
  * Objective Function: A function describing the "quality" of a solution.
  * Cost Function: An expression evaluating the cost of a solution.
  * Current State: The current state of the system.
  * Neighbor State: States adjacent to the current state.

- Hill Climbing:
  * Local and Global Minima and Maxima: Local and global optimal points.
  * Hill Climbing Variants:
    + Steepest-Ascent
    + Stochastic
    + First Choice
    + Random Restart
    + Local Beam Search

- Simulated Annealing:
  * Traveling Salesman Problem: An optimization problem where a salesperson needs to find the shortest route through multiple cities.

- Linear Programming:
  * A optimization method using a linear objective function and linear constraints.

- Constraint Satisfaction:
  * Hard Constraint: A constraint that cannot be violated.
  * Soft Constraint: A constraint that can be violated with a cost.
  * Unary Constraint: A constraint on a single variable.
  * Binary Constraint: A constraint between two variables.

- Node Consistency:
A method to ensure that each variable satisfies all its constraints.

- Arc Consistency:
A method to ensure that every value assigned to each variable satisfies all constraints with other variables.

- Backtracking Search:
  * Inference: Using information from assigned variables to reduce search space.
  * MRV (Minimum Remaining Values): Choosing a variable with the fewest remaining values.
  * Degree Heuristic: Choosing a variable related to the most constraints.
  * Constraining Values: Prioritizing values with the fewest choices.
___

## Learning

```mermaid
graph LR
  A[Learning]
  B[Machine Learning]
  C[Supervised Learning]
  D[Nearest Neighbor Classification]
  E[Perceptron Learning]
  F[Perceptron Learning Rules]
  G[Hard Threshold]
  H[Soft Threshold]
  I[Support Vector Machine]
  J[Regression]
  K[Loss Functions]
  L[Overfitting]
  M[Regularization]
  N[Scikit-Learn]
  O[Reinforcement Learning]
  P[Markov Decision Process]
  Q[Q-Learning]
  R[Unsupervised Learning]
  S[Clustering]
  T[K-Means Clustering]

  A -->|Involves| B
  B -->|Includes| C
  C -->|Utilizes| D
  C -->|Involves| E
  E -->|Includes| F
  E -->|Describes| G
  E -->|Describes| H
  B -->|Involves| I
  B -->|Involves| J
  J -->|Includes| K
  J -->|Describes| L
  J -->|Involves| M
  B -->|Involves| N
  A -->|Involves| O
  O -->|Describes| P
  O -->|Involves| Q
  A -->|Involves| R
  R -->|Involves| S
  S -->|Includes| T
```

- Machine Learning:
A field of study in artificial intelligence where systems learn from data.

- Supervised Learning:
A machine learning method where a model is trained on a labeled dataset.

- Nearest Neighbor Classification:
A classification method based on labeling data points based on their nearest neighbors.

- Perceptron Learning:
  * Perceptron Learning Rules: Rules for updating weights in a perceptron.
  * Hard Threshold: A threshold used to determine the perceptron's output.
  * Soft Threshold: A variation of a soft threshold in a perception.

- Support Vector Machine:
A machine learning model using a separating line with maximum margin between classes.

- Regression:
A method for predicting continuous values.
  * Loss Functions: Functions measuring the difference between prediction and actual values.
  * Overfitting: The phenomenon of a model overly emphasizing noise in training data.
  * Regularization: Techniques to reduce overfitting, including holdout cross-validation, training set, test set, and k-hold cross-validation.

- Scikit-Learn:
An open-source library for machine learning in Python.

- Reinforcement Learning:
A type of machine learning where a system interacts with an environment and learns from experience.

- Markov Decision Process:
A decision model based on the Markov process.

- Q-Learning:
A reinforcement learning algorithm using a strategy of greedy decision making, exploration, and exploitation.

- Unsupervised Learning:
A machine learning method that does not require labeled training data.

- Clustering:
A method automatically classifying data points into groups.

- K-Means Clustering:
A popular clustering method assuming a certain number, k, of groups.

___

## Neural Networks

```mermaid
graph TD
  A[Activation Functions]
  B[Neural Network Structure]
  C[Gradient Descent]
  D[Multilayer Neural Networks]
  E[Backpropagation]
  F[Overfitting]
  G[Dropout]
  H[TensorFlow]
  I[Computer Vision]
  J[Image Convolution]
  K[Convolutional Neural Networks]
  L[Convolution]
  M[Pooling]
  N[Flattening]
  O[Recurrent Neural Networks]
  P[Feedforward Neural Networks]
  Q[Input]
  R[Network]
  S[Output]
  T[ReLU]
  U[Stochastic Gradient Descent]
  V[Mini-Batch Gradient Descent]

  A --> B
  C --> D
  D --> E
  D --> F
  F --> G
  F --> H
  C --> U
  C --> V
  D --> T
  D --> U
  D --> V
  I --> J
  K --> L
  K --> M
  K --> N
  O --> P
  P --> Q
  P --> R
  P --> S


```

___

## Language

```mermaid
graph TD
  A[Natural Language Processing]
  B[Syntax and Semantics]
  C[Syntax]
  D[Semantics]
  E[Context-Free Grammar]
  F[Formal Grammar]
  G[nltk]
  H[n-grams]
  I[Character n-gram]
  J[Word n-gram]
  K[Tokenization]
  L[Word Tokenization]
  M[Sentence Tokenization]
  N[Markov Models]
  O[Bag-of-Words Model]
  P[Naive Bayes]
  Q[Word Representation]
  R[word2vec]
  S[Neural Networks]
  T[Attention]
  U[Transformers]
  V[Position Encoding]
  W[Self-Attention]

  A --> B
  B --> C
  B --> D
  C --> E
  E --> F
  D --> G
  G --> H
  H --> I
  H --> J
  D --> K
  K --> L
  K --> M
  D --> N
  N --> O
  G --> P
  P --> Q
  Q --> R
  S --> T
  S --> U
  U --> V
  U --> W

```

___

