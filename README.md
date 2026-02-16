# AlphaZero implementation for single-player, turn-based Puyo Puyo
This package contains a custom implementation of AlphaZero that learns to play a single-player, turn-based version of
Puyo Puyo.

## Installation
Create and/or activate a virtual environment, then clone the package using
```
git clone https://github.com/rcapillon/simone2026.git
```
Then install the package using
```
cd simone_puyo/
pip install .
```

## Usage
For now, look in the scripts folder for usage examples.

## Limitations
- The MCTS tree is not reused after each move which hinders search performance. This will be fixed soon.