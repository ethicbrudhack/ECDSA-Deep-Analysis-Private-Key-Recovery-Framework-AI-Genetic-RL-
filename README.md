# 🧩 ECDSA Deep Analysis & Private Key Recovery Framework (AI + Genetic + RL)

This project implements an **AI-driven cryptanalysis framework for ECDSA**, combining:
- 🧬 **Genetic Algorithms**
- 🧠 **Machine Learning (XGBoost, LSTM, TensorFlow)**
- 🔥 **Simulated Annealing Optimization**
- 🤖 **Reinforcement Learning Environment (OpenAI Gym)**
- 📈 **Linear and Reuse Nonce Detection**
- 🪄 **Hybrid AI-supervised attacks**

It is designed to **research vulnerabilities in ECDSA nonce usage**, **simulate key recovery**, and **analyze cryptographic patterns** in Bitcoin-like signatures.

> ⚠️ **Educational and cryptographic research use only.**  
> This framework is meant for testing, auditing, and learning — **not for unauthorized key recovery**.

---

## ⚙️ Features

✅ Parses real ECDSA signatures from transaction data  
✅ Detects reused or linearly related nonces (`k`)  
✅ Reconstructs candidate private keys (`d`)  
✅ AI-assisted search for optimal `k` values  
✅ Generates all address formats from recovered keys  
✅ Integrates:
   - XGBoost regression & classification
   - TensorFlow LSTM networks
   - Genetic optimization (DEAP)
   - Simulated annealing refinement
   - Reinforcement learning exploration  
✅ Parallelized computation (multiprocessing)

---

## 🧠 Technical Overview

ECDSA signature equation:
\[
s = k^{-1}(z + d \cdot r) \pmod{n}
\]

If multiple signatures share related or repeated `k`, or if `k` can be estimated,
then `d` can be recovered by:
\[
d = (s \cdot k - z) \cdot r^{-1} \pmod{n}
\]

This framework searches for `k` that minimizes the **variance between recovered `d` values**, using AI-assisted optimization loops.

---

## 📂 File Structure

| File | Description |
|------|--------------|
| `main.py` | The main recovery and analysis script |
| `vulnerabilities.txt` | Input file with extracted signatures (`r`, `s`, `z`) |
| `method_results/` | Directory for intermediate results (LSTM, GAN, ML analysis) |
| `optimization.log` | Log file with runtime details |
| `analysis.json`, `all_results.csv` | ML analysis outputs |

---

## 🧮 Input Format (`vulnerabilities.txt`)

Each block of data should contain a set of signatures separated by a delimiter:

r1: <hex>
s1: <hex>
z1: <hex>

r2: <hex>
s2: <hex>
z2: <hex>


The script automatically parses all blocks into ECDSA signature structures.

---

## 🚀 How to Run

### 1️⃣ Install dependencies
```bash
pip install requests numpy sympy pandas deap xgboost tensorflow gym ecdsa bech32 base58

2️⃣ Prepare transaction data

Add your extracted ECDSA signatures into vulnerabilities.txt using the above format.

3️⃣ Run the main analysis
python3 main.py


The program will:

Parse all transaction signatures

Train ML models (XGBoost + LSTM)

Generate candidate k values via multiple methods

Recover potential private keys (d)

Verify derived addresses against target Bitcoin address

⚙️ Algorithmic Components
Module	Description
read_transactions_from_file()	Reads and parses all signatures from file
recover_d_cached()	Efficient modular inverse-based key recovery
objective(k)	Objective function measuring consistency of recovered d
genetic_algorithm_k()	Genetic search for optimal nonce values
simulated_annealing_k()	Temperature-based local optimization
train_ml_model()	ML-assisted nonce prediction (XGBoost)
train_lstm()	LSTM model for temporal nonce pattern learning
ECDSAEnv	Custom OpenAI Gym environment for RL-based optimization
generate_addresses_from_private_key()	Generates P2PKH, Bech32, and P2SH Bitcoin addresses
🧬 Recovery Workflow
vulnerabilities.txt  ──►  read_transactions()
       │
       ▼
  extract features → train ML model (XGBoost / LSTM)
       │
       ▼
  predict(k) → refine (GA, Simulated Annealing)
       │
       ▼
  recover(d) → generate addresses → compare with target

📈 Advanced AI Components
🔹 Machine Learning

Uses XGBoostRegressor and XGBoostClassifier for predicting likely nonces and analyzing success patterns.

🔹 Deep Learning

Trains an LSTM neural network on transaction sequences to model time-based nonce relations.

🔹 Reinforcement Learning

Implements a custom Gym environment (ECDSAEnv) where an RL agent learns to adjust k values to minimize cryptographic error.

🔹 Genetic Algorithm

Evolves candidate nonces (k) by crossover, mutation, and selection to minimize the variance in recovered d values.

🔹 Simulated Annealing

Performs fine-grained optimization around previously successful k values using thermal decay control.

⚠️ Ethical Disclaimer

This project is built for academic research and blockchain cryptography education.
Do not use it to recover or analyze private keys for wallets that you do not own.
The full ECDSA keyspace is computationally infeasible to brute-force.
Use responsibly and in compliance with ethical hacking and cryptographic research laws.

🧩 Example Output
==== Rozpoczynam analizę ataków na ECDSA ====
Test generowania adresów:
Adres P2PKH: 1BgGZ9tcN4rm9KBzDn7KprQz87SZ26SAMH
Adres Bech32: bc1qw508d6qejxtdg4y5r3zarvary0c5xw7kygt080
Adres P2SH: 3J98t1WpEZ73CNmQviecrnyiWrnqRhWNLy
Iteracja 0: generowanie kandydatów...
✅ Wykryto liniową zależność k z pary transakcji: k = 0x1b3a4c5...
🎉 ZNALEZIONO POPRAWNY KLUCZ!
🔑 Klucz prywatny: 0x518b3f5...
🔑 Kandydat k: 0x1b3a4c5...

📊 Output Files
File	Purpose
optimization.log	Logs all algorithm progress, candidate k, and errors
method_results/all_results.csv	Stores cumulative results from AI and heuristic runs
method_results/analysis.json	JSON summary of ML attack results
method_results/gan_results.txt	GAN output log
method_results/nonce_data.csv	LSTM training data snapshot
🧰 Requirements

Python ≥ 3.9

TensorFlow ≥ 2.10

XGBoost ≥ 1.6

NumPy ≥ 1.24

DEAP ≥ 1.3

Gym ≥ 0.26

ecdsa ≥ 0.18

bech32, base58

📚 References

NIST FIPS 186-4 – Digital Signature Standard

Bitcoin ECDSA Vulnerability Analysis

DEAP Evolutionary Algorithms Framework

OpenAI Gym Documentation

TensorFlow LSTM Layers

BTC donation address: bc1q4nyq7kr4nwq6zw35pg0zl0k9jmdmtmadlfvqhr
