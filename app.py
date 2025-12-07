# app.py
# Movie Recommender GUI with Colorful Theme
# Run: python app.py

import os
import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------
# Configuration / Constants
# ---------------------------
DATA_FILE = "movierecomm.csv"
DEFAULT_N = 5
BG_COLOR = "#87CEEB"      # Sky Blue
FRAME_COLOR = "#F0F8FF"   # Alice Blue (lighter white-blue)
BUTTON_COLOR = "#4682B4"  # Steel Blue
BUTTON_TEXT = "white"

# ---------------------------
# Helper: Load and prepare data
# ---------------------------
def load_and_prepare(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at '{path}'.")

    df = pd.read_csv(path)
    required = {"title", "genres", "overview"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"Dataset missing required columns. Required: {required}.")

    df["genres"] = df["genres"].fillna("unknown genre").astype(str)
    df["overview"] = df["overview"].fillna("no overview available").astype(str)

    for col in ["rating", "popularity", "year", "revenue"]:
        if col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                df[col] = df[col].fillna(df[col].median())
            except:
                df[col] = df[col].fillna(0)

    df["combined_text"] = df["genres"].str.lower() + " " + df["overview"].str.lower()
    df = df.reset_index(drop=True)
    return df

# ---------------------------
# Helper: Build TF-IDF + Similarity
# ---------------------------
def build_model(df):
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(df["combined_text"])
    similarity = cosine_similarity(tfidf_matrix)
    return vectorizer, tfidf_matrix, similarity

# ---------------------------
# Recommendation functions
# ---------------------------
def recommend_by_genre(df, genre_query, n=DEFAULT_N):
    genre_query = str(genre_query).strip().lower()
    mask = df["genres"].str.lower().str.contains(genre_query, na=False)
    matches = df[mask]
    if matches.empty:
        words = [w for w in genre_query.split() if len(w) > 2]
        for w in words:
            mask = df["genres"].str.lower().str.contains(w, na=False)
            matches = df[mask]
            if not matches.empty:
                break
    if "popularity" in df.columns:
        matches_sorted = matches.sort_values(by="popularity", ascending=False)
    else:
        matches_sorted = matches
    return matches_sorted["title"].head(n).tolist()

def recommend_by_movie(df, similarity_matrix, movie_title, n=DEFAULT_N):
    movie_title = str(movie_title).strip().lower()
    titles_lower = df["title"].str.lower().values
    if movie_title not in titles_lower:
        mask = df["title"].str.lower().str.contains(movie_title, na=False)
        if mask.any():
            idx = df[mask].index[0]
        else:
            return []
    else:
        idx = int(np.where(titles_lower == movie_title)[0][0])
    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    recommended_indices = [i for i, score in sim_scores if i != idx][:n]
    return df["title"].iloc[recommended_indices].tolist()

def recommend(df, similarity_matrix, query, n=DEFAULT_N):
    q = str(query).strip()
    if not q:
        return []
    genre_mask = df["genres"].str.lower().str.contains(q.lower(), na=False)
    if genre_mask.any():
        return recommend_by_genre(df, q, n)
    recs = recommend_by_movie(df, similarity_matrix, q, n)
    if recs:
        return recs
    words = [w for w in q.lower().split() if len(w) > 2]
    for w in words:
        genre_mask = df["genres"].str.lower().str.contains(w, na=False)
        if genre_mask.any():
            return recommend_by_genre(df, w, n)
    return []

# ---------------------------
# GUI: Tkinter App
# ---------------------------
class MovieRecommenderApp:
    def __init__(self, root, df, similarity_matrix):
        self.root = root
        self.df = df
        self.similarity_matrix = similarity_matrix

        root.title("Movie Recommendation System")
        root.geometry("750x500")
        root.configure(bg=BG_COLOR)
        root.resizable(False, False)

        # --- Title ---
        title_lbl = tk.Label(root, text="Movie Recommendation System", font=("Helvetica", 18, "bold"), bg=BG_COLOR, fg="white")
        title_lbl.pack(pady=10)

        # --- Input frame ---
        frame = tk.Frame(root, bg=FRAME_COLOR, bd=2, relief="groove", padx=10, pady=10)
        frame.pack(fill="x", padx=20, pady=10)

        tk.Label(frame, text="Enter movie title OR genre:", bg=FRAME_COLOR, font=("Helvetica", 10, "bold")).grid(row=0, column=0, sticky="w")
        self.query_entry = tk.Entry(frame, width=50, font=("Helvetica", 10))
        self.query_entry.grid(row=1, column=0, padx=(0,10), pady=5, sticky="w")

        tk.Label(frame, text="Number of recommendations:", bg=FRAME_COLOR, font=("Helvetica", 10, "bold")).grid(row=0, column=1, sticky="w", padx=(10,0))
        
        # --- Fixed Spinbox with StringVar ---
        self.n_var = tk.StringVar(value=str(DEFAULT_N))
        self.n_spin = tk.Spinbox(frame, from_=1, to=20, width=5, font=("Helvetica", 10), textvariable=self.n_var)
        self.n_spin.grid(row=1, column=1, sticky="w", padx=(10,0))

        # --- Buttons ---
        btn_frame = tk.Frame(root, bg=BG_COLOR)
        btn_frame.pack(fill="x", pady=(0,10))

        recommend_btn = tk.Button(btn_frame, text="Recommend", command=self.on_recommend, bg=BUTTON_COLOR, fg="black", font=("Helvetica", 10, "bold"))
        recommend_btn.pack(side="left", padx=(12,6), pady=5)

        clear_btn = tk.Button(btn_frame, text="Clear Results", command=self.clear_results, bg=BUTTON_COLOR, fg="black", font=("Helvetica", 10, "bold"))
        clear_btn.pack(side="left", padx=(6,6), pady=5)

        sample_btn = tk.Button(btn_frame, text="Sample Inputs", command=self.show_sample_inputs, bg=BUTTON_COLOR, fg="black", font=("Helvetica", 10, "bold"))
        sample_btn.pack(side="left", padx=(6,6), pady=5)

        # --- Results area ---
        res_frame = tk.Frame(root, bg=FRAME_COLOR, bd=2, relief="groove", padx=10, pady=10)
        res_frame.pack(fill="both", expand=True, padx=20, pady=(0,10))

        tk.Label(res_frame, text="Recommendations:", font=("Helvetica", 12, "bold"), bg=FRAME_COLOR).pack(anchor="w")

        self.result_list = tk.Listbox(res_frame, height=12, width=90, font=("Helvetica", 10))
        self.result_list.pack(side="left", fill="both", expand=True, pady=(6,0))

        scrollbar = tk.Scrollbar(res_frame)
        scrollbar.pack(side="right", fill="y")
        self.result_list.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.result_list.yview)

        # --- Status bar ---
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status = tk.Label(root, textvariable=self.status_var, relief="sunken", anchor="w", bg="white")
        status.pack(side="bottom", fill="x")

    def on_recommend(self):
        q = self.query_entry.get().strip()
        try:
            n = int(self.n_var.get())
            if n <= 0:
                raise ValueError
        except:
            messagebox.showerror("Invalid input", "Please enter a valid positive integer for number of recommendations.")
            return
        if not q:
            messagebox.showinfo("Input required", "Please enter a movie title or a genre.")
            return

        self.status_var.set("Computing recommendations...")
        self.root.update_idletasks()

        recs = recommend(self.df, self.similarity_matrix, q, n)
        self.result_list.delete(0, tk.END)
        if not recs:
            self.result_list.insert(tk.END, "No recommendations found for this input.")
            self.status_var.set("No matches found.")
        else:
            for i, title in enumerate(recs, start=1):
                self.result_list.insert(tk.END, f"{i}. {title}")
            self.status_var.set(f"Displayed top {len(recs)} recommendations for '{q}'")

    def clear_results(self):
        self.result_list.delete(0, tk.END)
        self.status_var.set("Cleared results.")

    def show_sample_inputs(self):
        examples = ["action", "romance", "Inception", "Dangal", "animation", "thriller"]
        messagebox.showinfo("Sample Inputs", "Try these:\n\n" + "\n".join(examples))

# ---------------------------
# Main
# ---------------------------
def main():
    try:
        df = load_and_prepare(DATA_FILE)
    except Exception as e:
        messagebox.showerror("Startup error", f"Error loading data: {e}")
        return

    vectorizer, tfidf_matrix, similarity_matrix = build_model(df)
    root = tk.Tk()
    app = MovieRecommenderApp(root, df, similarity_matrix)
    root.mainloop()


if __name__ == "__main__":
    main()
