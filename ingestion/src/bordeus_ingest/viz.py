"""Step 5: visualizzazione degli embedding con t-SNE (2D o 3D).

Riduce vettori ad alta dimensione (embed.EMBEDDING_DIM) a 2 o 3 dimensioni
per ispezionarli visivamente — utile per verificare a occhio se i chunk
formano cluster sensati (per categoria, per documento) o se qualcosa
nell'estrazione/chunking sta producendo rumore. Non usato dal bot in
produzione: è uno strumento di ispezione per chi cura la pipeline.

Legge direttamente dalle tabelle di `langchain-postgres`
(`langchain_pg_collection`, `langchain_pg_embedding`), le stesse su cui
scrive vectorstore.py e da cui legge il bot.
"""

from __future__ import annotations

import numpy as np
import psycopg
from pgvector.psycopg import register_vector


def fetch_embeddings_for_collection(database_url: str, collection_name: str) -> list[dict]:
    """Recupera contenuto, embedding e metadati di tutti i chunk di una
    collection (= comune). Il tipo `vector` viene registrato qui perché
    questa funzione gira sempre DOPO che `langchain-postgres` ha già
    creato l'estensione (in fase di ingestion) — nessun problema di
    ordine "uovo e gallina" da gestire."""
    conn = psycopg.connect(database_url, autocommit=True)
    try:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT e.document, e.embedding, e.cmetadata
                FROM langchain_pg_embedding e
                JOIN langchain_pg_collection c ON c.uuid = e.collection_id
                WHERE c.name = %s
                ORDER BY e.id
                """,
                (collection_name,),
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    return [
        {
            "content": document,
            "embedding": embedding.tolist(),
            "metadata": metadata or {},
        }
        for document, embedding, metadata in rows
    ]


def reduce(
    embeddings: list[list[float]],
    n_components: int = 2,
    perplexity: float = 30.0,
    random_state: int = 42,
) -> np.ndarray:
    """Riduce una lista di embedding a coordinate 2D o 3D con t-SNE.

    La perplexity di t-SNE deve essere minore del numero di campioni:
    con pochi chunk la abbassiamo automaticamente, invece di lasciare che
    scikit-learn sollevi un errore criptico o produca un risultato
    degenere.
    """
    if n_components not in (2, 3):
        raise ValueError("n_components deve essere 2 o 3")

    # Import lazy: scikit-learn è una dipendenza di sviluppo (gruppo
    # `dev`, vedi pyproject.toml), non runtime.
    from sklearn.manifold import TSNE

    X = np.asarray(embeddings, dtype=np.float32)
    n_samples = X.shape[0]
    if n_samples < 3:
        raise ValueError(
            f"Servono almeno 3 embedding per una proiezione t-SNE sensata "
            f"(trovati {n_samples}). Ingerisci più contenuto prima di visualizzare."
        )

    effective_perplexity = min(perplexity, max(1.0, (n_samples - 1) / 3))

    tsne = TSNE(
        n_components=n_components,
        perplexity=effective_perplexity,
        random_state=random_state,
        init="pca",
    )
    return tsne.fit_transform(X)


def plot(coords: np.ndarray, labels: list[str], title: str = "Chunk embedding (t-SNE)"):
    """Scatter plot 2D o 3D (dedotto da coords.shape[1]) colorato per
    etichetta. Ritorna la Figure matplotlib senza chiamare plt.show(): si
    visualizza da sola nell'output di una cella Jupyter, e può essere
    salvata con `fig.savefig(...)` anche fuori da un notebook."""
    # Import lazy per lo stesso motivo di sklearn in reduce().
    import matplotlib.pyplot as plt

    n_components = coords.shape[1]
    unique_labels = sorted(set(labels))
    cmap = plt.get_cmap("tab10" if len(unique_labels) <= 10 else "tab20")

    fig = plt.figure(figsize=(9, 7))
    if n_components == 3:
        ax = fig.add_subplot(111, projection="3d")
    else:
        ax = fig.add_subplot(111)

    for i, label in enumerate(unique_labels):
        mask = [l == label for l in labels]
        pts = coords[mask]
        coords_args = (pts[:, 0], pts[:, 1], pts[:, 2]) if n_components == 3 else (pts[:, 0], pts[:, 1])
        ax.scatter(*coords_args, label=label, color=cmap(i % cmap.N), alpha=0.75, s=40)

    ax.set_title(title)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    if n_components == 3:
        ax.set_zlabel("t-SNE 3")
    ax.legend(loc="best", fontsize=8, markerscale=1.2, title="Legenda")
    fig.tight_layout()
    return fig
