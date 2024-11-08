import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import search_data as search_data

# Read the data from CSV file and preprocess the corpus
data = pd.read_csv("data.csv")
processed_corpus = search_data.processed_corpus(data)

def plot_TSNE(embeddings, perplexity=2, n_iter=3000, name="lsi.png"):
    """
    Generate and save a t-SNE plot for the provided embeddings.
    
    Args:
        embeddings (list): The list of embeddings to plot.
        perplexity (int): The perplexity parameter for t-SNE.
        n_iter (int): The number of iterations for t-SNE.
        name (str): The name of the output image file.
    """
    # Apply t-SNE to reduce dimensions to 2D
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter)
    data_2d = tsne.fit_transform(np.array(embeddings))

    # Prepare DataFrame for plotting
    df = pd.DataFrame(data=data_2d, columns=['Dimension 1', 'Dimension 2'])
    labels = []

    # Assign group labels to points (for visualization)
    for i in range(10):
        labels += [i] * (5 if i < 5 else 1)
    df["Group"] = labels

    # Plot the scatterplot
    sns.scatterplot(data=df[:50], x="Dimension 1", y="Dimension 2", hue="Group", palette="deep")
    sns.scatterplot(data=df[50:], x="Dimension 1", y="Dimension 2", hue="Group", palette="deep", marker="X", legend=False, edgecolor="black")

    # Adjust plot limits and appearance for doc2vec model
    if name == "doc2vec.png":
        plt.xlim(left=-200, right=200)
        plt.ylim(bottom=-200, top=200)

    # Add title and legend
    plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left")
    plt.title(f"TSNE of the embeddings of {name}\nperplexity = {perplexity}\nn_iter = {n_iter}")
    plt.tight_layout()

    # Save the plot as PNG file
    plt.savefig(name)
    plt.show()

def give_me_models():
    """
    Retrieve the LSI and Doc2Vec models using search_data functions.
    """
    lsi_corpus = search_data.search_lsi(data, " ", processed_corpus, return_corpus=True)
    doc2vec_model = search_data.search_doc2vec(data, " ", processed_corpus, return_model=True)
    return lsi_corpus, doc2vec_model

def give_me_indeces(query):
    """
    Get the indices of the 5 most similar results for each model based on the query.
    
    Args:
        query (str): The search query string.
        
    Returns:
        tuple: Four lists of indices for frequency, tfidf, lsi, and doc2vec models.
    """
    frequency_sims = search_data.search_freq(data, query.lower(), processed_corpus)
    tfidf_sims = search_data.search_tfidf(data, query.lower(), processed_corpus)
    lsi_sims = search_data.search_lsi(data, query.lower(), processed_corpus)
    doc2vec_sims = search_data.search_doc2vec(data, query.lower(), processed_corpus)
    
    return frequency_sims, tfidf_sims, lsi_sims, doc2vec_sims

# Read and preprocess the ground truth file
with open("ground-truth-unique.txt", "r") as f:
    ground_truth = [line.strip() for line in f.readlines() if line.strip()]

# Split the ground truth into sublists of 3 elements
ground_truth = [ground_truth[i:i + 3] for i in range(0, len(ground_truth), 3)]

# Clean up the paths in the ground truth
for elem in ground_truth:
    elem[2] = elem[2][2:]  # Remove './' from path

# Get the indices of the ground truth entries in the data
indeces = []
for elem in ground_truth:
    node, path = elem[1], elem[2]
    idx = np.where((data['Name'] == node) & (data['Path'] == path))[0][0]
    indeces.append(idx)

# Initialize variables for matching and precision calculation
matching_frequency_sims = matching_tfidf_sims = matching_lsi_sims = matching_doc2vec_sims = 0
frequency_precision, tfidf_precision, lsi_precision, doc2vec_precision = [], [], [], []

# Lists to store indices for plotting LSI and Doc2Vec embeddings
lsi_indeces, doc2vec_indeces = [], []

def calculate_recall_precision(index, matching_count, sims, precision, model_name=""):
    """
    Calculate the recall and precision for a given model based on its similarity results.
    
    Args:
        index (int): The index of the query in the ground truth.
        matching_count (int): The current count of matching results.
        sims (list): The list of most similar document indices from the model.
        precision (list): The list to store precision values.
        model_name (str): The name of the model being evaluated (LSI or Doc2Vec).
    
    Returns:
        tuple: The updated precision list and matching count.
    """
    values = [value[0] for value in sims]

    # Store LSI and Doc2Vec indices for plotting
    if model_name == "LSI":
        lsi_indeces += values
    elif model_name == "Doc2Vec":
        doc2vec_indeces += values

    # Check if the query index is in the similarity list
    if indeces[index] in values:
        matching_count += 1
        index_in_values = values.index(indeces[index])
        precision.append(1 / (index_in_values + 1))  # Precision based on position
    else:
        precision.append(0)

    return precision, matching_count

# Evaluate the models for each query in the ground truth
for index, elem in enumerate(ground_truth):
    frequency_sims, tfidf_sims, lsi_sims, doc2vec_sims = give_me_indeces(elem[0])

    # Calculate precision for each model
    frequency_precision, matching_frequency_sims = calculate_recall_precision(index, matching_frequency_sims, frequency_sims, frequency_precision)
    tfidf_precision, matching_tfidf_sims = calculate_recall_precision(index, matching_tfidf_sims, tfidf_sims, tfidf_precision)
    lsi_precision, matching_lsi_sims = calculate_recall_precision(index, matching_lsi_sims, lsi_sims, lsi_precision, "LSI")
    doc2vec_precision, matching_doc2vec_sims = calculate_recall_precision(index, matching_doc2vec_sims, doc2vec_sims, doc2vec_precision, "Doc2Vec")

# Print matching results and precision for each model
print(f"Matching frequency sims: {matching_frequency_sims}")
print(f"Matching tfidf sims: {matching_tfidf_sims}")
print(f"Matching lsi sims: {matching_lsi_sims}")
print(f"Matching doc2vec sims: {matching_doc2vec_sims}\n")
print(f"Precision frequency sims: {frequency_precision}, average: {np.mean(frequency_precision)}")
print(f"Precision tfidf sims: {tfidf_precision}, average: {np.mean(tfidf_precision)}")
print(f"Precision lsi sims: {lsi_precision}, average: {np.mean(lsi_precision)}")
print(f"Precision doc2vec sims: {doc2vec_precision}, average: {np.mean(doc2vec_precision)}")

# Save precision and recall results to a file
with open("precision_recall.txt", "w") as f:
    f.writelines(f"frequency sims:\n recall: {matching_frequency_sims/10}\n precision: {np.mean(frequency_precision)}\n\n")
    f.writelines(f"tfidf sims:\n recall: {matching_tfidf_sims/10}\n precision: {np.mean(tfidf_precision)}\n\n")
    f.writelines(f"lsi sims:\n recall: {matching_lsi_sims/10}\n precision: {np.mean(lsi_precision)}\n\n")
    f.writelines(f"doc2vec sims:\n recall: {matching_doc2vec_sims/10}\n precision: {np.mean(doc2vec_precision)}\n\n")

# Retrieve models and embeddings
lsi_corpus, doc2vec_model = give_me_models()
lsi_embeddings = [[val[1] for val in lsi_corpus[i]] for i in lsi_indeces]
doc2vec_embeddings = [doc2vec_model.infer_vector(processed_corpus[i]) for i in doc2vec_indeces]
ground_truth_embeddings_lsi = [[elem[1] for elem in lsi_corpus[idx]] for idx in indeces]
ground_truth_embeddings_doc2vec = [doc2vec_model.infer_vector(processed_corpus[idx]) for idx in indeces]

# Combine embeddings for visualization
lsi_embeddings += ground_truth_embeddings_lsi
doc2vec_embeddings += ground_truth_embeddings_doc2vec

# Plot the t-SNE of the embeddings
plot_TSNE(lsi_embeddings)
plot_TSNE(doc2vec_embeddings, name="doc2vec.png")
