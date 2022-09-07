import streamlit as st

# import numpy as np
# import pandas as pd
import networkx as nx

# from random import sample
import matplotlib.pyplot as plt
from matplotlib import cm
from collections import Counter
import base64

# import nltk
from nltk.corpus import wordnet as wn

# from nltk.corpus import words
# from nltk.corpus.reader import WordListCorpusReader


st.title("Hyponyms Exploration")


@st.cache
def recursive_hyponyms(depth):
    entity_graph_list = []
    entity_layers_dict = {"entity.n.01": 0}
    entity = wn.synset("entity.n.01")
    depth = 10
    attempt = 0

    def entity_hyponyms(n, max_depth, synset=wn.synset("entity.n.01")):
        if n < max_depth:
            for syn in synset.hyponyms():
                n += 1
                entity_graph_list.append((synset.name(), syn.name()))
                entity_layers_dict[syn.name()] = n
                entity_hyponyms(n, max_depth, synset=syn)
        return 0

    entity_hyponyms(0, depth, entity)
    return entity_graph_list, entity_layers_dict, attempt


@st.cache
def make_network_graph(entity_graph_list, entity_layers_dict, attempt, depth):
    dG = nx.DiGraph()
    dG.add_edges_from(entity_graph_list)
    root_edge_list = [edge[0] for edge in entity_graph_list]
    root_edge_counter_dict = dict(Counter(root_edge_list))
    node_size_list = [
        300 * root_edge_counter_dict[node_name]
        if node_name in root_edge_counter_dict.keys()
        else 300
        for node_name in list(dG.nodes())
    ]
    node_names_short = {
        node_name: node_name.split(".")[0] for node_name in list(dG.nodes())
    }
    # Adding the layers and associated colors to the graph

    # Get the color map of Blues, and segmenting it into the number of layers (depth) the data has
    blues = cm.get_cmap(
        "Blues", depth
    )  # --> List of RGBA tuples, callable with number, not indexable
    colors = []

    # looping through all nodes
    for node in list(dG.nodes()):
        # Adding nodes with specified layer/depth
        dG.add_node(node, layer=entity_layers_dict[node])
        # Appending the correct color for associated depth to a colors list
        colors.append(blues(entity_layers_dict[node]))

    # Generating the positions for a multilayered graph
    pos = nx.multipartite_layout(dG, subset_key="layer", align="horizontal", scale=5)
    attempt += 1
    fig = plt.figure(figsize=(depth * 20, depth * 20))
    nx.draw(
        dG, pos, node_color=colors, node_size=node_size_list, edge_color="lightgray"
    )
    nx.draw_networkx_labels(dG, pos, labels=node_names_short)
    filename = f"multipartite_entity_{depth}_attempt_{attempt}" + ".pdf"
    plt.savefig(filename, transparent=True)
    return filename


@st.cache
def show_pdf(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode("utf-8")
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="800" height="800" type="application/pdf"></iframe>'
    return pdf_display


entity_graph_list, entity_layers_dict, attempt = recursive_hyponyms(5)

filename = make_network_graph(entity_graph_list, entity_layers_dict, attempt, 5)


pdf_display = show_pdf("multipartite_entity_5_attempt_1.pdf")

st.markdown(pdf_display, unsafe_allow_html=True)
