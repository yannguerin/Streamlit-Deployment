import streamlit as st

# import numpy as np
# import pandas as pd
import networkx as nx

# from random import sample
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from collections import Counter
import base64

from nltk import download

download("wordnet")
download("omw-1.4")
from nltk.corpus import wordnet as wn

# from nltk.corpus import words
# from nltk.corpus.reader import WordListCorpusReader


@st.cache
def recursive_hyponyms(depth, starting_name):
    entity_graph_list = []
    entity_layers_dict = {starting_name: 0}
    starting_synset = wn.synset(starting_name)
    # depth = 10
    attempt = 0

    def entity_hyponyms(n, max_depth, synset):
        if n < max_depth:
            n += 1
            for syn in synset.hyponyms():
                entity_graph_list.append((synset.name(), syn.name()))
                entity_layers_dict[syn.name()] = n
                entity_hyponyms(n, max_depth, synset=syn)
        return 0

    entity_hyponyms(0, depth, starting_synset)
    return entity_graph_list, entity_layers_dict, attempt


@st.cache(allow_output_mutation=True)
def make_network_graph(
    entity_graph_list, entity_layers_dict, attempt, depth, starting_name
):
    dG = nx.DiGraph()
    dG.add_edges_from(entity_graph_list)
    root_edge_list = [edge[0] for edge in entity_graph_list]
    root_edge_counter_dict = dict(Counter(root_edge_list))
    node_size_list = [
        10 * root_edge_counter_dict[node_name]
        if node_name in root_edge_counter_dict.keys()
        else 10
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
    pos_pre_scaling = nx.multipartite_layout(
        dG, subset_key="layer", align="horizontal", scale=5
    )
    pos = {
        key: (
            np.array(
                [
                    val[0] * 25,
                    (val[1] / 500) + (entity_layers_dict[key] * (val[1] / 5000)),
                ]
            )
            if (index % 2) == 0
            else np.array(
                [
                    val[0] * 25,
                    (val[1] / 500) - (entity_layers_dict[key] * (val[1] / 5000)),
                ]
            )
        )
        for index, (key, val) in enumerate(pos_pre_scaling.items())
    }
    # No ZigZag
    # pos = {
    #     key: np.array([val[0] * 75000 + 1, val[1] / 1000])
    #     for key, val in pos_pre_scaling.items()
    # }
    attempt += 1
    return (
        dG,
        pos,
        colors,
        node_size_list,
        node_names_short,
        starting_name,
        depth,
        attempt,
    )


@st.cache
def draw_network_graph_as_pdf(
    dG, pos, colors, node_size_list, node_names_short, starting_name, depth, attempt
):
    fig = plt.figure(figsize=(depth * 30, depth * 30))
    nx.draw(
        dG,
        pos,
        node_color=colors,
        node_size=node_size_list,
        edge_color="lightgray",
        width=0.5,
    )
    nx.draw_networkx_labels(dG, pos, labels=node_names_short, font_size=2)
    filename = (
        f"multipartite_{starting_name.split('.')[0]}_{depth}_attempt_{attempt}" + ".pdf"
    )
    plt.savefig(filename, transparent=True)
    return filename, dG.number_of_nodes()


# Solution for showing pdfs obtained from https://towardsdatascience.com/display-and-download-pdf-in-streamlit-a-blog-use-case-5fc1ac87d4b1
@st.cache
def show_pdf(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode("utf-8")
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="800" height="800" type="application/pdf"></iframe>'
    return pdf_display


st.title("Hyponyms Exploration")

search_word = st.text_input("Search Synsets for given word:", value="")

synset_options = wn.synsets(str(search_word))
string_synset_options = [str(syn.name()) for syn in synset_options]

synset_to_add = None
if len(string_synset_options) > 0:
    synset_to_add = st.selectbox(
        "Choose the Synset to add.", list(string_synset_options)
    )

if st.button("Click me to see the definition of your chosen synset."):
    st.write(wn.synset(synset_to_add).definition())


options = [
    "entity.n.01",
    "communication.n.01",
    "cognition.n.01",
    "belief.n.01",
    "doctrine.n.01",
    "car.n.01",
    "dog.n.01",
]

if synset_to_add:
    options.append(synset_to_add)


starting_synset = st.selectbox("Choose the starting Synset.", options)

depth = st.slider("Choose the recursive depth.", 3, 20, 5)
if depth >= 10:
    st.warning(
        "Large Depth Values may take a very long time to run depending on your computers hardware."
    )

if st.button("Run"):
    st.balloons()
    entity_graph_list, entity_layers_dict, attempt = recursive_hyponyms(
        depth, starting_synset
    )

    (
        dG,
        pos,
        colors,
        node_size_list,
        node_names_short,
        starting_name,
        depth,
        attempt,
    ) = make_network_graph(
        entity_graph_list, entity_layers_dict, attempt, depth, starting_synset
    )

    filename, nNodes = draw_network_graph_as_pdf(
        dG, pos, colors, node_size_list, node_names_short, starting_name, depth, attempt
    )

    pdf_display = show_pdf(
        f"multipartite_{starting_synset.split('.')[0]}_{depth}_attempt_1.pdf"
    )

    st.markdown(pdf_display, unsafe_allow_html=True)
else:
    st.write("Click Run to build and view the graph")
