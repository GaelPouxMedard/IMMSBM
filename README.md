# IMMSBM

This repository contains the code for building the matrix of observations of pairs of entities giving rise to output entities (MMSBM_buildObservations.py). 
It also contains the code to fit the data using an EM algorithm (MMSBM_interactionPaires.py).
Finally, we also provide the code used to run the baseline model that do not account for interactions in the same dataset (MMSBM_noInteractionPaires.py)

Abstract of the corresponding publication: 
In most of the real-world applications, it is seldom the case that a given observable evolves independently of its environment. In social networks, users' behaviour results from the people they interact with, news in their history feed, or trending topics. In natural language, the meaning of phrases emerges from the combination of words. In general medicine, a diagnosis is established on the basis of symptoms' interaction.
We here describe the new model IMMSBM that investigates the role of interactions and quantifies their importance within the very corpus we just mentioned. We find that interactions play an important role in those corpus. In inference tasks, taking them into account leads to average relative changes up to 150\% of the prior probability of an outcome. Besides, their role greatly improves the predictive power of the model, and provides a principle method to deal with cold start problems.
Our findings suggest that neglecting interaction while modeling real-world phenomenons might lead to draw incorrect conclusion.
