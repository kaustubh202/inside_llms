# Domain knowledge per layer :

Now that we have obtained the v-usable information and the fisher separability scores by probing through a simple Logistic regressor between each pair of dataset (10 in total as we are currently using 5 datasets) for each layer(28 in total as the model in use is Llama-3.2-3B variant) by using the activations after MLP,Attention Blocks and using the Residual Stream activations we needed to find metrics which help us give a sense of "Domain knowledge stored in a layer" using the information obtained .This lays down our objective.

## Proposed Methodology:
To obtain "Stored domain knowledge per layer" we propose the following metric using the fisher separability scores and v-usable information obtained:

1) We begin with creating lists  for each pair of dataset (cpp,python,math_think,math_solution and physics), for each metric (either v-usable information or fisher separability scores) and for each component Blocks (MLP/Attention/Residual stream) : This results in total of 10x2x3 total number of lists containing the v-usable information and fisher separability score layer wise so each list contains 28 elements.

2) Now we process these lists by dividing each one by their L-2 norm (The why of normalizing the lists is explained later) obtaining 60 normalized lists.

3) In accordance to our objective we now create 30 Heatmaps(for each dataset , each component block , each metric used) each of size (n_layers)x(n_datasets used) in the following way:

* For each dataset (say for e.g. cpp) We create a heatmap of size 28x5 (n_layers=28 and n_datasets used =5) whose first 4 columns will contain the values of the L-2 normalized lists against the cpp dataset for the remaining 4 other datasets (so 28 values) for a specific metric (either fisher score or the v-usable info).

* Now the 5th column contains the average over other 4 columns for a specific layer (i.e. for the corresponding row representing the layer) and we hypothesize this is  a  representation in some form of domain knowledge for the layer with explanation below.

## Why this works?:
To explain why this is correct and represents "Domain knowledge stored in a layer" we first explain why we need l-2 normalized lists:

#### Why normalize?

We use the L-2 normalized lists due to the way we interpret "Domain knowledge stored in a layer" in a form of 1 vs all (datasets) scores for fisher separability , so think of "domain knowledge" be a representation of the fact that if for our chosen dataset if any other randomly chosen dataset is pitted against it and we calculate the fisher separability score against it (in the same methods through the probes) then the higher expectation of fisher separability score represents the layer has learnt to separate out the dataset against any other more than other layers on average , representing "domain knowledge" for that layer.So as rather than treating any other dataset "special" than the one for which we are probing (which could happen if fisher separability scores for our chosen dataset and some randomly chosen dataset is much higher than others) then the so called "special" random dataset can misrepresent the layer-wise domain knowledge by dominating it's value's over other datasets and representing overall domain knowledge as it's own separability score with our chosen dataset.Also as we only have comparisions with a limiting number of datasets (5) and not hundereds of them so normalization is a must to overturn domination of fisher scores in mean values for layers by a single random dataset and our chosen dataset. Also normalization still preserves the relative order of fisher scores "between layers" so this works.

Also it is obvious that it would be better to check with k-fold cross validation (K< number of datasets-1) to check wether our order of domain knowledge for layers for a specific dataset converges with number of datasets we have chosen and take mean against (by choosing combinations of k number of datasets at a time and checking wether all of or most of the k combinations and taking their means of L-2 normalized fisher scores converge with results when using all datasets representing pinpoint accuracy of order in which layers store "domain knowledge" rather than what we claim here which is more general due to limitation of number of datasets for which we have info here).


The results obtained are quite interesting for the fisher separability heatmaps but not for the v-usable information but that was already quite expected from our previous experiments in probe separability so we propose what fischer separbility scores used in the way here can represent and why even if v-usable information gives us no conclusion we can still use fisher separability scores to obtain our heatmaps and say they are a valid representation in some form of "domain knowledge stored in a layer"

## Why fisher separability score is a useful metric here:
Yaha hypothesize karna padega 

#  Results Obatined and Hypothesis:
### * For datasets like 'python' and 'cpp':
* We obtain extremely interesting and insightful results for Attention blocks fisher separability scores:
  * Certain Intermediate layers (i.e. layer 16,17,19  for cpp dataset and python dataset both) give out much higher scores than other layers and other layers really are relatively consistent in their domain knowledge(note layer 0 many times contains bogus results so we are not factoring that in).

* We also obtain insightful information for MLP blocks  fisher values:
  * For the MLP block , the python dataset "domain knowledge values" are fluctuating in the initial layers , from layer 15 onwards the value increases gradually with final layers with more knowledge than starting layers with few exceptions in between
  * For the cpp dataset MLP block, the trend is somewhat similar in the starting layers to the python dataset but there is a spike of "domain knowledge value" at around layer 15-16 , from then onwards the value drops a bit and maintains gradual increase of it's value till it's maximum value at layer 27.
* Also these results are obtained for Residual stream activation fisher separability scores:
  * For both datasets The values peak around layer 3,4,5 then go down a bit , maintain till intermediate layers , from then onwards (layer 14) value spikes then  maintains value (with slight fluctuations in between).

### * For datasets like 'physics':

* For the Attention block The starting layers 0-7 contain the most "domain knowledge" and then value drops , maintains till intermediate layers (14) from there is a spike at intermediate layers like (15,16,17,19) and few of the last layers like 26-27.

* For MLP block The intermediate layers

  

      



# Possible needed modifications and imporvements in robustness:
1) Number of datasets used are very less (5) we may need more datasets for convergence of our results as it's essential that we get robust results on exactly which layers will be the ones who store domain knowledge of a dataset that's the results are written in a broad and general manner ,not pinpointing layers.

2) We need to think more robustly on how exactly does this "1 vs all domain layer wise separability knowledge mean" and wether it translates perfectly to representing "domain knowledge stored in that layer".