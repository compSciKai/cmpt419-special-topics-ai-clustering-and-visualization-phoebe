# cmpt419-special-topics-ai-clustering-and-visualization-phoebe

In this project, I was given static image-based data of facial expressions from Phoebe Buffay from the T.V. show, Friends. 
Using this data, I analyzed the facial action units (UA), and input AUs into a gaussian mixture model (GMM) which is a soft clustering algorithm.
Optimally, 8 clusters were found, each determining a distinct mood for Pheobe. An example of clusters can be seen below. 

[img](mood_clusters.png)

First, we see happy (high activation) in purple, next to excitement/surprise in red.
I hypothesized two clusters sharing a border will share similar emotional states. 
Using the first two emotions as an example, happiness is quite similar to excitement/surprise.
In yellow is sadness, bluish green is happy (low activation)/embarassment. 
Embarassment can might be linked to sadness, and might be similar to excitement in red.
Lime green is contempt, maybe linking with embarassment and excitement at borders, and orange being fear/neutral expressions. F
ear/Neutral seems link a strange mix, we could call it low activation fear.
Green is upset/anger, and light orange is disgust.

It is interesting to see happiness (high activation) at one end of data cluster, and upset, disgust, fear, and sadness at the other end.
