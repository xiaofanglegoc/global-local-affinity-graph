# global-local-affinity-graph
source code for the paper "A global/local affinity graph for image segmentation"

here is the code to generate the table IV for our paper
- you could also extract only part of this code easily to port single

  different graph tp generate the table I 
 - to get our global/local affinity graph, you could just keep the local
 graph and LO graph, for example:
 ```
 W_L0 = compute_region_similarity_Sparse_penalty(feature,3,centroid,Area);
 W_GLG=assignGraphValue(W,W_L0,global_nodes);
 ```

- to combine different feature descriptor, 
     - first just change the following 
         feature=feat{k}.mlab;
      - here is the list you could change:
```
 {'mlab';

    'ch';

     'lbp';

     'siftbow100';

     'siftbow150';

     'siftbow200';

     'siftbow300'};
```
 - then you compute the affinity graph W_GLG_mlab,W_GLG_lbp, etc., 

 - finally, you combine them according to the fusion equation in our paper Eq.11-12

> Please Note that you may not generate the exact performance listed in our paper,
