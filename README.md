# PointTransformerCardinalityReduction
By Zach Eichenberger, Parth Raut, Mehar Singh

This is our final project for EECS 598-004: Systems for Generative AI class, taught by Mosharaf Chowdhury. 

Point clouds are an abundant source of data, being produced Lidar systems, 3d scans and model renderings. They can be used for tasks including point cloud classification, registration, and generative tasks including generating 3d models from sparse input views. 

Recent models have been developed to process this form of data, particularly using transformer architectures.  Unfortunately, due due to the high cardinality of the data representation, processing such point clouds can be computationally intensive. We seek to reduce this complexity by borrowing the concept of token pruning and merging from recent transformer architectures in LLMs and ViTs. 

This works by choosing tokens to prune, and merging similar or unimportant tokens together, merging their hidden state resenations. In doing so, evaluation computational cost is reduced by 2-3x. 