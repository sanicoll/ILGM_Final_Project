# ILGM_Final_Project
Final Project for ECE69500: Inference and Learning in Generative Models.

## About
This project investigates the feasibility of applying variational autoencoders within a distributed setting: specifically, a Social Learning Network (SLN) comprised of students in an online classroom. It compares results from centralized and distributed cases to find that while centralized VAEs perform well on all datasets, the distributed VAE architecture is able to learn similar levels of clustering with fewer training epochs per user (drastically reducing computing resource needs). Overall, VAEs provide a way to model students in a personalized manner such that outlier students are not disadvantaged by the model.

Furthermore, the project investigates the possibility of adding a categorical discriminator to the VAE architecture such that a specific feature subset may be more explicitly encoded. Referencing the graphical model in the provided presentation file, we add an "observed" (self-supervised, extracted from the data itself without explicit labels passed) feature representing the type of action a student will next take. However, we are unable to see much distinction between encodings when this node is added to the model versus when it is not. Further experimentation is necessary to better incorporate this discriminative step into the process.

## Running the Code
This code requires the latest version of Tensorflow/Keras. 
For the centralized VAE: 
- Code can be found under the src folder in 'VAE_central'
- 'run_sim.py' is the main driver file. It expects three arguments: the path (relative or absolute) to the data file to be analyzed. The number of epochs to train the model for. And the type of VAE to build (allowed terms = 'central', 'heads')

*central*: Builds a single, convolutional VAE with a latent space of 8 dimensions
*heads*: Builds a hierarchical VAE with # of data heads equal to the number of data types found in the provided data file. Each data head is a convolutional VAE with units dictated by feature vector size of input. The shared, second-level VAE is built from fully-connected layers and a latent space of 8 dimensions.

For the distributed VAE:
- Code can be found under the src folder in 'VAE_federated'
- 'sim_VAE.py' is the main driver file. It expects two arguments: the path (relative or absolute) to the data file to be analyzed, and the number of aggregations to perform.


## Notes:
Klingler et al. (https://files.eric.ed.gov/fulltext/ED596590.pdf) provides the description of architecture for the convolutional VAE (not code) and motivates the use of VAEs in education
All code is original

