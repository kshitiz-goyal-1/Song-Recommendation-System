# Sound recommendation system based on sound content
 This github repository consist of my major project for btech cse. i have made a sound based song recommendation system which recommends songs based on the previews of songs using mel spectograms and creating latent space.
 For running this particular project you may run backend.py .
 But before doing so you may have to do some more work .
 First get api keys from spotify for developers and create a folder dat and in that apikeys then add these secret keys and client key in json format.
 Further you need to get the feather files to train your model for that you have to run the jupyter notebooks one by one.
 If got stuck you may forward the issue to sahilsaklani74177@gmail.com. 
 After all training and getting the model up to date then run the backend.py and open the index.html to search that particular song you will get recommendations.
 Note: The accuracy of the projects depends upon the pool of songs you have used in traing encoder and decoder model. If your latent space pool have less no. of songs then you may not get good recommendations.
 Project pipeline: 
• Scrape song information from Spotify's Public API.
• Convert waveforms from mp3 previews to Mel spectrograms.
• Train an autoencoder network to extract latent features from the audio information.
• Use UMAP for dimensionality reduction to view the latent space.
• Make recommendations based on cosine similarity in the latent space.
In this model training we have used encoder nad decoder model for creating the latent space.
