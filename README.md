# ScenarioBasedHistoryMatching


The GAN we train here is a WGAN-GP. 

Key Practices for stable training GANs:
1. Use Wasserstein as the cost function: This function gives an idea about the quality of realizations compared to traditional GAN approach of reak/fake classification
2. Use spectral normalization to stabilize training
3. Always use small batch size for stable training
4. Use grandient penalty to avoid exploding gradients

The code reported here for now only involve training process. we train conditional GANs to generate 2D fluvial channels with different orientations, proportion of facies and channels width. 

You can access UT box link to download a checkpoint during training and test the workflow: https://utexas.box.com/s/aebriobls1mugu7d5dd2wmd2kj4zd7gz. 
