# MLOps

I used **Docker** and **GitHub Actions** to implement CI/CD principles for Machine Learning workflow and applied it to a personnal NLP project. My goal is also to implement continuous trainign (CT) so that the model can be retrained easily. 

To do this we need to have a dedicated Git repository per model. The main branch of this repository contains the code in production. The reposiroy does not contain a trained model, it contains the code necessary to build the full training / inference pipeline. When someone wants to add a new feature he branches the repository. He can then test the modification in a dedicated environment. When ready he merges his code which trigers the following workflow:
- Test the code
- Build a python package
- Send the package to a dedicated server
- Use the package to train / evaluate the model 
  - Use Docker to guaranty compatibility between train / inference environments
  - Output a joblib file that contains the entire pipeline
  - Compare the new model to the one already in production on validation data
- If the new model is better: Serve it (replace the previous one)

I was inspired by this Google's blog post on the topic: [MLOps: Continuous delivery and automation pipelines in machine learning](https://cloud.google.com/solutions/machine-learning/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning#mlops_level_2_cicd_pipeline_automation).

![Elements for ML systems. Adapted from Hidden Technical Debt in Machine Learning Systems.](https://cloud.google.com/solutions/images/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning-1-elements-of-ml.png)
**Figure 1.** Elements for ML systems. [Adapted from Hidden Technical Debt in Machine Learning Systems](https://papers.nips.cc/paper/2015/file/86df7dcfd896fcaf2674f757a2463eba-Paper.pdf).

#### Sources:
- [MLOps: Continuous delivery and automation pipelines in machine learning](https://cloud.google.com/solutions/machine-learning/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning#mlops_level_2_cicd_pipeline_automation)
- [Hidden Technical Debt in Machine Learning Systems](https://papers.nips.cc/paper/2015/file/86df7dcfd896fcaf2674f757a2463eba-Paper.pdf)
