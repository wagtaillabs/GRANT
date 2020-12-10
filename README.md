# GRANT

Graft, Reassemble, Answer delta, Neighbour sensitivity, Training delta (GRANT)

GRANT has been created by Wagtail Labs to remove the guesswork in using tree based models, assisting the creation of faster, more accurate models with satisfying explanations. It provides a deep understanding of the model's internal behaviour, shows prediction sensitivities, and helps data scientists improve inaccurate predictions.

For a detailed introduction to the theory of GRANT visit [wagtaillabs.com](https://wagtaillabs.com).

For a detailed introduction to the implementation of GRANT see the GRANT_walkthrough.ipynb notebook in this repo.

## Installation

Clone the WagtailLabs/GRANT/grant.py file.

## Usage

For a detailed walkthrough of the usage see the GRANT.ipynb notebook in this repo.

```python
from WagtailLabs.grant import grant

grant_rf = grant(rf, train_features, train_labels)
grant_rf.graft()
grafted_df = grant_rf.get_graft() #Returns the Graft (dataset containing all decision boundaries) of the Tree Ensemble

grafted_dt = grant_rf.reassemble() #Returns a decision tree that produces the exact same results of the Tree Ensemble in less time

grant_rf.amalgamate(threshold)
amalgamated_df = grant_rf.get_amalgamated() #Returns a pruned (simplified) copy of the Graft where contigious decision boundaries with a prediction difference less than the supplied threshold are merged

sensitivities = grant_rf.neighbour_sensitivity(val_feature) #Returns the change required in explanatory data required to reach each neighbouring decision boundary

trainer_delta = grant_rf.training_delta_trainer(trainer_feature, trainer_label) #Returns the incremental change in result from a given training record to all predictions

grant_rf.training_delta_trainee(trainee_feature, trainee_label) #Returns the incremental change in result from each training record that contributed any given prediction

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[Apache License 2.0](https://choosealicense.com/licenses/apache-2.0/)
