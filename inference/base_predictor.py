class BasePredictor:
    """Base class. All predictors must implement predict()."""

    def predict(self, model_input):
        raise NotImplementedError
