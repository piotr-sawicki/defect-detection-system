class Predictor:
    def predict(self, image_id: str) -> dict:
        # TODO: replace with real model inference
        return {
            "defect_detected": False,
            "confidence": 0.0,
        }


predictor = Predictor()