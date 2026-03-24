class Predictor:
    def predict(self, image_path: str) -> dict:
        # TODO: replace with real model inference
        return {
            "defect_detected": False,
            "confidence": 0.0,
        }

    def predict_bytes(self, image_bytes: bytes) -> dict:
        # TODO: replace with real model inference
        return {
            "defect_detected": False,
            "confidence": 0.0,
        }


predictor = Predictor()