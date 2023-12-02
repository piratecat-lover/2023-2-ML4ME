class InfraredObjectDetection:
    def __init__(self):
        self.model = SVC()

    def preprocess_image(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray

    def extract_features(self, image):
        hog = cv2.HOGDescriptor()
        features = hog.compute(image)
        return features

    def train(self, X, y):
        self.model.fit(X, y)

    def detect_people(self, image):
        processed_image = self.preprocess_image(image)
        features = self.extract_features(processed_image)
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.1, random_state=42)
        self.train(X_train, y_train)

        window_size = (50, 50)
        step_size = 5

        predictions = []
        for y in range(0, processed_image.shape[0] - window_size[1], step_size):
            for x in range(0, processed_image.shape[1] - window_size[0], step_size):
                window = processed_image[y:y + window_size[1], x:x + window_size[0]]
                window_features = self.extract_features(window)
                if self.model.predict(window_features.reshape(1, -1)):
                    predictions.append((x, y, x + window_size[0], y + window_size[1]))
                
        return predictions

# Usage:
# Instantiate the class and use the detect_people method on an infrared image
detector = InfraredObjectDetection()
image = cv2.imread('path_to_infrared_image.jpg')
detected_boxes = detector.detect_people(image)
print(detected_boxes)