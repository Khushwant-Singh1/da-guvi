class SummaryStats:
    def __init__(self, data):
        self.data = data

    def get_summary(self):
        return self.data.describe()

    def get_insights(self):
        # Placeholder for insights logic
        return {}
