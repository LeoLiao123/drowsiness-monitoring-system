from ubidots import ApiClient
from config.settings import UBIDOTS_API_KEY

class DataPublisher:
    def __init__(self):
        """Initialize Ubidots API client"""
        self.api = ApiClient(token=UBIDOTS_API_KEY)
        self.variable = None
        
    def set_variable(self, variable_id):
        """
        Set the Ubidots variable for publishing
        
        Args:
            variable_id (str): Ubidots variable ID
        """
        self.variable = self.api.get_variable(variable_id)
        
    def publish_drowsiness_rate(self, rate):
        """
        Publish drowsiness rate to Ubidots
        
        Args:
            rate (float): Drowsiness rate value
        """
        if self.variable:
            self.variable.save_value({'value': rate * 100})