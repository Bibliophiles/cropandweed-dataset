#!/usr/bin/env python3

import logging
import os
from pathlib import Path
from datetime import datetime
import sys
from typing import Optional

class LoggingManager:
    """Manages the logging setup for the entire application."""
    
    # Class variable to store the global instance
    _instance = None

    def __init__(self, output_dir: str, model_name: str, log_level: str = "INFO"):
        """
        Initializes the logging manager.
        
        Args:
            output_dir: The base directory for all experiment outputs.
            model_name: The name of the model, used to create a unique log file name.
            log_level: The minimum logging level to display.
        """
        self.output_dir = Path(output_dir)
        self.model_name = model_name
        self.log_level = log_level
        self._root_logger_setup = False
        self._log_file_path = None

        # Ensure the output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a unique log file for this experiment
        self._create_log_file()
        
        # Set as global instance
        LoggingManager._instance = self
        
    def _create_log_file(self):
        """Creates a unique log file for the current experiment run."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = self.output_dir / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        self._log_file_path = log_dir / f"{self.model_name}_{timestamp}.log"
        print(f"📄 Logging to file: {self._log_file_path}")

    def setup_root_logger(self):
        """Configures the root logger with a console and file handler."""
        if self._root_logger_setup:
            return

        # Get the root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, self.log_level.upper()))
        
        # Clear existing handlers to prevent duplicate output
        if root_logger.hasHandlers():
            root_logger.handlers.clear()

        # Create a formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
        
        # File handler
        file_handler = logging.FileHandler(self._log_file_path)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        
        # Set the flag to prevent re-setup
        self._root_logger_setup = True
        
        # Test the logger
        root_logger.info("Logging system initialized.")

    def get_logger(self, name: Optional[str] = None) -> logging.Logger:
        """
        Returns a logger instance. If the root logger has not been set up, it sets it up first.
        
        Args:
            name: The name of the logger. Defaults to the root logger.
            
        Returns:
            A configured logging.Logger instance.
        """
        if not self._root_logger_setup:
            self.setup_root_logger()
            
        return logging.getLogger(name)
    
    @classmethod
    def get_logger_instance(cls, name: str) -> logging.Logger:
        """Class method to get logger from global instance"""
        if cls._instance is None:
            # Create a basic logger if no manager instance exists
            logger = logging.getLogger(name)
            if not logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                handler.setFormatter(formatter)
                logger.addHandler(handler)
                logger.setLevel(logging.INFO)
            return logger
        else:
            return cls._instance.get_logger(name)

# Expose a simple setup function for convenience in other modules
def setup_logger(name: str) -> logging.Logger:
    """Helper function to get a logger instance from the global manager."""
    return LoggingManager.get_logger_instance(name)