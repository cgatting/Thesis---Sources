"""
Main entry point for RefScore academic application GUI.

This module provides the main application launcher and entry point
for the graphical user interface.
"""

import sys
import logging
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt

from .gui.main_window import MainWindow
from .utils.config import Config
from .utils.exceptions import RefScoreError


def setup_logging(level: int = logging.INFO) -> None:
    """
    Set up application logging.
    
    Args:
        level: Logging level
    """
    logging.basicConfig(
        level=level,
        format='[%(asctime)s] %(levelname)s - %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def create_application() -> QApplication:
    """
    Create and configure the Qt application.
    
    Returns:
        Configured QApplication instance
    """
    # Enable high DPI support
    if hasattr(Qt, 'AA_EnableHighDpiScaling'):
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    app = QApplication(sys.argv)
    app.setApplicationName("RefScore Academic")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("Academic Project")
    app.setOrganizationDomain("university.edu")
    
    return app


def main() -> int:
    """
    Main application entry point.
    
    Returns:
        Application exit code
    """
    try:
        # Set up logging
        setup_logging()
        logger = logging.getLogger(__name__)
        logger.info("Starting RefScore Academic Application")
        
        # Load configuration
        config = Config()
        logger.info("Configuration loaded")
        
        # Create Qt application
        app = create_application()
        
        # Create and show main window
        main_window = MainWindow(config)
        main_window.show()
        
        logger.info("Application started successfully")
        
        # Run the application
        return app.exec_()
        
    except RefScoreError as e:
        logger.error(f"RefScore error: {e}")
        return 1
    except Exception as e:
        logger.critical(f"Unexpected error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())