import logging

# Set up the logger
logger = logging.getLogger('elo_bench_logger')
logger.setLevel(logging.DEBUG)

# Create a console handler and set its level
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# # Create formatter
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# # Add formatter to console_handler
# console_handler.setFormatter(formatter)

# Add console_handler to logger
logger.addHandler(console_handler)